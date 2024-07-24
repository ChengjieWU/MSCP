from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .hiql import JointTrainAgent
from .hiql import compute_value_loss, compute_actor_loss, compute_high_actor_loss, compute_high_actor_loss_mixed
from .hiql import compute_value_loss_mixed, conservative_value_loss, conservative_value_loss_v2, conservative_value_loss_v3
from .hiql import actor_loss_with_L2_distance
from .hiql import calculate_guiding_reward, compute_guiding_value_loss_mixed, compute_guiding_value_loss
from .hiql import compute_vae_loss


def compute_fast_high_actor_loss(agent, batch, network_params, seed: jax.random.PRNGKey = None):
    key1, key2, key3 = jax.random.split(seed, 3)

    cur_goals = batch['high_goals']
    v1, v2 = agent.network(batch['observations'], cur_goals, method='value', seed=key1)
    nv1, nv2 = agent.network(batch['fast_high_targets'], cur_goals, method='value', seed=key2)
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2

    adv = nv - v
    exp_a = jnp.exp(adv * agent.config['high_temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=True, method='fast_high_actor', params=network_params)
    if agent.config['use_rep']:
        target = agent.network(targets=batch['fast_high_targets'], bases=batch['observations'], method='value_goal_encoder', seed=key3)
    else:
        target = batch['fast_high_targets'] - batch['observations']
    log_probs = dist.log_prob(target)
    actor_loss = -(exp_a * log_probs).mean()

    return actor_loss, {
        'high_actor_loss': actor_loss,
        'high_adv': adv.mean(),
        'high_bc_log_probs': log_probs.mean(),
        'high_adv_median': jnp.median(adv),
        'high_mse': jnp.mean((dist.mode() - target)**2),
        'high_scale': dist.scale_diag.mean(),
    }


def compute_fast_high_actor_loss_mixed(agent, batch, dataset_batch, network_params, seed: jax.random.PRNGKey = None):
    key1, key2 = jax.random.split(seed, 2)

    info = {}
    actor_loss, actor_info = compute_fast_high_actor_loss(agent, batch, network_params, seed=key1)
    for k, v in actor_info.items():
        info[f'online/{k}'] = v

    dataset_actor_loss, dataset_actor_info = compute_fast_high_actor_loss(agent, dataset_batch, network_params, seed=key2)
    for k, v in dataset_actor_info.items():
        info[f'offline/{k}'] = v
    
    loss = (actor_loss + dataset_actor_loss) * 0.5
    info['fast_high_actor_loss'] = loss

    return loss, info


class SCPIQLAgent(JointTrainAgent):

    @partial(jax.jit, static_argnames=('num_samples',))
    def sample_fast_high_actions(agent,
                                 observations: np.ndarray,
                                 goals: np.ndarray,
                                 *,
                                 seed: jax.random.PRNGKey,
                                 temperature: float = 1.0,
                                 num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, temperature=temperature, method='fast_high_actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        return actions

    def get_one_step_subgoal(agent, observations, goals, temperature, seed):
        if agent.config['one_step_mode'] == "v2":
            immediate_subgoal = agent.sample_fast_high_actions(observations, goals, seed=seed, temperature=temperature)
            if agent.config['use_rep']:
                immediate_subgoal = immediate_subgoal / jnp.linalg.norm(immediate_subgoal, axis=-1, keepdims=True) * jnp.sqrt(immediate_subgoal.shape[-1])
            else:
                immediate_subgoal = observations + immediate_subgoal
            return immediate_subgoal
        raise NotImplementedError(f"One step mode {agent.config['one_step_mode']} not implemented")

    @partial(jax.jit, static_argnames=('value_update', 'actor_update', 'high_actor_update'))
    def pretrain_update(agent, pretrain_batch, seed=None, value_update=True, actor_update=True, high_actor_update=True):
        value_seed, actor_seed, high_actor_seed, fast_high_actor_seed, vae_seed = jax.random.split(seed, 5)
        def loss_fn(network_params):
            info = {}

            # Value
            if value_update:
                value_loss, value_info = compute_value_loss(agent, pretrain_batch, network_params, seed=value_seed)
                for k, v in value_info.items():
                    info[f'value/{k}'] = v
            else:
                value_loss = 0.
            
            # Actor
            if actor_update:
                actor_loss, actor_info = compute_actor_loss(agent, pretrain_batch, network_params, seed=actor_seed)
                for k, v in actor_info.items():
                    info[f'actor/{k}'] = v
            else:
                actor_loss = 0.

            # High Actor
            if high_actor_update and agent.config['use_waypoints']:
                high_actor_loss, high_actor_info = compute_high_actor_loss(agent, pretrain_batch, network_params, seed=high_actor_seed)
                for k, v in high_actor_info.items():
                    info[f'high_actor/{k}'] = v
            else:
                high_actor_loss = 0.

            # Fast High Actor
            if high_actor_update and agent.config['use_waypoints']:
                fast_high_actor_loss, fast_high_actor_info = compute_fast_high_actor_loss(agent, pretrain_batch, network_params, seed=fast_high_actor_seed)
                for k, v in fast_high_actor_info.items():
                    info[f'fast_high_actor/{k}'] = v
            else:
                fast_high_actor_loss = 0.
            
            # VAE reconstruction
            if agent.config['use_vae']:
                vae_loss, vae_info = compute_vae_loss(agent, pretrain_batch, network_params, KL_coefficient=agent.config['vae_KL_coefficient'], seed=vae_seed)
                for k, v in vae_info.items():
                    info[f'vae/{k}'] = v
            else:
                vae_loss = 0.

            loss = value_loss + actor_loss + high_actor_loss + fast_high_actor_loss + vae_loss * agent.config['vae_coefficient']
            return loss, info
        
        if value_update:
            new_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
            )
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        if value_update:
            params = unfreeze(new_network.params)
            params['networks_target_value'] = new_target_params
            new_network = new_network.replace(params=freeze(params))
        
        return agent.replace(network=new_network), info

    @partial(jax.jit, static_argnames=('actor_supervision', 'high_actor_update', 'value_update'))
    def finetune_mixed_update(
        agent,
        batch,
        dataset_batch,
        seed: jax.random.PRNGKey,
        actor_supervision: str,
        high_actor_update: bool,
        value_update: bool = True,
    ):
        def loss_fn(network_params, seed):
            seed, key_gv, key_v, key_actor, key_high_actor, key_fast_high_actor = jax.random.split(seed, 6)

            info = {}
            # Guiding Value
            if agent.config['guiding_reward']:
                seed, sub_seed = jax.random.split(seed)
                immediate_subgoal = agent.get_one_step_subgoal(batch['observations'], batch['high_goals'], seed=sub_seed, temperature=0.)
                seed, sub_seed = jax.random.split(seed)
                immediate_subgoal_dataset = agent.get_one_step_subgoal(dataset_batch['observations'], dataset_batch['high_goals'], seed=sub_seed, temperature=0.)
                if 'antmaze' in agent.config['env_name'] and not agent.config['use_rep']:
                    xy_guiding_reward = calculate_guiding_reward(False, batch['next_observations'][:, :2], immediate_subgoal[:, :2])   # always calculate it for logging
                    xy_guiding_reward_dataset = calculate_guiding_reward(False, dataset_batch['next_observations'][:, :2], immediate_subgoal_dataset[:, :2])
                if agent.config['guiding_reward_xy']:
                    guiding_reward = xy_guiding_reward
                    guiding_reward_dataset = xy_guiding_reward_dataset
                else:
                    guiding_reward = calculate_guiding_reward(agent.config['use_rep'], batch['next_observations'], immediate_subgoal,
                                                              observations=batch['observations'], agent=agent)
                    guiding_reward_dataset = calculate_guiding_reward(agent.config['use_rep'], dataset_batch['next_observations'], immediate_subgoal_dataset,
                                                                      observations=dataset_batch['observations'], agent=agent)
                guiding_v_loss, guiding_v_info = compute_guiding_value_loss_mixed(agent, batch, dataset_batch, guiding_reward, guiding_reward_dataset, network_params, seed=key_gv)
                for k, v in guiding_v_info.items():
                    info[f'guiding_v/{k}'] = v
                if 'antmaze' in agent.config['env_name'] and not agent.config['use_rep']:
                    info[f'guiding_v/online/xy_guiding_r'] = xy_guiding_reward.mean()
                    info[f'guiding_v/offline/xy_guiding_r'] = xy_guiding_reward_dataset.mean()
            else:
                guiding_v_loss = 0.

            # Value
            if value_update and agent.config['hiql_actor_loss_coefficient'] > 0:
                if agent.config['mixed_finetune_value_loss'] == 'hiql':
                    value_loss, value_info = compute_value_loss_mixed(agent, batch, dataset_batch, network_params, seed=key_v)
                elif agent.config['mixed_finetune_value_loss'] == 'hiql_cql':
                    value_loss, value_info = conservative_value_loss(agent, batch, dataset_batch, network_params, seed=key_v)
                elif agent.config['mixed_finetune_value_loss'] == 'hiql_cql_v2':
                    value_loss, value_info = conservative_value_loss_v2(agent, batch, dataset_batch, network_params, seed=key_v)
                elif agent.config['mixed_finetune_value_loss'] == 'hiql_cql_v3':
                    value_loss, value_info = conservative_value_loss_v3(agent, batch, dataset_batch, network_params, seed=key_v)
                else:
                    raise NotImplementedError
                for k, v in value_info.items():
                    info[f'value/{k}'] = v
            else:
                value_loss = 0.

            # Actor
            if actor_supervision == 'value':
                if agent.config['guiding_reward']:
                    actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, guiding_r=guiding_reward, seed=key_actor)
                else:
                    actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, seed=key_actor)
            elif actor_supervision == 'subgoal_L2':
                actor_loss, actor_info = actor_loss_with_L2_distance(agent, batch, network_params, seed=key_actor)
            else:
                raise NotImplementedError
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            # High Actor
            if high_actor_update and agent.config['use_waypoints']:
                high_actor_loss, high_actor_info = compute_high_actor_loss_mixed(agent, batch, dataset_batch, network_params, seed=key_high_actor)
                for k, v in high_actor_info.items():
                    info[f'high_actor/{k}'] = v
            else:
                high_actor_loss = 0.
            
            # Fast High Actor
            if high_actor_update and agent.config['use_waypoints']:
                fast_high_actor_loss, fast_high_actor_info = compute_fast_high_actor_loss_mixed(agent, batch, dataset_batch, network_params, seed=key_fast_high_actor)
                for k, v in fast_high_actor_info.items():
                    info[f'fast_high_actor/{k}'] = v
            else:
                fast_high_actor_loss = 0.
            
            loss = value_loss + actor_loss + high_actor_loss + guiding_v_loss + fast_high_actor_loss
            return loss, info


        new_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
        )

        seed, sub_seed = jax.random.split(seed)
        new_network, info = agent.network.apply_loss_fn(loss_fn=partial(loss_fn, seed=sub_seed), has_aux=True)

        params = unfreeze(new_network.params)
        params['networks_target_value'] = new_target_params
        new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info

    @partial(jax.jit, static_argnames=('actor_supervision', 'high_actor_update'))
    def finetune_full_online_update(
        agent,
        batch,
        seed: jax.random.PRNGKey,
        actor_supervision: str,
        high_actor_update: bool,
    ):
        def loss_fn(network_params, seed):
            seed, key_gv, key_v, key_actor, key_high_actor, key_fast_high_actor = jax.random.split(seed, 6)

            info = {}
            # Guiding Value
            if agent.config['guiding_reward']:
                seed, sub_seed = jax.random.split(seed)
                immediate_subgoal = agent.get_one_step_subgoal(batch['observations'], batch['high_goals'], seed=sub_seed, temperature=0.)
                if 'antmaze' in agent.config['env_name'] and not agent.config['use_rep']:
                    xy_guiding_reward = calculate_guiding_reward(False, batch['next_observations'][:, :2], immediate_subgoal[:, :2])   # always calculate it for logging
                if agent.config['guiding_reward_xy']:
                    guiding_reward = xy_guiding_reward
                else:
                    guiding_reward = calculate_guiding_reward(agent.config['use_rep'], batch['next_observations'], immediate_subgoal,
                                                              observations=batch['observations'], agent=agent)
                guiding_v_loss, guiding_v_info = compute_guiding_value_loss(agent, batch, guiding_reward, network_params, expectile_param=agent.config['guiding_v_expectile'], seed=key_gv)
                for k, v in guiding_v_info.items():
                    info[f'guiding_v/{k}'] = v
                if 'antmaze' in agent.config['env_name'] and not agent.config['use_rep']:
                    info[f'guiding_v/online/xy_guiding_r'] = xy_guiding_reward.mean()
            else:
                guiding_v_loss = 0.

            # Value
            if agent.config['hiql_actor_loss_coefficient'] > 0:
                if agent.config['mixed_finetune_value_loss'] == 'hiql':
                    value_loss, value_info = compute_value_loss(agent, batch, network_params, seed=key_v)
                elif agent.config['mixed_finetune_value_loss'] == 'hiql_cql_v3':
                    value_loss, value_info = compute_value_loss(agent, batch, network_params, expectile_param=0.5, seed=key_v)
                else:
                    raise NotImplementedError
                for k, v in value_info.items():
                    info[f'value/{k}'] = v
            else:
                value_loss = 0.

            # Actor
            if actor_supervision == 'value':
                if agent.config['guiding_reward']:
                    actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, guiding_r=guiding_reward, seed=key_actor)
                else:
                    actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, seed=key_actor)
            elif actor_supervision == 'subgoal_L2':
                actor_loss, actor_info = actor_loss_with_L2_distance(agent, batch, network_params, seed=key_actor)
            else:
                raise NotImplementedError
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            # High Actor
            if high_actor_update and agent.config['use_waypoints']:
                high_actor_loss, high_actor_info = compute_high_actor_loss(agent, batch, network_params, seed=key_high_actor)
                for k, v in high_actor_info.items():
                    info[f'high_actor/{k}'] = v
            else:
                high_actor_loss = 0.

            # Fast High Actor
            if high_actor_update and agent.config['use_waypoints']:
                fast_high_actor_loss, fast_high_actor_info = compute_fast_high_actor_loss(agent, batch, network_params, seed=key_fast_high_actor)
                for k, v in fast_high_actor_info.items():
                    info[f'fast_high_actor/{k}'] = v
            else:
                fast_high_actor_loss = 0.
            
            loss = value_loss + actor_loss + high_actor_loss + guiding_v_loss + fast_high_actor_loss
            return loss, info


        new_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
        )

        seed, sub_seed = jax.random.split(seed)
        new_network, info = agent.network.apply_loss_fn(loss_fn=partial(loss_fn, seed=sub_seed), has_aux=True)

        params = unfreeze(new_network.params)
        params['networks_target_value'] = new_target_params
        new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info
