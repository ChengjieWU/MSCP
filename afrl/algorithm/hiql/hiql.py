from functools import partial

from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
from jaxrl_m.common import TrainState

from flax.core import freeze, unfreeze

from .. import iql


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


def compute_actor_loss(agent, batch, network_params, ensemble_agents=None, ensemble_type="mean", guiding_r=None, seed: jax.random.PRNGKey = None):
    if agent.config['use_waypoints']:  # Use waypoint states as goals (for hierarchical policies)
        cur_goals = batch['low_goals']
    else:  # Use randomized last observations as goals (for flat policies)
        cur_goals = batch['high_goals']
    
    if agent.config['hiql_actor_loss_coefficient'] > 0:
        def calc_adv(net, key):
            sub_key1, sub_key2 = jax.random.split(key)
            v1, v2 = net(batch['observations'], cur_goals, method='value', seed=sub_key1)
            nv1, nv2 = net(batch['next_observations'], cur_goals, method='value', seed=sub_key2)
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2
            adv = nv - v
            return adv
        
        if ensemble_agents is None:
            seed, key = jax.random.split(seed)
            adv = calc_adv(agent.network, key)
        else:
            adv_list = []   # agent should already be included in ensemble_agents
            for ensemble_agent in ensemble_agents:
                seed, key = jax.random.split(seed)
                adv_list.append(calc_adv(ensemble_agent.network, key))
            if ensemble_type == "mean":
                adv = jnp.stack(adv_list, axis=1).mean(axis=1)  # use average advantage over ensemble agents (including self)
            elif ensemble_type == "min":
                adv = jnp.stack(adv_list, axis=1).min(axis=1)
            else:
                raise NotImplementedError(f"Ensemble type {ensemble_type} not implemented")
    else:
        adv = 0.
    
    if agent.config['guiding_reward']:
        # assert agent.config['use_waypoints'], f"Guiding reward requires hierarchical policy"
        assert guiding_r is not None, f"Guiding reward requires guiding_r"
        seed, key = jax.random.split(seed)
        Vg = agent.network(batch['observations'], cur_goals, method='guiding_v', seed=key)
        adv_g = guiding_r - Vg
    else:
        adv_g = 0.

    exp_a = jnp.exp((adv * agent.config['hiql_actor_loss_coefficient'] + agent.config['guiding_reward'] * adv_g) * agent.config['temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    if agent.config['use_waypoints']:
        goal_rep_grad = agent.config['policy_train_rep']
        state_rep_grad = True if not agent.config['policy_share_value_state'] else agent.config['policy_train_rep']
    else:
        goal_rep_grad = True
        state_rep_grad = True
    seed, key = jax.random.split(seed)
    dist = agent.network(batch['observations'], cur_goals, state_rep_grad=state_rep_grad, goal_rep_grad=goal_rep_grad, method='actor', params=network_params, seed=key)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()

    info = {
        'actor_loss': actor_loss,
        'adv': jnp.mean(adv),
        'bc_log_probs': log_probs.mean(),
        'adv_median': jnp.median(adv),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
        'adv_x_coeff': jnp.mean(adv * agent.config['hiql_actor_loss_coefficient']),
    }
    if agent.config['guiding_reward']:
        info['adv_g'] = adv_g.mean()
    return actor_loss, info


def actor_loss_with_L2_distance(agent, batch, network_params, seed: jax.random.PRNGKey = None):
    assert agent.config['use_waypoints']    # only considering the hierarchical setting
    assert 'antmaze' in agent.config['env_name'], f'Only support antmaze for now'
    cur_goals = batch['low_goals']

    d = jnp.linalg.norm(batch['observations'][:, :2] - cur_goals[:, :2], axis=1)
    nd = jnp.linalg.norm(batch['next_observations'][:, :2] - cur_goals[:, :2], axis=1)
    reward = d - nd

    exp_a = jnp.exp(reward * agent.config['temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    if agent.config['use_waypoints']:
        goal_rep_grad = agent.config['policy_train_rep']
        state_rep_grad = True if not agent.config['policy_share_value_state'] else agent.config['policy_train_rep']
    else:
        goal_rep_grad = True
        state_rep_grad = True
    dist = agent.network(batch['observations'], cur_goals, state_rep_grad=state_rep_grad, goal_rep_grad=goal_rep_grad, method='actor', params=network_params, seed=seed)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()

    return actor_loss, {
        'actor_loss': actor_loss,
        'bc_log_probs': log_probs.mean(),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
        'reward': reward.mean(),
        'reward max': reward.max(),
        'reward min': reward.min(),
    }


def compute_high_actor_loss(agent, batch, network_params, seed: jax.random.PRNGKey = None):
    key1, key2, key3 = jax.random.split(seed, 3)

    cur_goals = batch['high_goals']
    v1, v2 = agent.network(batch['observations'], cur_goals, method='value', seed=key1)
    nv1, nv2 = agent.network(batch['high_targets'], cur_goals, method='value', seed=key2)
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2

    adv = nv - v
    exp_a = jnp.exp(adv * agent.config['high_temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=True, method='high_actor', params=network_params)
    if agent.config['use_rep']:
        target = agent.network(targets=batch['high_targets'], bases=batch['observations'], method='value_goal_encoder', seed=key3)
    else:
        target = batch['high_targets'] - batch['observations']
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


def compute_high_actor_loss_mixed(agent, batch, dataset_batch, network_params, seed: jax.random.PRNGKey = None):
    key1, key2 = jax.random.split(seed, 2)

    info = {}
    actor_loss, actor_info = compute_high_actor_loss(agent, batch, network_params, seed=key1)
    for k, v in actor_info.items():
        info[f'online/{k}'] = v
    
    dataset_actor_loss, dataset_actor_info = compute_high_actor_loss(agent, dataset_batch, network_params, seed=key2)
    for k, v in dataset_actor_info.items():
        info[f'offline/{k}'] = v
    
    loss = (actor_loss + dataset_actor_loss) * 0.5
    info['high_actor_loss'] = loss

    return loss, info


def compute_value_loss(agent, batch, network_params, expectile_param: float = None, repr_grad: bool = True, seed: jax.random.PRNGKey = None):
    key1, key2, key3 = jax.random.split(seed, 3)

    # # masks are 0 if terminal, 1 otherwise
    # batch['masks'] = 1.0 - batch['rewards']
    # # rewards are 0 if terminal, -1 otherwise
    # batch['rewards'] = batch['rewards'] - 1.0
    expectile_param = expectile_param if expectile_param is not None else agent.config['pretrain_expectile']

    (next_v1, next_v2) = agent.network(batch['next_observations'], batch['goals'], method='target_value', seed=key1)
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    (v1_t, v2_t) = agent.network(batch['observations'], batch['goals'], method='target_value', seed=key2)
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    (v1, v2) = agent.network(batch['observations'], batch['goals'], method='value', params=network_params, state_rep_grad=repr_grad, goal_rep_grad=repr_grad, seed=key3)

    value_loss1 = expectile_loss(adv, q1 - v1, expectile_param).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, expectile_param).mean()
    value_loss = value_loss1 + value_loss2

    advantage = adv
    return value_loss, {
        'value_loss': value_loss,
        'v max': v1.max(),
        'v min': v1.min(),
        'v mean': v1.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
    }


def compute_value_loss_mixed(agent, batch, dataset_batch, network_params, seed: jax.random.PRNGKey = None):
    key1, key2 = jax.random.split(seed, 2)

    info = {}
    repr_grad = agent.config['grad_value_repr']

    value_loss, value_info = compute_value_loss(agent, batch, network_params, repr_grad=repr_grad, seed=key1)
    for k, v in value_info.items():
        info[f'online/{k}'] = v

    dataset_loss, dataset_info = compute_value_loss(agent, dataset_batch, network_params, repr_grad=repr_grad, seed=key2)
    for k, v in dataset_info.items():
        info[f'offline/{k}'] = v
    
    loss = (value_loss + dataset_loss) * 0.5
    info['value_loss'] = loss
    
    return loss, info


def conservative_value_loss(agent, batch, dataset_batch, network_params, seed: jax.random.PRNGKey = None):
    key1, key2, key3 = jax.random.split(seed, 3)

    info = {}
    repr_grad = agent.config['grad_value_repr']

    dataset_loss, dataset_info = compute_value_loss(agent, dataset_batch, network_params, repr_grad=repr_grad, seed=key1)
    for k, v in dataset_info.items():
        info[f'offline/{k}'] = v
    
    dataset_v1, dataset_v2 = agent.network(dataset_batch['observations'], batch['goals'], method='value', params=network_params,
                                           state_rep_grad=repr_grad, goal_rep_grad=repr_grad, seed=key2)
    dataset_v = (dataset_v1 + dataset_v2) * 0.5
    online_v1, online_v2 = agent.network(batch['observations'], batch['goals'], method='value', params=network_params,
                                         state_rep_grad=repr_grad, goal_rep_grad=repr_grad, seed=key3)
    online_v = (online_v1 + online_v2) * 0.5
    conservative_loss = -dataset_v.mean() + jax.scipy.special.logsumexp(online_v) - jnp.log(online_v.shape[0])

    loss = dataset_loss + agent.config['cql_alpha'] * conservative_loss
    info['value_loss'] = loss
    info['conservative_loss'] = conservative_loss
    info['dataset_v'] = dataset_v.mean()
    info['online_v'] = online_v.mean()
    
    return loss, info


def conservative_value_loss_v2(agent, batch, dataset_batch, network_params, seed: jax.random.PRNGKey = None):
    key1, key2, key3 = jax.random.split(seed, 3)

    info = {}
    repr_grad = agent.config['grad_value_repr']

    dataset_loss, dataset_info = compute_value_loss(agent, dataset_batch, network_params, repr_grad=repr_grad, seed=key1)
    for k, v in dataset_info.items():
        info[f'offline/{k}'] = v
    
    dataset_v1, dataset_v2 = agent.network(dataset_batch['observations'], batch['goals'], method='value', params=network_params,
                                           state_rep_grad=repr_grad, goal_rep_grad=repr_grad, seed=key2)
    dataset_v = (dataset_v1 + dataset_v2) * 0.5
    online_v1, online_v2 = agent.network(batch['observations'], batch['goals'], method='value', params=network_params,
                                         state_rep_grad=repr_grad, goal_rep_grad=repr_grad, seed=key3)
    online_v = (online_v1 + online_v2) * 0.5
    conservative_loss = -dataset_v.mean() + online_v.mean()

    loss = dataset_loss + agent.config['cql_alpha'] * conservative_loss
    info['value_loss'] = loss
    info['conservative_loss'] = conservative_loss
    info['dataset_v'] = dataset_v.mean()
    info['online_v'] = online_v.mean()
    
    return loss, info


def conservative_value_loss_v3(agent, batch, dataset_batch, network_params, seed: jax.random.PRNGKey = None):
    key1, key2 = jax.random.split(seed, 2)

    info = {}
    repr_grad = agent.config['grad_value_repr']

    value_loss, value_info = compute_value_loss(agent, batch, network_params, expectile_param=0.5, repr_grad=repr_grad, seed=key1)
    for k, v in value_info.items():
        info[f'online/{k}'] = v

    dataset_loss, dataset_info = compute_value_loss(agent, dataset_batch, network_params, repr_grad=repr_grad, seed=key2)
    for k, v in dataset_info.items():
        info[f'offline/{k}'] = v
    
    loss = (value_loss + dataset_loss) * 0.5
    info['value_loss'] = loss
    
    return loss, info


def compute_guiding_value_loss(agent, batch, guiding_r, network_params, expectile_param: float = None, seed: jax.random.PRNGKey = None):
    if agent.config['use_waypoints']:  # Use waypoint states as goals (for hierarchical policies)
        cur_goals = batch['low_goals']
    else:  # Use randomized last observations as goals (for flat policies)
        cur_goals = batch['high_goals']

    expectile_param = expectile_param if expectile_param is not None else agent.config['pretrain_expectile']

    # Stop gradient to avoid influencing pre-trained state representation & goal representation
    gv = agent.network(batch['observations'], cur_goals, state_rep_grad=False, goal_rep_grad=False, method='guiding_v', params=network_params, seed=seed)
    diff = guiding_r - gv

    gv_loss = expectile_loss(diff, diff, expectile_param).mean()
    return gv_loss, {
        'guiding_v_loss': gv_loss,
        'guiding_r': guiding_r.mean(),
        'guiding_v': gv.mean(),
    }


def compute_guiding_value_loss_mixed(agent, batch, dataset_batch, guiding_r, guiding_r_dataset, network_params, seed: jax.random.PRNGKey = None):
    key1, key2 = jax.random.split(seed, 2)
    
    info = {}
    if agent.config['guiding_v_expectile'] > 0:
        gv_loss, gv_info = compute_guiding_value_loss(agent, batch, guiding_r, network_params, expectile_param=agent.config['guiding_v_expectile'], seed=key1)
        for k, v in gv_info.items():
            info[f'online/{k}'] = v
    else:
        gv_loss = 0.
    
    if agent.config['guiding_v_dataset_expectile'] > 0:
        dataset_gv_loss, dataset_gv_info = compute_guiding_value_loss(agent, dataset_batch, guiding_r_dataset, network_params, expectile_param=agent.config['guiding_v_dataset_expectile'], seed=key2)
        for k, v in dataset_gv_info.items():
            info[f'offline/{k}'] = v
    else:
        dataset_gv_loss = 0.
    
    loss = (gv_loss + dataset_gv_loss) * 0.5
    info['guiding_v_loss'] = loss

    return loss, info


def calculate_guiding_reward(use_rep, next_observations, immediate_subgoal, observations=None, agent=None, seed: jax.random.PRNGKey = None):
    if use_rep:
        assert agent.config['use_waypoints'] and agent.config['use_rep']
        assert agent is not None and observations is not None
        actual_subgoal_repr = agent.network(targets=next_observations, bases=observations, method='value_goal_encoder', seed=seed)
    else:
        actual_subgoal_repr = next_observations
    assert actual_subgoal_repr.shape == immediate_subgoal.shape
    return -jnp.sqrt(jnp.sum((actual_subgoal_repr - immediate_subgoal) ** 2, axis=-1))


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def compute_vae_loss(agent, batch, network_params, KL_coefficient: float, seed: jax.random.PRNGKey = None):
    if agent.config['rep_type'] == 'state':
        targets = batch['goals']
    elif agent.config['rep_type'] == 'diff':
        targets = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, batch['goals'], batch['observations'])
    elif agent.config['rep_type'] == 'concat':
        targets = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), batch['goals'], batch['observations'])
    else:
        raise NotImplementedError
    targets = jax.tree_map(symlog, targets)
    
    x_recon, z_mean, z_logvar = agent.network(batch['observations'], batch['goals'], method='vae', params=network_params, seed=seed)
    recon_loss = jnp.reshape((x_recon - targets)**2, (batch['observations'].shape[0], -1)).sum(axis=-1).mean()
    kl_loss = -0.5 * jnp.reshape(1 + z_logvar - z_mean**2 - jnp.exp(z_logvar), (batch['observations'].shape[0], -1)).sum(axis=-1).mean()
    total_loss = recon_loss + kl_loss * KL_coefficient
    info = {
        'vae_loss': total_loss,
        'vae_recon_loss': recon_loss,
        'vae_kl_loss': kl_loss,
        'vae_logvar': z_logvar.mean(),
    }
    return total_loss, info


class JointTrainAgent(iql.IQLAgent):
    network: TrainState = None

    @partial(jax.jit, static_argnames=('value_update', 'actor_update', 'high_actor_update'))
    def pretrain_update(agent, pretrain_batch, seed=None, value_update=True, actor_update=True, high_actor_update=True):
        value_seed, actor_seed, high_actor_seed, vae_seed = jax.random.split(seed, 4)
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

            # VAE reconstruction
            if agent.config['use_vae']:
                vae_loss, vae_info = compute_vae_loss(agent, pretrain_batch, network_params, KL_coefficient=agent.config['vae_KL_coefficient'], seed=vae_seed)
                for k, v in vae_info.items():
                    info[f'vae/{k}'] = v
            else:
                vae_loss = 0.

            loss = value_loss + actor_loss + high_actor_loss + vae_loss * agent.config['vae_coefficient']

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

    @partial(jax.jit, static_argnames=('num_samples', 'low_dim_goals', 'discrete'))
    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       *,
                       low_dim_goals: bool = False,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None) -> jnp.ndarray:
        seed, sub_seed = jax.random.split(seed)
        dist = agent.network(observations, goals, low_dim_goals=low_dim_goals, temperature=temperature, method='actor', seed=sub_seed)
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        if not discrete:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnames=('num_samples',))
    def sample_high_actions(agent,
                            observations: np.ndarray,
                            goals: np.ndarray,
                            *,
                            seed: PRNGKey,
                            temperature: float = 1.0,
                            num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, temperature=temperature, method='high_actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        return actions

    @partial(jax.jit, static_argnames=('actor_supervision',))
    def finetune_update(agent, batch, seed: jax.random.PRNGKey, actor_supervision: str):
        key1, key2 = jax.random.split(seed, 2)
        def loss_fn(network_params):
            info = {}

            if actor_supervision == 'value':
                actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, seed=key1)
            elif actor_supervision == 'subgoal_L2':
                actor_loss, actor_info = actor_loss_with_L2_distance(agent, batch, network_params, seed=key2)
            else:
                raise NotImplementedError
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            loss = actor_loss
            return loss, info
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        
        return agent.replace(network=new_network), info

    def get_one_step_subgoal(agent, single_step_agent, observations, goals, temperature, seed):
        if agent.config['one_step_mode'] == "legacy":
            immediate_subgoal = single_step_agent.sample_high_actions(observations, goals, seed=seed, temperature=temperature)
            return immediate_subgoal
        elif agent.config['one_step_mode'] == "v2":
            immediate_subgoal = single_step_agent.sample_high_actions(observations, goals, seed=seed, temperature=temperature)
            if agent.config['use_rep']:
                immediate_subgoal = immediate_subgoal / jnp.linalg.norm(immediate_subgoal, axis=-1, keepdims=True) * jnp.sqrt(immediate_subgoal.shape[-1])
            else:
                immediate_subgoal = observations + immediate_subgoal
            return immediate_subgoal
        raise NotImplementedError(f"One step mode {agent.config['one_step_mode']} not implemented")

    @partial(jax.jit, static_argnames=('actor_supervision', 'high_actor_update', 'value_update', 'ensemble_type'))
    def finetune_mixed_update(
        agent,
        batch,
        dataset_batch,
        seed: jax.random.PRNGKey,
        actor_supervision: str,
        high_actor_update: bool,
        value_update: bool = True,
        ensemble_agents=None,
        ensemble_type="mean",
        single_step_agent=None,
    ):
        if ensemble_agents is not None:
            assert actor_supervision == 'value', f"Ensemble only supports using value to train actor"

        def loss_fn(network_params, seed):
            seed, key_gv, key_v, key_actor, key_high_actor = jax.random.split(seed, 5)

            info = {}
            # Guiding Value
            if agent.config['guiding_reward']:
                assert single_step_agent is not None, f"Guiding reward requires single step agent"
                seed, sub_seed = jax.random.split(seed)
                immediate_subgoal = agent.get_one_step_subgoal(single_step_agent, batch['observations'], batch['high_goals'], seed=sub_seed, temperature=0.)
                seed, sub_seed = jax.random.split(seed)
                immediate_subgoal_dataset = agent.get_one_step_subgoal(single_step_agent, dataset_batch['observations'], dataset_batch['high_goals'], seed=sub_seed, temperature=0.)
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
                if 'antmaze' in agent.config['env_name']:
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
                    actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, ensemble_agents, ensemble_type, guiding_reward, seed=key_actor)
                else:
                    actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, ensemble_agents, ensemble_type, seed=key_actor)
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

            loss = value_loss + actor_loss + high_actor_loss + guiding_v_loss
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

    @jax.jit
    def get_policy_rep(agent,
                       *,
                       targets: np.ndarray,
                       bases: np.ndarray = None,
                       ) -> jnp.ndarray:
        return agent.network(targets=targets, bases=bases, method='policy_goal_encoder')
