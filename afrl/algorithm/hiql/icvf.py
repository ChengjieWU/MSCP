from functools import partial

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.core import freeze, unfreeze
import optax

from jaxrl_m.common import TrainState

from .hiql import JointTrainAgent
from .hiql import compute_value_loss, compute_actor_loss, compute_high_actor_loss
from .hiql import compute_value_loss_mixed, conservative_value_loss, conservative_value_loss_v2, conservative_value_loss_v3
from .hiql import actor_loss_with_L2_distance
from .hiql import compute_high_actor_loss_mixed


def expectile_loss(adv, diff, expectile=0.8):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * diff ** 2


def cal_icvf_loss(value_fn, target_value_fn, batch, config):

    assert all([k in config for k in ['no_intent', 'min_q', 'expectile', 'discount']]), 'Missing ICVF config keys'

    if config['no_intent']:
        batch['desired_goals'] = jax.tree_map(jnp.ones_like, batch['desired_goals'])

    ###
    # Compute TD error for outcome s_+
    # 1(s == s_+) + V(s', s_+, z) - V(s, s_+, z)
    ###

    (next_v1_gz, next_v2_gz) = target_value_fn(batch['next_observations'], batch['goals'], batch['desired_goals'])
    q1_gz = batch['rewards'] + config['discount'] * batch['masks'] * next_v1_gz
    q2_gz = batch['rewards'] + config['discount'] * batch['masks'] * next_v2_gz
    q1_gz, q2_gz = jax.lax.stop_gradient(q1_gz), jax.lax.stop_gradient(q2_gz)

    (v1_gz, v2_gz) = value_fn(batch['observations'], batch['goals'], batch['desired_goals'])

    ###
    # Compute the advantage of s -> s' under z
    # r(s, z) + V(s', z, z) - V(s, z, z)
    ###

    (next_v1_zz, next_v2_zz) = target_value_fn(batch['next_observations'], batch['desired_goals'], batch['desired_goals'])
    if config['min_q']:
        next_v_zz = jnp.minimum(next_v1_zz, next_v2_zz)
    else:
        next_v_zz = (next_v1_zz + next_v2_zz) / 2
    
    q_zz = batch['desired_rewards'] + config['discount'] * batch['desired_masks'] * next_v_zz

    (v1_zz, v2_zz) = target_value_fn(batch['observations'], batch['desired_goals'], batch['desired_goals'])
    v_zz = (v1_zz + v2_zz) / 2
    adv = q_zz - v_zz

    if config['no_intent']:
        adv = jnp.zeros_like(adv)
    
    ###
    #
    # If advantage is positive (next state is better than current state), then place additional weight on
    # the value loss. 
    #
    ##
    value_loss1 = expectile_loss(adv, q1_gz-v1_gz, config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2_gz-v2_gz, config['expectile']).mean()
    value_loss = value_loss1 + value_loss2

    def masked_mean(x, mask):
        return (x * mask).sum() / (1e-5 + mask.sum())

    advantage = adv
    return value_loss, {
        'value_loss': value_loss,
        'v_gz max': v1_gz.max(),
        'v_gz min': v1_gz.min(),
        'v_zz': v_zz.mean(),
        'v_gz': v1_gz.mean(),
        # 'v_g': v1_g.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
        'reward mean': batch['rewards'].mean(),
        'mask mean': batch['masks'].mean(),
        'q_gz max': q1_gz.max(),
        'value_loss1': masked_mean((q1_gz-v1_gz)**2, batch['masks']), # Loss on s \neq s_+
        'value_loss2': masked_mean((q1_gz-v1_gz)**2, 1.0 - batch['masks']), # Loss on s = s_+
    }


def periodic_target_update(
    model: TrainState, target_model: TrainState, period: int
) -> TrainState:
    new_target_params = jax.tree_map(
        lambda p, tp: optax.periodic_update(p, tp, model.step, period),
        model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


class ICVFAgent(JointTrainAgent):
    
    @partial(jax.jit, static_argnames=('value_update', 'actor_update', 'high_actor_update'))
    def pretrain_update(agent, pretrain_batch, icvf_batch, seed=None, value_update=True, actor_update=True, high_actor_update=True):
        value_seed, actor_seed, high_actor_seed = jax.random.split(seed, 3)
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

            # ICVF loss
            # hard-coded ICVF configs
            icvf_additional_configs = {
                'no_intent': False,
                'discount': 0.99,
                'min_q': True,
                'expectile': 0.9,
            }
            value_fn = lambda s, g, z: agent.network(s, g, z, method='icvf_value_fn', params=network_params)
            target_value_fn = lambda s, g, z: agent.network(s, g, z, method='target_icvf_value_fn')
            icvf_loss, icvf_info = cal_icvf_loss(value_fn, target_value_fn, icvf_batch, icvf_additional_configs)
            for k, v in icvf_info.items():
                info[f'icvf/{k}'] = v

            loss = value_loss + actor_loss + high_actor_loss + icvf_loss * agent.config['icvf_constraint_alpha']

            return loss, info

        if value_update:
            new_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
            )
            new_icvf_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['icvfs_icvf_value_state_encoder'], agent.network.params['icvfs_target_icvf_value_state_encoder']
            )

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        if value_update:
            params = unfreeze(new_network.params)
            params['networks_target_value'] = new_target_params
            params['icvfs_target_icvf_value_state_encoder'] = new_icvf_target_params
            new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info

    @partial(jax.jit, static_argnames=('actor_supervision', 'high_actor_update'))
    def icvf_finetune_mixed_update(
        agent,
        batch,
        dataset_batch,
        icvf_batch,
        seed: jax.random.PRNGKey,
        actor_supervision: str,
        high_actor_update: bool,
    ):
        def loss_fn(network_params, seed):
            seed, key_v, key_actor, key_high_actor = jax.random.split(seed, 4)

            info = {}

            # Value
            if agent.config['hiql_actor_loss_coefficient'] > 0:
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
            
            # ICVF loss
            # hard-coded ICVF configs
            icvf_additional_configs = {
                'no_intent': False,
                'discount': 0.99,
                'min_q': True,
                'expectile': 0.9,
            }
            value_fn = lambda s, g, z: agent.network(s, g, z, method='icvf_value_fn', params=network_params)
            target_value_fn = lambda s, g, z: agent.network(s, g, z, method='target_icvf_value_fn')
            icvf_loss, icvf_info = cal_icvf_loss(value_fn, target_value_fn, icvf_batch, icvf_additional_configs)
            for k, v in icvf_info.items():
                info[f'icvf/{k}'] = v

            loss = value_loss + actor_loss + high_actor_loss + icvf_loss * agent.config['icvf_constraint_alpha']
            return loss, info
        
        new_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
        )
        new_icvf_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['icvfs_icvf_value_state_encoder'], agent.network.params['icvfs_target_icvf_value_state_encoder']
        )

        seed, sub_seed = jax.random.split(seed)
        new_network, info = agent.network.apply_loss_fn(loss_fn=partial(loss_fn, seed=sub_seed), has_aux=True)

        params = unfreeze(new_network.params)
        params['networks_target_value'] = new_target_params
        params['icvfs_target_icvf_value_state_encoder'] = new_icvf_target_params
        new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info

    @jax.jit
    def icvf_update(
            agent,
            batch,
            dataset_batch,
            seed: jax.random.PRNGKey,
    ):
        # ICVF loss on offline data (dataset_batch)
        # HIQL loss on online data (batch)
        value_seed, actor_seed, high_actor_seed = jax.random.split(seed, 3)
        def loss_fn(network_params):
            info = {}

            # Value
            repr_grad = agent.config['grad_value_repr']
            # value_loss, value_info = compute_value_loss(agent, batch, network_params, expectile_param=0.5, repr_grad=repr_grad, seed=value_seed)
            value_loss, value_info = compute_value_loss(agent, batch, network_params, expectile_param=agent.config['pretrain_expectile'], repr_grad=repr_grad, seed=value_seed)
            for k, v in value_info.items():
                info[f'value/{k}'] = v
            
            # Actor
            actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, seed=actor_seed)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            
            # High Actor
            if agent.config['use_waypoints']:
                high_actor_loss, high_actor_info = compute_high_actor_loss(agent, batch, network_params, seed=high_actor_seed)
                for k, v in high_actor_info.items():
                    info[f'high_actor/{k}'] = v
            else:
                high_actor_loss = 0.
            
            # ICVF loss
            # hard-coded ICVF configs
            icvf_additional_configs = {
                'no_intent': False,
                'discount': 0.99,
                'min_q': True,
                'expectile': 0.9,
            }
            value_fn = lambda s, g, z: agent.network(s, g, z, method='icvf_value_fn', params=network_params)
            target_value_fn = lambda s, g, z: agent.network(s, g, z, method='target_icvf_value_fn')
            icvf_loss, icvf_info = cal_icvf_loss(value_fn, target_value_fn, dataset_batch, icvf_additional_configs)
            for k, v in icvf_info.items():
                info[f'icvf/{k}'] = v

            loss = value_loss + actor_loss + high_actor_loss + icvf_loss * agent.config['icvf_constraint_alpha']

            return loss, info
        
        new_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
        )
        new_icvf_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['icvfs_icvf_value_state_encoder'], agent.network.params['icvfs_target_icvf_value_state_encoder']
        )

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        params = unfreeze(new_network.params)
        params['networks_target_value'] = new_target_params
        params['icvfs_target_icvf_value_state_encoder'] = new_icvf_target_params
        new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info 
