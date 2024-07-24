import copy
from functools import partial
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.core import freeze, unfreeze
import optax

from jaxrl_m.common import TrainState
from jaxrl_m.networks import Policy, DiscretePolicy

from .iql import IQLAgent
from .networks import RelativeRepresentation, GoalFreeMonolithicVF, ActorCritic


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


def compute_value_loss(agent, batch, network_params, expectile_param: float = None):
    # TODO: this is strange, this is actually not consistent with action-free IQL
    # masks are 0 if terminal, 1 otherwise
    # batch['masks'] = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    # batch['rewards'] = batch['rewards'] - 1.0
    expectile_param = expectile_param if expectile_param is not None else agent.config['pretrain_expectile']

    (next_v1, next_v2) = agent.network(batch['next_observations'], method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    (v1_t, v2_t) = agent.network(batch['observations'], method='target_value')
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    (v1, v2) = agent.network(batch['observations'], method='value', params=network_params)

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


def compute_value_loss_mixed(agent, batch, dataset_batch, network_params):
    info = {}
    value_loss, value_info = compute_value_loss(agent, batch, network_params)
    for k, v in value_info.items():
        info[f'online/{k}'] = v

    dataset_loss, dataset_info = compute_value_loss(agent, dataset_batch, network_params)
    for k, v in dataset_info.items():
        info[f'offline/{k}'] = v
    
    loss = (value_loss + dataset_loss) * 0.5
    info['value_loss'] = loss
    
    return loss, info


def conservative_value_loss(agent, batch, dataset_batch, network_params):
    info = {}
    dataset_loss, dataset_info = compute_value_loss(agent, dataset_batch, network_params)
    for k, v in dataset_info.items():
        info[f'offline/{k}'] = v
    
    dataset_v1, dataset_v2 = agent.network(dataset_batch['observations'], method='value', params=network_params)
    dataset_v = (dataset_v1 + dataset_v2) * 0.5
    online_v1, online_v2 = agent.network(batch['observations'], method='value', params=network_params)
    online_v = (online_v1 + online_v2) * 0.5
    conservative_loss = -dataset_v.mean() + jax.scipy.special.logsumexp(online_v) - jnp.log(online_v.shape[0])

    loss = dataset_loss + agent.config['cql_alpha'] * conservative_loss
    info['value_loss'] = loss
    info['conservative_loss'] = conservative_loss
    info['dataset_v'] = dataset_v.mean()
    info['online_v'] = online_v.mean()

    return loss, info


def conservative_value_loss_v2(agent, batch, dataset_batch, network_params):
    info = {}
    dataset_loss, dataset_info = compute_value_loss(agent, dataset_batch, network_params)
    for k, v in dataset_info.items():
        info[f'offline/{k}'] = v
    
    dataset_v1, dataset_v2 = agent.network(dataset_batch['observations'], method='value', params=network_params)
    dataset_v = (dataset_v1 + dataset_v2) * 0.5
    online_v1, online_v2 = agent.network(batch['observations'], method='value', params=network_params)
    online_v = (online_v1 + online_v2) * 0.5

    conservative_loss = -dataset_v.mean() + online_v.mean()
    loss = dataset_loss + agent.config['cql_alpha'] * conservative_loss
    info['value_loss'] = loss
    info['conservative_loss'] = conservative_loss
    info['dataset_v'] = dataset_v.mean()
    info['online_v'] = online_v.mean()
    
    return loss, info


def conservative_value_loss_v3(agent, batch, dataset_batch, network_params):
    info = {}
    value_loss, value_info = compute_value_loss(agent, batch, network_params, expectile_param=0.5)
    for k, v in value_info.items():
        info[f'online/{k}'] = v

    dataset_loss, dataset_info = compute_value_loss(agent, dataset_batch, network_params)
    for k, v in dataset_info.items():
        info[f'offline/{k}'] = v
    
    loss = (value_loss + dataset_loss) * 0.5
    info['value_loss'] = loss
    
    return loss, info


def compute_actor_loss(agent, batch, network_params):
    # TODO: this is strange, this is actually not consistent with action-free IQL
    v1, v2 = agent.network(batch['observations'], method='value')
    nv1, nv2 = agent.network(batch['next_observations'], method='value')
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2

    adv = nv - v
    exp_a = jnp.exp(adv * agent.config['temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    dist = agent.network(batch['observations'], state_rep_grad=True, method='actor', params=network_params)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()

    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'nv-v': adv.mean(),
        'v': v.mean(),
        'bc_log_probs': log_probs.mean(),
        'adv_median': jnp.median(adv),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
        'entropy': dist.entropy().mean(),
    }


def vem_actor_loss(agent, batch, network_params):
    v1, v2 = agent.network(batch['observations'], method='value')
    nv1, nv2 = agent.network(batch['next_observations'], method='value')
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2

    nv_v = nv - v

    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * nv
    adv = q - v

    weights = jnp.squeeze(jax.nn.softmax(adv, axis=0))

    dist = agent.network(batch['observations'], state_rep_grad=True, method='actor', params=network_params)
    log_probs = dist.log_prob(batch['actions'])

    per_actor_loss = -log_probs * len(weights) * weights
    actor_loss = per_actor_loss.mean()
    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'nv-v': nv_v.mean(),
        'v': v.mean(),
        'q': q.mean(),
        'adv_median': jnp.median(adv),
        'bc_log_probs': log_probs.mean(),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
        'entropy': dist.entropy().mean(),
    }


class AFIQLAgent(IQLAgent):
    network: TrainState = None

    @partial(jax.jit, static_argnames=('actor_loss', 'value_update', 'actor_update'))
    def pretrain_update(agent,
                        pretrain_batch,
                        seed: jax.random.PRNGKey,
                        actor_loss: str, 
                        value_update: bool = True,
                        actor_update: bool = True):
        actor_loss_fn = {
            "hiql_original": compute_actor_loss,
            "vem": vem_actor_loss,
        }[actor_loss]

        def loss_fn(network_params):
            info = {}
            
            # Value
            if value_update:
                value_loss, value_info = compute_value_loss(agent, pretrain_batch, network_params)
                for k, v in value_info.items():
                    info[f'value/{k}'] = v
            else:
                value_loss = 0.

            # Actor
            if actor_update:
                actor_loss, actor_info = actor_loss_fn(agent, pretrain_batch, network_params)
                for k, v in actor_info.items():
                    info[f'actor/{k}'] = v
            else:
                actor_loss = 0.

            loss = value_loss + actor_loss
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

    @partial(jax.jit, static_argnames=('num_samples', 'discrete'))
    def sample_actions(agent,
                       observations: np.ndarray,
                       *,
                       seed: jax.random.PRNGKey,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, temperature=temperature, method='actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        if not discrete:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnames=('actor_loss', ))
    def finetune_update(agent,
                        batch,
                        seed: jax.random.PRNGKey,
                        actor_loss: str):
        actor_loss_fn = {
            "hiql_original": compute_actor_loss,
            "vem": vem_actor_loss,
        }[actor_loss]

        def loss_fn(network_params):
            info = {}

            # Actor
            actor_loss, actor_info = actor_loss_fn(agent, batch, network_params)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            loss = actor_loss
            return loss, info
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        
        return agent.replace(network=new_network), info
    
    @partial(jax.jit, static_argnames=('actor_loss', ))
    def finetune_mixed_update(agent,
                              batch,
                              dataset_batch,
                              seed: jax.random.PRNGKey,
                              actor_loss: str):
        actor_loss_fn = {
            "hiql_original": compute_actor_loss,
            "vem": vem_actor_loss,
        }[actor_loss]

        def loss_fn(network_params):
            info = {}
            # Value
            if agent.config['mixed_finetune_value_loss'] == 'hiql':
                value_loss, value_info = compute_value_loss_mixed(agent, batch, dataset_batch, network_params)
            elif agent.config['mixed_finetune_value_loss'] == 'hiql_cql':
                value_loss, value_info = conservative_value_loss(agent, batch, dataset_batch, network_params)
            elif agent.config['mixed_finetune_value_loss'] == 'hiql_cql_v2':
                value_loss, value_info = conservative_value_loss_v2(agent, batch, dataset_batch, network_params)
            elif agent.config['mixed_finetune_value_loss'] == 'hiql_cql_v3':
                value_loss, value_info = conservative_value_loss_v3(agent, batch, dataset_batch, network_params)
            else:
                raise NotImplementedError
            for k, v in value_info.items():
                info[f'value/{k}'] = v

            # Actor
            actor_loss, actor_info = actor_loss_fn(agent, batch, network_params)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            loss = value_loss + actor_loss
            return loss, info
        
        new_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
        )
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        params = unfreeze(new_network.params)
        params['networks_target_value'] = new_target_params
        new_network = new_network.replace(params=freeze(params))
        
        return agent.replace(network=new_network), info

    @jax.jit
    def predict_value(agent, observations: np.ndarray, seed: jax.random.PRNGKey):
        v1, v2 = agent.network(observations, method='value')
        return v1


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        mixed_finetune_value_loss: str = 'hiql',
        cql_alpha: float = 0.005,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 1,
        pretrain_expectile: float = 0.7,
        way_steps: int = 0,
        rep_dim: int = 10,
        use_rep: int = 0,
        policy_train_rep: float = 0,
        visual: int = 0,
        encoder: str = 'impala',
        discrete: int = 0,
        use_layer_norm: int = 0,
        rep_type: str = 'state',
        use_waypoints: int = 0,
        **kwargs) -> AFIQLAgent:

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, high_actor_key, critic_key, value_key = jax.random.split(rng, 5)

        value_state_encoder = None
        # value_goal_encoder = None
        policy_state_encoder = None
        # policy_goal_encoder = None
        # high_policy_state_encoder = None
        # high_policy_goal_encoder = None
        if visual:
            assert use_rep
            from jaxrl_m.vision import encoders

            visual_encoder = encoders[encoder]
            def make_encoder(bottleneck):
                if bottleneck:
                    return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(rep_dim,), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(rep_dim=value_hidden_dims[-1], hidden_dims=(value_hidden_dims[-1],), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

            value_state_encoder = make_encoder(bottleneck=False)
            # value_goal_encoder = make_encoder(bottleneck=use_waypoints)
            policy_state_encoder = make_encoder(bottleneck=False)
            # policy_goal_encoder = make_encoder(bottleneck=False)
            # high_policy_state_encoder = make_encoder(bottleneck=False)
            # high_policy_goal_encoder = make_encoder(bottleneck=False)
        # else:
        #     def make_encoder(bottleneck):
        #         if bottleneck:
        #             return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(*value_hidden_dims, rep_dim), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
        #         else:
        #             return RelativeRepresentation(rep_dim=value_hidden_dims[-1], hidden_dims=(*value_hidden_dims, value_hidden_dims[-1]), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

        #     if use_rep:
        #         value_goal_encoder = make_encoder(bottleneck=True)

        value_def = GoalFreeMonolithicVF(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim)

        if discrete:
            action_dim = actions[0] + 1
            actor_def = DiscretePolicy(actor_hidden_dims, action_dim=action_dim)
        else:
            action_dim = actions.shape[-1]
            actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        # high_action_dim = observations.shape[-1] if not use_rep else rep_dim
        # high_actor_def = Policy(actor_hidden_dims, action_dim=high_action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        network_def = ActorCritic(
            encoders={
                'value_state': value_state_encoder,
                # 'value_goal': value_goal_encoder,
                'policy_state': policy_state_encoder,
                # 'policy_goal': policy_goal_encoder,
                # 'high_policy_state': high_policy_state_encoder,
                # 'high_policy_goal': high_policy_goal_encoder,
            },
            networks={
                'value': value_def,
                'target_value': copy.deepcopy(value_def),
                'actor': actor_def,
                # 'high_actor': high_actor_def,
            },
        )
        network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
        network_params = network_def.init(value_key, observations)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = unfreeze(network.params)
        params['networks_target_value'] = params['networks_value']
        network = network.replace(params=freeze(params))

        config = flax.core.FrozenDict(dict(
            discount=discount, temperature=temperature, mixed_finetune_value_loss=mixed_finetune_value_loss, cql_alpha=cql_alpha,
            target_update_rate=tau, pretrain_expectile=pretrain_expectile, way_steps=way_steps, rep_dim=rep_dim,
            policy_train_rep=policy_train_rep,
            use_rep=use_rep, use_waypoints=use_waypoints,
        ))

        return AFIQLAgent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config)
