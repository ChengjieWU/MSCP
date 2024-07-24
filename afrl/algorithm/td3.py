import copy
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import flax
from flax.core import freeze, unfreeze
import optax

from jaxrl_m.common import TrainState

from .afiql import AFIQLAgent
from .networks import GoalFreeMonolithicVF, DoubleQValueFunction, TD3NetworkWithValue, TD3Policy, TD3Network


def TD3_critic_loss_value_shaped(agent, batch, seed, network_params):
    v1, v2 = agent.network(batch['observations'], method='value')
    nv1, nv2 = agent.network(batch['next_observations'], method='value')
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2

    shaped_rewards = batch['rewards'] + agent.config['discount'] * batch['masks'] * nv - batch['masks'] * v

    noise = jnp.clip(jax.random.normal(seed, shape=batch['actions'].shape) * agent.config['policy_noise'], -agent.config['noise_clip'], agent.config['noise_clip'])
    next_target_a = noise + agent.network(batch['next_observations'], method='target_actor')
    next_target_a = jnp.clip(next_target_a, -agent.config['max_action'], agent.config['max_action'])

    nq1, nq2 = agent.network(batch['next_observations'], next_target_a, method='target_q')
    y = shaped_rewards + agent.config['discount'] * batch['masks'] * jnp.minimum(nq1, nq2)

    q1, q2 = agent.network(batch['observations'], batch['actions'], method='q', params=network_params)
    q_loss1 = ((y - q1) ** 2).mean()
    q_loss2 = ((y - q2) ** 2).mean()
    q_loss = q_loss1 + q_loss2

    return q_loss, {
        'q_loss': q_loss,
        'q_max': q1.max(),
        'q_min': q1.min(),
        'q_mean': q1.mean(),
        'y_mean': y.mean(),
        'v_mean': v.mean(),
        'shaped_rewards': shaped_rewards.mean(),
    }


def TD3_critic_loss(agent, batch, seed, network_params):
    noise = jnp.clip(jax.random.normal(seed, shape=batch['actions'].shape) * agent.config['policy_noise'], -agent.config['noise_clip'], agent.config['noise_clip'])
    next_target_a = noise + agent.network(batch['next_observations'], method='target_actor')
    next_target_a = jnp.clip(next_target_a, -agent.config['max_action'], agent.config['max_action'])

    nq1, nq2 = agent.network(batch['next_observations'], next_target_a, method='target_q')
    y = batch['rewards'] + agent.config['discount'] * batch['masks'] * jnp.minimum(nq1, nq2)

    q1, q2 = agent.network(batch['observations'], batch['actions'], method='q', params=network_params)
    q_loss1 = ((y - q1) ** 2).mean()
    q_loss2 = ((y - q2) ** 2).mean()
    q_loss = q_loss1 + q_loss2

    return q_loss, {
        'q_loss': q_loss,
        'q_max': q1.max(),
        'q_min': q1.min(),
        'q_mean': q1.mean(),
        'y_mean': y.mean(),
        'r_mean': batch['rewards'].mean(),
    }


def TD3_actor_loss(agent, batch, network_params):
    pred_action = agent.network(batch['observations'], method='actor', params=network_params)
    q1, q2 = agent.network(batch['observations'], pred_action, method='q')
    actor_loss = -q1.mean()

    return actor_loss, {
        'actor_loss': actor_loss
    }


class AFIQLFinetuneTD3Agent(AFIQLAgent):
    def pretrain_update(agent, *args, **kwargs):
        raise AssertionError(f"The 'pretrain_upadte' method of AFIQLFinetuneTD3Agent should never be called.")

    def TD3_update(agent, batch, seed: jax.random.PRNGKey, epoch: int):
        rng_key, q_seed = jax.random.split(seed)
        agent_new, info = agent.update_q(batch, q_seed)
        if epoch % agent.config['policy_freq'] == 0:
            rng_key, actor_seed = jax.random.split(rng_key)
            agent_new, actor_info = agent_new.update_actor(batch, actor_seed)
            info.update(actor_info)
        return agent_new, info

    def finetune_update(agent, batch, seed: jax.random.PRNGKey, epoch: int):
        rng_key, q_seed = jax.random.split(seed)
        agent_new, info = agent.update_q_with_shaped_reward(batch, q_seed)
        if epoch % agent.config['policy_freq'] == 0:
            rng_key, actor_seed = jax.random.split(rng_key)
            agent_new, actor_info = agent_new.update_actor(batch, actor_seed)
            info.update(actor_info)
        return agent_new, info

    @partial(jax.jit, static_argnames=('explore', ))
    def sample_actions(agent,
                       observations,
                       seed: jax.random.PRNGKey,
                       explore: bool) -> jnp.ndarray:
        actions = agent.network(observations, method='actor')
        if explore:
            noise = jax.random.normal(seed, shape=actions.shape) * agent.config['expl_noise']
            actions = jnp.clip(actions + noise, -agent.config['max_action'], agent.config['max_action'])
        return actions

    @jax.jit
    def update_q_with_shaped_reward(agent,
                                    batch,
                                    seed: jax.random.PRNGKey):
        def loss_fn(network_params):
            info = {}
            q_loss, q_info = TD3_critic_loss_value_shaped(agent, batch, seed, network_params)
            for k, v in q_info.items():
                info[f'q/{k}'] = v
            return q_loss, info
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        return agent.replace(network=new_network), info

    @jax.jit
    def update_q(agent,
                 batch,
                 seed: jax.random.PRNGKey):
        def loss_fn(network_params):
            info = {}
            q_loss, q_info = TD3_critic_loss(agent, batch, seed, network_params)
            for k, v in q_info.items():
                info[f'q/{k}'] = v
            return q_loss, info
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        return agent.replace(network=new_network), info

    @jax.jit
    def update_actor(agent,
                     batch,
                     seed: jax.random.PRNGKey):
        def loss_fn(network_params):
            info = {}
            actor_loss, actor_info = TD3_actor_loss(agent, batch, network_params)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            return actor_loss, info
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        new_target_q_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']),
            agent.network.params['networks_q'], agent.network.params['networks_target_q']
        )
        new_target_actor_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']),
            agent.network.params['networks_actor'], agent.network.params['networks_target_actor']
        )
        params = unfreeze(new_network.params)
        params['networks_target_q'] = new_target_q_params
        params['networks_target_actor'] = new_target_actor_params
        new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info


def create_learner(
        seed: jax.random.PRNGKey,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        use_pretrained_v: bool,
        max_action: float,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        expl_noise: float = 0.1,
        policy_freq: int = 2,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        q_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        rep_dim: int = 10,
        use_rep: int = 0,
        policy_train_rep: float = 0,
        visual: int = 0,
        encoder: str = 'impala',
        use_layer_norm: int = 0,
        rep_type: str = 'state',
) -> AFIQLFinetuneTD3Agent:
    rng = jax.random.PRNGKey(seed)
    rng, weight_init_seed = jax.random.split(rng)

    if use_pretrained_v:
        value_state_encoder = None
    q_state_encoder = None
    policy_state_encoder = None

    if visual:
        assert use_rep
        from jaxrl_m.vision import encoders
        from .networks import RelativeRepresentation

        visual_encoder = encoders[encoder]
        def make_encoder(bottleneck):
            if bottleneck:
                return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(rep_dim,), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
            else:
                return RelativeRepresentation(rep_dim=value_hidden_dims[-1], hidden_dims=(value_hidden_dims[-1],), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)
        if use_pretrained_v:
            value_state_encoder = make_encoder(bottleneck=False)
        q_state_encoder = make_encoder(bottleneck=False)
        policy_state_encoder = make_encoder(bottleneck=False)
    
    if use_pretrained_v:
        value_def = GoalFreeMonolithicVF(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim)
    q_def = DoubleQValueFunction(hidden_dims=q_hidden_dims, use_layer_norm=use_layer_norm)

    action_dim = actions.shape[-1]
    actor_def = TD3Policy(actor_hidden_dims, action_dim=action_dim, max_action=max_action)

    if use_pretrained_v:
        network_def = TD3NetworkWithValue(
            encoders={
                'value_state': value_state_encoder,
                'q_state': q_state_encoder,
                'policy_state': policy_state_encoder,
            },
            networks={
                'value': value_def,
                'target_value': copy.deepcopy(value_def),
                'q': q_def,
                'target_q': copy.deepcopy(q_def),
                'actor': actor_def,
                'target_actor': copy.deepcopy(actor_def),
            }
        )
    else:
        network_def = TD3Network(
            encoders={
                'q_state': q_state_encoder,
                'policy_state': policy_state_encoder,
            },
            networks={
                'q': q_def,
                'target_q': copy.deepcopy(q_def),
                'actor': actor_def,
                'target_actor': copy.deepcopy(actor_def),
            }
        )

    network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))   # nn.Module
    network_params = network_def.init(weight_init_seed, observations, actions)['params']

    network = TrainState.create(network_def, network_params, tx=network_tx)     # TrainState
    params = unfreeze(network.params)
    if use_pretrained_v:
        params['networks_target_value'] = params['networks_value']
    params['networks_target_q'] = params['networks_q']
    params['networks_target_actor'] = params['networks_actor']
    network = network.replace(params=freeze(params))

    config = flax.core.FrozenDict(dict(
        discount=discount, target_update_rate=tau, max_action=max_action, policy_freq=policy_freq,
        policy_noise=policy_noise, noise_clip=noise_clip, expl_noise=expl_noise,
        rep_dim=rep_dim,
        policy_train_rep=policy_train_rep,
        use_rep=use_rep,
    ))
    
    return AFIQLFinetuneTD3Agent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config)
