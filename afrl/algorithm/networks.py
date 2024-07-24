from typing import Callable, Dict, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from jaxrl_m.typing import PRNGKey, Shape, Dtype, Array
from jaxrl_m.networks import MLP, ensemblize, default_init

from .vae_nets import vae_encode, reparameterize


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)


class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu)(observations)


class RelativeRepresentation(nn.Module):
    rep_dim: int = 256
    hidden_dims: tuple = (256, 256)
    module: nn.Module = None
    visual: bool = False
    layer_norm: bool = False
    rep_type: str = 'state'
    bottleneck: bool = True  # Meaning that we're using this representation for high-level actions

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        if self.visual:
            inputs = self.module()(inputs)
        if self.layer_norm:
            rep = LayerNormMLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)
        else:
            rep = MLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)

        if self.bottleneck:
            rep = rep / jnp.linalg.norm(rep, axis=-1, keepdims=True) * jnp.sqrt(self.rep_dim)

        return rep


class HybridRelativeRepresentation(nn.Module):
    rep_dim: int = 256
    hidden_dims: tuple = (256, 256)
    module: nn.Module = None
    visual: bool = True # actually not used
    layer_norm: bool = False
    rep_type: str = 'state'
    bottleneck: bool = True  # Meaning that we're using this representation for high-level actions

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        visual_part = self.module()(inputs['image'])
        inputs = jnp.concatenate([visual_part, inputs['state']], axis=-1)

        if self.layer_norm:
            rep = LayerNormMLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)
        else:
            rep = MLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)

        if self.bottleneck:
            rep = rep / jnp.linalg.norm(rep, axis=-1, keepdims=True) * jnp.sqrt(self.rep_dim)

        return rep


class MonolithicVF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    rep_dim: int = None
    obs_rep: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False)

    def __call__(self, observations, goals=None, info=False):
        phi = observations
        psi = goals

        v1, v2 = self.value_net(jnp.concatenate([phi, psi], axis=-1)).squeeze(-1)

        if info:
            return {
                'v': (v1 + v2) / 2,
            }
        return v1, v2


class SingleVF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    rep_dim: int = None
    obs_rep: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False, ensemble=False)
    
    def __call__(self, observations, goals=None):
        phi = observations
        psi = goals

        v = self.value_net(jnp.concatenate([phi, psi], axis=-1)).squeeze(-1)
        return v


class GoalFreeMonolithicVF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    rep_dim: int = None
    obs_rep: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False)

    def __call__(self, observations, info=False):
        phi = observations

        v1, v2 = self.value_net(phi).squeeze(-1)

        if info:
            return {
                'v': (v1 + v2) / 2,
            }
        return v1, v2


class DoubleQValueFunction(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = True

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.q_value_net = repr_class((*self.hidden_dims, 1), activate_final=False, ensemble=True)
    
    def __call__(self, observations, actions):
        assert len(observations.shape) == len(actions.shape) == 2
        inputs = jnp.concatenate([observations, actions], axis=-1)
        q1, q2 = self.q_value_net(inputs).squeeze(-1)
        return q1, q2


def get_rep(
        encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)


class HierarchicalActorCritic(nn.Module):
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]
    use_waypoints: int

    def value(self, observations, goals, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['value'](state_reps, goal_reps, **kwargs)

    def target_value(self, observations, goals, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['target_value'](state_reps, goal_reps, **kwargs)

    def actor(self, observations, goals, low_dim_goals=False, state_rep_grad=True, goal_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)

        if low_dim_goals:
            goal_reps = goals
        else:
            if self.use_waypoints:
                # Use the value_goal representation
                goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
            else:
                goal_reps = get_rep(self.encoders['policy_goal'], targets=goals, bases=observations)
            if not goal_rep_grad:
                goal_reps = jax.lax.stop_gradient(goal_reps)

        return self.networks['actor'](jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs)

    def high_actor(self, observations, goals, state_rep_grad=True, goal_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['high_policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)

        goal_reps = get_rep(self.encoders['high_policy_goal'], targets=goals, bases=observations)
        if not goal_rep_grad:
            goal_reps = jax.lax.stop_gradient(goal_reps)

        return self.networks['high_actor'](jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs)

    def value_goal_encoder(self, targets, bases, **kwargs):
        return get_rep(self.encoders['value_goal'], targets=targets, bases=bases)

    def policy_goal_encoder(self, targets, bases, **kwargs):
        assert not self.use_waypoints
        return get_rep(self.encoders['policy_goal'], targets=targets, bases=bases)

    def __call__(self, observations, goals):
        # Only for initialization
        rets = {
            'value': self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
            'actor': self.actor(observations, goals),
            'high_actor': self.high_actor(observations, goals),
        }
        return rets


class HierarchicalActorCriticV2(nn.Module):
    """ This resembles HierarchicalActorCritic, but it
        1. has additional low_level_delta feature.
            When low_level_delta is set to True, the low level actor directly uses currrent position - goal position as input.
            This was used and only used by HIQL.
        2. enables stop gradient in value training.
        3. enables fast high actor support.
        4. supports policy_share_value_state.
        5. supports decoders (VAE with reconstruction)
        6. supports pretrained ICVF
    """
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]
    decoders: Dict[str, nn.Module]
    icvfs: Dict[str, nn.Module]
    use_waypoints: int
    low_level_delta: bool
    policy_share_value_state: bool
    use_vae: bool
    use_icvf: bool
    icvf_use_psi: bool

    def get_rep(
            self, encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
            seed: jax.random.PRNGKey = None,
    ):
        if self.use_vae and self.use_icvf and self.icvf_use_psi:
            raise ValueError(f"VAE is not compatible with ICVF pretrained psi!")
        if self.use_icvf and self.encoders['value_state'] is not None:
            raise ValueError(f"ICVF phi is not compatible with value state encoder")
        if self.use_icvf and self.icvf_use_psi and self.encoders['value_goal'] is not None:
            raise ValueError(f"value goal encoder is not compatible with ICVF pretrained psi")
        
        if encoder is None:
            return targets
        else:
            if self.use_vae and seed is not None:
                seed, sub_seed = jax.random.split(seed)
                z = vae_encode(sub_seed, encoder, targets, bases)
                z = z / jnp.linalg.norm(z, axis=-1, keepdims=True) * z.shape[-1]
                return z
            if bases is None:
                return encoder(targets)
            else:
                return encoder(targets, bases)

    def get_value_state_rep(self, observations):
        if not self.use_icvf:
            return self.get_rep(self.encoders['value_state'], targets=observations)
        else:
            phis = self.icvfs['icvf_value_state_encoder'].get_phi(observations)
            # return self.icvfs['icvf_value_state_connector'](phis[0])
            return phis[0]
    
    def get_value_goal_rep(self, goals, observations, seed: jax.random.PRNGKey = None):
        if not self.use_icvf or not self.icvf_use_psi:
            return self.get_rep(self.encoders['value_goal'], targets=goals, bases=observations, seed=seed)
        else:
            psis = self.icvfs['icvf_value_state_encoder'].get_psi(goals)
            return psis[0]

    def icvf_value_fn(self, observations, outcomes, intents):
        return self.icvfs['icvf_value_state_encoder'](observations, outcomes, intents)
    
    def target_icvf_value_fn(self, observations, outcomes, intents):
        return self.icvfs['target_icvf_value_state_encoder'](observations, outcomes, intents)

    def value(self, observations, goals, state_rep_grad=True, goal_rep_grad=True,
              seed: jax.random.PRNGKey = None, **kwargs):
        state_reps = self.get_value_state_rep(observations)
        goal_reps = self.get_value_goal_rep(goals, observations, seed=seed)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)
        if not goal_rep_grad:
            goal_reps = jax.lax.stop_gradient(goal_reps)
        return self.networks['value'](state_reps, goal_reps, **kwargs)

    def vae(self, observations, goals,
            seed: jax.random.PRNGKey = None, **kwargs):
        z_mean, z_logvar = self.encoders['value_goal'](targets=goals, bases=observations)
        z = reparameterize(seed, z_mean, z_logvar)
        x_recon = self.decoders['value_goal'](z)
        return x_recon, z_mean, z_logvar

    def guiding_v(self, observations, goals, state_rep_grad=False, goal_rep_grad=False, can_skip_on_purpose: bool=False,
                  seed: jax.random.PRNGKey = None, **kwargs):
        if 'guiding_v' in self.networks:
            state_reps = self.get_value_state_rep(observations)
            goal_reps = self.get_value_goal_rep(goals, observations, seed=seed)
            if not state_rep_grad:
                state_reps = jax.lax.stop_gradient(state_reps)
            if not goal_rep_grad:
                goal_reps = jax.lax.stop_gradient(goal_reps)
            return self.networks['guiding_v'](state_reps, goal_reps, **kwargs)
        elif not can_skip_on_purpose:
            raise AssertionError(f'Guiding value function was not defined in this model!')
        return None

    def target_value(self, observations, goals,
                     seed: jax.random.PRNGKey = None, **kwargs):
        state_reps = self.get_value_state_rep(observations)
        goal_reps = self.get_value_goal_rep(goals, observations, seed=seed)
        return self.networks['target_value'](state_reps, goal_reps, **kwargs)

    def actor(self, observations, goals, low_dim_goals=False, state_rep_grad=True, goal_rep_grad=True,
              seed: jax.random.PRNGKey = None, **kwargs):
        if self.low_level_delta:
            # We cannot use any kind of representation under this circumstance.
            assert self.encoders['policy_state'] is None and self.encoders['value_goal'] is None and self.encoders['policy_goal'] is None

        if not self.policy_share_value_state:
            state_reps = self.get_rep(self.encoders['policy_state'], targets=observations)
        else:
            state_reps = self.get_value_state_rep(observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)

        if low_dim_goals:
            goal_reps = goals
            if self.use_waypoints and self.use_icvf and self.icvf_use_psi:
                goal_reps = self.get_value_goal_rep(goals, observations, seed=seed)
        else:
            if self.use_waypoints:
                # Use the value_goal representation
                # goal_reps = self.get_rep(self.encoders['value_goal'], targets=goals, bases=observations, seed=seed)
                goal_reps = self.get_value_goal_rep(goals, observations, seed=seed)
            else:
                goal_reps = self.get_rep(self.encoders['policy_goal'], targets=goals, bases=observations)
            if not goal_rep_grad:
                goal_reps = jax.lax.stop_gradient(goal_reps)

        if not self.low_level_delta:
            return self.networks['actor'](jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs)
        else:
            if len(state_reps.shape) == 1:
                if isinstance(state_reps, jax.Array):
                    state_reps = state_reps.at[:2].set(state_reps[:2] - goal_reps[:2])
                else:
                    state_reps[:2] = state_reps[:2] - goal_reps[:2]
            elif len(state_reps.shape) == 2:
                if isinstance(state_reps, jax.Array):
                    state_reps = state_reps.at[:, :2].set(state_reps[:, :2] - goal_reps[:, :2]) # jax arrays are immutable
                else:
                    state_reps[:, :2] = state_reps[:, :2] - goal_reps[:, :2]
            else:
                raise RuntimeError
            return self.networks['actor'](state_reps, **kwargs)

    def high_actor(self, observations, goals, state_rep_grad=True, goal_rep_grad=True,
                   seed: jax.random.PRNGKey = None, **kwargs):
        state_reps = self.get_rep(self.encoders['high_policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)

        goal_reps = self.get_rep(self.encoders['high_policy_goal'], targets=goals, bases=observations)
        if not goal_rep_grad:
            goal_reps = jax.lax.stop_gradient(goal_reps)

        return self.networks['high_actor'](jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs)

    def fast_high_actor(self, observations, goals, state_rep_grad=True, goal_rep_grad=True, can_skip_on_purpose: bool=False,
                        seed: jax.random.PRNGKey = None, **kwargs):
        """NOTICE: fast high actor share the same representation network as the high actor."""
        if 'fast_high_actor' in self.networks:
            state_reps = self.get_rep(self.encoders['high_policy_state'], targets=observations)
            if not state_rep_grad:
                state_reps = jax.lax.stop_gradient(state_reps)

            goal_reps = self.get_rep(self.encoders['high_policy_goal'], targets=goals, bases=observations)
            if not goal_rep_grad:
                goal_reps = jax.lax.stop_gradient(goal_reps)
            
            return self.networks['fast_high_actor'](jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs)
        elif not can_skip_on_purpose:
            raise AssertionError(f"Fast high actor was not defined in this model!")
        return None

    def value_goal_encoder(self, targets, bases,
                           seed: jax.random.PRNGKey = None, **kwargs):
        return self.get_rep(self.encoders['value_goal'], targets=targets, bases=bases, seed=seed)

    def policy_goal_encoder(self, targets, bases,
                            seed: jax.random.PRNGKey = None, **kwargs):
        assert not self.use_waypoints
        return self.get_rep(self.encoders['policy_goal'], targets=targets, bases=bases)

    def __call__(self, observations, goals):
        # Only for initialization
        rng = jax.random.PRNGKey(0)
        rets = {
            'value': self.value(observations, goals, seed=rng),
            'target_value': self.target_value(observations, goals, seed=rng),
            'actor': self.actor(observations, goals, seed=rng),
            'high_actor': self.high_actor(observations, goals),
            'guiding_v': self.guiding_v(observations, goals, can_skip_on_purpose=True, seed=rng),
            'fast_high_actor': self.fast_high_actor(observations, goals, can_skip_on_purpose=True),
        }
        if self.use_vae:
            rets['vae'] = self.vae(observations, goals, seed=rng)
        if self.use_icvf:
            rets['icvf_vf'] = self.icvf_value_fn(observations, observations, observations)
            rets['target_icvf_vf'] = self.target_icvf_value_fn(observations, observations, observations)
            rets['phi'] = self.get_value_state_rep(observations)
            if self.icvf_use_psi:
                rets['psi'] = self.get_value_goal_rep(goals, observations, seed=rng)
        return rets


class ActorCritic(nn.Module):
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]

    def value(self, observations, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        return self.networks['value'](state_reps, **kwargs)
    
    def target_value(self, observations, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        return self.networks['target_value'](state_reps, **kwargs)
    
    def actor(self, observations, state_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)
        return self.networks['actor'](state_reps, **kwargs)

    def __call__(self, observations):
        # Only for initialization
        rets = {
            'value': self.value(observations),
            'target_value': self.target_value(observations),
            'actor': self.actor(observations),
        }
        return rets


class TD3Network(nn.Module):
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]

    def q(self, observations, actions, **kwargs):
        state_reps = get_rep(self.encoders['q_state'], targets=observations)
        return self.networks['q'](state_reps, actions, **kwargs)
    
    def target_q(self, observations, actions, **kwargs):
        state_reps = get_rep(self.encoders['q_state'], targets=observations)
        return self.networks['target_q'](state_reps, actions, **kwargs)

    def actor(self, observations, state_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)
        return self.networks['actor'](state_reps, **kwargs)
    
    def target_actor(self, observations, state_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)
        return self.networks['target_actor'](state_reps, **kwargs)
    
    def __call__(self, observations, actions):
        # Only for initialization
        rets = {
            'q': self.q(observations, actions),
            'target_q': self.target_q(observations, actions),
            'actor': self.actor(observations),
            'target_actor': self.target_actor(observations),
        }


class TD3NetworkWithValue(TD3Network):
    def value(self, observations, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        return self.networks['value'](state_reps, **kwargs)

    def target_value(self, observations, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        return self.networks['target_value'](state_reps, **kwargs)
    
    def __call__(self, observations, actions):
        # Only for initialization
        rets = {
            'value': self.value(observations),
            'target_value': self.target_value(observations),
            'q': self.q(observations, actions),
            'target_q': self.target_q(observations, actions),
            'actor': self.actor(observations),
            'target_actor': self.target_actor(observations),
        }


class TD3Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(self, observations) -> jax.Array:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        action = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        action = self.max_action * nn.activation.tanh(action)
        return action
