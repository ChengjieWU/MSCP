import copy

from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState
from jaxrl_m.networks import Policy, Critic, ensemblize, DiscretePolicy
from loguru import logger

import flax
from flax.core import freeze, unfreeze

from . import hiql, scpiql, icvf
from ..networks import Representation, HierarchicalActorCriticV2, RelativeRepresentation, MonolithicVF, SingleVF, HybridRelativeRepresentation
from ..vae_nets import RelativeVAEEncoder, VAEDecoder
from ..icvf_nets import create_icvf


def create_learner(
        env_name: str,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        fast_state_centric_planner: bool,
        mixed_finetune_value_loss: str = 'hiql',
        cql_alpha: float = 0.005,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 1,
        high_temperature: float = 1,
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
        low_level_delta: bool = False,
        guiding_reward: float = 0.,
        guiding_reward_xy: bool = False,
        one_step_mode: str = "legacy",
        guiding_v_expectile: float = None,
        guiding_v_dataset_expectile: float = None,
        grad_value_repr: bool = False,
        policy_share_value_state: bool = False,
        use_reconstruction: bool = False,
        vae_coefficient: float = 1.,
        vae_KL_coefficient: float = 0.01,
        use_icvf: bool = False,
        icvf_constraint_alpha: float = 100.,
        icvf_use_psi: bool = False,
        hiql_actor_loss_coefficient: float = 1.0,
        debug: bool = False,
    ):
        if env_name.startswith('topview'):
            visual_hybrid = True
            logger.info(f'Environment is {env_name}, using hybrid visual networks.')
        else:
            visual_hybrid = False

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, high_actor_key, critic_key, value_key = jax.random.split(rng, 5)

        value_state_encoder = None
        value_goal_encoder = None
        policy_state_encoder = None
        policy_goal_encoder = None
        high_policy_state_encoder = None
        high_policy_goal_encoder = None
        if use_reconstruction:
            if visual or visual_hybrid:
                assert use_rep
                raise NotImplementedError
            else:
                def make_encoder():
                    # return RelativeVAEEncoder(rep_dim=rep_dim, hidden_dims=value_hidden_dims, visual=False, layer_norm=use_layer_norm, rep_type=rep_type)
                    return RelativeVAEEncoder(rep_dim=rep_dim, hidden_dims=(512, 256, 128), visual=False, layer_norm=use_layer_norm, rep_type=rep_type)
                def make_decoder():
                    # return VAEDecoder(original_dim=observations.shape[-1] * 2 if rep_type == 'concat' else observations.shape[-1],
                    #                   hidden_dims=value_hidden_dims, visual=False, layer_norm=use_layer_norm)
                    return VAEDecoder(original_dim=observations.shape[-1] * 2 if rep_type == 'concat' else observations.shape[-1],
                                      hidden_dims=(128, 256, 512), visual=False, layer_norm=use_layer_norm)
                if use_rep:
                    value_goal_encoder = make_encoder()
                    value_goal_decoder = make_decoder()
        else:
            if visual or visual_hybrid:
                assert use_rep
                from jaxrl_m.vision import encoders

                visual_encoder = encoders[encoder]
                relative_representation_module = HybridRelativeRepresentation if visual_hybrid else RelativeRepresentation
                def make_encoder(bottleneck):
                    if bottleneck:
                        return relative_representation_module(rep_dim=rep_dim, hidden_dims=(rep_dim,), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
                    else:
                        return relative_representation_module(rep_dim=value_hidden_dims[-1], hidden_dims=(value_hidden_dims[-1],), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

                value_state_encoder = make_encoder(bottleneck=False)
                value_goal_encoder = make_encoder(bottleneck=use_waypoints)
                if not policy_share_value_state:
                    policy_state_encoder = make_encoder(bottleneck=False)
                policy_goal_encoder = make_encoder(bottleneck=False)
                high_policy_state_encoder = make_encoder(bottleneck=False)
                high_policy_goal_encoder = make_encoder(bottleneck=False)
            else:
                def make_encoder(bottleneck):
                    if bottleneck:
                        return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(*value_hidden_dims, rep_dim), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
                    else:
                        return RelativeRepresentation(rep_dim=value_hidden_dims[-1], hidden_dims=(*value_hidden_dims, value_hidden_dims[-1]), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

                if use_rep:
                    value_goal_encoder = make_encoder(bottleneck=True)

        if use_icvf:
            assert not (visual or visual_hybrid)
            # hard-coded pretrained ICVF net size
            # icvf_value_state_encoder, icvf_value_state_connector = create_icvf(
            #     rep_dim=rep_dim, hidden_dims=[256, 256], use_layer_norm=False, create_connector=True)
            icvf_value_state_encoder = create_icvf(
                rep_dim=rep_dim, hidden_dims=[256, 256], use_layer_norm=False, create_connector=False)

        value_def = MonolithicVF(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim)
        if guiding_reward:
            guiding_v_def = SingleVF(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim)

        if discrete:
            action_dim = actions[0] + 1
            actor_def = DiscretePolicy(actor_hidden_dims, action_dim=action_dim)
        else:
            action_dim = actions.shape[-1]
            actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        high_action_dim = observations.shape[-1] if not use_rep else rep_dim
        high_actor_def = Policy(actor_hidden_dims, action_dim=high_action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        if fast_state_centric_planner:
            fast_high_actor_def = Policy(actor_hidden_dims, action_dim=high_action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        networks = {
            'value': value_def,
            'target_value': copy.deepcopy(value_def),
            'actor': actor_def,
            'high_actor': high_actor_def,
        }
        if guiding_reward:
            networks['guiding_v'] = guiding_v_def
        if fast_state_centric_planner:
            networks['fast_high_actor'] = fast_high_actor_def

        decoders = {}
        if use_reconstruction:
            if use_rep:
                decoders['value_goal'] = value_goal_decoder

        icvfs = {}
        if use_icvf:
            icvfs['icvf_value_state_encoder'] = icvf_value_state_encoder
            icvfs['target_icvf_value_state_encoder'] = copy.deepcopy(icvf_value_state_encoder)
            # icvfs['icvf_value_state_connector'] = icvf_value_state_connector

        network_def = HierarchicalActorCriticV2(
            encoders={
                'value_state': value_state_encoder,
                'value_goal': value_goal_encoder,
                'policy_state': policy_state_encoder,
                'policy_goal': policy_goal_encoder,
                'high_policy_state': high_policy_state_encoder,
                'high_policy_goal': high_policy_goal_encoder,
            },
            networks=networks,
            decoders=decoders,
            icvfs=icvfs,
            use_waypoints=use_waypoints,
            low_level_delta=low_level_delta,
            policy_share_value_state=policy_share_value_state,
            use_vae=use_reconstruction,
            use_icvf=use_icvf,
            icvf_use_psi=icvf_use_psi,
        )
        network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
        network_params = network_def.init(value_key, observations, observations)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = unfreeze(network.params)
        params['networks_target_value'] = params['networks_value']
        if use_icvf:
            params['icvfs_target_icvf_value_state_encoder'] = params['icvfs_icvf_value_state_encoder']
        network = network.replace(params=freeze(params))

        config = flax.core.FrozenDict(dict(
            discount=discount, temperature=temperature, high_temperature=high_temperature,
            mixed_finetune_value_loss=mixed_finetune_value_loss, cql_alpha=cql_alpha,
            target_update_rate=tau, pretrain_expectile=pretrain_expectile, way_steps=way_steps, rep_dim=rep_dim,
            policy_train_rep=policy_train_rep,
            use_rep=use_rep, use_waypoints=use_waypoints,
            guiding_reward=guiding_reward, guiding_reward_xy=guiding_reward_xy, one_step_mode=one_step_mode,
            guiding_v_expectile=guiding_v_expectile, guiding_v_dataset_expectile=guiding_v_dataset_expectile,
            grad_value_repr=grad_value_repr, policy_share_value_state=policy_share_value_state,
            use_vae=use_reconstruction, rep_type=rep_type, vae_coefficient=vae_coefficient, vae_KL_coefficient=vae_KL_coefficient,
            use_icvf=use_icvf, icvf_constraint_alpha=icvf_constraint_alpha,
            hiql_actor_loss_coefficient=hiql_actor_loss_coefficient,
            env_name=env_name,
        ))

        if fast_state_centric_planner:
            return scpiql.SCPIQLAgent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config)
        elif use_icvf:
            return icvf.ICVFAgent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config)
        else:
            return hiql.JointTrainAgent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config)
