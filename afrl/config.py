from typing import Optional

import argparse


def get_parser_env(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    parser = parser if parser is not None else argparse.ArgumentParser(f"environment args")
    parser.add_argument('--env_name', type=str, default='antmaze-large-diverse-v2', help='environment name (default antmaze-large-diverse-v2)')
    parser.add_argument('--append_target', action='store_true', help="(antmaze) append target goal to observation (default: False)")
    parser.add_argument('--frame_skip', type=int, default=1, help='frame skip (default: 1)')
    return parser


def get_parser_wandb(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    parser = parser if parser is not None else argparse.ArgumentParser(f"wandb args")
    parser.add_argument('--wandb_mode', type=str, default="online", choices=["online", "offline", "disabled"], help='wandb mode')
    parser.add_argument('--run_name', type=str, default=None, help="wandb run name")
    parser.add_argument('--job_type', type=str, default=None, help="wandb job_type")
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")
    parser.add_argument('--wandb_tags', nargs='*', type=str, help="wandb tags")
    parser.add_argument('--wandb_project', type=str, default='action-free rl', help='wandb project')
    parser.add_argument('--wandb_entity', type=str, default='set your own wandb entity', help='wandb entity')
    parser.add_argument('--wandb_group', type=str, default=None, help='wandb group, default to environment name')
    return parser


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Action-Free RL Pre-Training")

    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

    # Env
    parser = get_parser_env(parser)

    # Training (General)
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for pre-training (default: 256)')
    parser.add_argument('--log_interval', type=int, default=1000, help='interval for logging (default: 1000)')
    parser.add_argument('--eval_interval', type=int, default=25000, help='interval for evaluation (default: 25000)')
    parser.add_argument('--eval_episodes', type=int, default=50, help='number of episodes for evaluation (default: 50)')
    parser.add_argument('--num_video_episodes', type=int, default=2, help='number of video episodes for evaluation (default: 2)')
    parser.add_argument('--save_interval', type=int, default=250000, help='interval for saving (default: 250000)')

    # Pretraining
    parser.add_argument('--pretrain_alg', type=str, default='afiql', choices=['afiql', 'hiql', 'scpiql'], help='pre-training algorithm (default: afiql)')
    parser.add_argument('--pretrain_steps', type=int, default=1000000, help='number of pre-training steps (default: 1000000)')

    # Finetuning
    parser.add_argument('--data_source', type=str, choices=['online', 'offline', 'mixed'], default='online', help='data source for finetuning (default: online)')
    parser.add_argument('--no_pretrain', action='store_true', help='update all components when data_source=online, do not load checkpoints this is used for pretrain ablation (default: False)')
    parser.add_argument('--no_updata_value', action='store_true', help='do not update value function (default: False)')

    parser.add_argument('--finetune_alg', type=str, choices=['iql', 'td3', 'hiql', 'guided_sac', 'scpiql'], default='iql', help='finetune algorithm (default: iql)')
    parser.add_argument('--mixed_finetune_value_loss', type=str, choices=['hiql', 'hiql_cql', 'hiql_cql_v2', 'hiql_cql_v3'], default='hiql', help='loss for training value function in finetuning (default: hiql)')
    parser.add_argument('--cql_alpha', type=float, default=0.005, help='alpha in CQL loss')

    parser.add_argument('--auto_find_checkpoint', action='store_true', help='automatically find checkpoint (it overrides checkpoint_file and wandb_checkpoint_run) (default: False)')
    parser.add_argument('--checkpoint_file', type=str, default=None, help='checkpoint file (default: None)')
    parser.add_argument('--wandb_checkpoint_run', type=str, default=None, help='wandb run id for checkpoint (default: None)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of environment workers (default: 8)')
    parser.add_argument('--max_env_steps', type=int, default=10000000, help='maximum number of environment steps (default: 10000000)')
    parser.add_argument('--epoch_steps', type=int, default=200, help='number of environment steps per epoch per worker (default: 200)')
    parser.add_argument('--warmup_steps', type=int, default=200, help='number of environment steps for replay buffer warmup (default: 200)')
    parser.add_argument('--replay_buffer_size', type=int, default=300000, help='size of replay buffer (default: 300000)')
    parser.add_argument('--train_iters', type=int, default=4, help='number of training iterations per epoch (default: 4)')
    parser.add_argument('--explore_temperature', type=float, default=1.0, help='temperature for exploration (default: 1.0)')

    # Algorithm
    parser.add_argument('--pretrain_expectile', type=float, default=0.7, help='expectile for pre-training (default: 0.7)')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for IQL (default: 1.0)')
    parser.add_argument('--actor_loss', type=str, default='hiql_original', choices=['hiql_original', 'vem'], help='loss for training actor (default: hiql_original)')

    # Algorithm: TD3
    parser.add_argument('--td3_max_action', type=float, default=1.0, help='td3 max action (default: 1.0)')
    parser.add_argument('--td3_policy_noise', type=float, default=0.2, help='td3 policy noise (default: 0.2)')
    parser.add_argument('--td3_noise_clip', type=float, default=0.5, help='td3 noise clip (default: 0.5)')
    parser.add_argument('--td3_expl_noise', type=float, default=0.1, help='td3 exploration noise (default: 0.1)')
    parser.add_argument('--td3_policy_freq', type=int, default=2, help='td3 policy update frequency (default: 2)')

    # Finetuning: ICVF
    parser.add_argument('--icvf_constraint_alpha', type=float, default=100., help='parameter alpha of ICVF constraint in finetuning (default: 100)')
    parser.add_argument('--icvf_use_psi', action='store_true', help='whether to use ICVF psi as value goal representation')

    # Algorithm: HIQL:finetune
    parser.add_argument('--hiql_finetune_actor_supervision', type=str, default='value', choices=['value', 'subgoal_L2'], help='which source of supervision for training actor (default: value)')
    parser.add_argument('--low_level_delta', action='store_true', help='whether to use state_xy - goal_xy as part of state representation for low-level policy')
    parser.add_argument('--sample_scheme', type=str, default='original', choices=['original', 'her', 'plain', 'herV2'], help='sample scheme for HIQL (default: original)')
    parser.add_argument('--relabelling_prob', type=float, default=0.8, help='probability for goal relabelling (default: 0.8)')
    parser.add_argument('--ensemble_checkpoint_file', nargs='*', type=str, default=None, help='checkpoint file for ensemble (default: None)')
    parser.add_argument('--ensemble_wandb_checkpoint_run', nargs='*', type=str, default=None, help='wandb run id for ensemble checkpoint (default: None)')
    parser.add_argument('--ensemble_type', type=str, default='mean', choices=['mean', 'min'], help='ensemble type (default: mean)')
    parser.add_argument('--high_actor_update', action='store_true', help='whether to update high-level actor (default: False)')
    parser.add_argument('--guiding_reward', type=float, default=0.0, help='guiding reward (default: 0.0)')
    parser.add_argument('--guiding_reward_xy', action='store_true', help='whether to use guiding reward in xy space (default: False)')
    parser.add_argument('--hiql_actor_loss_coefficient', type=float, default=1.0, help='hiql actor loss coefficient')
    parser.add_argument('--single_step_checkpoint_file', type=str, default=None, help='checkpoint file for single step (default: None)')
    parser.add_argument('--debug', action='store_true', help='whether to use debug mode (default: False)')
    parser.add_argument('--one_step_mode', type=str, default='legacy', choices=['legacy', 'v2'], help='one step mode (default: None)')
    parser.add_argument('--guiding_v_expectile', type=float, default=None,
                        help='expectile for guiding V, same as pretrain_expectile if set to None, disabled if set to 0 (default: None)')
    parser.add_argument('--guiding_v_dataset_expectile', type=float, default=None,
                        help='expectile for guiding V on offline dataset, same as pretrain_expectile if set to None, disabled if set to 0 (default: None)')
    parser.add_argument('--grad_value_repr', action='store_true', help='whether to finetune representation in value training (default: False)')
    parser.add_argument('--policy_share_value_state', action='store_true', help='whether to share state representation between policy and value (default: False)')
    
    parser.add_argument('--use_fast_planner_as_high_policy', action='store_true', help='use fast state centric planner to propose subgoal (default: False)')

    # Network
    parser.add_argument('--value_hidden_dim', type=int, default=512, help='hidden dimension for value network (default: 512)')
    parser.add_argument('--value_num_layers', type=int, default=3, help='number of layers for value network (default: 3)')
    parser.add_argument('--use_rep', type=int, default=0, help='whether to use representation (default: 0)')
    parser.add_argument('--rep_dim', type=int, default=None, help='dimension of representation (default: None)')
    parser.add_argument('--rep_type', type=str, default="state", choices=['state', 'diff', 'concat'], help='type of representation (default: state)')
    parser.add_argument('--policy_train_rep', type=int, default=0, help='whether to train representation for policy (default: 0)')
    parser.add_argument('--use_layer_norm', type=int, default=1, help='whether to use layer norm (default: 1)')
    parser.add_argument('--visual', type=int, default=0, help='whether the environment observation is visual (default: 0)')
    parser.add_argument('--use_reconstruction', action='store_true', help='whether to use VAE and reconstruction loss to train (default: False)')
    parser.add_argument('--vae_coefficient', type=float, default=1.0, help='coefficient for VAE loss (default: 1.0)')
    parser.add_argument('--vae_KL_coefficient', type=float, default=0.01, help='coefficient for VAE KL loss (default: 0.01)')

    # Goal-conditioned (used by HIQL)
    parser.add_argument('--p_randomgoal', type=float, default=0.3, help='probability of random goal (default: 0.3)')
    parser.add_argument('--p_trajgoal', type=float, default=0.5, help='probability of trajectory goal (default: 0.5)')
    parser.add_argument('--p_currgoal', type=float, default=0.2, help='probability of current goal (default: 0.2)')
    parser.add_argument('--geom_sample', type=int, default=1, help='whether to sample geometry (default: 1)')
    parser.add_argument('--high_p_randomgoal', type=float, default=0.3, help='probability of random goal (default: 0.3)')
    parser.add_argument('--way_steps', type=int, default=25, help='number of steps for waypoint (default: 25)')
    parser.add_argument('--use_waypoints', type=int, default=1, help='whether to use waypoints (default: 1)')
    parser.add_argument('--high_temperature', type=float, default=1.0, help='temperature for high-level policy (default: 1.0)')
    parser.add_argument('--p_aug', type=float, default=None, help='probability of augmentation (default: 0.0)')

    # wandb
    parser = get_parser_wandb(parser)
    return parser
