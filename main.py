import os
# Disable JAX GPU memory preallocation behavior
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from datetime import datetime
from functools import partial
from pathlib import Path
import time

import numpy as np
import jax
import flax
import wandb
from loguru import logger
import tqdm
import pickle

from jaxrl_m.evaluation import evaluate_with_trajectories as hiql_evaluate_with_trajectories

from afrl.config import get_parser
from afrl.d4rl_utils import make_env, get_dataset
from afrl.env_utils import other_make_env, other_get_dataset
from afrl.env_utils import make_didactic_maze_env, get_didactic_maze_dataset
from afrl.algorithm import afiql, hiql
from afrl.utils import CsvLogger
from afrl.video_utils import record_video
from afrl.evaluation import evaluate_with_trajectories
from afrl.ant_utils import d4rl_ant, ant_diagnostics
from afrl.plot_utils import get_traj_v, hiql_get_traj_v
from afrl import viz_utils
from afrl.gc_dataset import GCSDataset


def setup_wandb(conf):
    wandb.init(
        project=conf.wandb_project,
        entity=conf.wandb_entity,  # change to your own entity name
        config=conf,
        group=f"{conf.env_name}" if conf.wandb_group is None else conf.wandb_group,
        job_type=conf.job_type if conf.job_type is not None else "train",
        name=conf.run_name if conf.run_name is not None else f"seed{conf.seed}",
        notes=conf.notes,
        dir=conf.workdir,
        mode=conf.wandb_mode,
        tags=conf.wandb_tags,
        save_code=True,
    )


@jax.jit
def get_debug_statistics(agent, batch):
    def get_info(s):
        return agent.network(s, info=True, method='value', seed=jax.random.PRNGKey(0))

    s = batch['observations']

    info = get_info(s)

    stats = {}

    stats.update({
        'v': info['v'].mean(),
    })

    return stats


@jax.jit
def hiql_get_debug_statistics(agent, batch):
    def get_info(s, g):
        return agent.network(s, g, info=True, method='value', seed=jax.random.PRNGKey(0))

    s = batch['observations']
    g = batch['goals']

    info = get_info(s, g)

    stats = {}

    stats.update({
        'v': info['v'].mean(),
    })

    return stats


def main():
    parser = get_parser()
    args = parser.parse_args()

    # arguments check
    if args.pretrain_alg in {'hiql', 'scpiql'} and args.append_target is True:
        raise ValueError(f"For consistency with HIQL, append_target must be set to False!")

    tag = f'{datetime.now().strftime("%y%m%d_%H%M%S")}'
    workdir = Path(f'logs/{tag}')
    workdir.mkdir(parents=False, exist_ok=False)
    args.workdir = workdir.as_posix()

    args.config = {}
    args.config['pretrain_expectile'] = args.pretrain_expectile
    args.config['discount'] = args.discount
    args.config['temperature'] = args.temperature
    args.config['value_hidden_dims'] = (args.value_hidden_dim,) * args.value_num_layers
    args.config['use_rep'] = args.use_rep
    args.config['rep_dim'] = args.rep_dim
    args.config['policy_train_rep'] = args.policy_train_rep
    if args.pretrain_alg in {'hiql', 'scpiql'}:
        args.config['high_temperature'] = args.high_temperature
        args.config['use_waypoints'] = args.use_waypoints
        args.config['way_steps'] = args.way_steps

    setup_wandb(args)
    wandb_run_dir = Path(str(wandb.run.dir))
    save_dir = wandb_run_dir.joinpath('checkpoints')
    save_dir.mkdir(parents=False, exist_ok=False)

    if 'antmaze' in args.env_name:
        env = make_env(args.env_name, append_target=args.append_target, set_render=True)
        dataset = get_dataset(env, env_name=args.env_name, append_target=args.append_target)
        dataset_info = {'discrete': False}
    elif 'kitchen' in args.env_name:
        env = make_env(args.env_name, set_render=True)
        dataset = get_dataset(env, env_name=args.env_name, filter_terminals=True)
        dataset_info = {'discrete': False}
    elif 'calvin' in args.env_name:
        env = other_make_env(args.env_name)
        dataset, dataset_info = other_get_dataset(args.env_name)
    elif 'procgen' in args.env_name:
        env = other_make_env(args.env_name)
        dataset, dataset_info = other_get_dataset(args.env_name)
    elif 'didactic_maze' in args.env_name:
        env = make_didactic_maze_env(args.env_name)
        dataset, dataset_info = get_didactic_maze_dataset(args.env_name)
    else:
        raise NotImplementedError(f"Environment {args.env_name} has not been implemented.")
    
    """ Create learning agent """
    logger.info(f"Creating learning agent...")
    example_batch = dataset.sample(1)
    if args.pretrain_alg == 'afiql':
        agent = afiql.create_learner(
            args.seed,
            example_batch['observations'],
            example_batch['actions'] if not dataset_info['discrete'] else dataset_info['example_action'],
            visual=args.visual,
            encoder="impala",
            discrete=dataset_info['discrete'],
            use_layer_norm=args.use_layer_norm,
            rep_type=args.rep_type,
            **args.config,
        )
    elif args.pretrain_alg in {'hiql', 'scpiql'}:
        goal_info = None
        args.gcdataset = {}
        args.gcdataset['p_randomgoal'] = args.p_randomgoal
        args.gcdataset['p_trajgoal'] = args.p_trajgoal
        args.gcdataset['p_currgoal'] = args.p_currgoal
        args.gcdataset['geom_sample'] = args.geom_sample
        args.gcdataset['high_p_randomgoal'] = args.high_p_randomgoal
        args.gcdataset['way_steps'] = args.way_steps
        args.gcdataset['discount'] = args.discount
        args.gcdataset['reward_scale'] = 1.0    # WARNING: in original HIQL, GCSDataset.reward_shift = 0.0.
        args.gcdataset['reward_shift'] = -1.0   # HIQL minuse 1 from reward in hiql compute_value_loss.
        args.gcdataset['terminal'] = True       # We set reward and terminal right in dataset sampling.
        args.gcdataset['p_aug'] = args.p_aug
        pretrain_dataset = GCSDataset(dataset, **args.gcdataset)
        agent = hiql.create_learner(
            args.env_name,
            args.seed,
            example_batch['observations'],
            example_batch['actions'] if not dataset_info['discrete'] else dataset_info['example_action'],
            fast_state_centric_planner=(args.pretrain_alg == 'scpiql'),
            visual=args.visual,
            encoder="impala",
            discrete=dataset_info['discrete'],
            use_layer_norm=args.use_layer_norm,
            rep_type=args.rep_type,
            use_reconstruction=args.use_reconstruction,
            vae_coefficient=args.vae_coefficient,
            vae_KL_coefficient=args.vae_KL_coefficient,
            **args.config
        )

    env.reset()

    rng_key = jax.random.PRNGKey(seed=args.seed)

    # For debugging metrics
    if 'antmaze' in args.env_name:
        if args.pretrain_alg == 'afiql':
            example_trajectory = dataset.sample(50, indx=np.arange(1000, 1050))
        elif args.pretrain_alg in {'hiql', 'scpiql'}:
            example_trajectory = pretrain_dataset.sample(50, indx=np.arange(1000, 1050))
    elif 'kitchen' in args.env_name or 'calvin' in args.env_name:
        if args.pretrain_alg == 'afiql':
            example_trajectory = dataset.sample(50, indx=np.arange(0, 50))
        elif args.pretrain_alg in {'hiql', 'scpiql'}:
            example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    elif 'procgen-500' in args.env_name or 'procgen-1000' in args.env_name:
        if args.pretrain_alg == 'afiql':
            example_trajectory = dataset.sample(50, indx=np.arange(5000, 5050))
        elif args.pretrain_alg in {'hiql', 'scpiql'}:
            example_trajectory = pretrain_dataset.sample(50, indx=np.arange(5000, 5050))
    elif 'didactic_maze' in args.env_name:
        if args.pretrain_alg == 'afiql':
            example_trajectory = dataset.sample(10, indx=np.arange(0, 10))
        elif args.pretrain_alg in {'hiql', 'scpiql'}:
            example_trajectory = pretrain_dataset.sample(10, indx=np.arange(0, 10))
    else:
        raise NotImplementedError
    
    train_logger = CsvLogger(wandb_run_dir.joinpath('train.csv'))
    eval_logger = CsvLogger(wandb_run_dir.joinpath('eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, args.pretrain_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        rng_key, agent_train_seed = jax.random.split(rng_key)
        if args.pretrain_alg == 'afiql':
            pretrain_batch = dataset.sample(args.batch_size)
            agent, update_info = agent.pretrain_update(pretrain_batch, seed=agent_train_seed, actor_loss=args.actor_loss, value_update=True, actor_update=True)
        elif args.pretrain_alg in {'hiql', 'scpiql'}:
            pretrain_batch = pretrain_dataset.sample(args.batch_size)
            agent, update_info = agent.pretrain_update(pretrain_batch, seed=agent_train_seed, value_update=True, actor_update=True, high_actor_update=True)
        data_metrics = {}
        data_metrics[f'data/dataset_reward'] = pretrain_batch['rewards'].mean()

        # Logging
        if i % args.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update(data_metrics)
            if args.pretrain_alg == 'afiql':
                debug_statistics = get_debug_statistics(agent, pretrain_batch)
            elif args.pretrain_alg in {'hiql', 'scpiql'}:
                debug_statistics = hiql_get_debug_statistics(agent, pretrain_batch)
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / args.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)
        
        # Evaluation
        if i == 1 or i % args.eval_interval == 0:
            logger.info(f"Running evaluation at epoch {i}: {args.eval_episodes} episodes + {args.num_video_episodes} video episodes")
            if 'procgen' in args.env_name:
                assert args.pretrain_alg in {'hiql', 'scpiql'}, f"procgen only supports HIQL and SCPIQL"
                policy_fn = partial(agent.sample_actions, discrete=dataset_info['discrete'], temperature=0.)
                high_policy_fn = partial(agent.sample_high_actions, temperature=0.)
                policy_rep_fn = agent.get_policy_rep
                base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
                eval_metrics = {}
                for goal_info in dataset_info['goal_infos']:
                    rng_key, eval_seed = jax.random.split(rng_key)
                    eval_info, trajs, renders = hiql_evaluate_with_trajectories(
                        policy_fn, high_policy_fn, policy_rep_fn, env, env_name=args.env_name, seed=eval_seed,
                        num_episodes=args.eval_episodes,
                        base_observation=base_observation, num_video_episodes=args.num_video_episodes,
                        use_waypoints=args.use_waypoints,
                        epsilon=0.05,
                        goal_info=goal_info, config=args.config,
                    )
                    eval_metrics.update({f'evaluation/level{goal_info["eval_level_name"]}_{k}': v for k, v in eval_info.items()})
            else:
                if args.pretrain_alg == 'afiql':
                    policy_fn = partial(agent.sample_actions, discrete=dataset_info['discrete'], temperature=0.)
                    rng_key, eval_seed = jax.random.split(rng_key)
                    eval_info, trajs, renders, value_preds = evaluate_with_trajectories(
                        policy_fn, env, env_name=args.env_name, seed=eval_seed,
                        num_episodes=args.eval_episodes, num_video_episodes=args.num_video_episodes)
                elif args.pretrain_alg in {'hiql', 'scpiql'}:
                    policy_fn = partial(agent.sample_actions, discrete=dataset_info['discrete'], temperature=0.)
                    high_policy_fn = partial(agent.sample_high_actions, temperature=0.)
                    policy_rep_fn = agent.get_policy_rep
                    base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
                    rng_key, eval_seed = jax.random.split(rng_key)
                    eval_info, trajs, renders = hiql_evaluate_with_trajectories(
                        policy_fn, high_policy_fn, policy_rep_fn, env, env_name=args.env_name, seed=eval_seed,
                        num_episodes=args.eval_episodes,
                        base_observation=base_observation, num_video_episodes=args.num_video_episodes,
                        use_waypoints=args.use_waypoints,
                        goal_info=goal_info, config=args.config)
                else:
                    raise NotImplementedError

                eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}

            if args.num_video_episodes > 0:
                video = record_video('Video', i, renders=renders)
                eval_metrics['video'] = video

            if args.pretrain_alg == 'afiql':
                traj_values = {}
                traj_values["example_trajectory"] = get_traj_v(agent, example_trajectory)['value_preds']
                for j in range(0, args.num_video_episodes):
                    traj_values[f'trajectory_{j}'] = get_traj_v(agent, trajs[args.eval_episodes + j])['value_preds']
                value_viz = viz_utils.make_visual_no_image(
                    traj_values,
                    [partial(viz_utils.visualize_metric, metric_name=k) for k in traj_values.keys()]
                )
                eval_metrics['value_traj_viz'] = wandb.Image(value_viz)
            elif args.pretrain_alg in {'hiql', 'scpiql'}:
                traj_metrics = hiql_get_traj_v(agent, example_trajectory)
                value_viz = viz_utils.make_visual_no_image(
                    traj_metrics,
                    [partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
                )
                eval_metrics['value_traj_viz'] = wandb.Image(value_viz)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Checkpoint
        if args.save_interval > 0 and i % args.save_interval == 0:
            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
                config=args.config
            )
            fname = save_dir.joinpath(f'params_{i}.pkl')
            logger.info(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)
    
    # Save final model
    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
        config=args.config
    )
    fname = save_dir.joinpath(f'params_final.pkl')
    logger.info(f'Saving to {fname}')
    with open(fname, "wb") as f:
        pickle.dump(save_dict, f)

    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    main()
