import os
# Disable JAX GPU memory preallocation behavior
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from collections import defaultdict
import datetime
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
from afrl.d4rl_utils import make_env, get_dataset, subproc_venv_env_fn
from afrl.env_utils import other_make_env, other_get_dataset, other_subproc_venv_env_fn, make_procgen_env
from afrl.algorithm import afiql, td3, hiql
from afrl.utils import CsvLogger
from afrl.video_utils import record_video
from afrl.evaluation import evaluate_with_trajectories, add_to, dict_mean
from afrl.envs.venvs import SubprocVectorEnv, DummyVectorEnv
from afrl.collector import Collector, DummyDatasetCollector, GCCollector
from afrl.plot_utils import get_traj_v, hiql_get_traj_v
from afrl import viz_utils
from afrl.gc_dataset import GCSDataset
from afrl.algorithm.checkpoint import check_hiql_loading, load_hiql_checkpoint, find_pretrained_checkpoint

from main import setup_wandb, get_debug_statistics, hiql_get_debug_statistics


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    # arguments check
    if args.finetune_alg in {'hiql', 'scpiql'} and args.append_target is True:
        raise ValueError(f"For consistency with HIQL, append_target must be set to False!")
    if args.use_fast_planner_as_high_policy:
        assert args.finetune_alg in {'hiql', 'scpiql'}, "Fast planner only works for HIQL and SCPIQL."
        assert args.guiding_reward, "Fast planner only works with guiding reward."
        assert args.use_waypoints, "Using fast planner as high policy only works with waypoints."
        assert args.way_steps == 1, "Using fast planner as high policy only works with way_steps == 1."

    tag = f'{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}'
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
    if args.finetune_alg in {'hiql', 'scpiql'}:
        args.config['high_temperature'] = args.high_temperature
        args.config['use_waypoints'] = args.use_waypoints
        args.config['way_steps'] = args.way_steps

    setup_wandb(args)
    wand_run_dir = Path(str(wandb.run.dir))
    save_dir = wand_run_dir.joinpath('checkpoints')
    save_dir.mkdir(parents=False, exist_ok=False)

    if 'antmaze' in args.env_name:
        env = make_env(args.env_name, append_target=args.append_target, set_render=True, frame_skip=args.frame_skip)
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
    else:
        raise NotImplementedError(f"Environment {args.env_name} has not been implemented.")
    

    """ Create learning agent """
    logger.info(f"Creating learning agent...")
    example_batch = dataset.sample(1)
    if args.finetune_alg == 'iql':
        agent = afiql.create_learner(
            args.seed,
            example_batch['observations'],
            example_batch['actions'] if not dataset_info['discrete'] else dataset_info['example_action'],
            mixed_finetune_value_loss=args.mixed_finetune_value_loss,
            cql_alpha=args.cql_alpha,
            visual=args.visual,
            encoder="impala",
            discrete=dataset_info['discrete'],
            use_layer_norm=args.use_layer_norm,
            rep_type=args.rep_type,
            **args.config,
        )
        pretrained_agent = agent
    elif args.finetune_alg == 'td3':
        assert 'antmaze' in args.env_name
        agent = td3.create_learner(
            args.seed,
            example_batch['observations'],
            example_batch['actions'] if not dataset_info['discrete'] else dataset_info['example_action'],
            use_pretrained_v=True,
            max_action=args.td3_max_action - 1e-5,
            policy_noise=args.td3_policy_noise,
            noise_clip=args.td3_noise_clip,
            expl_noise=args.td3_expl_noise,
            policy_freq=args.td3_policy_freq,
            discount=args.discount,
            value_hidden_dims=(args.value_hidden_dim,) * args.value_num_layers,
            rep_dim=args.rep_dim,
            use_rep=args.use_rep,
            policy_train_rep=args.policy_train_rep,
            visual=args.visual,
            use_layer_norm=args.use_layer_norm,
        )
        pretrained_agent = afiql.create_learner(
            args.seed,
            example_batch['observations'],
            example_batch['actions'] if not dataset_info['discrete'] else dataset_info['example_action'],
            mixed_finetune_value_loss=args.mixed_finetune_value_loss,
            cql_alpha=args.cql_alpha,
            visual=args.visual,
            encoder="impala",
            discrete=dataset_info['discrete'],
            use_layer_norm=args.use_layer_norm,
            rep_type=args.rep_type,
            **args.config,
        )   # We use iql for pretraining
    elif args.finetune_alg in {'hiql', 'scpiql'}:
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
        args.gcdataset['reward_shift'] = -1.0   # HIQL minus 1 from reward in hiql compute_value_loss.
        args.gcdataset['terminal'] = True       # We set reward and terminal right in dataset sampling.
        args.gcdataset['p_aug'] = args.p_aug
        pretrain_dataset = GCSDataset(dataset, **args.gcdataset)
        agent = hiql.create_learner(
            args.env_name,
            args.seed,
            example_batch['observations'],
            example_batch['actions'] if not dataset_info['discrete'] else dataset_info['example_action'],
            fast_state_centric_planner=(args.finetune_alg == 'scpiql'),
            mixed_finetune_value_loss=args.mixed_finetune_value_loss,
            cql_alpha=args.cql_alpha,
            visual=args.visual,
            encoder="impala",
            discrete=dataset_info['discrete'],
            use_layer_norm=args.use_layer_norm,
            rep_type=args.rep_type,
            low_level_delta=args.low_level_delta,
            guiding_reward=args.guiding_reward,
            guiding_reward_xy=args.guiding_reward_xy,
            one_step_mode=args.one_step_mode,
            guiding_v_expectile=args.guiding_v_expectile,
            guiding_v_dataset_expectile=args.guiding_v_dataset_expectile,
            grad_value_repr=args.grad_value_repr,
            policy_share_value_state=args.policy_share_value_state,
            hiql_actor_loss_coefficient=args.hiql_actor_loss_coefficient,
            debug=args.debug,
            **args.config
        )
        pretrained_agent = agent
        if args.low_level_delta or args.guiding_reward:    # the pretrained_agent does not turn on low_level_delta or guiding_reward
            pretrained_agent = hiql.create_learner(
                args.env_name,
                args.seed,
                example_batch['observations'],
                example_batch['actions'] if not dataset_info['discrete'] else dataset_info['example_action'],
                fast_state_centric_planner=(args.finetune_alg == 'scpiql'),
                visual=args.visual,
                encoder="impala",
                discrete=dataset_info['discrete'],
                use_layer_norm=args.use_layer_norm,
                rep_type=args.rep_type,
                policy_share_value_state=args.policy_share_value_state,
                debug=args.debug,
                **args.config
            )
    else:
        raise NotImplementedError
    logger.info(f"Agent created.")


    """ Load pre-trained weights """
    if not (args.no_pretrain and args.finetune_alg == 'scpiql'):
        logger.info(f"Loading checkpoint...")
        
        if args.auto_find_checkpoint:
            checkpoint_file = find_pretrained_checkpoint(base_path=Path('').absolute(), conf=args)
            agent, pretrained_agent = load_hiql_checkpoint(agent, pretrained_agent, f"{args.wandb_entity}/{args.wandb_project}", checkpoint_file, wandb_checkpoint_run=None,
                                                        finetune_alg=args.finetune_alg, use_rep=args.use_rep, visual=args.visual, use_waypoints=args.use_waypoints,
                                                        policy_share_value_state=args.policy_share_value_state)
        else:
            agent, pretrained_agent = load_hiql_checkpoint(agent, pretrained_agent, f"{args.wandb_entity}/{args.wandb_project}", args.checkpoint_file, args.wandb_checkpoint_run,
                                                        finetune_alg=args.finetune_alg, use_rep=args.use_rep, visual=args.visual, use_waypoints=args.use_waypoints,
                                                        policy_share_value_state=args.policy_share_value_state)
        check_hiql_loading(agent, pretrained_agent, finetune_alg=args.finetune_alg, use_rep=args.use_rep, visual=args.visual, use_waypoints=args.use_waypoints,
                        policy_share_value_state=args.policy_share_value_state)
        # logger.warning('NO CHECKPOINTS!!!')
        logger.info(f"Checkpoint loaded.")
    else:
        logger.warning(f"NO_PRETRAIN is True, no checkpoint is loaded!")
    del pretrained_agent

    if args.guiding_reward and args.finetune_alg == 'hiql':
        single_step_agent = hiql.create_learner(
            args.env_name,
            args.seed,
            example_batch['observations'],
            example_batch['actions'] if not dataset_info['discrete'] else dataset_info['example_action'],
            fast_state_centric_planner=False,
            mixed_finetune_value_loss=args.mixed_finetune_value_loss,
            cql_alpha=args.cql_alpha,
            visual=args.visual,
            encoder="impala",
            discrete=dataset_info['discrete'],
            use_layer_norm=args.use_layer_norm,
            rep_type=args.rep_type,
            debug=args.debug,
            **args.config
        )
        single_step_pretrained = single_step_agent
        single_step_agent, single_step_pretrained = load_hiql_checkpoint(single_step_agent, single_step_pretrained, f"{args.wandb_entity}/{args.wandb_project}",
                                                                         args.single_step_checkpoint_file, None,
                                                                         finetune_alg=args.finetune_alg, use_rep=args.use_rep, visual=args.visual, use_waypoints=args.use_waypoints,
                                                                         policy_share_value_state=False)
        check_hiql_loading(single_step_agent, single_step_pretrained, finetune_alg=args.finetune_alg, use_rep=args.use_rep, visual=args.visual, use_waypoints=args.use_waypoints,
                           policy_share_value_state=False)
        del single_step_pretrained
        logger.info(f"Single-step agent loaded.")
    else:
        single_step_agent = None

    ensemble_agents = None
    if (args.ensemble_checkpoint_file is not None or args.ensemble_wandb_checkpoint_run is not None) and args.finetune_alg == 'hiql':
        if args.ensemble_checkpoint_file is not None and args.ensemble_wandb_checkpoint_run is not None:
            raise ValueError("Must specify exactly one of ensemble_checkpoint_file and ensemble_wandb_checkpoint_run.")
        if args.ensemble_checkpoint_file is not None:
            checkpoint_file_list = args.ensemble_checkpoint_file
            wandb_checkpoint_run_list = [None] * len(checkpoint_file_list)
        else:
            wandb_checkpoint_run_list = args.ensemble_wandb_checkpoint_run
            checkpoint_file_list = [None] * len(wandb_checkpoint_run_list)
        assert args.finetune_alg == 'hiql' and not args.low_level_delta, "Ensemble only works for HIQL without low_level_delta."
        logger.info(f"Loading ensemble checkpoints...")
        ensemble_agents = []
        for i in range(len(checkpoint_file_list)):
            agent_i = hiql.create_learner(
                args.env_name,
                args.seed,
                example_batch['observations'],
                example_batch['actions'] if not dataset_info['discrete'] else dataset_info['example_action'],
                fast_state_centric_planner=False,
                mixed_finetune_value_loss=args.mixed_finetune_value_loss,
                cql_alpha=args.cql_alpha,
                visual=args.visual,
                encoder="impala",
                discrete=dataset_info['discrete'],
                use_layer_norm=args.use_layer_norm,
                rep_type=args.rep_type,
                low_level_delta=args.low_level_delta,   # TODO: pre-trained model should not have this, this should throw an error
                policy_share_value_state=args.policy_share_value_state,
                debug=args.debug,
                **args.config
            )
            pretrained_agent_i = agent_i
            agent_i, pretrained_agent_i = load_hiql_checkpoint(agent_i, pretrained_agent_i, f"{args.wandb_entity}/{args.wandb_project}",
                                                               checkpoint_file_list[i], wandb_checkpoint_run_list[i],
                                                               finetune_alg=args.finetune_alg, use_rep=args.use_rep, visual=args.visual, use_waypoints=args.use_waypoints,
                                                               policy_share_value_state=args.policy_share_value_state)
            check_hiql_loading(agent_i, pretrained_agent_i, finetune_alg=args.finetune_alg, use_rep=args.use_rep, visual=args.visual, use_waypoints=args.use_waypoints,
                               policy_share_value_state=args.policy_share_value_state)
            ensemble_agents.append(agent_i)
        logger.info(f"Ensemble checkpoints loaded.")


    """ Create subproc vectorized environments """
    logger.info(f"Creating subproc vectorized envs...")
    child_seeds = [args.seed * 1000 + i for i in range(args.num_workers)]
    if 'antmaze' in args.env_name or 'kitchen' in args.env_name:
        if 'topview' in args.env_name:
            # We need rendering, so cannot use subproc vectorized envs
            venv = DummyVectorEnv([partial(subproc_venv_env_fn, args.env_name, seed=child_seeds[i], append_target=args.append_target, set_render=True, frame_skip=args.frame_skip) for i in range(args.num_workers)])
        else:
            venv = SubprocVectorEnv([partial(subproc_venv_env_fn, args.env_name, seed=child_seeds[i], append_target=args.append_target, set_render=False, frame_skip=args.frame_skip) for i in range(args.num_workers)])
    elif 'calvin' in args.env_name:
        venv = DummyVectorEnv([partial(other_subproc_venv_env_fn, args.env_name, seed=child_seeds[i]) for i in range(args.num_workers)])
    elif 'procgen' in args.env_name:
        venv = make_procgen_env(args.env_name, num_envs=args.num_workers, seed=child_seeds[-1])
    else:
        raise NotImplementedError(f"Environment {args.env_name} has not been implemented.")
    logger.info(f"Subproc vectorized envs created.")


    """ Create data collector & replay buffer """
    rng_key = jax.random.PRNGKey(seed=args.seed)
    
    rng_key, collector_seed = jax.random.split(rng_key)
    if args.data_source == 'offline':
        if args.finetune_alg in {'hiql', 'scpiql'}:
            collector = DummyDatasetCollector(pretrain_dataset, rng_key=collector_seed)
        else:
            collector = DummyDatasetCollector(dataset, rng_key=collector_seed)
    else:
        if args.finetune_alg in {'hiql', 'scpiql'}:
            collector = GCCollector(args.env_name, venv, rng_key=collector_seed, replay_buffer_size=args.replay_buffer_size, 
                                    sample_scheme=args.sample_scheme, relabelling_prob=args.relabelling_prob, gc_configs=args.gcdataset, use_rep=args.use_rep)
        else:
            collector = Collector(venv, rng_key=collector_seed, replay_buffer_size=args.replay_buffer_size)
    collector.reset()
    replay_buffer = collector.replay_buffer
    # warmup
    logger.info(f"Warming up replay buffer...")
    collector.warmup(args.warmup_steps)     # uniformly random warmup
    logger.info(f"Replay buffer warmup finished.")

    """ Online finetuning """
    # For debugging metrics
    if 'antmaze' in args.env_name:
        if args.finetune_alg in {'hiql', 'scpiql'}:
            example_trajectory = pretrain_dataset.sample(50, indx=np.arange(1000, 1050))
        else:
            example_trajectory = dataset.sample(50, indx=np.arange(1000, 1050))
    elif 'kitchen' in args.env_name or 'calvin' in args.env_name:
        if args.finetune_alg in ['iql', 'td3']:
            example_trajectory = dataset.sample(50, indx=np.arange(0, 50))
        elif args.finetune_alg in {'hiql', 'scpiql'}:
            example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    elif 'procgen-500' in args.env_name or 'procgen-1000' in args.env_name:
        if args.finetune_alg in ['iql', 'td3']:
            example_trajectory = dataset.sample(50, indx=np.arange(5000, 5050))
        elif args.finetune_alg in {'hiql', 'scpiql'}:
            example_trajectory = pretrain_dataset.sample(50, indx=np.arange(5000, 5050))
    else:
        raise NotImplementedError

    train_logger = CsvLogger(wand_run_dir.joinpath('train.csv'))
    eval_logger = CsvLogger(wand_run_dir.joinpath('eval.csv'))
    first_time = time.time()
    last_time = time.time()

    num_epochs = args.max_env_steps // (args.epoch_steps * args.num_workers)
    for i in tqdm.tqdm(range(1, num_epochs + 1), smoothing=True, dynamic_ncols=True):
        start_time = time.time()

        # data collection
        if args.finetune_alg in {'hiql', 'scpiql'}:
            collect_policy_fn = partial(agent.sample_actions, discrete=dataset_info['discrete'], temperature=args.explore_temperature)
            if not args.use_fast_planner_as_high_policy:
                collect_high_policy_fn = partial(agent.sample_high_actions, temperature=args.explore_temperature)
            else:
                if args.finetune_alg == 'hiql':
                    collect_high_policy_fn = partial(agent.get_one_step_subgoal, single_step_agent, temperature=0.)
                elif args.finetune_alg == 'scpiql':
                    collect_high_policy_fn = partial(agent.get_one_step_subgoal, temperature=0.)
                else:
                    raise AssertionError(f"Should not reach here")
            collect_policy_rep_fn = agent.get_policy_rep
            base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
            collector.collect(args.epoch_steps, policy_fn=collect_policy_fn, high_policy_fn=collect_high_policy_fn, 
                              policy_rep_fn=collect_policy_rep_fn, base_observation=base_observation,
                              use_waypoints=args.use_waypoints, use_rep=args.use_rep)
        else:
            if args.finetune_alg == 'iql':
                train_policy_fn = partial(agent.sample_actions, discrete=dataset_info['discrete'], temperature=args.explore_temperature)
            elif args.finetune_alg == 'td3':
                train_policy_fn = partial(agent.sample_actions, explore=True)
            else:
                raise NotImplementedError
            collector.collect(args.epoch_steps, policy_fn=train_policy_fn)
        collect_finish_time = time.time()

        # model learning
        data_metrics = {}
        update_info = defaultdict(list)
        if replay_buffer.size >= args.batch_size:
            for _ in range(args.train_iters):
                mini_batch = replay_buffer.sample(args.batch_size)
                data_metrics[f'data/reward'] = mini_batch['rewards'].mean()
                rng_key, agent_train_seed = jax.random.split(rng_key)
                if args.finetune_alg == 'iql':
                    if args.data_source == 'mixed':
                        pretrain_batch = dataset.sample(args.batch_size)
                        data_metrics[f'data/dataset_reward'] = pretrain_batch['rewards'].mean()
                        agent, update_info_c = agent.finetune_mixed_update(mini_batch, pretrain_batch, seed=agent_train_seed, actor_loss=args.actor_loss)
                    else:
                        agent, update_info_c = agent.finetune_update(mini_batch, seed=agent_train_seed, actor_loss=args.actor_loss)
                elif args.finetune_alg == 'td3':
                    agent, update_info_c = agent.finetune_update(mini_batch, seed=agent_train_seed, epoch=i)
                elif args.finetune_alg in {'hiql', 'scpiql'}:
                    if args.data_source == 'mixed':
                        pretrain_batch = pretrain_dataset.sample(args.batch_size)
                        data_metrics[f'data/dataset_reward'] = pretrain_batch['rewards'].mean()
                        if args.finetune_alg == 'hiql':
                            agent, update_info_c = agent.finetune_mixed_update(mini_batch, pretrain_batch, seed=agent_train_seed, 
                                                                               actor_supervision=args.hiql_finetune_actor_supervision,
                                                                               high_actor_update=args.high_actor_update,
                                                                               value_update=not args.no_updata_value,
                                                                               ensemble_agents=[agent,] + ensemble_agents if ensemble_agents is not None else None,
                                                                               ensemble_type=args.ensemble_type,
                                                                               single_step_agent=single_step_agent)
                        elif args.finetune_alg == 'scpiql':
                            agent, update_info_c = agent.finetune_mixed_update(mini_batch, pretrain_batch, seed=agent_train_seed, 
                                                                               actor_supervision=args.hiql_finetune_actor_supervision,
                                                                               high_actor_update=args.high_actor_update,
                                                                               value_update=not args.no_updata_value)
                        else:
                            raise AssertionError(f"Should not reach here")
                    elif args.data_source == 'online' and args.no_pretrain:
                        if args.finetune_alg == 'scpiql':
                            agent, update_info_c = agent.finetune_full_online_update(mini_batch, seed=agent_train_seed,
                                                                                     actor_supervision=args.hiql_finetune_actor_supervision,
                                                                                     high_actor_update=True)
                        else:
                            raise NotImplementedError
                    else:
                        agent, update_info_c = agent.finetune_update(mini_batch, seed=agent_train_seed, actor_supervision=args.hiql_finetune_actor_supervision)
                else:
                    raise NotImplementedError
                add_to(update_info, update_info_c)

                if ensemble_agents is not None:
                    assert args.finetune_alg == 'hiql' and not args.low_level_delta, "Ensemble only works for HIQL without low_level_delta."
                    assert args.data_source == 'mixed', "Ensemble only works for mixed data source."
                    for agent_i in ensemble_agents:
                        rng_key, agent_i_train_seed = jax.random.split(rng_key)
                        # mini_batch_i = replay_buffer.sample(args.batch_size)          # independently sampled training data
                        # pretrain_batch_i = pretrain_dataset.sample(args.batch_size)
                        mini_batch_i, pretrain_batch_i = mini_batch, pretrain_batch     # same training data
                        agent_i, _ = agent_i.finetune_mixed_update(mini_batch_i, pretrain_batch_i, seed=agent_i_train_seed, 
                                                                   actor_supervision=args.hiql_finetune_actor_supervision,
                                                                   high_actor_update=args.high_actor_update,
                                                                   ensemble_agents=[agent,] + ensemble_agents,
                                                                   ensemble_type=args.ensemble_type)

        learn_finish_time = time.time()
        
        current_steps = i * args.epoch_steps * args.num_workers

        # Logging
        if i % args.log_interval == 0:
            update_info = dict_mean(update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            
            if args.finetune_alg in ['iql', 'td3']:
                debug_statistics = get_debug_statistics(agent, mini_batch)  # TODO: only the last mini batch
            elif args.finetune_alg in {'hiql', 'scpiql'}:
                debug_statistics = hiql_get_debug_statistics(agent, mini_batch) # TODO: only the last mini batch
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            
            train_metrics.update(data_metrics)  # TODO: only the last mini batch
            train_metrics['time/epoch_time'] = (time.time() - last_time) / args.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            train_metrics['time/collect_time'] = (collect_finish_time - start_time)
            train_metrics['time/learn_time'] = (learn_finish_time - collect_finish_time)
            train_metrics['time/fps'] = args.epoch_steps * args.num_workers / (learn_finish_time - start_time)
            consumed_time = datetime.timedelta(seconds=train_metrics['time/total_time'])
            remaining_time = datetime.timedelta(seconds=train_metrics['time/epoch_time'] * (num_epochs - i))
            logger.info(f"Epoch {i}: Time consumed {str(consumed_time)}, current FPS {train_metrics['time/fps']}, estimated remaining time {str(remaining_time)}")
            last_time = time.time()
            wandb.log(train_metrics, step=current_steps)
            train_logger.log(train_metrics, step=current_steps)

        # Evaluation
        if (i == 1) or (i % args.eval_interval == 0) or (i == num_epochs):
        # if i % args.eval_interval == 0:
            logger.info(f"Running evaluation at epoch {i}: {args.eval_episodes} episodes + {args.num_video_episodes} video episodes")
            if 'procgen' in args.env_name:
                assert args.finetune_alg in {'hiql', 'scpiql'}, f"procgen only supports HIQL and SCPIQL"
                policy_fn = partial(agent.sample_actions, discrete=dataset_info['discrete'], temperature=0.)
                high_policy_fn = partial(agent.sample_high_actions, temperature=0.)
                policy_rep_fn = agent.get_policy_rep
                base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
                single_step_fn = None
                if args.guiding_reward:
                    if args.finetune_alg == 'hiql':
                        single_step_fn = partial(agent.get_one_step_subgoal, single_step_agent, temperature=0.)
                    elif args.finetune_alg == 'scpiql':
                        single_step_fn = partial(agent.get_one_step_subgoal, temperature=0.)
                if args.use_fast_planner_as_high_policy:
                    high_policy_fn = single_step_fn
                eval_metrics = {}
                for goal_info in dataset_info['goal_infos']:
                    rng_key, eval_seed = jax.random.split(rng_key)
                    eval_info, trajs, renders = hiql_evaluate_with_trajectories(
                        policy_fn, high_policy_fn, policy_rep_fn, env, env_name=args.env_name, seed=eval_seed,
                        num_episodes=args.eval_episodes,
                        base_observation=base_observation, num_video_episodes=args.num_video_episodes,
                        use_waypoints=args.use_waypoints,
                        epsilon=0.05,
                        goal_info=goal_info, config=args.config, single_step_fn=single_step_fn)
                    eval_metrics.update({f'evaluation/level{goal_info["eval_level_name"]}_{k}': v for k, v in eval_info.items()})
            else:
                if args.finetune_alg in ['iql', 'td3']:
                    if args.finetune_alg == 'iql':
                        policy_fn = partial(agent.sample_actions, discrete=dataset_info['discrete'], temperature=0.)
                    elif args.finetune_alg == 'td3':
                        policy_fn = partial(agent.sample_actions, explore=False)
                    rng_key, eval_seed = jax.random.split(rng_key)
                    eval_info, trajs, renders, value_preds = evaluate_with_trajectories(
                        policy_fn, env, env_name=args.env_name, seed=eval_seed,
                        num_episodes=args.eval_episodes, num_video_episodes=args.num_video_episodes)
                elif args.finetune_alg in {'hiql', 'scpiql'}:
                    policy_fn = partial(agent.sample_actions, discrete=dataset_info['discrete'], temperature=0.)
                    high_policy_fn = partial(agent.sample_high_actions, temperature=0.)
                    policy_rep_fn = agent.get_policy_rep
                    base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
                    rng_key, eval_seed = jax.random.split(rng_key)
                    single_step_fn = None
                    if args.guiding_reward:
                        if args.finetune_alg == 'hiql':
                            single_step_fn = partial(agent.get_one_step_subgoal, single_step_agent, temperature=0.)
                        elif args.finetune_alg == 'scpiql':
                            single_step_fn = partial(agent.get_one_step_subgoal, temperature=0.)
                    if args.use_fast_planner_as_high_policy:
                        high_policy_fn = single_step_fn
                    eval_info, trajs, renders = hiql_evaluate_with_trajectories(
                        policy_fn, high_policy_fn, policy_rep_fn, env, env_name=args.env_name, seed=eval_seed,
                        num_episodes=args.eval_episodes,
                        base_observation=base_observation, num_video_episodes=args.num_video_episodes,
                        use_waypoints=args.use_waypoints,
                        goal_info=goal_info, config=args.config, single_step_fn=single_step_fn)
                else:
                    raise NotImplementedError

                eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}

            if args.num_video_episodes > 0:
                video = record_video('Video', current_steps, renders=renders)
                eval_metrics['video'] = video
            
            if args.finetune_alg in ['iql', 'td3']:
                traj_values = {}
                traj_values["example_trajectory"] = get_traj_v(agent, example_trajectory)['value_preds']
                for j in range(0, args.num_video_episodes):
                    traj_values[f'trajectory_{j}'] = get_traj_v(agent, trajs[args.eval_episodes + j])['value_preds']
                value_viz = viz_utils.make_visual_no_image(
                    traj_values,
                    [partial(viz_utils.visualize_metric, metric_name=k) for k in traj_values.keys()]
                )
                eval_metrics['value_traj_viz'] = wandb.Image(value_viz)
            elif args.finetune_alg in {'hiql', 'scpiql'}:
                traj_metrics = hiql_get_traj_v(agent, example_trajectory)
                value_viz = viz_utils.make_visual_no_image(
                    traj_metrics,
                    [partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
                )
                eval_metrics['value_traj_viz'] = wandb.Image(value_viz)
            else:
                raise NotImplementedError

            wandb.log(eval_metrics, step=current_steps)
            eval_logger.log(eval_metrics, step=current_steps)

        # Checkpoint
        if args.save_interval > 0 and i % args.save_interval == 0:
            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
                config=args.config
            )
            fname = save_dir.joinpath(f'params_{current_steps}.pkl')
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
    venv.close()


if __name__ == '__main__':
    main()
