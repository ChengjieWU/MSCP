import abc
from collections import defaultdict
from typing import Optional, Callable, List, Dict, NamedTuple, Any

import numpy as np
import jax
import jax.numpy as jnp

from jaxrl_m.dataset import ReplayBuffer, Dataset

from afrl.envs.venvs import BaseVectorEnv
from afrl.gc_dataset import GCSReplayBuffer


class BaseCollector(abc.ABC):
    """Basic signatures for collector."""
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def warmup(self, n_step: int):
        pass

    @abc.abstractmethod
    def collect(self, n_step: int):
        pass


class Collector(BaseCollector):
    def __init__(self, venv: BaseVectorEnv, rng_key: jax.random.PRNGKey,
                 replay_buffer: Optional[ReplayBuffer] = None, replay_buffer_size: Optional[int] = None,
                 rewarder = None,
                 ) -> None:
        self.rng_key = rng_key
        self.venv = venv
        self.batch_size = venv.env_num
        self.replay_buffer = replay_buffer
        if replay_buffer is None:
            assert replay_buffer_size is not None, f"Must specify replay_buffer_size if replay_buffer is None."
            self.replay_buffer_size = replay_buffer_size
            self.replay_buffer = self.create_replay_buffer()
        else:
            self.replay_buffer_size = self.replay_buffer.max_size
        
        self.rewarder: Optional[Callable] = rewarder
        self._only_add_trajectory = (self.rewarder is not None)

        self._observations = None
        self._trajectory_buffer = None

    def reset(self):
        self._observations = self.venv.reset()
        assert len(self._observations) == self.batch_size
        self._trajectory_buffer: List[List[Dict[str, np.ndarray]]] = [[] for _ in range(self.batch_size)]

    def create_replay_buffer(self):
        observations = self.venv.reset()
        actions = np.stack([x.sample() for x in self.venv.action_space])
        next_observations, rewards, dones, infos = self.venv.step(actions)
        rewards -= 1.
        transition = dict(
            observations=observations[0],
            next_observations=next_observations[0],
            actions=actions[0],
            rewards=np.float32(rewards[0]),
            masks=np.float32(1-dones[0]),
            dones=dones[0],
            dones_float=np.float32(dones[0]),
        )
        replay_buffer = ReplayBuffer.create(transition, size=self.replay_buffer_size)
        return replay_buffer

    def warmup(self, n_step: int):
        # TODO: seed for randomness is not handled properly, action_space.seed() is not called in current implementation
        def random_explore_fn(*args, **kwargs):
            return np.stack([self.venv.action_space[i].sample() for i in range(self.batch_size)])
        return self.collect(n_step, random_explore_fn)

    def collect(self, n_step: int, policy_fn: Callable):
        collect_infos = defaultdict(list)

        for i in range(n_step):
            self.rng_key, seed = jax.random.split(self.rng_key)
            actions = policy_fn(self._observations, seed=seed)
            next_observations, rewards, dones, infos = self.venv.step(np.asarray(actions))  # to be compatible with VecEnv wrapper
            rewards -= 1.    # process reward

            for k in range(self.batch_size):
                transition = dict(
                    observations=self._observations[k],
                    next_observations=next_observations[k],
                    actions=actions[k],
                    rewards=np.float32(rewards[k]),
                    masks=np.float32(1-dones[0]),
                    dones=dones[k],
                    dones_float=np.float32(dones[k]),
                )
                if self._only_add_trajectory:
                    self._trajectory_buffer[k].append(transition)
                else:
                    self.replay_buffer.add_transition(transition)
            
            for k in range(self.batch_size):
                if dones[k]:
                    next_observations[k] = self.venv.reset(k)[0]
                    if self._only_add_trajectory:
                        if self.rewarder is not None:
                            raise AssertionError('Should not reach here')
                            relabelled_rewards = self.rewarder.relabel([traj, ])[0]
                            for t in range(len(self._trajectory_buffer[k])):
                                self._trajectory_buffer[k][t]["rewards"] = relabelled_rewards[t]
                            collect_infos["relabelled_episode_reward"].append(np.sum(relabelled_rewards))
                            collect_infos["episode_length"].append(len(self._trajectory_buffer[k]))
                        for t in range(len(self._trajectory_buffer[k])):
                            self.replay_buffer.add_transition(self._trajectory_buffer[k][t])
                        self._trajectory_buffer[k] = []
            
            self._observations = next_observations
        
        for k, v in collect_infos.items():
            collect_infos[k] = np.mean(v)
        return collect_infos


def correct_dict_batch(x):
    # [{k: v}, {k: v}, ...] -> {k: [v, v, ...]}
    return jax.tree_map(lambda *args: jnp.stack(args, axis=0), *x)


class GCCollector(BaseCollector):
    def __init__(self, env_name: str, venv: BaseVectorEnv, rng_key: jax.random.PRNGKey,
                 replay_buffer: Optional[GCSReplayBuffer] = None,
                 replay_buffer_size: Optional[int] = None, sample_scheme: str = 'original', relabelling_prob: float = 0.8,
                 rewarder = None, gc_configs: Dict[str, Any] = None, use_rep: bool = False,
                 ) -> None:
        self.env_name = env_name
        self.rng_key = rng_key
        self.venv = venv
        self.batch_size = venv.env_num
        self.use_rep = use_rep
        
        self.replay_buffer = replay_buffer
        if replay_buffer is None:
            assert replay_buffer_size is not None, f"Must specify replay_buffer_size if replay_buffer is None."
            self.replay_buffer_size = replay_buffer_size
            self.replay_buffer = self.create_replay_buffer(gc_configs, sample_scheme, relabelling_prob)
        else:
            self.replay_buffer_size = self.replay_buffer.max_size
        
        self.rewarder: Optional[Callable] = rewarder

        self._observations = None
        self._trajectory_buffer = None
        self._goals = None
    
    def _observation_postprocess(self, obs):
        if 'antmaze' in self.env_name:
            if 'topview' in self.env_name:
                assert set(obs[0].keys()) == {'image', 'state'}
            else:
                assert obs.ndim == 2, f"only support 2-dim observations"
            pass    # do nothing
        elif 'kitchen' in self.env_name:
            assert obs.ndim == 2, f"only support 2-dim observations"
            obs = obs[:, :30]
        elif 'calvin' in self.env_name:
            obs = np.array([x['ob'] for x in obs])
        elif 'procgen' in self.env_name:
            assert obs.ndim == 4, f"only support 4-dim observations"
            obs = obs   # do nothing
        else:
            raise NotImplementedError(f"Unknown env_name: {self.env_name}")
        return obs

    def reset(self):
        self._observations = self._observation_postprocess(self.venv.reset())
        self._goals = np.array(self.venv.get_env_attr('target_goal'))
        assert len(self._observations) == len(self._goals) == self.batch_size
        self._trajectory_buffer: List[List[Dict[str, np.ndarray]]] = [[] for _ in range(self.batch_size)]

    def create_replay_buffer(self, gc_configs: Dict[str, Any], sample_scheme, relabelling_prob) -> GCSReplayBuffer:
        observations = self._observation_postprocess(self.venv.reset())
        actions = np.stack([x.sample() for x in self.venv.action_space])
        next_observations, rewards, dones, infos = self.venv.step(actions)
        next_observations = self._observation_postprocess(next_observations)
        rewards -= 1.
        transition = dict(
            observations=observations[0],
            next_observations=next_observations[0],
            actions=actions[0],
            rewards=np.float32(rewards[0]),
            masks=np.float32(1-dones[0]),
            dones=dones[0],
            dones_float=np.float32(dones[0]),
        )
        if not self.use_rep:
            transition['traj_subgoals'] = observations[0]   # we also store the actual encountered subgoals, but only supports when not self.use_rep
        replay_buffer = ReplayBuffer.create(transition, size=self.replay_buffer_size)
        replay_buffer = GCSReplayBuffer(dataset=replay_buffer, sample_scheme=sample_scheme, relabelling_prob=relabelling_prob,
                                        **gc_configs)
        return replay_buffer

    def warmup(self, n_step: int):
        def random_explore_fn(*args, **kwargs):
            return np.stack([self.venv.action_space[i].sample() for i in range(self.batch_size)])
        return self.collect(n_step, random_explore_fn, None, None, self._observations[0], False, False)    #  self._observations won't be actually used

    def collect(self, n_step: int, policy_fn, high_policy_fn, policy_rep_fn, base_observation, use_waypoints: bool, use_rep: bool):
        collect_infos = defaultdict(list)

        for i in range(n_step):
            if 'antmaze' in self.env_name:
                if 'topview' in self.env_name:
                    obs_goal = self._goals
                else:
                    obs_goal = base_observation.copy()
                    assert len(obs_goal.shape) == 1, f"only support 1-dim observations"
                    obs_goal = np.tile(obs_goal, (self.batch_size, 1))
                    obs_goal[:, :2] = self._goals
            elif 'kitchen' in self.env_name:
                obs_goal = self._goals.copy()
                assert len(base_observation.shape) == 1, f"only support 1-dim observations"
                obs_goal[:, :9] = np.tile(base_observation[:9], (self.batch_size, 1))
            elif 'calvin' in self.env_name:
                obs_goal = base_observation.copy()
                assert len(obs_goal.shape) == 1, f"only support 1-dim observations"
                obs_goal = np.tile(obs_goal, (self.batch_size, 1))
                obs_goal[:, 15:21] = self._goals
            elif 'procgen' in self.env_name:
                obs_goal = self._goals
            else:
                raise NotImplementedError(f"Unknown env_name: {self.env_name}")

            if not use_waypoints:
                cur_obs_goal = obs_goal
                if use_rep:
                    if 'topview' in self.env_name:
                        cur_obs_goal_rep = policy_rep_fn(targets=correct_dict_batch(cur_obs_goal), bases=correct_dict_batch(self._observations))
                    else:
                        cur_obs_goal_rep = policy_rep_fn(targets=cur_obs_goal, bases=self._observations)
                else:
                    cur_obs_goal_rep = cur_obs_goal
            else:
                self.rng_key, sub_seed = jax.random.split(self.rng_key)
                if 'topview' in self.env_name:
                    cur_obs_goal = high_policy_fn(observations=correct_dict_batch(self._observations), goals=correct_dict_batch(obs_goal), seed=sub_seed)
                else:
                    cur_obs_goal = high_policy_fn(observations=self._observations, goals=obs_goal, seed=sub_seed)
                if use_rep:
                    cur_obs_goal = cur_obs_goal / np.linalg.norm(cur_obs_goal, axis=-1, keepdims=True) * np.sqrt(cur_obs_goal.shape[-1])
                else:
                    cur_obs_goal = self._observations + cur_obs_goal
                cur_obs_goal_rep = cur_obs_goal

            self.rng_key, sub_seed = jax.random.split(self.rng_key)
            if 'topview' in self.env_name:
                actions = policy_fn(observations=correct_dict_batch(self._observations), goals=cur_obs_goal_rep, low_dim_goals=True, seed=sub_seed)
            else:
                actions = policy_fn(observations=self._observations, goals=cur_obs_goal_rep, low_dim_goals=True, seed=sub_seed)

            next_observations, rewards, dones, infos = self.venv.step(np.asarray(actions))  # to be compatible with VecEnv wrapper
            next_observations = self._observation_postprocess(next_observations)
            if 'antmaze' in self.env_name:
                rewards -= 1.    # process reward
            elif 'kitchen' in self.env_name:
                pass    # do nothing
            elif 'calvin' in self.env_name:
                pass    # do nothing
            elif 'procgen' in self.env_name:
                pass    # we do not use reward, so it does not matter
            else:
                raise NotImplementedError(f"Unknown env_name: {self.env_name}")

            for k in range(self.batch_size):
                transition = dict(
                    observations=self._observations[k],
                    next_observations=next_observations[k],
                    actions=actions[k],
                    rewards=np.float32(rewards[k]),
                    masks=np.float32(1-dones[0]),
                    dones=dones[k],
                    dones_float=np.float32(dones[k]),
                )
                if not self.use_rep:
                    transition['traj_subgoals'] = cur_obs_goal_rep[k]
                self._trajectory_buffer[k].append(transition)
            
            for k in range(self.batch_size):
                if dones[k]:
                    next_observations[k] = self._observation_postprocess(self.venv.reset(k))[0]
                    self._goals[k] = np.array(self.venv.get_env_attr('target_goal'))[k]
                    
                    if self.rewarder is not None:
                        raise AssertionError('Should not reach here')
                        traj = [
                            Transition(
                                observation=t["observations"],
                                action=t["actions"],
                                reward=t["rewards"],
                                discount=t["masks"],
                                next_observation=t["next_observations"],
                            ) for t in self._trajectory_buffer[k]
                        ]
                        relabelled_rewards = self.rewarder.relabel([traj, ])[0]
                        for t in range(len(self._trajectory_buffer[k])):
                            self._trajectory_buffer[k][t]["rewards"] = relabelled_rewards[t]
                        collect_infos["relabelled_episode_reward"].append(np.sum(relabelled_rewards))
                    collect_infos["episode_length"].append(len(self._trajectory_buffer[k]))
                    self.replay_buffer.add_trajectory(self._trajectory_buffer[k])
                    self._trajectory_buffer[k] = []
            
            self._observations = next_observations
        
        for k, v in collect_infos.items():
            collect_infos[k] = np.mean(v)
        return collect_infos


class DummyDatasetCollector(BaseCollector):
    def __init__(self, dataset: Dataset, rng_key: jax.random.PRNGKey) -> None:
        self.replay_buffer = dataset

    def reset(self):
        pass

    def warmup(self, n_step: int):
        return {}

    def collect(self, n_step: int, *args, **kwargs):
        return {}
