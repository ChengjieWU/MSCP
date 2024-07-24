import pickle

import d4rl
import gym
from gym.core import Env
import numpy as np

from jaxrl_m.dataset import Dataset
from jaxrl_m.evaluation import EpisodeMonitor


class TargetGoalObservation(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        obs_low = np.concatenate((np.array((-np.inf, -np.inf)),self.env.observation_space.low))
        obs_high = np.concatenate((np.array((np.inf, np.inf)),self.env.observation_space.high))
        self.observation_space = gym.spaces.box.Box(low=obs_low, high=obs_high)

    def observation(self, observation):
        return np.concatenate((np.array(self.unwrapped.target_goal), observation))


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env: Env, skip: int = 1):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GetTargetWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
    
    @property
    def target_goal(self):
        return self.env.wrapped_env.target_goal


class KitchenWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        # self.observation_space = gym.spaces.Box(low=self.observation_space.low[:30], high=self.observation_space.high[:30], shape=(30,))
        self._target_goal = None
    
    @property
    def target_goal(self):
        if self._target_goal is None:
            raise RuntimeError("Target goal is not set. Reset the environment first.")
        return self._target_goal
    
    def reset(self):
        obs = self.env.reset()
        self._target_goal = obs[30:]
        # return obs[:30]
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # return obs[:30], reward, done, info
        return obs, reward, done, info


class VisualAntMazeWrapper(gym.Wrapper):
    def __init__(self, env: Env, env_name: str):
        super().__init__(env)
        self._env_name = env_name
        assert self._env_name in {'topview-antmaze-large-diverse-v2'}

        """We save the precomputed goal info into a file.
        # (Precomputed index) The closest observation to the original goal
        if 'large-diverse' in FLAGS.env_name:
            target_idx = 38190
        elif 'large-play' in FLAGS.env_name:
            target_idx = 798118
        elif 'ultra-diverse' in FLAGS.env_name:
            target_idx = 352934
        elif 'ultra-play' in FLAGS.env_name:
            target_idx = 77798
        else:
            raise NotImplementedError
        goal_info = {
            'ob': {
                'image': dataset['observations']['image'][target_idx],
                'state': dataset['observations']['state'][target_idx],
            }
        }
        """
        with open(f'data/antmaze_topview_6_60/{self._env_name[8:]}_goal.npz', 'rb') as f:
            self._goal_info = pickle.load(f)['ob']  # {'image': ..., 'state': ...}
        
        image_observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        state_observation_space = gym.spaces.Box(low=self.env.observation_space.low[2:], high=self.env.observation_space.high[2:],
                                                 shape=(self.env.observation_space.shape[0] - 2,), dtype=self.env.observation_space.dtype)
        self.observation_space = gym.spaces.Dict({'image': image_observation_space, 'state': state_observation_space})
    
    @property
    def target_goal(self):
        return self._goal_info  # {'image': ..., 'state': ...}

    def obs(self, observation):
        self.env.viewer.cam.lookat[0] = observation[0]
        self.env.viewer.cam.lookat[1] = observation[1]
        self.env.viewer.cam.lookat[2] = 0
        observation = {
            'image': self.env.render(mode='rgb_array', width=64, height=64),
            'state': observation[2:],
        }
        return observation

    def reset(self):
        observation = self.env.reset()
        return self.obs(observation)    # {'image': ..., 'state': ...}

    def step(self, a):
        observation, reward, done, info = self.env.step(a)
        observation = self.obs(observation)
        return observation, reward, done, info


def make_env(env_name: str, append_target: bool = False, set_render: bool = False, frame_skip: int = 1):
    if 'antmaze' in env_name:
        if env_name.startswith('antmaze'):  # original antmaze environments
            if 'ultra' in env_name:
                import d4rl_ext
                # import gym
                env = gym.make(env_name)
            else:
                env = gym.make(env_name)
            env = EpisodeMonitor(env)
            env = GetTargetWrapper(env)
            if append_target:
                env = TargetGoalObservation(env)
            if frame_skip > 1:
                env = FrameSkipWrapper(env, skip=frame_skip)
        else:
            assert env_name == 'topview-antmaze-large-diverse-v2'
            assert not append_target and frame_skip == 1
            env = gym.make('antmaze-large-diverse-v2')
            env = EpisodeMonitor(env)
            env = VisualAntMazeWrapper(env, env_name=env_name)
        
            # Update colors
            l = len(env.model.tex_type)
            # amz-large
            sx, sy, ex, ey = 15, 45, 55, 100
            for i in range(l):
                if env.model.tex_type[i] == 0:
                    height = env.model.tex_height[i]
                    width = env.model.tex_width[i]
                    s = env.model.tex_adr[i]
                    for x in range(height):
                        for y in range(width):
                            cur_s = s + (x * width + y) * 3
                            R = 192
                            r = int((ex - x) / (ex - sx) * R)
                            g = int((y - sy) / (ey - sy) * R)
                            r = np.clip(r, 0, R)
                            g = np.clip(g, 0, R)
                            env.model.tex_rgb[cur_s:cur_s + 3] = [r, g, 128]
            env.model.mat_texrepeat[0, :] = 1

        # set camera position
        if set_render or 'topview' in env_name:
            env.render(mode='rgb_array', width=200, height=200)
            if 'large' in env_name:
                if 'topview' not in env_name:
                    env.viewer.cam.lookat[0] = 18
                    env.viewer.cam.lookat[1] = 12
                    env.viewer.cam.distance = 50
                    env.viewer.cam.elevation = -90
                else:
                    env.viewer.cam.azimuth = 90.
                    env.viewer.cam.distance = 6
                    env.viewer.cam.elevation = -60
                # viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
                # viz = ant_diagnostics.Visualizer(env_name, viz_env, viz_dataset, discount=FLAGS.discount)
            elif 'ultra' in env_name:
                env.viewer.cam.lookat[0] = 26
                env.viewer.cam.lookat[1] = 18
                env.viewer.cam.distance = 70
                env.viewer.cam.elevation = -90
            elif 'umaze' in env_name:
                env.viewer.cam.lookat[0] = 4
                env.viewer.cam.lookat[1] = 4
                env.viewer.cam.distance = 30
                env.viewer.cam.elevation = -90
            elif 'medium' in env_name:
                env.viewer.cam.lookat[0] = 9
                env.viewer.cam.lookat[1] = 9
                env.viewer.cam.distance = 40
                env.viewer.cam.elevation = -90
            else:
                raise NotImplementedError(f"Camera view for env {env_name} is not implemented.")
    elif 'kitchen' in env_name:
        env = gym.make(env_name)
        env = EpisodeMonitor(env)
        env = KitchenWrapper(env)
        assert not append_target and frame_skip == 1, f"append_target and frame_skip are not supported for {env_name}"
    else:
        raise NotImplementedError(f"Env {env_name} is not implemented.")
    
    return env


def subproc_venv_env_fn(env_name: str, seed: int, append_target: bool, set_render: bool = False, frame_skip: int = 1):
    # handle proper seeding for parallel environments
    np.random.seed(seed)
    env = make_env(env_name, append_target=append_target, set_render=False, frame_skip=frame_skip)
    env.seed(seed)
    return env


def venv_env_fn(env_name: str, seed: int, append_target: bool, frame_skip: int = 1):
    return subproc_venv_env_fn(env_name, seed, append_target, frame_skip)


def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Modified from d4rl.qlearning_dataset
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    # goal_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        # goal = dataset['infos/goal'][i].astype(np.float32)

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        # goal_.append(goal)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        # 'goals': np.array(goal_),
    }


def get_dataset(env: gym.Env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                filter_terminals: bool = False,
                obs_dtype=np.float32,
                append_target: bool = False,
                dataset=None,
                ):
    if 'topview' in env_name:
        assert env_name == 'topview-antmaze-large-diverse-v2'
        dataset = np.load('data/antmaze_topview_6_60/antmaze-large-diverse-v2.npz')
        dataset = Dataset.create(
            observations={
                'image': dataset['images'],
                'state': dataset['observations'][:, 2:],
            },
            actions=dataset['actions'],
            rewards=dataset['rewards'],
            masks=dataset['masks'],
            dones_float=dataset['dones_float'],
            next_observations={
                'image': dataset['next_images'],
                'state': dataset['next_observations'][:, 2:],
            },
        )
        return dataset

    if dataset is None:
        dataset = qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset['actions'] = np.clip(dataset['actions'], -lim, lim)     # clip actions to [-1 + eps, 1 - eps]

    dataset['terminals'][-1] = 1
    if filter_terminals:
        # drop terminal transitions
        non_last_idx = np.nonzero(~dataset['terminals'])[0]
        last_idx = np.nonzero(dataset['terminals'])[0]
        penult_idx = last_idx - 1
        new_dataset = dict()
        for k, v in dataset.items():
            if k == 'terminals':
                v[penult_idx] = 1
            new_dataset[k] = v[non_last_idx]
        dataset = new_dataset

    if 'antmaze' in env_name:
        # dataset['terminals'][:] = 0.
        dones_float = np.zeros_like(dataset['rewards'], dtype=np.float32)
        split_position = np.linalg.norm(dataset['observations'][1:] - dataset['next_observations'][:-1], axis=1) > 1e-6
        split_position = np.append(split_position, True)
        split_position = np.logical_or(split_position, dataset['terminals'] == 1)
        dones_float[split_position] = 1.
    else:
        dones_float = dataset['terminals'].copy()
    
    if 'antmaze' in env_name:
        dataset['rewards'] = dataset['rewards'] - 1.0   # process reward
    elif 'kitchen' in env_name:
        dataset['observations'] = dataset['observations'][:, :30]
        dataset['next_observations'] = dataset['next_observations'][:, :30]

    if append_target:
        assert 'antmaze' in env_name, f"append_target is not supported for {env_name}"
        dataset['observations'] = np.concatenate((dataset['goals'], dataset['observations']), axis=1)
        dataset['next_observations'] = np.concatenate((dataset['goals'], dataset['next_observations']), axis=1)

    return Dataset.create(
        observations=dataset['observations'].astype(obs_dtype),
        actions=dataset['actions'].astype(np.float32),
        rewards=dataset['rewards'].astype(np.float32),
        masks=1.0 - dones_float.astype(np.float32),
        dones_float=dones_float.astype(np.float32),
        next_observations=dataset['next_observations'].astype(obs_dtype),
    )
