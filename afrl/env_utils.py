import gzip
import pickle

import numpy as np
import gym
from gym.core import Env
from gym import spaces
from loguru import logger

from jaxrl_m.evaluation import EpisodeMonitor
from jaxrl_m.dataset import Dataset
from .d4rl_utils import get_dataset


_hydra_initialized = False


class CalvinWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = self.action_space['ac']

    @property
    def target_goal(self):
        return np.array([0.25, 0.15, 0, 0.088, 1, 1])
    
    def get_normalized_score(self, x):
        return x / 4.0
    
    def step(self, action):
        return self.env.step({'ac': np.array(action)})


def other_make_env(env_name: str):
    global _hydra_initialized
    if 'calvin' in env_name:
        from afrl.envs.calvin import CalvinEnv
        from hydra import compose, initialize
        from afrl.envs.gym_env import GymWrapper
        from afrl.envs.gym_env import wrap_env
        if not _hydra_initialized:
            initialize(config_path='envs/conf')
            _hydra_initialized = True
        cfg = compose(config_name='calvin')
        env = CalvinEnv(**cfg)
        env.max_episode_steps = cfg.max_episode_steps = 360
        env = GymWrapper(
            env=env,
            from_pixels=cfg.pixel_ob,
            from_state=cfg.state_ob,
            height=cfg.screen_size[0],
            width=cfg.screen_size[1],
            channels_first=False,
            frame_skip=cfg.action_repeat,
            return_state=False,
        )
        env = wrap_env(env, cfg)
        env = CalvinWrapper(env)
        env = EpisodeMonitor(env)
    elif 'procgen' in env_name:
        from afrl.envs.procgen_env import ProcgenWrappedEnv
        import matplotlib

        matplotlib.use('Agg')

        n_processes = 1
        env = ProcgenWrappedEnv(n_processes, 'maze', 1, 1)
    else:
        raise NotImplementedError(f"Env {env_name} is not implemented.")
    
    return env


def calculate_target_goal(level: int, rng: np.random.Generator = None):
        from afrl.envs.procgen_viz import ProcgenLevel

        level_details = ProcgenLevel.create(int(level))
        goal_states = list(level_details.done_locs)
        if rng is not None:
            target_state = goal_states[rng.choice(len(goal_states))]
        else:
            target_state = goal_states[np.random.choice(len(goal_states))]

        goal_img = level_details.states[target_state][1]
        goal_loc = np.array(target_state)
        return goal_img, goal_loc


class GCProcgenMazeEnv:
    def __init__(self, num_envs: int, start_level: int, num_levels: int, distribution_mode='easy', seed=None):
        from procgen import ProcgenEnv

        self.env_num = num_envs
        self.start_level = start_level
        self.num_levels = num_levels
        self.dist_mode = distribution_mode
        self.envs = ProcgenEnv(num_envs=num_envs,
                               env_name='maze',
                               distribution_mode=distribution_mode,
                               start_level=start_level,
                               num_levels=num_levels,
                               rand_seed=int(seed))
        self._rng = np.random.default_rng(seed + 100)
        self._reset_statistics()

    def _reset_statistics(self):
        self._level_seeds = np.array([x['level_seed'] for x in self.envs.env.get_info()], dtype=int)
        self._done = np.ones(self.env_num, dtype=bool)
        self._r = np.zeros(self.env_num)
        self._t = np.zeros(self.env_num)
        goal_img_list, goal_loc_list = [], []
        for level_seed in self._level_seeds:
            goal_img, goal_loc = calculate_target_goal(level_seed, rng=self._rng)
            goal_img_list.append(goal_img)
            goal_loc_list.append(goal_loc)
        self._goal_imgs = np.array(goal_img_list)
        self._goal_locs = np.array(goal_loc_list)

    def obs(self, x):
        return x['rgb']

    def reset(self, idx: int = None):
        # ProcgenEnv does not support manually resetting.
        # It automatically resets an env instance once it is done.
        if idx is not None:
            # This is for API compatibility.
            # We have already handled the things after done in 'step' function.
            assert 0 <= idx < self.env_num
            assert self._done[idx], f"Resetting the environment while it is not done. Prohibited!"
            rew, ob, first = self.envs.env.observe()
            assert np.all(self._done == first)  # Check
            return self.obs(ob)[idx:idx+1]
        if not np.all(self._done):
            logger.warning(f"Resetting the environment while some environments are not done. Be cautious in Procgen!")
            from procgen import ProcgenEnv
            # We close the intial envs, and create new ones.
            self.envs.close()
            self.envs = ProcgenEnv(num_envs=self.env_num,
                                   env_name='maze',
                                   distribution_mode=self.dist_mode,
                                   start_level=self.start_level,
                                   num_levels=self.num_levels,
                                   rand_seed=int(self._rng.integers(1000000)))
        observation = self.obs(self.envs.reset())
        self._reset_statistics()
        return observation
    
    def step(self, action):
        obs, reward, done, info = self.envs.step(action)
        self._done = done
        self._r += reward
        self._t += 1
        for i in range(len(done)):
            if done[i]:
                info[i]['episode'] = dict(r=self._r[i], t=self._t[i], time_r=max(-500, -1 * self._t[i]))
                new_level_seed = int(self.envs.env.get_info()[i]['level_seed'])
                # assert new_level_seed != self._level_seeds[i], f"This is for initial debugging!!! Should be turned off."
                self._level_seeds[i] = new_level_seed
                self._r[i] = 0
                self._t[i] = 0
                goal_img, goal_loc = calculate_target_goal(new_level_seed, rng=self._rng)
                self._goal_imgs[i] = goal_img
                self._goal_locs[i] = goal_loc
        return self.obs(obs), reward, done, info

    @property
    def target_goal(self):
        return self._goal_imgs

    def close(self):
        return self.envs.close()
    
    @property
    def observation_space(self):
        return [self.envs.observation_space['rgb']] * self.env_num

    @property
    def action_space(self):
        return [self.envs.action_space] * self.env_num
    
    def get_env_attr(self, attr):
        return getattr(self, attr)


def make_procgen_env(env_name: str, num_envs: int, seed=None):
    assert 'procgen' in env_name
    import matplotlib
    matplotlib.use('Agg')

    num_levels = int(env_name.split('-')[1])
    assert num_levels in {500, 1000}
    env = GCProcgenMazeEnv(num_envs=num_envs, start_level=0, num_levels=num_levels,
                           distribution_mode='easy', seed=seed)
    return env


def other_get_dataset(env_name: str):
    if 'calvin' in env_name:
        data = pickle.load(gzip.open('data/calvin.gz', "rb"))
        ds = []
        for i, d in enumerate(data):
            if len(d['obs']) < len(d['dones']):
                continue  # Skip incomplete trajectories.
            # Only use the first 21 states of non-floating objects.
            d['obs'] = d['obs'][:, :21]
            new_d = dict(
                observations=d['obs'][:-1],
                next_observations=d['obs'][1:],
                actions=d['actions'][:-1],
            )
            num_steps = new_d['observations'].shape[0]
            new_d['rewards'] = np.zeros(num_steps)
            new_d['terminals'] = np.zeros(num_steps, dtype=bool)
            new_d['terminals'][-1] = True
            ds.append(new_d)
        dataset = dict()
        for key in ds[0].keys():
            dataset[key] = np.concatenate([d[key] for d in ds], axis=0)
        dataset = get_dataset(None, env_name, dataset=dataset)
        dataset_info = {'discrete': False}
    elif 'procgen' in env_name:
        from afrl.envs.procgen_env import get_procgen_dataset

        if env_name == 'procgen-500':
            dataset = get_procgen_dataset('data/procgen/level500.npz', state_based=('state' in env_name))
            min_level, max_level = 0, 499
        elif env_name == 'procgen-1000':
            dataset = get_procgen_dataset('data/procgen/level1000.npz', state_based=('state' in env_name))
            min_level, max_level = 0, 999
        else:
            raise NotImplementedError

        # Test on large levels having >=20 border states
        large_levels = [12, 34, 35, 55, 96, 109, 129, 140, 143, 163, 176, 204, 234, 338, 344, 369, 370, 374, 410, 430, 468, 470, 476, 491] + [5034, 5046, 5052, 5080, 5082, 5142, 5244, 5245, 5268, 5272, 5283, 5335, 5342, 5366, 5375, 5413, 5430, 5474, 5491]
        goal_infos = []
        goal_infos.append({'eval_level': [level for level in large_levels if min_level <= level <= max_level], 'eval_level_name': 'train'})
        goal_infos.append({'eval_level': [level for level in large_levels if level > max_level], 'eval_level_name': 'test'})

        dones_float = 1.0 - dataset['masks']
        dones_float[-1] = 1.0
        dataset = dataset.copy({
            'dones_float': dones_float
        })
        discrete = True
        example_action = np.max(dataset['actions'], keepdims=True)
        dataset_info = {'discrete': discrete, 'example_action': example_action, 'goal_infos': goal_infos}
    else:
        raise NotImplementedError(f"Env {env_name} is not implemented.")
    
    return dataset, dataset_info


def other_subproc_venv_env_fn(env_name: str, seed: int):
    # handle proper seeding for parallel environments
    np.random.seed(seed)
    env = other_make_env(env_name)
    env.seed(seed)
    return env


class DidacticMaze(Env):
    _time_limit = 30

    def __init__(self):
        self.action_space = spaces.Discrete(n=4)
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32)

    @property
    def maze_size(self):
        return 6, 6    

    @property
    def target_goal(self):
        return np.array([0., 5.], dtype=np.float32)

    def reset(self):
        self._x = 0.
        self._y = 0.
        self._done = False
        self._t = 0
        return np.array([self._x, self._y], dtype=np.float32)
    
    def step(self, action):
        if self._done:
            raise RuntimeError(f"Episode is finished, reset first!")
        if action == 0:
            if self._x <= 3:
                if self._y <= 1 or (self._y >= 3 and self._y <= 4):
                    self._y += 1
            else:
                if self._y <= 4:
                    self._y += 1
        elif action == 1:
            if self._x <= 3:
                if (self._y >= 1 and self._y <= 2) or self._y >= 4:
                    self._y -= 1
            else:
                if self._y >= 1:
                    self._y -= 1
        elif action == 2:
            if self._x <= 4:
                self._x += 1
        elif action == 3:
            if self._x >= 1:
                self._x -= 1
        else:
            raise ValueError(f"Invalid action {action}")
        assert 0 <= self._x <= 5 and 0 <= self._y <= 5  # sanity check
        self._t += 1
        r = 0.
        if self._x == self.target_goal[0] and self._y == self.target_goal[1]:
            self._done = True
            r = 1.
        elif self._t >= self._time_limit:
            self._done = True
        return np.array([self._x, self._y], dtype=np.float32), r, self._done, {'t': self._t}


class DidacticMaze2(Env):
    _time_limit = 30

    def __init__(self):
        self.action_space = spaces.Discrete(n=4)
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32)
    
    @property
    def maze_size(self):
        return 8, 4

    @property
    def target_goal(self):
        return np.array([2., 0.], dtype=np.float32)

    def reset(self):
        self._x = 5.
        self._y = 0.
        self._done = False
        self._t = 0
        return np.array([self._x, self._y], dtype=np.float32)
    
    def step(self, action):
        if self._done:
            raise RuntimeError(f"Episode is finished, reset first!")
        xy = np.array([self._x, self._y])
        dxy = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])][action]
        new_xy = xy + dxy
        new_x, new_y = new_xy[0], new_xy[1]

        if new_x < 0 or new_x > 7 or new_y < 0 or new_y > 3:
            pass
        elif self._y == new_y and 0 <= self._y <= 1 and self._x + new_x == 7:
            pass
        elif self._x == new_x and 2 <= self._x <= 5 and self._y + new_y == 3:
            pass
        else:
            self._x, self._y = new_x, new_y

        assert 0 <= self._x <= 7 and 0 <= self._y <= 3  # sanity check
        self._t += 1
        r = 0.
        if self._x == self.target_goal[0] and self._y == self.target_goal[1]:
            self._done = True
            r = 1.
        elif self._t >= self._time_limit:
            self._done = True
        return np.array([self._x, self._y], dtype=np.float32), r, self._done, {'t': self._t}


class DidacticMaze3(DidacticMaze2):
    @property
    def target_goal(self):
        return np.array([0., 0.], dtype=np.float32)
    
    def reset(self):
        self._x = 7.
        self._y = 0.
        self._done = False
        self._t = 0
        return np.array([self._x, self._y], dtype=np.float32)


def make_didactic_maze_env(env_name: str):
    return {'didactic_maze': DidacticMaze, 'didactic_maze2': DidacticMaze2, 'didactic_maze3': DidacticMaze3}[env_name]()


def get_didactic_maze_dataset(env_name: str):
    env = make_didactic_maze_env(env_name)
    observations = []
    if env_name == 'didactic_maze':
        actions = np.array([2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3])
    elif env_name == 'didactic_maze2':
        actions = np.array([2, 2, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 2, 2])
    elif env_name == 'didactic_maze3':
        actions = np.array([0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1])
    rewards = []
    
    obs = env.reset()
    observations.append(obs)
    
    for action in actions:
        obs, r, done, info = env.step(action)
        observations.append(obs)
        rewards.append(r)

    observations = np.array(observations)
    rewards = np.array(rewards)

    next_observations = observations[1:].copy()
    observations = observations[:-1].copy()

    dones_float = np.zeros_like(rewards, dtype=np.float32)
    dones_float[-1] = 1.0

    return Dataset.create(
        observations=observations,
        actions=actions,
        rewards=rewards,
        masks=1.0 - dones_float,
        dones_float=dones_float,
        next_observations=next_observations,
    ), {'discrete': True, 'example_action': np.array([3])}
