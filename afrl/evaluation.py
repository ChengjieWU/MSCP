from collections import defaultdict
from typing import Dict

import gym
import numpy as np
import jax


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def dict_mean(d):
    return {k: np.mean(v) for k, v in d.items()}


def evaluate_with_trajectories(
        policy_fn, env: gym.Env, env_name: str, seed: jax.random.PRNGKey,
        num_episodes: int, num_video_episodes: int = 0,
        value_fn = None,
) -> Dict[str, float]:
    rng_key = seed

    trajectories = []
    stats = defaultdict(list)

    value_preds = []
    renders = []
    for i in range(num_episodes + num_video_episodes):
        trajectory = defaultdict(list)

        observation, done = env.reset(), False

        value_pred = []
        render = []
        step = 0
        while not done:
            rng_key, sub_seed = jax.random.split(rng_key)
            action = policy_fn(observations=observation, seed=sub_seed)
            next_observation, r, done, info = env.step(action)

            step += 1

            if value_fn is not None and i >= num_episodes:
                rng_key, sub_seed = jax.random.split(rng_key)
                cur_value = value_fn(observations=observation, seed=sub_seed)
                value_pred.append(cur_value)

            # Render
            if i >= num_episodes and step % 3 == 0:
                if 'antmaze' in env_name:
                    size = 200
                    cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
                    render.append(cur_frame)
                else:
                    raise NotImplementedError

            transition = dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=r,
                dones=done,
                infos=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation

        add_to(stats, flatten(info, parent_key="final"))
        for k, v in trajectory.items():
            trajectory[k] = np.array(v)
        trajectories.append(trajectory)
        if i >= num_episodes:
            value_preds.append(np.array(value_pred))
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories, renders, value_preds


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img
