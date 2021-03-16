import pickle

import numpy as np
from stable_baselines import SAC
from copy import deepcopy

from deep_rlsp.util.video import save_video
from deep_rlsp.util.mujoco import initialize_mujoco_from_obs


def load_data(filename):
    with open(filename, "rb") as f:
        play_data = pickle.load(f)
    return play_data


def init_env_from_obs(env, obs):
    id = env.spec.id
    if (
        "InvertedPendulum" in id
        or "HalfCheetah" in id
        or "Hopper" in id
        or "Ant" in id
        or "Fetch" in id
    ):
        return initialize_mujoco_from_obs(env, obs)
    else:
        # gridworld env
        state = env.obs_to_s(obs)
        env.reset()
        env.unwrapped.s = deepcopy(state)
        return env


def get_trajectory(
    env,
    policy,
    get_observations=False,
    get_rgbs=False,
    get_return=False,
    print_debug=False,
):
    observations = [] if get_observations else None
    trajectory_rgbs = [] if get_rgbs else None
    total_reward = 0 if get_return else None

    obs = env.reset()
    done = False
    while not done:
        if isinstance(policy, SAC):
            a = policy.predict(np.expand_dims(obs, 0), deterministic=False)[0][0]
        else:
            a, _ = policy.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(a)
        if print_debug:
            print("action")
            print(a)
            print("obs")
            print(obs)
            print("reward")
            print(reward)
        if get_observations:
            observations.append(obs)
        if get_rgbs:
            rgb = env.render("rgb_array")
            trajectory_rgbs.append(rgb)
        if get_return:
            total_reward += reward
    return observations, trajectory_rgbs, total_reward


def evaluate_policy(env, policy, n_rollouts, video_out=None, print_debug=False):
    total_reward = 0
    for i in range(n_rollouts):
        get_rgbs = i == 0 and video_out is not None
        _, trajectory_rgbs, total_reward_episode = get_trajectory(
            env, policy, False, get_rgbs, True, print_debug=(print_debug and i == 0)
        )
        total_reward += total_reward_episode
        if get_rgbs:
            save_video(trajectory_rgbs, video_out, fps=20.0)
            print("Saved video to", video_out)
    return total_reward / n_rollouts


def memoize(f):
    # Assumes that all inputs to f are 1-D Numpy arrays
    memo = {}

    def helper(*args):
        key = tuple((tuple(x) for x in args))
        if key not in memo:
            memo[key] = f(*args)
        return memo[key]

    return helper


def sample_obs_from_trajectory(observations, n_samples):
    idx = np.random.choice(np.arange(len(observations)), n_samples)
    return np.array(observations)[idx]
