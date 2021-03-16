import dataclasses
import numpy as np
import tensorflow as tf
import gym
import pickle
import sys

from stable_baselines import SAC

from imitation.data import types
from imitation.data.rollout import unwrap_traj
import deep_rlsp


def convert_trajs(filename, traj_len):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    assert traj_len < len(data["observations"][0])
    obs = np.array(data["observations"][0][: traj_len + 1])
    acts = np.array(data["actions"][0][:traj_len])
    rews = np.array([0 for _ in range(traj_len)])
    infos = [{} for _ in range(traj_len)]
    traj = types.TrajectoryWithRew(obs=obs, acts=acts, infos=infos, rews=rews)
    return [traj]


def rollout_policy(filename, traj_len, seed, env_name, n_trajs=1):
    model = SAC.load(filename)
    env = gym.make(env_name)
    env.seed(seed)

    trajs = []
    for _ in range(int(n_trajs)):
        obs_list, acts_list, rews_list = [], [], []
        obs = env.reset()
        obs_list.append(obs)
        for _ in range(traj_len):
            act = model.predict(obs, deterministic=True)[0]
            obs, r, done, _ = env.step(act)
            # assert not done
            acts_list.append(act)
            obs_list.append(obs)
            rews_list.append(r)

        infos = [{} for _ in range(traj_len)]
        traj = types.TrajectoryWithRew(
            obs=np.array(obs_list),
            acts=np.array(acts_list),
            infos=infos,
            rews=np.array(rews_list),
        )
        trajs.append(traj)

    return trajs


def recode_and_save_trajectories(traj_or_policy_file, save_loc, traj_len, seed, args):
    if "skills" in traj_or_policy_file:
        trajs = convert_trajs(traj_or_policy_file, traj_len, *args)
    else:
        trajs = rollout_policy(traj_or_policy_file, traj_len, seed, *args)

    # assert len(trajs) == 1
    for traj in trajs:
        assert traj.obs.shape[0] == traj_len + 1
        assert traj.acts.shape[0] == traj_len
    trajs = [dataclasses.replace(traj, infos=None) for traj in trajs]
    types.save(save_loc, trajs)


if __name__ == "__main__":
    _, traj_or_policy_file, save_loc, traj_len, seed = sys.argv[:5]
    if seed == "generate_seed":
        seed = np.random.randint(0, 1e9)
    else:
        seed = int(seed)
    save_loc = save_loc.format(traj_len, seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    traj_len = int(traj_len)
    recode_and_save_trajectories(
        traj_or_policy_file, save_loc, traj_len, seed, sys.argv[5:]
    )
