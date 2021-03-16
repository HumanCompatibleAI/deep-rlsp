import numpy as np


class MujocoObsClipper:
    def __init__(self, env_id):
        if env_id.startswith("InvertedPendulum"):
            self.low = -300
            self.high = 300
        elif env_id.startswith("HalfCheetah"):
            self.low = -300
            self.high = 300
        elif env_id.startswith("Hopper"):
            self.low = -300
            self.high = 300
        elif env_id.startswith("Ant"):
            self.low = -100
            self.high = 100
        else:
            self.low = -float("inf")
            self.high = float("inf")
        self.counter = 0

    def clip(self, obs):
        obs_c = np.clip(obs, self.low, self.high)
        clipped = np.any(obs != obs_c)
        if clipped:
            self.counter += 1
        return obs_c, clipped


def initialize_mujoco_from_obs(env, obs):
    """
    Initialize a given mujoco environment to a state conditioned on an observation.

    Missing information in the observation (which is usually the torso-coordinates)
    is filled with zeros.
    """
    env_id = env.unwrapped.spec.id
    if env_id == "InvertedPendulum-v2":
        nfill = 0
    elif env_id in (
        "HalfCheetah-v2",
        "HalfCheetah-FW-v2",
        "HalfCheetah-BW-v2",
        "HalfCheetah-Plot-v2",
        "Hopper-v2",
        "Hopper-FW-v2",
    ):
        nfill = 1
    elif env_id == "Ant-FW-v2":
        nfill = 2
    elif env_id == "FetchReachStack-v1":
        env.set_state_from_obs(obs)
        return env
    else:
        raise NotImplementedError(f"{env_id} not supported")

    nq = env.model.nq
    nv = env.model.nv
    obs_qpos = np.zeros(nq)
    obs_qpos[nfill:] = obs[: nq - nfill]
    obs_qvel = obs[nq - nfill : nq - nfill + nv]
    env.set_state(obs_qpos, obs_qvel)
    return env


def get_reward_done_from_obs_act(env, obs, act):
    """
    Returns a reward and done variable from an obs and action in a mujoco environment.

    This is a hacky way to get some information about the reward of a state and whether
    the episode is done, if you just have an observation. However, it is only used for
    evaluating the latent space model by directly training a policy on this signal.
    """
    env = initialize_mujoco_from_obs(env, obs)
    obs, reward, done, info = env.step(act)
    return reward, done


def compute_reward_done_from_obs(env, last_obs, action, next_obs):
    """
    Returns a reward and done variable from a transition in a mujoco environment.

    Both are manually computed from the observation.
    """
    env_id = env.unwrapped.spec.id
    if env_id == "InvertedPendulum-v2":
        # https://github.com/openai/gym/blob/master/gym/envs/mujoco/inverted_pendulum.py
        reward = 1.0
        notdone = np.isfinite(next_obs).all() and (np.abs(next_obs[1]) <= 0.2)
        done = not notdone
    elif env_id == "HalfCheetah-v2":
        # Note: This does not work for the default environment because the xposition
        # is being removed from the observation by the environment.
        nv = env.model.nv
        last_obs_qpos = last_obs[:-nv]
        next_obs_qpos = next_obs[:-nv]
        xposbefore = last_obs_qpos[0]
        xposafter = next_obs_qpos[0]
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / env.dt
        reward = reward_ctrl + reward_run
        done = False
    else:
        raise NotImplementedError("{} not supported".format(env_id))
    return reward, done
