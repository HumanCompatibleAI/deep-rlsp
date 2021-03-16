import gym
import numpy as np

from gym.envs.registration import register
from gym.wrappers import TimeLimit

from deep_rlsp.envs.gridworlds import TOY_PROBLEMS
from deep_rlsp.envs.gridworlds.gym_envs import get_gym_gridworld_id_time_limit


for env_name in TOY_PROBLEMS.keys():
    for env_spec in TOY_PROBLEMS[env_name]:  # type: ignore
        id, time_limit = get_gym_gridworld_id_time_limit(env_name, env_spec)
        register(
            id=id,
            entry_point="deep_rlsp.envs.gridworlds.gym_envs:make_gym_gridworld",
            max_episode_steps=time_limit,
            kwargs={"env_name": env_name, "env_spec": env_spec},
        )


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.

    See
    https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    Original implementation:
    https://github.com/araffin/rl-baselines-zoo/blob/master/utils/wrappers.py

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.0]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


class TerminationToPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=1):
        self.penalty = penalty
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            reward -= self.penalty
            done = False
        return obs, reward, done, info


class RobotObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs = self.unwrapped._get_obs()
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            [
                obs["observation"].shape[0]
                + obs["achieved_goal"].shape[0]
                + obs["desired_goal"].shape[0]
            ],
        )

    def observation(self, obs):
        return np.concatenate(
            [obs["observation"], obs["achieved_goal"], obs["desired_goal"]]
        )


try:
    from deep_rlsp.envs.mujoco.half_cheetah import HalfCheetahEnv
    from deep_rlsp.envs.robotics.reach_side_effects import FetchReachSideEffectEnv

    def get_cheetah(time_feature=True, **kwargs):
        env = HalfCheetahEnv(**kwargs)
        if time_feature:
            env = TimeFeatureWrapper(env)
        return env

    def get_ant(**kwargs):
        env = gym.make("Ant-v2")
        env = TimeFeatureWrapper(env)
        return env

    def get_hopper(**kwargs):
        env = gym.make("Hopper-v2")
        env = TimeFeatureWrapper(env)
        return env

    def get_hopper_penalty(**kwargs):
        env = gym.make("Hopper-v2")
        env = TimeFeatureWrapper(env)
        env = TerminationToPenaltyWrapper(env, penalty=1)
        return env

    def get_swimmer(**kwargs):
        env = gym.make("Swimmer-v2")
        env = TimeFeatureWrapper(env)
        return env

    def get_reach(**kwargs):
        env = FetchReachSideEffectEnv()
        # env = TimeFeatureWrapper(env)
        # env = gym.make("FetchReachDense-v1")
        # env = RobotObservationWrapper(env)
        return env

    register(
        id="HalfCheetah-FW-v2",
        entry_point="deep_rlsp:get_cheetah",
        max_episode_steps=1000,
        kwargs={"expose_all_qpos": False, "task": "default"},
    )

    register(
        id="HalfCheetah-QPOS-v2",
        entry_point="deep_rlsp:get_cheetah",
        max_episode_steps=1000,
        kwargs={"expose_all_qpos": True, "task": "default"},
    )

    register(
        id="HalfCheetah-BW-v2",
        entry_point="deep_rlsp:get_cheetah",
        max_episode_steps=1000,
        kwargs={"expose_all_qpos": False, "task": "run_back"},
    )

    register(
        id="HalfCheetah-Plot-v2",
        entry_point="deep_rlsp:get_cheetah",
        max_episode_steps=1000,
        kwargs={
            "time_feature": False,
            "expose_all_qpos": False,
            "task": "default",
            "model_path": "half_cheetah_plot.xml",
        },
    )

    register(
        id="Ant-FW-v2",
        entry_point="deep_rlsp:get_ant",
        max_episode_steps=1000,
        kwargs={"expose_all_qpos": False, "task": "forward"},
    )

    register(
        id="Ant-FW-NB-v2",
        entry_point="deep_rlsp:get_ant",
        max_episode_steps=1000,
        kwargs={
            "expose_all_qpos": False,
            "task": "forward",
            "model_path": "ant_nobackground.xml",
        },
    )

    register(
        id="Hopper-FW-v2", entry_point="deep_rlsp:get_hopper", max_episode_steps=1000
    )

    register(
        id="Hopper-FW-Penalty-v2",
        entry_point="deep_rlsp:get_hopper_penalty",
        max_episode_steps=1000,
    )

    register(
        id="Swimmer-FW-v2", entry_point="deep_rlsp:get_swimmer", max_episode_steps=1000
    )

    register(
        id="FetchReachStack-v1", entry_point="deep_rlsp:get_reach", max_episode_steps=50
    )

    mujoco_available = True
except Exception as e:
    print(e)
    mujoco_available = False
