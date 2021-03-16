from typing import Tuple, Dict, Union, Optional

import numpy as np
import gym

from deep_rlsp.envs.gridworlds.env import Env
from deep_rlsp.model import LatentSpaceModel
from deep_rlsp.util.parameter_checks import check_between
from deep_rlsp.util.helper import init_env_from_obs


class RewardWeightWrapper(gym.Wrapper):
    def __init__(self, gridworld_env: Env, reward_weights: np.ndarray):
        self.gridworld_env = gridworld_env
        self.reward_weights = reward_weights
        super().__init__(gridworld_env)

    def update_reward_weights(self, reward_weights):
        self.reward_weights = reward_weights

    def step(self, action: int) -> Tuple[Union[int, np.ndarray], float, bool, Dict]:
        return self.gridworld_env.step(action, r_vec=self.reward_weights)


class LatentSpaceRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: Env,
        latent_space: LatentSpaceModel,
        r_inferred: Optional[np.ndarray],
        inferred_weight: Optional[float] = None,
        init_observations: Optional[list] = None,
        time_horizon: Optional[int] = None,
        init_prob: float = 0.2,
        use_task_reward: bool = False,
        reward_action_norm_factor: float = 0,
    ):
        if inferred_weight is not None:
            check_between("inferred_weight", inferred_weight, 0, 1)
        self.env = env
        self.latent_space = latent_space
        self.r_inferred = r_inferred
        if inferred_weight is None:
            inferred_weight = 1
        self.inferred_weight = inferred_weight
        self.state = None
        self.timestep = 0
        self.use_task_reward = use_task_reward
        self.reward_action_norm_factor = reward_action_norm_factor

        self.init_prob = init_prob
        self.init_observations = init_observations

        if time_horizon is None:
            self.time_horizon = self.env.spec.max_episode_steps
        else:
            self.time_horizon = time_horizon
        super().__init__(env)

    def reset(self) -> Tuple[np.ndarray, float, bool, Dict]:
        obs = super().reset()

        if self.init_observations is not None:
            if np.random.random() < self.init_prob:
                idx = np.random.randint(0, len(self.init_observations))
                obs = self.init_observations[idx]
                self.env = init_env_from_obs(self.env, obs)

        self.state = self.latent_space.encoder(obs)
        self.timestep = 0
        return obs

    def _get_reward(self, action, info):
        if self.r_inferred is not None:
            inferred = np.dot(self.r_inferred, self.state)
            info["inferred"] = inferred
        else:
            inferred = 0

        if self.use_task_reward and "task_reward" in info:
            task = info["task_reward"]
        else:
            task = 0

        action_norm = np.square(action).sum()

        reward = self.inferred_weight * inferred
        reward += task
        reward += self.reward_action_norm_factor * action_norm
        return reward

    def step(
        self, action, return_true_reward: bool = False,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, true_reward, done, info = self.env.step(action)
        self.state = self.latent_space.encoder(obs)

        if return_true_reward:
            assert "true_reward" in info
            reward = info["true_reward"]
        else:
            reward = self._get_reward(action, info)

        # ignore termination criterion from mujoco environments
        # to avoid reward information leak
        self.timestep += 1
        done = self.timestep > self.time_horizon

        return obs, reward, done, info
