from typing import Optional

import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.bench.monitor import Monitor

from deep_rlsp.envs.reward_wrapper import RewardWeightWrapper


class PPOSolver:
    def __init__(
        self, env, temperature: float = 1, tensorboard_log: Optional[str] = None
    ):
        self.env = RewardWeightWrapper(env, None)
        self.temperature = temperature
        # Monitor allows PPO to log the reward it achieves
        monitored_env = Monitor(self.env, None, allow_early_resets=True)
        self.vec_env = DummyVecEnv([lambda: monitored_env])
        self.tensorboard_log = tensorboard_log
        self._reset_model()

    def _reset_model(self):
        self.model = PPO2(
            MlpPolicy,
            self.vec_env,
            verbose=1,
            ent_coef=self.temperature,
            tensorboard_log=self.tensorboard_log,
        )

    def _get_tabular_policy(self):
        """
        Extracts a tabular policy representation from the PPO2 model.
        """
        policy = np.zeros((self.env.nS, self.env.nA))
        for state_id in range(self.env.nS):
            state = self.env.get_state_from_num(state_id)
            obs = self.env.s_to_obs(state)
            probs = self.model.action_probability(obs)
            policy[state_id, :] = probs

        # `action_probability` sometimes returns slightly unnormalized distributions
        # (probably numerical issues)  Hence, we normalize manually.
        policy /= policy.sum(axis=1, keepdims=True)
        assert np.allclose(policy.sum(axis=1), 1)
        return policy

    def learn(
        self, reward_weights, total_timesteps: int = 1000, reset_model: bool = False
    ):
        """
        Performs the PPO algorithm using the implementation from `stable_baselines`.

        Returns (np.ndarray):
            Array of shape (nS, nA) containing the action probabilites for each state.
        """
        if reset_model:
            self._reset_model()
        self.env.update_reward_weights(reward_weights)
        self.model.learn(total_timesteps=total_timesteps, log_interval=10)
        return self._get_tabular_policy()
