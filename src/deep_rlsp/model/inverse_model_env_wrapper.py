from typing import Tuple, Dict, Union

import numpy as np
import gym

from deep_rlsp.model import LatentSpaceModel, InverseDynamicsMDN
from deep_rlsp.util.mujoco import compute_reward_done_from_obs

MIN_FLOAT64 = np.finfo(np.float64).min
MAX_FLOAT64 = np.finfo(np.float64).max


class InverseModelGymEnv(gym.Env):
    """
    Allows to treat an inverse dynamics model as an MDP.

    Used for model evaluation and debugging.
    """

    def __init__(
        self,
        latent_space: LatentSpaceModel,
        inverse_model: InverseDynamicsMDN,
        initial_obs: np.ndarray,
        time_horizon: int,
    ):
        self.latent_space = latent_space
        self.inverse_model = inverse_model

        self.time_horizon = time_horizon
        self.update_initial_state(obs=initial_obs)
        self.action_space = self.latent_space.env.action_space
        self.observation_space = gym.spaces.Box(
            MIN_FLOAT64, MAX_FLOAT64, shape=(self.latent_space.state_size,)
        )
        if hasattr(self.latent_space.env, "nA"):
            self.nA = self.latent_space.env.nA
        self.reset()

    def update_initial_state(self, state=None, obs=None):
        if (state is not None and obs is not None) or (state is None and obs is None):
            raise ValueError("Exactly one of state and obs should be None.")
        if state is None:
            obs = np.expand_dims(obs, 0)
            state = self.latent_space.encoder(obs)[0]
        self.initial_state = state

    def reset(self) -> np.ndarray:
        self.timestep = 0
        self.state = self.initial_state
        return self.state

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        last_state = self.state
        last_obs = self.latent_space.decoder(last_state)
        self.state = self.inverse_model.step(self.state, action)
        obs = self.latent_space.decoder(self.state)
        # invert transition to get reward
        reward, done = compute_reward_done_from_obs(
            self.latent_space.env, obs, action, last_obs
        )
        return self.state, reward, done, dict()
