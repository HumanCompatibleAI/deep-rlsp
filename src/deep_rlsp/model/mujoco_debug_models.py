"""
Exact dynamics models and handcoded features used for debugging in the pendulum env.
"""

import numpy as np

from deep_rlsp.model.exact_dynamics_mujoco import ExactDynamicsMujoco


class IdentityFeatures:
    def __init__(self, env):
        self.encoder = lambda x: x
        self.decoder = lambda x: x


class MujocoDebugFeatures:
    def __init__(self, env):
        self.env_id = env.unwrapped.spec.id
        assert env.unwrapped.spec.id in ("InvertedPendulum-v2", "FetchReachStack-v1")
        self.env = env
        if self.env_id == "InvertedPendulum-v2":
            self.state_size = 5
        elif self.env_id == "FetchReachStack-v1":
            self.state_size = env.observation_space.shape[0] + 1

    def encoder(self, obs):
        if self.env_id == "InvertedPendulum-v2":
            # feature = np.isfinite(obs).all() and (np.abs(obs[1]) <= 0.2)
            feature = np.abs(obs[1])
        elif self.env_id == "FetchReachStack-v1":
            feature = int(obs[-4] < 0.5)
        obs = np.concatenate([obs, [feature]])
        return obs

    def decoder(self, state):
        return state[:-1]


class PendulumDynamics:
    def __init__(self, latent_space, backward=False):
        assert latent_space.env.unwrapped.spec.id == "InvertedPendulum-v2"
        self.latent_space = latent_space
        self.backward = backward
        self.dynamics = ExactDynamicsMujoco(
            self.latent_space.env.unwrapped.spec.id, tolerance=1e-2, max_iters=100
        )
        self.low = np.array([-1, -np.pi, -10, -10])
        self.high = np.array([1, np.pi, 10, 10])

    def step(self, state, action, sample=True):
        obs = state  # self.latent_space.decoder(state)
        obs = np.clip(obs, self.low, self.high)
        if self.backward:
            obs = self.dynamics.inverse_dynamics(obs, action)
        else:
            obs = self.dynamics.dynamics(obs, action)
        state = obs  # self.latent_space.encoder(obs)
        return state

    def learn(self, *args, return_initial_loss=False, **kwargs):
        print("Using exact dynamics...")
        if return_initial_loss:
            return 0, 0
        return 0
