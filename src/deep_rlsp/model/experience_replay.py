import collections

import numpy as np

from deep_rlsp.model.exact_dynamics_mujoco import ExactDynamicsMujoco


class Normalizer:
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.std = np.ones(num_inputs)

    def add(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        var = self.mean_diff / self.n
        var[var < 1e-8] = 1
        self.std = np.sqrt(var)

    def normalize(self, inputs):
        return (inputs - self.mean) / self.std

    def unnormalize(self, inputs):
        return inputs * self.std + self.mean


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.obs_normalizer = None
        self.act_normalizer = None
        self.delta_normalizer = None
        self.max_delta = None
        self.max_delta = None

    def __len__(self):
        return len(self.buffer)

    def append(self, obs, action, next_obs):
        delta = obs - next_obs
        if self.obs_normalizer is None:
            self.obs_normalizer = Normalizer(len(obs))
        if self.delta_normalizer is None:
            self.delta_normalizer = Normalizer(len(obs))
            self.max_delta = delta
            self.min_delta = delta
        if self.act_normalizer is None:
            self.act_normalizer = Normalizer(len(action))
        self.obs_normalizer.add(obs)
        self.obs_normalizer.add(next_obs)
        self.act_normalizer.add(action)
        self.delta_normalizer.add(delta)
        self.buffer.append((obs, action, next_obs))
        self.max_delta = np.maximum(self.max_delta, delta)
        self.min_delta = np.minimum(self.min_delta, delta)

    def sample(self, batch_size, normalize=False):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, act, next_obs = zip(*[self.buffer[idx] for idx in indices])
        obs, act, next_obs = np.array(obs), np.array(act), np.array(next_obs)
        if normalize:
            obs = self.normalize_obs(obs)
            act = self.normalize_act(act)
            next_obs = self.normalize_obs(next_obs)
        return obs, act, next_obs

    def clip_delta(self, delta):
        return np.clip(delta, self.min_delta, self.max_delta)

    def normalize_obs(self, obs):
        return self.obs_normalizer.normalize(obs)

    def unnormalize_obs(self, obs):
        return self.obs_normalizer.unnormalize(obs)

    def normalize_delta(self, delta):
        return self.delta_normalizer.normalize(delta)

    def unnormalize_delta(self, delta):
        return self.delta_normalizer.unnormalize(delta)

    def normalize_act(self, act):
        return self.act_normalizer.normalize(act)

    def unnormalize_act(self, act):
        return self.act_normalizer.unnormalize(act)

    def add_random_rollouts(self, env, timesteps, n_rollouts):
        for _ in range(n_rollouts):
            obs = env.reset()
            for t in range(timesteps):
                action = env.action_space.sample()
                next_obs, _, _, _ = env.step(action)
                self.append(obs, action, next_obs)
                obs = next_obs

    def add_play_data(self, env, play_data):
        dynamics = ExactDynamicsMujoco(env.unwrapped.spec.id)
        observations, actions = play_data["observations"], play_data["actions"]
        n_traj = len(observations)
        assert len(actions) == n_traj
        for i in range(n_traj):
            l_traj = len(observations[i])
            for t in range(l_traj):
                obs = observations[i][t]
                action = actions[i][t]
                next_obs = dynamics.dynamics(obs, action)
                self.append(obs, action, next_obs)

    def add_policy_rollouts(self, env, policy, n_rollouts, horizon, eps_greedy=0):
        for i in range(n_rollouts):
            obs = env.reset()
            for t in range(horizon):
                if eps_greedy > 0 and np.random.random() < eps_greedy:
                    action = env.action_space.sample()
                else:
                    action = policy.predict(obs)[0]
                next_obs, reward, done, info = env.step(action)
                self.append(obs, action, next_obs)
                obs = next_obs
