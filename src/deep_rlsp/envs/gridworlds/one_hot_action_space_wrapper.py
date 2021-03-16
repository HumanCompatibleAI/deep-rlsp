import gym
import numpy as np


class OneHotActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self.n_actions = env.action_space.n
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (self.n_actions,))

    def step(self, action, **kwargs):
        return self.env.step(self.action(action), **kwargs)

    def action(self, one_hot_action):
        assert one_hot_action.shape == (self.n_actions,)
        action = np.argmax(one_hot_action)
        return action

    def reverse_action(self, action):
        return np.arange(self.n_actions) == action
