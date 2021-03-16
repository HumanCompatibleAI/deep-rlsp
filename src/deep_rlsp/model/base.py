import abc


class TransitionModel(abc.ABC):
    def __init__(self, env):
        pass

    @abc.abstractmethod
    def models_observations(self):
        pass

    @abc.abstractmethod
    def forward_sample(self, state):
        pass

    @abc.abstractmethod
    def backward_sample(self, state, t):
        pass


class InversePolicy(abc.ABC):
    def __init__(self, env, policy):
        pass

    @abc.abstractmethod
    def step(self, next_state):
        pass

    @abc.abstractmethod
    def sample(self, next_state):
        pass
