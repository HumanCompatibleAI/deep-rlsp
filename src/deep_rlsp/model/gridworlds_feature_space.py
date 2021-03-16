class GridworldsFeatureSpace:
    def __init__(self, env):
        self.env = env
        s = self.env.init_state
        f = self.env.s_to_f(s)
        assert len(f.shape) == 1
        self.state_size = f.shape[0]

    def encoder(self, obs):
        s = self.env.obs_to_s(obs)
        f = self.env.s_to_f(s)
        return f

    def decoder(self, state):
        raise NotImplementedError()
