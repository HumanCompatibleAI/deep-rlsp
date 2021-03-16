import numpy as np
from itertools import product

from deep_rlsp.envs.gridworlds.env import Env, Direction, get_grid_representation


class BasicRoomEnv(Env):
    """
    Basic empty room with stochastic transitions. Used for debugging.
    """

    def __init__(self, prob, use_pixels_as_observations=True):
        self.height = 3
        self.width = 3
        self.init_state = (1, 1)
        self.prob = prob
        self.nS = self.height * self.width
        self.nA = 5

        super().__init__(1, use_pixels_as_observations=use_pixels_as_observations)

        self.num_features = 2
        self.default_action = Direction.get_number_from_direction(Direction.STAY)
        self.num_features = len(self.s_to_f(self.init_state))

        self.reset()

        states = self.enumerate_states()
        self.make_transition_matrices(states, range(self.nA), self.nS, self.nA)
        self.make_f_matrix(self.nS, self.num_features)

    def enumerate_states(self):
        return product(range(self.width), range(self.height))

    def get_num_from_state(self, state):
        return np.ravel_multi_index(state, (self.width, self.height))

    def get_state_from_num(self, num):
        return np.unravel_index(num, (self.width, self.height))

    def s_to_f(self, s):
        return s

    def _obs_to_f(self, obs):
        return np.unravel_index(obs[0].argmax(), obs[0].shape)

    def _s_to_obs(self, s):
        layers = [[s]]
        obs = get_grid_representation(self.width, self.height, layers)
        return np.array(obs, dtype=np.float32)

        # render_width = 64
        # render_height = 64
        # x, y = s
        # obs = np.zeros((3, render_height, render_width), dtype=np.float32)
        # obs[
        #     :,
        #     y * render_height : (y + 1) * render_height,
        #     x * render_width : (x + 1) * render_width,
        # ] = 1
        # return obs

    def get_next_states(self, state, action):
        # next_states = []
        # for a in range(self.nA):
        #     next_s = self.get_next_state(state, a)
        # p = 1 - self.prob if a == action else self.prob / (self.nA - 1)
        # next_states.append((p, next_s, 0))

        next_s = self.get_next_state(state, action)
        next_states = [(self.prob, next_s, 0), (1 - self.prob, state, 0)]
        return next_states

    def get_next_state(self, state, action):
        """Returns the next state given a state and an action."""
        action = int(action)

        if action == Direction.get_number_from_direction(Direction.STAY):
            pass
        elif action < len(Direction.ALL_DIRECTIONS):
            move_x, move_y = Direction.move_in_direction_number(state, action)
            # New position is legal
            if 0 <= move_x < self.width and 0 <= move_y < self.height:
                state = move_x, move_y
            else:
                # Move only changes orientation, which we already handled
                pass
        else:
            raise ValueError("Invalid action {}".format(action))

        return state

    def s_to_ansi(self, state):
        return str(self.s_to_obs(state))


if __name__ == "__main__":
    from gym.utils.play import play

    env = BasicRoomEnv(1)
    play(env, fps=5)
