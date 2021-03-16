from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

from matplotlib.colors import to_rgba_array
import seaborn as sns

COLOR_CYCLE = ["#0000ff"] + list(sns.color_palette("deep"))


def action_id_to_string(action_id):
    if action_id == 5:
        return "MOVE"
    else:
        return Direction.INDEX_TO_DIRECTION_STRING[action_id]


def get_grid_representation(width, height, layers):
    """Returns a 3d representation of a gridworld to be used as an observation."""
    grid = np.zeros((height, width, len(layers)), dtype=np.uint8)
    for i, layer in enumerate(layers):
        for x, y in layer:
            grid[y, x, i] = 1
    return grid


def grid_to_rgb(grid, render_scale):
    height, width, n_layers = grid.shape
    # rgb = np.zeros((height * render_scale, width * render_scale, 3), dtype=np.int)
    rgb = np.ones((64, 64, 3), dtype=np.int) * 200
    for layer in list(range(1, n_layers)) + [0]:
        for y in range(height):
            for x in range(width):
                color = COLOR_CYCLE[layer % len(COLOR_CYCLE)]
                if grid[y, x, layer]:
                    rgb[
                        y * render_scale : (y + 1) * render_scale,
                        x * render_scale : (x + 1) * render_scale,
                        :,
                    ] = to_rgba_array(color)[0, :3] * int(255)
    return rgb


class Env(ABC, gym.Env):
    metadata = {"render.modes": ["ansi", "rgb_array", "human"]}

    def __init__(self, obs_max, use_pixels_as_observations=False):
        self.use_states_as_observations = False
        self.use_pixels_as_observations = use_pixels_as_observations
        self.obs_shape = self._s_to_obs(self.init_state).shape
        self.observation_space = gym.spaces.Box(
            0, obs_max, shape=self.s_to_obs(self.init_state).shape
        )
        self.action_space = gym.spaces.Discrete(self.nA)
        self.time_horizon = 20
        self.render_scale = 64 // max(self.height, self.width)

    @abstractmethod
    def get_num_from_state(self, state):
        raise NotImplementedError()

    @abstractmethod
    def get_state_from_num(self, state_id):
        raise NotImplementedError()

    @abstractmethod
    def get_next_states(self, state, action):
        raise NotImplementedError()

    @abstractmethod
    def enumerate_states(self):
        raise NotImplementedError()

    @abstractmethod
    def s_to_f(self, s):
        """Returns features of the state."""
        raise NotImplementedError()

    @abstractmethod
    def s_to_ansi(self, state):
        """Returns a string to render the state."""
        raise NotImplementedError()

    def is_deterministic(self):
        return False

    def make_reward_heatmaps(self, rewards, out_prefix):
        pass

    def get_initial_state_distribution(self, known_initial_state=True):
        if known_initial_state:
            p_0 = np.zeros(self.nS)
            p_0[self.get_num_from_state(self.init_state)] = 1
        else:
            p_0 = np.ones(self.nS) / self.nS
        return p_0

    def make_transition_matrices(self, states_iter, actions_iter, nS, nA):
        """
        states_iter: ITERATOR of states (i.e. can only be used once)
        actions_iter: ITERATOR of actions (i.e. can only be used once)
        """
        P = {}
        T_matrix = lil_matrix((nS * nA, nS))
        baseline_matrix = lil_matrix((nS, nS))
        actions = list(actions_iter)

        for state in states_iter:
            state_id = self.get_num_from_state(state)
            P[state_id] = {}
            for _, action in enumerate(actions):
                next_s = self.get_next_states(state, action)
                next_s = [(p, self.get_num_from_state(s), r) for p, s, r in next_s]
                P[state_id][action] = next_s
                state_action_index = state_id * nA + action
                for prob, next_state_id, _ in next_s:
                    T_matrix[state_action_index, next_state_id] = prob
                    if action == self.default_action:
                        baseline_matrix[state_id, next_state_id] = prob
        self.P = P
        self.T_matrix = T_matrix.tocsr()
        self.T_matrix_transpose = T_matrix.transpose().tocsr()
        self.baseline_matrix_transpose = baseline_matrix.transpose().tocsr()

    def make_f_matrix(self, nS, num_features):
        self.f_matrix = np.zeros((nS, num_features))
        for state_id in self.P.keys():
            state = self.get_state_from_num(state_id)
            self.f_matrix[state_id, :] = self.s_to_f(state)

    def reset(self, state=None):
        if state is None:
            state = self.init_state
        self.timestep = 0
        self.s = deepcopy(state)
        obs = self.s_to_obs(state)
        if self.use_pixels_as_observations:
            obs = grid_to_rgb(np.reshape(obs, self.obs_shape), self.render_scale)
        return obs

    def state_step(self, action, state=None):
        if state is None:
            state = self.s
        next_states = self.get_next_states(state, action)
        probabilities = [p for p, _, _ in next_states]
        idx = np.random.choice(np.arange(len(next_states)), p=probabilities)
        return next_states[idx][1]

    def step(self, action, r_vec=None):
        """
        given an action, takes a step from self.s, updates self.s and returns:
        - the observation (features of the next state)
        - the associated reward
        - done, the indicator of completed episode
        - info
        """
        self.s = self.state_step(action)
        self.timestep += 1

        features = self.s_to_f(self.s)
        obs = self.s_to_obs(self.s)

        if self.use_pixels_as_observations:
            obs = grid_to_rgb(np.reshape(obs, self.obs_shape), self.render_scale)

        reward = 0 if r_vec is None else np.array(features.T @ r_vec)
        done = self.time_horizon is not None and self.timestep >= self.time_horizon
        info = defaultdict(lambda: "")

        return (obs, reward, np.array(done, dtype="bool"), info)

    def s_to_obs(self, s):
        if self.use_states_as_observations:
            return s
        obs = self._s_to_obs(s).flatten()
        return obs.copy()

    def obs_to_s(self, obs):
        if self.use_states_as_observations:
            return obs
        obs = np.reshape(obs, self.obs_shape)
        return self._obs_to_s(obs)

    def obs_to_f(self, obs):
        if self.use_states_as_observations:
            return self.s_to_f(obs)
        return self._obs_to_f(obs)

    def render(self, mode="ansi"):
        """Renders the environment."""
        if mode == "ansi":
            return self.s_to_ansi(self.s)
        elif mode == "rgb_array" or mode == "human":
            assert not self.use_states_as_observations
            obs = self.s_to_obs(self.s)
            rgb = grid_to_rgb(np.reshape(obs, self.obs_shape), self.render_scale)
            if mode == "human":
                plt.axis("off")
                plt.imshow(rgb, origin="upper", extent=(0, self.width, self.height, 0))
                plt.tight_layout()
                plt.pause(0.05)
                plt.clf()
                return None
            return rgb
        else:
            return super().render(mode=mode)  # just raise an exception

    def get_keys_to_action(self):
        """
        Provides the controls for using the environment with gym.util.play
        """
        KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN = 275, 276, 273, 274
        return {
            (): Direction.get_number_from_direction(Direction.STAY),
            (KEY_UP,): Direction.get_number_from_direction(Direction.NORTH),
            (KEY_DOWN,): Direction.get_number_from_direction(Direction.SOUTH),
            (KEY_UP, KEY_DOWN): Direction.get_number_from_direction(Direction.STAY),
            (KEY_LEFT,): Direction.get_number_from_direction(Direction.WEST),
            (KEY_RIGHT,): Direction.get_number_from_direction(Direction.EAST),
            (KEY_LEFT, KEY_RIGHT): Direction.get_number_from_direction(Direction.STAY),
        }  # control with arrow keys


class DeterministicEnv(Env):
    def is_deterministic(self):
        return True

    def make_transition_matrices(self, states_iter, actions_iter, nS, nA):
        """
        states_iter: ITERATOR of states (i.e. can only be used once)
        actions_iter: ITERATOR of actions (i.e. can only be used once)
        nS: Number of states
        nA: Number of actions
        """
        super().make_transition_matrices(states_iter, actions_iter, nS, nA)
        self._make_deterministic_transition_matrix(nS, nA)
        self._make_deterministic_transition_transpose_matrix(nS, nA)

    def get_next_states(self, state, action):
        return [(1.0, self.get_next_state(state, action), 0)]

    def state_step(self, action, state=None):
        if state is None:
            state = self.s
        return self.get_next_state(state, action)

    def _make_deterministic_transition_matrix(self, nS, nA):
        """Create self.deterministic_T, a matrix with index S,A -> S'   """
        self.deterministic_T = np.zeros((nS, nA), dtype="int32")
        for s in range(nS):
            for a in range(nA):
                self.deterministic_T[s, a] = self.P[s][a][0][1]

    def _make_deterministic_transition_transpose_matrix(self, nS, nA):
        """
        Create self.deterministic_transpose, a matrix with index S, A -> S',
        containing the inverse dynamics
        """
        self.deterministic_transpose = np.zeros((nS, nA), dtype="int32")
        for s in range(nS):
            for a in range(nA):
                self.deterministic_transpose[self.P[s][a][0][1], a] = s


class Direction(object):
    """A class that contains the five actions available in Gridworlds.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """

    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    STAY = (0, 0)
    INDEX_TO_DIRECTION = [NORTH, SOUTH, EAST, WEST, STAY]
    INDEX_TO_DIRECTION_STRING = ["NORTH", "SOUTH", "EAST", "WEST", "STAY"]
    DIRECTION_TO_INDEX = {a: i for i, a in enumerate(INDEX_TO_DIRECTION)}
    ALL_DIRECTIONS = INDEX_TO_DIRECTION
    N_DIRECTIONS = len(ALL_DIRECTIONS)

    @staticmethod
    def move_in_direction(point, direction):
        """Takes a step in the given direction and returns the new point.

        point: Tuple (x, y) representing a point in the x-y plane.
        direction: One of the Directions.
        """
        x, y = point
        dx, dy = direction
        return (x + dx, y + dy)

    @staticmethod
    def move_in_direction_number(point, num):
        direction = Direction.get_direction_from_number(num)
        return Direction.move_in_direction(point, direction)

    @staticmethod
    def get_number_from_direction(direction):
        return Direction.DIRECTION_TO_INDEX[direction]

    @staticmethod
    def get_direction_from_number(number):
        return Direction.INDEX_TO_DIRECTION[number]

    @staticmethod
    def get_onehot_from_direction(direction):
        num = Direction.get_number_from_direction(direction)
        return np.arange(Direction.N_DIRECTIONS) == num
