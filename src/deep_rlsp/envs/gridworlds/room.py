from copy import deepcopy
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from deep_rlsp.envs.gridworlds.env import (
    DeterministicEnv,
    Direction,
    get_grid_representation,
)


class RoomState:
    """State of the environment that describes the positions of all objects in the env.

    Attributes:
        agent_pos (tuple): (x, y) tuple for the agent's location.
        vase_states (dict): Dictionary mapping (x, y) tuples to booleans, where True
            means that the vase is intact.
    """

    def __init__(self, agent_pos, vase_states):
        self.agent_pos = agent_pos
        self.vase_states = vase_states

    def __eq__(self, other):
        return (
            isinstance(other, RoomState)
            and self.agent_pos == other.agent_pos
            and self.vase_states == other.vase_states
        )

    def __hash__(self):
        def get_vals(dictionary):
            return tuple([dictionary[loc] for loc in sorted(dictionary.keys())])

        return hash(self.agent_pos + get_vals(self.vase_states))


class RoomEnv(DeterministicEnv):
    """Room with a breakable vase and multiple goals.

    Attributes
        height (int): Height of the grid. Y coordinates are in [0, height).
        width (int): Width of the grid. X coordinates are in [0, width).
        init_state (RoomState): Initial state of the environment.
        vase_locations (list): (x, y) tuples describing the locations of vases.
        num_vases (int): Number of vases.
        carpet_locations (set): (x, y) tuples describing the locations of carpets.
        feature_locations (list): (x, y) tuples describing the locations of the goals.
        s (RoomState): Current state.
        nA (int): Number of actions.
    """

    def __init__(self, spec):
        self.height = spec.height
        self.width = spec.width
        self.init_state = deepcopy(spec.init_state)
        self.vase_locations = list(self.init_state.vase_states.keys())
        self.num_vases = len(self.vase_locations)
        self.carpet_locations = set(spec.carpet_locations)
        self.feature_locations = list(spec.feature_locations)
        self.nA = 5

        super().__init__(1, use_pixels_as_observations=spec.use_pixels_as_observations)

        self.default_action = Direction.get_number_from_direction(Direction.STAY)
        self.num_features = len(self.s_to_f(self.init_state))

        states = self.enumerate_states()
        self.reset()
        self.make_transition_matrices(states, range(self.nA), self.nS, self.nA)
        self.make_f_matrix(self.nS, self.num_features)

    def enumerate_states(self):
        state_num = {}
        # Possible vase states
        for vase_intact_bools in product([True, False], repeat=self.num_vases):
            vase_states = dict(zip(self.vase_locations, vase_intact_bools))
            # Possible agent positions
            for y in range(self.height):
                for x in range(self.width):
                    pos = (x, y)
                    if pos in vase_states and vase_states[pos]:
                        # Can't have the agent on an intact vase
                        continue
                    state = RoomState(pos, vase_states)
                    if state not in state_num:
                        state_num[state] = len(state_num)

        self.state_num = state_num
        self.num_state = {v: k for k, v in self.state_num.items()}
        self.nS = len(state_num)

        return state_num.keys()

    def get_num_from_state(self, state):
        return self.state_num[state]

    def get_state_from_num(self, num):
        return self.num_state[num]

    def s_to_f(self, s):
        """Returns features of the state.

        Feature vector is given by:
            - Number of broken vases
            - Whether the agent is on a carpet
            - For each feature location, whether the agent is on that location
        """
        num_broken_vases = list(s.vase_states.values()).count(False)
        carpet_feature = int(s.agent_pos in self.carpet_locations)
        features = [int(s.agent_pos == fpos) for fpos in self.feature_locations]
        features = [num_broken_vases, carpet_feature] + features
        return np.array(features, dtype=np.float32)

    def _obs_to_f(self, obs):
        """Returns features of the state given its observation.

        Feature vector is given by:
            - Number of broken vases
            - Whether the agent is on a carpet
            - For each feature location, whether the agent is on that location
        """
        agent_pos = np.unravel_index(obs[0].argmax(), obs[0].shape)
        num_broken_vases = np.sum(obs[2])
        carpet_feature = int(agent_pos in self.carpet_locations)
        features = [num_broken_vases, carpet_feature] + [
            int(agent_pos == fpos) for fpos in self.feature_locations
        ]
        return np.array(features, dtype=np.float32)

    def _s_to_obs(self, s):
        """Returns an array representation of the env to be used as an observation.

        The representation has dimensions (5, height, width) and consist of one-hot
        encodings of:
            - the agent's position
            - intact vases
            - broken vases
            - carpets
            - goals
        """
        layers = [
            [s.agent_pos],
            [pos for pos, intact in s.vase_states.items() if intact],
            [pos for pos, intact in s.vase_states.items() if not intact],
            self.carpet_locations,
            self.feature_locations,
        ]
        obs = get_grid_representation(self.width, self.height, layers)
        return np.array(obs, dtype=np.float32)

    def _obs_to_s(self, obs):
        obs = obs.copy()
        # agent_pos
        agent_y, agent_x = np.unravel_index(obs[:, :, 0].argmax(), obs[:, :, 0].shape)
        # vase
        vase_states = dict()
        for (vase_x, vase_y) in self.vase_locations:
            vase_state = obs[vase_y, vase_x, 1:3].argmax()
            vase_intact = vase_state == 0 and (agent_x, agent_y) != (vase_x, vase_y)
            vase_states[(vase_x, vase_y)] = vase_intact
        state = RoomState((agent_x, agent_y), vase_states)
        return state

    def get_next_state(self, state, action):
        """Returns the next state given a state and an action."""
        action = int(action)
        new_x, new_y = Direction.move_in_direction_number(state.agent_pos, action)
        # New position is still in bounds:
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            new_x, new_y = state.agent_pos
        new_agent_pos = new_x, new_y
        new_vase_states = deepcopy(state.vase_states)
        if new_agent_pos in new_vase_states:
            new_vase_states[new_agent_pos] = False  # Break the vase
        return RoomState(new_agent_pos, new_vase_states)

    def make_reward_heatmaps(self, rewards, out_prefix):
        possible_vase_states = product([False, True], repeat=self.num_vases)
        for vases_broken in possible_vase_states:
            reward_matrix = np.zeros((self.height, self.width))
            vase_states = dict(zip(self.vase_locations, vases_broken))
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) in vase_states and vase_states[(x, y)]:
                        # can't be on top of an intact vase
                        continue
                    state = RoomState((x, y), vase_states=vase_states)
                    state_id = self.get_num_from_state(state)
                    reward_matrix[y, x] = rewards[state_id]
            plt.imshow(reward_matrix)
            # Loop over data dimensions and create text annotations.
            # ax = plt.gca()
            # for y in range(self.height):
            #     for x in range(self.width):
            #         text = ax.text(
            #             x,
            #             y,
            #             np.round(reward_matrix[y, x], 2),
            #             ha="center",
            #             va="center",
            #             color="w",
            #             fontsize=14,
            #         )
            plt.colorbar()
            plt.savefig(
                out_prefix
                + "vases_"
                + "_".join([str(vb) for vb in vases_broken])
                + ".png"
            )
            plt.clf()

    def s_to_ansi(self, state):
        """Returns a string to render the state."""
        h, w = self.height, self.width
        canvas = np.zeros(tuple([2 * h - 1, 3 * w + 1]), dtype="int8")

        # cell borders
        for y in range(1, canvas.shape[0], 2):
            canvas[y, :] = 1
        for x in range(0, canvas.shape[1], 3):
            canvas[:, x] = 2

        # vases
        for x, y in self.vase_locations:
            if state.vase_states[(x, y)]:
                canvas[2 * y, 3 * x + 1] = 4
            else:
                canvas[2 * y, 3 * x + 1] = 6

        # goals
        for x, y in self.feature_locations:
            canvas[2 * y, 3 * x + 1] = 5

        # agent
        x, y = state.agent_pos
        canvas[2 * y, 3 * x + 2] = 3

        black_color = "\x1b[0m"
        purple_background_color = "\x1b[0;35;85m"

        lines = []
        for line in canvas:
            chars = []
            for char_num in line:
                if char_num == 0:
                    chars.append("\u2003")
                elif char_num == 1:
                    chars.append("─")
                elif char_num == 2:
                    chars.append("│")
                elif char_num == 3:
                    chars.append("\x1b[0;33;85m█" + black_color)
                elif char_num == 4:
                    chars.append("\x1b[0;32;85m█" + black_color)
                elif char_num == 5:
                    chars.append(purple_background_color + "█" + black_color)
                elif char_num == 6:
                    chars.append("\033[91m█" + black_color)
            lines.append("".join(chars))
        return "\n".join(lines)


if __name__ == "__main__":
    from deep_rlsp.run import get_problem_parameters

    env, s_current, r_task, r_true = get_problem_parameters("room", "default")
    rewards_task = env.f_matrix @ r_task
    rewards_true = env.f_matrix @ r_true
    env.make_reward_heatmaps(rewards_task, "task")
    env.make_reward_heatmaps(rewards_true, "true")
