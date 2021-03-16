from copy import deepcopy
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from deep_rlsp.envs.gridworlds.env import (
    DeterministicEnv,
    Direction,
    get_grid_representation,
)


class TrainState:
    """State of the environment that describes the positions of all objects in the env.

    Attributes:
        agent_pos (tuple): (x, y) tuple for the agent's location.
        vase_states (dict): Dictionary mapping (x, y) tuples to booleans, where True
            means that the vase is intact.
        train_pos (tuple): (x, y) tuple for the train's initial location.
        train_intact (bool): True if the train is still intact.
    """

    def __init__(self, agent_pos, vase_states, train_pos, train_intact):
        self.agent_pos = agent_pos
        self.vase_states = vase_states
        self.train_pos = train_pos
        self.train_intact = train_intact

    def is_valid(self):
        pos = self.agent_pos
        # Can't be standing on the vase and have the vase intact
        if pos in self.vase_states and self.vase_states[pos]:
            return False
        # Can't be standing on the train and have the train intact
        if pos == self.train_pos and self.train_intact:
            return False
        return True

    def __eq__(self, other):
        return (
            isinstance(other, TrainState)
            and self.agent_pos == other.agent_pos
            and self.vase_states == other.vase_states
            and self.train_pos == other.train_pos
            and self.train_intact == other.train_intact
        )

    def __hash__(self):
        def get_vals(dictionary):
            return tuple([dictionary[loc] for loc in sorted(dictionary.keys())])

        return hash(
            self.agent_pos
            + get_vals(self.vase_states)
            + self.train_pos
            + (self.train_intact,)
        )


class TrainEnv(DeterministicEnv):
    """Room with breakable vases and a breakable tran that moves.

    Attributes
        height (int): Height of the grid. Y coordinates are in [0, height).
        width (int): Width of the grid. X coordinates are in [0, width).
        init_state (TrainState): Initial state of the environment.
        vase_locations (list): (x, y) tuples describing the locations of vases.
        num_vases (int): Number of vases.
        carpet_locations (set): (x, y) tuples describing the locations of carpets.
        feature_locations (list): (x, y) tuples describing the locations of the goals.
        train_transition (dict): Dictionary from (x, y) tuples to (x, y) tuples that
            describes the (deterministic) transitions of the train.
        train_locations (list): (x, y) tuples of potential location of the train.
        s (TrainState): Current state.
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
        self.train_transition = spec.train_transition
        self.train_locations = list(self.train_transition.keys())
        assert set(self.train_locations) == set(self.train_transition.values())
        self.nA = 5

        super().__init__(1)

        self.default_action = Direction.get_number_from_direction(Direction.STAY)
        self.num_features = len(self.s_to_f(self.init_state))

        self.reset()

        states = self.enumerate_states()
        self.make_transition_matrices(states, range(self.nA), self.nS, self.nA)
        self.make_f_matrix(self.nS, self.num_features)

    def enumerate_states(self):
        state_num = {}
        all_agent_positions = product(range(self.width), range(self.height))
        all_vase_states = map(
            lambda vase_vals: dict(zip(self.vase_locations, vase_vals)),
            product([True, False], repeat=self.num_vases),
        )
        all_states = map(
            lambda x: TrainState(*x),
            product(
                all_agent_positions,
                all_vase_states,
                self.train_locations,
                [True, False],
            ),
        )
        all_states = filter(lambda state: state.is_valid(), all_states)

        state_num = {}
        for state in all_states:
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
            - Whether the agent is on any carpet
            - Whether the agent has broken the train already
            - For each potential train position, whether the train is in that position
            - For each goal, whether the agent is on that goal
        """
        num_broken_vases = list(s.vase_states.values()).count(False)
        carpet_feature = int(s.agent_pos in self.carpet_locations)
        train_broken_feature = int(not s.train_intact)
        train_pos_features = [int(s.train_pos == pos) for pos in self.train_locations]
        loc_features = [int(s.agent_pos == fpos) for fpos in self.feature_locations]
        features = (
            [num_broken_vases, carpet_feature, train_broken_feature]
            + train_pos_features
            + loc_features
        )
        return np.array(features, dtype=np.float32)

    def _obs_to_f(self, obs):
        """Returns features of the state given its observation.

        Feature vector is given by:
            - Number of broken vases
            - Whether the agent is on any carpet
            - Whether the agent has broken the train already
            - For each potential train position, whether the train is in that position
            - For each goal, whether the agent is on that goal
        """
        agent_pos = np.unravel_index(obs[0].argmax(), obs[0].shape)
        num_broken_vases = np.sum(obs[2])
        carpet_feature = int(agent_pos in self.carpet_locations)
        train_broken_feature = int(np.sum([4]) >= 0.5)
        if train_broken_feature == 1:
            train_pos = np.unravel_index(obs[3].argmax(), obs[3].shape)
        else:
            train_pos = np.unravel_index(obs[4].argmax(), obs[4].shape)
        train_pos_features = [int(train_pos == pos) for pos in self.train_locations]
        loc_features = [int(agent_pos == fpos) for fpos in self.feature_locations]
        features = (
            [num_broken_vases, carpet_feature, train_broken_feature]
            + train_pos_features
            + loc_features
        )
        return np.array(features, dtype=np.float32)

    def _s_to_obs(self, s):
        """Returns an array representation of the env to be used as an observation.

        The representation has dimensions (7, height, width) and consist of one-hot
        encodings of:
            - the agent's position
            - intact vases
            - broken vases
            - intact trains (can only be one)
            - broken trains (can only be one)
            - carpets
            - goals
        """
        layers = [
            [s.agent_pos],
            [pos for pos, intact in s.vase_states.items() if intact],
            [pos for pos, intact in s.vase_states.items() if not intact],
            [s.train_pos] if s.train_intact else [],
            [s.train_pos] if not s.train_intact else [],
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
        # train
        max_val = -np.inf
        for (x, y) in self.train_locations:
            for z in (3, 4):
                val = obs[y, x, z]
                if val > max_val:
                    train_x, train_y, train_state = x, y, z
                    max_val = val
        train_intact = train_state == 3 and (agent_x, agent_y) != (train_x, train_y)
        state = TrainState(
            (agent_x, agent_y), vase_states, (train_x, train_y), train_intact
        )
        return state

    def get_next_state(self, state, action):
        """Return the next state given a state and an action."""
        action = int(action)
        new_x, new_y = Direction.move_in_direction_number(state.agent_pos, action)
        # New position is still in bounds:
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            new_x, new_y = state.agent_pos
        new_agent_pos = new_x, new_y
        new_vase_states = deepcopy(state.vase_states)
        new_train_pos, new_train_intact = state.train_pos, state.train_intact
        if state.train_intact:
            new_train_pos = self.train_transition[state.train_pos]

        # Break the vase and train if appropriate
        if new_agent_pos in new_vase_states:
            new_vase_states[new_agent_pos] = False
        if new_agent_pos == new_train_pos:
            new_train_intact = False
        return TrainState(
            new_agent_pos, new_vase_states, new_train_pos, new_train_intact
        )

    def make_reward_heatmaps(self, rewards, out_prefix):
        possible_vase_states = product([False, True], repeat=self.num_vases)
        for vases_broken in possible_vase_states:
            vase_states = dict(zip(self.vase_locations, vases_broken))
            for train_pos in self.train_locations:
                for train_intact in [False, True]:
                    reward_matrix = np.zeros((self.height, self.width))
                    for y in range(self.height):
                        for x in range(self.width):
                            state = TrainState(
                                (x, y),
                                vase_states=vase_states,
                                train_pos=train_pos,
                                train_intact=train_intact,
                            )
                            if state.is_valid():
                                state_id = self.get_num_from_state(state)
                                reward_matrix[y, x] = rewards[state_id]
                    plt.imshow(reward_matrix)
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
                        + "_train_{}_{}_{}".format(
                            train_pos[0], train_pos[1], train_intact
                        )
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

        # agent
        x, y = state.agent_pos
        canvas[2 * y, 3 * x + 2] = 3

        # train
        x, y = state.train_pos
        if state.train_intact:
            canvas[2 * y, 3 * x + 1] = 5
        else:
            canvas[2 * y, 3 * x + 1] = 6

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

    env, s_current, r_task, r_true = get_problem_parameters("train", "default")
    rewards_task = env.f_matrix @ r_task
    rewards_true = env.f_matrix @ r_true
    env.make_reward_heatmaps(rewards_task, "task")
    env.make_reward_heatmaps(rewards_true, "true")
