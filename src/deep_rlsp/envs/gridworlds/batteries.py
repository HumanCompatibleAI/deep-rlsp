from copy import deepcopy
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from deep_rlsp.envs.gridworlds.env import (
    DeterministicEnv,
    Direction,
    get_grid_representation,
)


class BatteriesState:
    """State of the environment that describes the positions of all objects in the env.

    Attributes:
        agent_pos (tuple): (x, y) tuple for the agent's location.
        train_pos (tuple): (x, y) tuple for the train's initial location.
        train_life (int): Amount of energy left for the train
            (reduces by 1 in each timestep).
        battery_present (dict): Dictionary from (x, y) tuples to booleans describing the
            batteries present in the environment and wheather they have been collected
            (False) or not (True).
        carrying_battery (bool): Wheather the agent is currently carrying a battery.
    """

    def __init__(
        self, agent_pos, train_pos, train_life, battery_present, carrying_battery
    ):
        self.agent_pos = agent_pos
        self.train_pos = train_pos
        self.train_life = train_life
        self.battery_present = battery_present
        self.carrying_battery = carrying_battery

    def is_valid(self):
        pos = self.agent_pos
        # Can't be standing on a battery and not carrying a battery
        if (
            pos in self.battery_present
            and self.battery_present[pos]
            and not self.carrying_battery
        ):
            return False
        return True

    def __eq__(self, other):
        return (
            isinstance(other, BatteriesState)
            and self.agent_pos == other.agent_pos
            and self.train_pos == other.train_pos
            and self.train_life == other.train_life
            and self.battery_present == other.battery_present
            and self.carrying_battery == other.carrying_battery
        )

    def __hash__(self):
        def get_vals(dictionary):
            return tuple([dictionary[loc] for loc in sorted(dictionary.keys())])

        return hash(
            self.agent_pos
            + self.train_pos
            + (self.train_life,)
            + get_vals(self.battery_present)
            + (self.carrying_battery,)
        )


class BatteriesEnv(DeterministicEnv):
    """Room with breakable, moving train and collectible batteries for the train.

    Attributes
        height (int): Height of the grid. Y coordinates are in [0, height).
        width (int): Width of the grid. X coordinates are in [0, width).
        init_state (BatteriesState): Initial state of the environment.
        battery_locations (list): (x, y) tuples describing the locations of batteries.
        num_batteries (int): Number of batteries.
        feature_locations (list): (x, y) tuples describing the locations of the goals.
        train_transition (dict): Dictionary from (x, y) tuples to (x, y) tuples that
            describes the (deterministic) transitions of the train.
        train_locations (list): (x, y) tuples of potential location of the train.
        s (BatteriesState): Current state.
        nA (int): Number of actions.
    """

    def __init__(self, spec):
        self.init_state = deepcopy(spec.init_state)

        self.height = spec.height
        self.width = spec.width
        self.battery_locations = sorted(list(self.init_state.battery_present.keys()))
        self.num_batteries = len(self.battery_locations)
        self.feature_locations = list(spec.feature_locations)
        self.train_transition = spec.train_transition
        self.train_locations = list(self.train_transition.keys())
        assert set(self.train_locations) == set(self.train_transition.values())
        self.nA = 5

        super().__init__(10)

        self.default_action = Direction.get_number_from_direction(Direction.STAY)
        self.num_features = len(self.s_to_f(self.init_state))

        self.reset()

        states = self.enumerate_states()
        self.make_transition_matrices(states, range(self.nA), self.nS, self.nA)
        self.make_f_matrix(self.nS, self.num_features)

    def enumerate_states(self):
        state_num = {}
        all_agent_positions = product(range(self.width), range(self.height))
        all_battery_states = map(
            lambda battery_vals: dict(zip(self.battery_locations, battery_vals)),
            product([True, False], repeat=self.num_batteries),
        )
        all_states = map(
            lambda x: BatteriesState(*x),
            product(
                all_agent_positions,
                self.train_locations,
                range(10),
                all_battery_states,
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
            - Number of batteries
            - Whether the train is still alive
            - For each train location, whether the train is at that location
            - For each feature location, whether the agent is on that location
        """
        num_batteries = list(s.battery_present.values()).count(True)
        train_dead_feature = int(s.train_life == 0)
        train_pos_features = [int(s.train_pos == pos) for pos in self.train_locations]
        loc_features = [int(s.agent_pos == fpos) for fpos in self.feature_locations]
        features = train_pos_features + loc_features
        features = [num_batteries, train_dead_feature] + features
        return np.array(features, dtype=np.float32)

    def _obs_to_f(self, obs):
        """Returns features of the state given its observation.

        Feature vector is given by:
            - Number of batteries
            - Whether the train is still alive
            - For each train location, whether the train is at that location
            - For each feature location, whether the agent is on that location
        """
        agent_pos = np.unravel_index(obs[0].argmax(), obs[0].shape)
        num_batteries = np.sum(obs[3])
        train_dead_feature = int(np.sum(obs[2]) > 0.5)
        if train_dead_feature == 0:
            train_pos = np.unravel_index(obs[1].argmax(), obs[1].shape)
        else:
            train_pos = np.unravel_index(obs[2].argmax(), obs[2].shape)
        train_pos_features = [int(train_pos == pos) for pos in self.train_locations]
        loc_features = [int(agent_pos == fpos) for fpos in self.feature_locations]
        features = train_pos_features + loc_features
        features = [num_batteries, train_dead_feature] + features
        return np.array(features, dtype=np.float32)

    def _s_to_obs(self, s):
        """Returns an array representation of the env to be used as an observation.

        The representation has dimensions (height, width, 6) and consist of:
            - 2d grid with 1 in the agent's position if it does not carry a battery and
                2 if it does, and 0 everywhere else
            - 2d grid with life of each train in the train's position
            - one-hot encoding of all trains with 0 life
            - one-hot encoding of all batteries that have not been collected
            - one-hot encoding of all batteries that have been collected
            - one-hot encoding of the goals
        """
        layers = [
            [s.agent_pos],
            [s.train_pos] if s.train_life > 0 else [],
            [s.train_pos] if s.train_life == 0 else [],
            [pos for pos, present in s.battery_present.items() if present],
            [pos for pos, present in s.battery_present.items() if not present],
            self.feature_locations,
        ]
        obs = get_grid_representation(self.width, self.height, layers)
        if s.carrying_battery:
            obs[:, :, 0] *= 2
        obs[:, :, 1] *= s.train_life
        return np.array(obs, dtype=np.float32)

    def _obs_to_s(self, obs):
        obs = obs.copy()
        # agent_pos
        agent_y, agent_x = np.unravel_index(obs[:, :, 0].argmax(), obs[:, :, 0].shape)
        carrying_battery = obs[agent_y, agent_x, 0] >= 1.5
        # train
        max_val = -np.inf
        for (x, y) in self.train_locations:
            for z in (1, 2):
                val = obs[y, x, z]
                if val > max_val:
                    train_x, train_y, train_state = x, y, z
                    max_val = val
        if train_state == 1:
            train_life = int(round(max_val))
            train_life = max(train_life, 1)
            train_life = min(train_life, 10)
        else:
            train_life = 0
        # batteries
        battery_present = dict()
        for (battery_x, battery_y) in self.battery_locations:
            battery_state = obs[battery_y, battery_x, 3:5].argmax()
            present = battery_state == 0
            if present and (agent_x, agent_y) == (battery_x, battery_y):
                carrying_battery = True  # make state consistent
            battery_present[(battery_x, battery_y)] = present
        state = BatteriesState(
            (agent_x, agent_y),
            (train_x, train_y),
            train_life,
            battery_present,
            carrying_battery,
        )
        return state

    def get_next_state(self, state, action):
        """Returns the next state given a state and an action."""
        action = int(action)
        new_x, new_y = Direction.move_in_direction_number(state.agent_pos, action)
        # New position is still in bounds:
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            new_x, new_y = state.agent_pos
        new_agent_pos = new_x, new_y

        new_train_pos, new_train_life = state.train_pos, state.train_life
        new_battery_present = deepcopy(state.battery_present)
        new_carrying_battery = state.carrying_battery
        if new_agent_pos == state.train_pos and state.carrying_battery:
            new_train_life = 10
            new_carrying_battery = False

        if new_train_life > 0:
            new_train_pos = self.train_transition[state.train_pos]
            new_train_life -= 1

        if (
            new_agent_pos in state.battery_present
            and state.battery_present[new_agent_pos]
            and not state.carrying_battery
        ):
            new_carrying_battery = True
            new_battery_present[new_agent_pos] = False

        result = BatteriesState(
            new_agent_pos,
            new_train_pos,
            new_train_life,
            new_battery_present,
            new_carrying_battery,
        )
        return result

    def s_to_ansi(self, state):
        """Returns a string to render the state."""
        h, w = self.height, self.width
        grid = [[" "] * w for _ in range(h)]
        x, y = state.agent_pos
        grid[y][x] = "A"
        x, y = state.train_pos
        grid[y][x] = "T"
        for (x, y), val in state.battery_present.items():
            if val:
                grid[y][x] = "B"
        return (
            "\n".join(["|".join(row) for row in grid])
            + "carrying_battery: "
            + str(state.carrying_battery)
        )

    def make_reward_heatmaps(self, rewards, out_prefix, train_life_values=[9, 7, 4, 0]):
        all_battery_states = map(
            lambda battery_vals: dict(zip(self.battery_locations, battery_vals)),
            product([True, False], repeat=self.num_batteries),
        )
        for battery_states in all_battery_states:
            for carrying_battery in [False, True]:
                for train_life in train_life_values:
                    for train_pos in self.train_locations:
                        reward_matrix = np.zeros((self.height, self.width))
                        for y in range(self.height):
                            for x in range(self.width):
                                state = BatteriesState(
                                    (x, y),
                                    train_pos,
                                    train_life,
                                    battery_states,
                                    carrying_battery,
                                )
                                if state.is_valid() and state in self.state_num:
                                    state_id = self.get_num_from_state(state)
                                    reward_matrix[y, x] = rewards[state_id]
                        if np.sum(reward_matrix ** 2) > 0:
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
                            batteries_str = "_".join(
                                [
                                    "{}_{}_{}".format(x, y, p)
                                    for (x, y), p in battery_states.items()
                                ]
                            )
                            plt.savefig(
                                out_prefix
                                + "batteries_"
                                + batteries_str
                                + "_carrying_"
                                + str(carrying_battery)
                                + "_train_{}_{}_{}".format(
                                    train_pos[0], train_pos[1], train_life
                                )
                                + ".png"
                            )
                            plt.clf()
