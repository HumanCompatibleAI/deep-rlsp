import numpy as np
from copy import deepcopy
from itertools import product

from deep_rlsp.envs.gridworlds.env import Env, Direction, get_grid_representation


def get_orientation_char(orientation):
    direction_to_char = {
        Direction.NORTH: "↑",
        Direction.SOUTH: "↓",
        Direction.WEST: "←",
        Direction.EAST: "→",
        Direction.STAY: "*",
    }
    direction = Direction.get_direction_from_number(orientation)
    return direction_to_char[direction]


class ApplesState:
    """State of the environment that describes the positions of all objects in the env.

    Attributes
        agent_pos (tuple): (orientation, x, y) tuple for the agent's location
        tree_states (dict): Dictionary mapping (x, y) tuples to booleans describing
            which tree has an apple.
        bucket_states (dict): Dictionary mapping (x, y) tuples to integers describing
            how many apples are contained in each bucket.
        carrying_apple (bool): True if agent is carrying an apple.
    """

    def __init__(self, agent_pos, tree_states, bucket_states, carrying_apple):
        self.agent_pos = agent_pos
        self.tree_states = tree_states
        self.bucket_states = bucket_states
        self.carrying_apple = carrying_apple

    def __eq__(self, other):
        return (
            isinstance(other, ApplesState)
            and self.agent_pos == other.agent_pos
            and self.tree_states == other.tree_states
            and self.bucket_states == other.bucket_states
            and self.carrying_apple == other.carrying_apple
        )

    def __hash__(self):
        def get_vals(dictionary):
            return tuple([dictionary[loc] for loc in sorted(dictionary.keys())])

        return hash(
            self.agent_pos
            + get_vals(self.tree_states)
            + get_vals(self.bucket_states)
            + (self.carrying_apple,)
        )


class ApplesEnv(Env):
    """Environment in which the agent is supposed move apples from trees to buckets.

    Attributes
        height (int): Height of the grid. Y coordinates are in [0, height).
        width (int): Width of the grid. X coordinates are in [0, width).
        init_state (ApplesState): Initial state of the environment.
        apple_regen_probability (float): Probability that for a given tree that does not
            have an apple, one will grow from one timestep to the next.
        bucket_capacity (int): Max. number of apples each bucket can contain.
        include_location_features (bool): If True, the feature representation will
            have one boolean feature for each potential position of the agent.
        tree_locations (list): (x, y) tuples describing the locations of trees.
        num_trees (int): Number of trees.
        bucket_locations (list): (x, y) tuples describing the locations of buckets.
        num_buckets (int): Number of buckets.
        possible_agent_locations (list): (x, y) tuples describing locations the agent
            can be in, i.e. that are not occupied by other objects.
        s (ApplesState): Current state.
        nA (int): Number of actions.
    """

    def __init__(self, spec):
        self.height = spec.height
        self.width = spec.width
        self.init_state = deepcopy(spec.init_state)
        self.apple_regen_probability = spec.apple_regen_probability
        self.bucket_capacity = spec.bucket_capacity
        self.include_location_features = spec.include_location_features

        self.tree_locations = list(self.init_state.tree_states.keys())
        self.num_trees = len(self.tree_locations)
        self.bucket_locations = list(self.init_state.bucket_states.keys())
        self.num_buckets = len(self.bucket_locations)
        used_locations = set(self.tree_locations + self.bucket_locations)
        self.possible_agent_locations = list(
            filter(
                lambda pos: pos not in used_locations,
                product(range(self.width), range(self.height)),
            )
        )
        self.nA = 6

        super().__init__(max(5, self.bucket_capacity))

        self.default_action = Direction.get_number_from_direction(Direction.STAY)
        self.num_features = len(self.s_to_f(self.init_state))

        self.reset()

        states = self.enumerate_states()
        self.make_transition_matrices(states, range(self.nA), self.nS, self.nA)
        self.make_f_matrix(self.nS, self.num_features)

    def enumerate_states(self):
        all_agent_positions = filter(
            lambda pos: (pos[1], pos[2]) in self.possible_agent_locations,
            product(range(4), range(self.width), range(self.height)),
        )
        all_tree_states = map(
            lambda tree_vals: dict(zip(self.tree_locations, tree_vals)),
            product([True, False], repeat=self.num_trees),
        )
        all_bucket_states = map(
            lambda bucket_vals: dict(zip(self.bucket_locations, bucket_vals)),
            product(range(self.bucket_capacity + 1), repeat=self.num_buckets),
        )
        all_states = map(
            lambda x: ApplesState(*x),
            product(
                all_agent_positions, all_tree_states, all_bucket_states, [True, False]
            ),
        )

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
            - Number of apples in buckets
            - Number of apples on trees
            - Whether the agent is carrying an apple
            - For each other location, whether the agent is on that location
        """
        num_bucket_apples = sum(s.bucket_states.values())
        num_tree_apples = sum(map(int, s.tree_states.values()))
        carrying_apple = int(s.carrying_apple)
        agent_pos = s.agent_pos[1], s.agent_pos[2]  # Drop orientation
        features = [num_bucket_apples, num_tree_apples, carrying_apple]
        if self.include_location_features:
            features = features + [
                int(agent_pos == pos) for pos in self.possible_agent_locations
            ]
        return np.array(features, dtype=np.float32)

    def _obs_to_f(self, obs):
        """Returns features of the state given its observation.

        Feature vector is given by:
            - Number of apples in buckets
            - Number of apples on trees
            - Whether the agent is carrying an apple
            - For each other location, whether the agent is on that location
        """
        num_bucket_apples = np.sum(obs[5])
        num_tree_apples = np.sum(obs[2])
        carrying_apple = np.sum(obs[1])
        agent_pos = np.unravel_index(obs[0].argmax(), obs[0].shape)
        features = [num_bucket_apples, num_tree_apples, carrying_apple]
        if self.include_location_features:
            features = features + [
                int(agent_pos == pos) for pos in self.possible_agent_locations
            ]
        return np.array(features, dtype=np.float32)

    def _s_to_obs(self, s):
        """Returns an array representation of the env to be used as an observation.

        The representation has dimensions (height, width, 6) and consist of:
            - 2d grid with the agents orientation at the agent's position and 0 else
            - 2d grid with 1 in the agent's position iff it is carrying an apple, and
                0 everywhere else
            - one-hot encoding of all trees that have apples
            - one-hot encoding of all trees that do not have apples
            - one-hot encoding of all buckets
            - 2d grid with the number of apples for each bucket in the bucket's location
                and 0 everywhere else
        """
        orientation, agent_x, agent_y = s.agent_pos
        layers = [
            [(agent_x, agent_y)],
            [(agent_x, agent_y)] if s.carrying_apple else [],
            [pos for pos, has_apple in s.tree_states.items() if has_apple],
            [pos for pos, has_apple in s.tree_states.items() if not has_apple],
            self.bucket_locations,
            self.bucket_locations,
        ]
        obs = get_grid_representation(self.width, self.height, layers)

        obs[:, :, 0] *= orientation + 1
        for (bucket_x, bucket_y), num_apples in s.bucket_states.items():
            obs[bucket_y, bucket_x, 5] = num_apples
        return np.array(obs, dtype=np.float32)

    def _obs_to_s(self, obs):
        obs = obs.copy()
        # agent_pos
        max_val = -np.inf
        for (x, y) in self.possible_agent_locations:
            val = obs[y, x, 0]
            if val > max_val:
                agent_x, agent_y = x, y
                max_val = val
        orientation = int(round(max_val))
        orientation = max(orientation, 1)
        orientation = min(orientation, 5)
        orientation -= 1
        # carrying apple
        carrying_apple = obs[agent_y, agent_x, 1] > 0.5
        # trees
        tree_states = dict()
        for (tree_x, tree_y) in self.tree_locations:
            tree_state = obs[tree_y, tree_x, 2:4].argmax()
            has_apple = tree_state == 0
            tree_states[(tree_x, tree_y)] = has_apple
        # buckets
        bucket_states = dict()
        for (bucket_x, bucket_y) in self.bucket_locations:
            num_apples = round(obs[bucket_y, bucket_x, 5])
            num_apples = max(num_apples, 0)
            num_apples = min(num_apples, self.bucket_capacity)
            bucket_states[(bucket_x, bucket_y)] = num_apples
        state = ApplesState(
            (orientation, agent_x, agent_y), tree_states, bucket_states, carrying_apple
        )
        return state

    def get_next_states(self, state, action):
        """Returns the next state given a state and an action."""
        action = int(action)
        orientation, x, y = state.agent_pos
        new_orientation, new_x, new_y = state.agent_pos
        new_tree_states = deepcopy(state.tree_states)
        new_bucket_states = deepcopy(state.bucket_states)
        new_carrying_apple = state.carrying_apple

        if action == Direction.get_number_from_direction(Direction.STAY):
            pass
        elif action < len(Direction.ALL_DIRECTIONS):
            new_orientation = action
            move_x, move_y = Direction.move_in_direction_number((x, y), action)
            # New position is legal
            if (
                0 <= move_x < self.width
                and 0 <= move_y < self.height
                and (move_x, move_y) in self.possible_agent_locations
            ):
                new_x, new_y = move_x, move_y
            else:
                # Move only changes orientation, which we already handled
                pass
        elif action == 5:
            obj_pos = Direction.move_in_direction_number((x, y), orientation)
            if state.carrying_apple:
                # We always drop the apple
                new_carrying_apple = False
                # If we're facing a bucket, it goes there
                if obj_pos in new_bucket_states:
                    prev_apples = new_bucket_states[obj_pos]
                    new_bucket_states[obj_pos] = min(
                        prev_apples + 1, self.bucket_capacity
                    )
            elif obj_pos in new_tree_states and new_tree_states[obj_pos]:
                new_carrying_apple = True
                new_tree_states[obj_pos] = False
            else:
                # Interact while holding nothing and not facing a tree.
                pass
        else:
            raise ValueError("Invalid action {}".format(action))

        new_pos = new_orientation, new_x, new_y

        def make_state(prob_apples_tuple):
            prob, tree_apples = prob_apples_tuple
            trees = dict(zip(self.tree_locations, tree_apples))
            s = ApplesState(new_pos, trees, new_bucket_states, new_carrying_apple)
            return (prob, s, 0)

        # For apple regeneration, don't regenerate apples that were just picked,
        # so use the apple booleans from the original state
        old_tree_apples = [state.tree_states[loc] for loc in self.tree_locations]
        new_tree_apples = [new_tree_states[loc] for loc in self.tree_locations]
        return list(
            map(make_state, self.regen_apples(old_tree_apples, new_tree_apples))
        )

    def regen_apples(self, old_tree_apples, new_tree_apples):
        if len(old_tree_apples) == 0:
            yield (1, [])
            return
        for prob, apples in self.regen_apples(old_tree_apples[1:], new_tree_apples[1:]):
            if old_tree_apples[0]:
                yield prob, [new_tree_apples[0]] + apples
            else:
                yield prob * self.apple_regen_probability, [True] + apples
                yield prob * (1 - self.apple_regen_probability), [False] + apples

    def s_to_ansi(self, state):
        """Returns a string to render the state."""
        h, w = self.height, self.width
        canvas = np.zeros(tuple([2 * h - 1, 2 * w + 1]), dtype="int8")

        # cell borders
        for y in range(1, canvas.shape[0], 2):
            canvas[y, :] = 1
        for x in range(0, canvas.shape[1], 2):
            canvas[:, x] = 2

        # trees
        for (x, y), has_apple in state.tree_states.items():
            canvas[2 * y, 2 * x + 1] = 3 if has_apple else 4

        for x, y in self.bucket_locations:
            canvas[2 * y, 2 * x + 1] = 5

        # agent
        orientation, x, y = state.agent_pos
        canvas[2 * y, 2 * x + 1] = 6

        black_color = "\x1b[0m"
        # purple_background_color = "\x1b[0;35;85m"

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
                    chars.append("\x1b[0;32;85m█" + black_color)
                elif char_num == 4:
                    chars.append("\033[91m█" + black_color)
                elif char_num == 5:
                    chars.append("\033[93m█" + black_color)
                elif char_num == 6:
                    orientation_char = get_orientation_char(orientation)
                    agent_color = "\x1b[1;42;42m" if state.carrying_apple else "\x1b[0m"
                    chars.append(agent_color + orientation_char + black_color)
            lines.append("".join(chars))
        return "\n".join(lines)
