import unittest

import numpy as np

from deep_rlsp.envs.gridworlds.apples import ApplesState, ApplesEnv
from deep_rlsp.envs.gridworlds.tests.common import BaseTests, get_directions


class TestApplesSpec(object):
    def __init__(self):
        """Test spec for the Apples environment.

        T is a tree, B is a bucket, C is a carpet, A is the agent.
        -----
        |T T|
        |   |
        |AB |
        -----
        """
        self.height = 3
        self.width = 5
        self.init_state = ApplesState(
            agent_pos=(0, 0, 2),
            tree_states={(0, 0): True, (2, 0): True},
            bucket_states={(1, 2): 0},
            carrying_apple=False,
        )
        # Use a power of 2, to avoid rounding issues
        self.apple_regen_probability = 1.0 / 4
        self.bucket_capacity = 10
        self.include_location_features = True


class TestApplesEnv(BaseTests.TestEnv):
    def setUp(self):
        self.env = ApplesEnv(TestApplesSpec())

        u, d, l, r, s = get_directions()
        i = 5  # interact action

        def make_state(agent_pos, tree1, tree2, bucket, carrying_apple):
            tree_states = {(0, 0): tree1, (2, 0): tree2}
            bucket_state = {(1, 2): bucket}
            return ApplesState(agent_pos, tree_states, bucket_state, carrying_apple)

        self.trajectories = [
            [
                (u, (make_state((u, 0, 1), True, True, 0, False), 1.0)),
                (i, (make_state((u, 0, 1), False, True, 0, True), 1.0)),
                (r, (make_state((r, 1, 1), False, True, 0, True), 3.0 / 4)),
                (d, (make_state((d, 1, 1), False, True, 0, True), 3.0 / 4)),
                (i, (make_state((d, 1, 1), False, True, 1, False), 3.0 / 4)),
                (u, (make_state((u, 1, 0), False, True, 1, False), 3.0 / 4)),
                (r, (make_state((r, 1, 0), False, True, 1, False), 3.0 / 4)),
                (i, (make_state((r, 1, 0), False, False, 1, True), 3.0 / 4)),
                (d, (make_state((d, 1, 1), False, False, 1, True), 9.0 / 16)),
                (i, (make_state((d, 1, 1), True, False, 2, False), 3.0 / 16)),
                (s, (make_state((d, 1, 1), True, True, 2, False), 1.0 / 4)),
            ]
        ]


class TestApplesModel(BaseTests.TestTabularTransitionModel):
    def setUp(self):
        self.env = ApplesEnv(TestApplesSpec())
        self.model_tests = []

        _, _, _, _, stay = get_directions()
        policy_stay = np.zeros((self.env.nS, self.env.nA))
        policy_stay[:, stay] = 1

        def make_state(apple1_present, apple2_present):
            return ApplesState(
                agent_pos=(0, 1, 1),
                tree_states={(0, 0): apple1_present, (2, 0): apple2_present},
                bucket_states={(1, 2): 0},
                carrying_apple=False,
            )

        state_0_0 = make_state(False, False)
        state_0_1 = make_state(False, True)
        state_1_0 = make_state(True, False)
        state_1_1 = make_state(True, True)

        forward_probs = np.zeros(self.env.nS)
        forward_probs[self.env.get_num_from_state(state_1_1)] = 1
        backward_probs = np.zeros(self.env.nS)
        backward_probs[self.env.get_num_from_state(state_0_0)] = 0.04
        backward_probs[self.env.get_num_from_state(state_0_1)] = 0.16
        backward_probs[self.env.get_num_from_state(state_1_0)] = 0.16
        backward_probs[self.env.get_num_from_state(state_1_1)] = 0.64
        transitions = [(state_1_1, 1, forward_probs, backward_probs)]

        unif = np.ones(self.env.nS) / self.env.nS
        self.model_tests.append(
            {
                "policy": policy_stay,
                "transitions": transitions,
                "initial_state_distribution": unif,
            }
        )


if __name__ == "__main__":
    unittest.main()
