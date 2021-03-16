import unittest

import numpy as np

from deep_rlsp.envs.gridworlds.room import RoomState, RoomEnv
from deep_rlsp.envs.gridworlds.tests.common import (
    BaseTests,
    get_directions,
    get_two_action_uniform_policy,
)


class TestRoomSpec(object):
    def __init__(self):
        """Test spec for the Room environment.

        G is a goal location, V is a vase, C is a carpet, A is the agent.
        -------
        |G G G|
        | CVC |
        |  A  |
        -------
        """
        self.height = 3
        self.width = 5
        self.init_state = RoomState((2, 2), {(2, 1): True})
        self.carpet_locations = [(1, 1), (3, 1)]
        self.feature_locations = [(0, 0), (2, 0), (4, 0)]
        self.use_pixels_as_observations = False


class TestRoomEnv(BaseTests.TestEnv):
    def setUp(self):
        self.env = RoomEnv(TestRoomSpec())
        u, d, l, r, s = get_directions()

        self.trajectories = [
            [
                (l, (RoomState((1, 2), {(2, 1): True}), 1.0)),
                (u, (RoomState((1, 1), {(2, 1): True}), 1.0)),
                (u, (RoomState((1, 0), {(2, 1): True}), 1.0)),
                (r, (RoomState((2, 0), {(2, 1): True}), 1.0)),
            ],
            [
                (u, (RoomState((2, 1), {(2, 1): False}), 1.0)),
                (u, (RoomState((2, 0), {(2, 1): False}), 1.0)),
            ],
            [
                (r, (RoomState((3, 2), {(2, 1): True}), 1.0)),
                (u, (RoomState((3, 1), {(2, 1): True}), 1.0)),
                (l, (RoomState((2, 1), {(2, 1): False}), 1.0)),
                (d, (RoomState((2, 2), {(2, 1): False}), 1.0)),
            ],
        ]


class TestRoomModel(BaseTests.TestTabularTransitionModel):
    def setUp(self):
        self.env = RoomEnv(TestRoomSpec())
        self.model_tests = []

        u, d, l, r, s = get_directions()
        policy_left_right = get_two_action_uniform_policy(
            self.env.nS, self.env.nA, l, r
        )
        state_middle = RoomState((2, 1), {(2, 1): False})
        state_left_vase = RoomState((1, 1), {(2, 1): True})
        state_right_vase = RoomState((3, 1), {(2, 1): True})
        state_left_novase = RoomState((1, 1), {(2, 1): False})
        state_right_novase = RoomState((3, 1), {(2, 1): False})
        forward_probs = np.zeros(self.env.nS)
        forward_probs[self.env.get_num_from_state(state_left_novase)] = 0.5
        forward_probs[self.env.get_num_from_state(state_right_novase)] = 0.5
        backward_probs = np.zeros(self.env.nS)
        backward_probs[self.env.get_num_from_state(state_left_vase)] = 0.25
        backward_probs[self.env.get_num_from_state(state_right_vase)] = 0.25
        backward_probs[self.env.get_num_from_state(state_left_novase)] = 0.25
        backward_probs[self.env.get_num_from_state(state_right_novase)] = 0.25
        transitions = [(state_middle, 1, forward_probs, backward_probs)]
        unif = np.ones(self.env.nS) / self.env.nS
        self.model_tests.append(
            {
                "policy": policy_left_right,
                "transitions": transitions,
                "initial_state_distribution": unif,
            }
        )

        self.setUpDeterministic()


if __name__ == "__main__":
    unittest.main()
