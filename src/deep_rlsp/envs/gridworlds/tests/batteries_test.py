import unittest

from deep_rlsp.envs.gridworlds.batteries import BatteriesState, BatteriesEnv
from deep_rlsp.envs.gridworlds.tests.common import BaseTests, get_directions


class TestBatteriesSpec(object):
    def __init__(self):
        """Test spec for the Batteries environment.

        G is a goal location, B is a battery, A is the agent, and T is the train.
        -------
        |B G  |
        |  TT |
        |  TTG|
        |     |
        |A   B|
        -------
        """
        self.height = 5
        self.width = 5
        self.init_state = BatteriesState(
            (0, 4), (2, 1), 8, {(0, 0): True, (4, 4): True}, False
        )
        self.feature_locations = [(2, 0), (4, 2)]
        self.train_transition = {
            (2, 1): (3, 1),
            (3, 1): (3, 2),
            (3, 2): (2, 2),
            (2, 2): (2, 1),
        }


class TestBatteriesEnv(BaseTests.TestEnv):
    def setUp(self):
        self.env = BatteriesEnv(TestBatteriesSpec())
        u, d, l, r, s = get_directions()

        def make_state(agent, train, life, battery_vals, carrying_battery):
            battery_present = dict(zip([(0, 0), (4, 4)], battery_vals))
            return BatteriesState(agent, train, life, battery_present, carrying_battery)

        self.trajectories = [
            [
                (u, (make_state((0, 3), (3, 1), 7, [True, True], False), 1.0)),
                (u, (make_state((0, 2), (3, 2), 6, [True, True], False), 1.0)),
                (u, (make_state((0, 1), (2, 2), 5, [True, True], False), 1.0)),
                (u, (make_state((0, 0), (2, 1), 4, [False, True], True), 1.0)),
                (r, (make_state((1, 0), (3, 1), 3, [False, True], True), 1.0)),
                (r, (make_state((2, 0), (3, 2), 2, [False, True], True), 1.0)),
                (s, (make_state((2, 0), (2, 2), 1, [False, True], True), 1.0)),
                (s, (make_state((2, 0), (2, 1), 0, [False, True], True), 1.0)),
                (d, (make_state((2, 1), (3, 1), 9, [False, True], False), 1.0)),
                (u, (make_state((2, 0), (3, 2), 8, [False, True], False), 1.0)),
            ]
        ]


@unittest.skip("runs very long")
class TestBatteriesModel(BaseTests.TestTabularTransitionModel):
    def setUp(self):
        self.env = BatteriesEnv(TestBatteriesSpec())
        self.model_tests = []
        self.setUpDeterministic()


if __name__ == "__main__":
    unittest.main()
