import unittest

from deep_rlsp.envs.gridworlds.train import TrainState, TrainEnv
from deep_rlsp.envs.gridworlds.tests.common import BaseTests, get_directions


class TestTrainSpec(object):
    def __init__(self):
        """Test spec for the Train environment.

        G is a goal location, V is a vase, C is a carpet, A is the agent.
        -------
        |  G C|
        |  TT |
        | VTTG|
        |     |
        |A    |
        -------
        """
        self.height = 5
        self.width = 5
        self.init_state = TrainState((0, 4), {(1, 2): True}, (2, 1), True)
        self.carpet_locations = [(4, 0)]
        self.feature_locations = ([(2, 0), (4, 2)],)
        self.train_transition = {
            (2, 1): (3, 1),
            (3, 1): (3, 2),
            (3, 2): (2, 2),
            (2, 2): (2, 1),
        }


class TestTrainEnv(BaseTests.TestEnv):
    def setUp(self):
        self.env = TrainEnv(TestTrainSpec())
        u, d, l, r, s = get_directions()

        self.trajectories = [
            [
                (u, (TrainState((0, 3), {(1, 2): True}, (3, 1), True), 1.0)),
                (u, (TrainState((0, 2), {(1, 2): True}, (3, 2), True), 1.0)),
                (u, (TrainState((0, 1), {(1, 2): True}, (2, 2), True), 1.0)),
                (r, (TrainState((1, 1), {(1, 2): True}, (2, 1), True), 1.0)),
                (u, (TrainState((1, 0), {(1, 2): True}, (3, 1), True), 1.0)),
                (r, (TrainState((2, 0), {(1, 2): True}, (3, 2), True), 1.0)),
                (s, (TrainState((2, 0), {(1, 2): True}, (2, 2), True), 1.0)),
                (s, (TrainState((2, 0), {(1, 2): True}, (2, 1), True), 1.0)),
            ],
            [
                (u, (TrainState((0, 3), {(1, 2): True}, (3, 1), True), 1.0)),
                (r, (TrainState((1, 3), {(1, 2): True}, (3, 2), True), 1.0)),
                (r, (TrainState((2, 3), {(1, 2): True}, (2, 2), True), 1.0)),
            ],
            [
                (r, (TrainState((1, 4), {(1, 2): True}, (3, 1), True), 1.0)),
                (r, (TrainState((2, 4), {(1, 2): True}, (3, 2), True), 1.0)),
                (r, (TrainState((3, 4), {(1, 2): True}, (2, 2), True), 1.0)),
                (u, (TrainState((3, 3), {(1, 2): True}, (2, 1), True), 1.0)),
                (u, (TrainState((3, 2), {(1, 2): True}, (3, 1), True), 1.0)),
                (s, (TrainState((3, 2), {(1, 2): True}, (3, 2), False), 1.0)),
                (s, (TrainState((3, 2), {(1, 2): True}, (3, 2), False), 1.0)),
                (u, (TrainState((3, 1), {(1, 2): True}, (3, 2), False), 1.0)),
                (l, (TrainState((2, 1), {(1, 2): True}, (3, 2), False), 1.0)),
            ],
        ]


class TestTrainModel(BaseTests.TestTabularTransitionModel):
    def setUp(self):
        self.env = TrainEnv(TestTrainSpec())
        self.model_tests = []
        self.setUpDeterministic()


if __name__ == "__main__":
    unittest.main()
