"""
Simple smoke tests that run some basic experiments and make sure they don't crash
"""


import unittest

from deep_rlsp.run import ex

ex.observers = []

ALL_ALGORITHMS_ON_ROOM = [
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "spec",
        "horizon": 7,
        "evaluation_horizon": 20,
        "epochs": 5,
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 7,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
        "epochs": 5,
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "reachability",
        "horizon": 7,
        "evaluation_horizon": 20,
        "epochs": 5,
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 7,
        "evaluation_horizon": 20,
        "solver": "value_iter",
        "epochs": 5,
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 7,
        "evaluation_horizon": 20,
        "solver": "ppo",
        "solver_iterations": 100,
        "reset_solver": False,
        "epochs": 5,
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 7,
        "evaluation_horizon": 20,
        "solver": "ppo",
        "solver_iterations": 100,
        "reset_solver": True,
        "epochs": 5,
    },
]

DEVIATION_ON_ALL_ENVS = [
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 7,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
        "epochs": 5,
    },
    {
        "env_name": "train",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 8,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
        "epochs": 5,
    },
    {
        "env_name": "apples",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 11,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
        "epochs": 5,
    },
    {
        "env_name": "batteries",
        "problem_spec": "easy",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 11,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
        "epochs": 5,
    },
    {
        "env_name": "batteries",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 11,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
        "epochs": 5,
    },
    {
        "env_name": "room",
        "problem_spec": "bad",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 5,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
        "epochs": 5,
    },
]


class TestToyExperiments(unittest.TestCase):
    def test_toy_experiments(self):
        """
        Runs experiments sequentially
        """
        for config_updates in ALL_ALGORITHMS_ON_ROOM + DEVIATION_ON_ALL_ENVS:
            ex.run(config_updates=config_updates)
