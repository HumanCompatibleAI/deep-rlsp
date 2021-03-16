"""
This script runs experiments on the toy environments from the original RLSP paper.
"""

from deep_rlsp.run_parallel import parallel_exp

ROOM_DEFAULT = [
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "spec",
        "horizon": 7,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 7,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "reachability",
        "horizon": 7,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 7,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 7,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 7,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
    {
        "env_name": "room",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 7,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
]

# Train:
TRAIN_DEFAULT = [
    {
        "env_name": "train",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "spec",
        "horizon": 8,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "train",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 8,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
    },
    {
        "env_name": "train",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "reachability",
        "horizon": 8,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "train",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 8,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "train",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 8,
        "evaluation_horizon": 20,
        "solver": "value_iter",
        "n_trajectories": 10,
    },
    {
        "env_name": "train",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 8,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
    {
        "env_name": "train",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 8,
        "evaluation_horizon": 20,
        "solver": "ppo",
        "n_trajectories": 10,
    },
]

# Apples:
APPLES_DEFAULT = [
    {
        "env_name": "apples",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "spec",
        "horizon": 11,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "apples",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 11,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
    },
    {
        "env_name": "apples",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "reachability",
        "horizon": 11,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "apples",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "apples",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "apples",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
    {
        "env_name": "apples",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
]

# Batteries, easy:
BATTERIES_EASY = [
    {
        "env_name": "batteries",
        "problem_spec": "easy",
        "combination_algorithm": "additive",
        "inference_algorithm": "spec",
        "horizon": 11,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "batteries",
        "problem_spec": "easy",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 11,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
    },
    {
        "env_name": "batteries",
        "problem_spec": "easy",
        "combination_algorithm": "additive",
        "inference_algorithm": "reachability",
        "horizon": 11,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "batteries",
        "problem_spec": "easy",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "batteries",
        "problem_spec": "easy",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "batteries",
        "problem_spec": "easy",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
    {
        "env_name": "batteries",
        "problem_spec": "easy",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
]

# Batteries, hard:
BATTERIES_HARD = [
    {
        "env_name": "batteries",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "spec",
        "horizon": 11,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "batteries",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 11,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
    },
    {
        "env_name": "batteries",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "reachability",
        "horizon": 11,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "batteries",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "batteries",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "batteries",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
    {
        "env_name": "batteries",
        "problem_spec": "default",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 11,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
]

# Far away vase:
ROOM_BAD = [
    {
        "env_name": "room",
        "problem_spec": "bad",
        "combination_algorithm": "additive",
        "inference_algorithm": "spec",
        "horizon": 5,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "room",
        "problem_spec": "bad",
        "combination_algorithm": "additive",
        "inference_algorithm": "deviation",
        "horizon": 5,
        "evaluation_horizon": 20,
        "inferred_weight": 0.5,
    },
    {
        "env_name": "room",
        "problem_spec": "bad",
        "combination_algorithm": "additive",
        "inference_algorithm": "reachability",
        "horizon": 5,
        "evaluation_horizon": 20,
    },
    {
        "env_name": "room",
        "problem_spec": "bad",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 5,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "room",
        "problem_spec": "bad",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 5,
        "evaluation_horizon": 20,
        "solver": "value_iter",
    },
    {
        "env_name": "room",
        "problem_spec": "bad",
        "combination_algorithm": "additive",
        "inference_algorithm": "rlsp",
        "horizon": 5,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
    {
        "env_name": "room",
        "problem_spec": "bad",
        "combination_algorithm": "additive",
        "inference_algorithm": "approx_rlsp",
        "horizon": 5,
        "evaluation_horizon": 20,
        "solver": "ppo",
    },
]


def main():
    experiments = (
        ROOM_DEFAULT
        + TRAIN_DEFAULT
        + APPLES_DEFAULT
        + BATTERIES_EASY
        + BATTERIES_HARD
        + ROOM_BAD
    )

    parallel_exp.run(
        config_updates={
            "config_permutations": [
                {"config_updates": config_updates} for config_updates in experiments
            ],
            # "num_cpus": 2,
            # "num_gpus": 0,
        }
    )


if __name__ == "__main__":
    main()
