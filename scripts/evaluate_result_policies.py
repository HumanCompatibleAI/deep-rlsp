import os
import argparse

import numpy as np
import gym

from stable_baselines import PPO2, SAC

from deep_rlsp.util.video import render_mujoco_from_obs, save_video
from deep_rlsp.model import StateVAE
from deep_rlsp.util.results import FileExperimentResults


def evaluate_policy(policy_file, policy_type, envname, num_rollouts):
    if policy_type == "ppo":
        model = PPO2.load(policy_file)

        def get_action(obs):
            return model.predict(obs)[0]

    elif policy_type == "sac":
        model = SAC.load(policy_file)

        def get_action(obs):
            return model.predict(obs, deterministic=True)[0]

    else:
        raise NotImplementedError()

    env = gym.make(envname)

    returns = []
    for i in range(num_rollouts):
        # print("iter", i, end=" ")
        obs = env.reset()
        done = False
        totalr = 0.0
        while not done:
            action = get_action(obs)
            obs, r, done, _ = env.step(action)
            totalr += r
        returns.append(totalr)

    return np.mean(returns), np.std(returns)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=str)
    parser.add_argument("envname", type=str)
    parser.add_argument("--policy_type", type=str, default="sac")
    parser.add_argument(
        "--num_rollouts", type=int, default=1, help="Number of expert rollouts"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base, root_dirs, _ = next(os.walk(args.results_folder))
    root_dirs = [os.path.join(base, dir) for dir in root_dirs]
    for root in root_dirs:
        print(root)
        _, dirs, files = next(os.walk(root))
        if "policy.zip" in files:
            policy_path = os.path.join(root, "policy.zip")
        elif any([f.startswith("rlsp_policy") for f in files]):
            policy_files = [f for f in files if f.startswith("rlsp_policy")]
            policy_numbers = [int(f.split(".")[0].split("_")[2]) for f in policy_files]
            # take second to last policy, in case a run crashed while writing
            # the last policy
            policy_file = f"rlsp_policy_{max(policy_numbers)-1}.zip"
            policy_path = policy_path = os.path.join(root, policy_file)
        else:
            policy_path = None

        if policy_path is not None:
            mean, std = evaluate_policy(
                policy_path, args.policy_type, args.envname, args.num_rollouts
            )
            print(policy_path)
            print(f"mean {mean}   std {std}")


if __name__ == "__main__":
    main()
