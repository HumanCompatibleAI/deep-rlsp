import argparse

import gym
from deep_rlsp.solvers import get_sac

# for envs
import deep_rlsp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("policy_out", type=str)
    parser.add_argument("--timesteps", default=int(2e6), type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env_id)

    solver = get_sac(env, learning_starts=1000)
    solver.learn(total_timesteps=args.timesteps)
    solver.save(args.policy_out)


if __name__ == "__main__":
    main()
