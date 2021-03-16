import argparse
import time

import numpy as np
import gym

from stable_baselines import PPO2, SAC

from deep_rlsp.util.video import render_mujoco_from_obs, save_video
from deep_rlsp.model.mujoco_debug_models import MujocoDebugFeatures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_file", type=str)
    parser.add_argument("policy_type", type=str)
    parser.add_argument("envname", type=str)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out_video", type=str, default=None)
    parser.add_argument(
        "--num_rollouts", type=int, default=1, help="Number of expert rollouts"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.envname)

    print("loading and building expert policy")

    if args.policy_type == "ppo":
        model = PPO2.load(args.policy_file)

        def get_action(obs):
            return model.predict(obs)[0]

    elif args.policy_type == "sac":
        model = SAC.load(args.policy_file)

        def get_action(obs):
            return model.predict(obs, deterministic=True)[0]

    elif args.policy_type == "gail":
        from imitation.policies import serialize
        from stable_baselines.common.vec_env import DummyVecEnv

        venv = DummyVecEnv([lambda: env])
        loading_context = serialize.load_policy("ppo2", args.policy_file, venv)
        model = loading_context.__enter__()

        def get_action(obs):
            return model.step(np.reshape(obs, (1, -1)))[0]

    else:
        raise NotImplementedError()

    # # env.unwrapped.viewer.cam.trackbodyid = 0
    # env.unwrapped.viewer.cam.fixedcamid = 0

    returns = []
    observations = []
    render_params = {}
    # render_params = {"width": 4000, "height":1000, "camera_id": -1}
    timesteps = 1000

    for i in range(args.num_rollouts):
        print("iter", i)
        if args.out_video is not None:
            rgbs = []
        obs = env.reset()
        done = False
        totalr = 0.0
        steps = 0
        for _ in range(timesteps):
            # print(obs)
            action = get_action(obs)
            action = env.action_space.sample()
            observations.append(obs)

            last_done = done
            obs, r, done, info = env.step(action)

            if not last_done:
                totalr += r
            steps += 1
            if args.render:
                env.render(mode="human", **render_params)
            if args.out_video is not None:
                rgb = env.render(mode="rgb_array", **render_params)
                rgbs.append(rgb)
            if steps % 100 == 0:
                print("%i/%i" % (steps, env.spec.max_episode_steps))
            if steps >= env.spec.max_episode_steps:
                break
        print("return", totalr)
        returns.append(totalr)

        if args.out_video is not None:
            save_video(rgbs, args.out_video, fps=20.0)

    print("returns", returns)
    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))


if __name__ == "__main__":
    main()
