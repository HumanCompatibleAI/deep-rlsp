import argparse
import datetime

import gym
import numpy as np

from deep_rlsp.model import ExperienceReplay, InverseDynamicsMLP, InverseDynamicsMDN
from deep_rlsp.util.video import render_mujoco_from_obs, save_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("--n_rollouts", default=1000, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--n_layers", default=5, type=int)
    parser.add_argument("--layer_size", default=1024, type=int)
    parser.add_argument("--gridworlds", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    env = gym.make(args.env_id)
    experience_replay = ExperienceReplay(None)
    experience_replay.add_random_rollouts(
        env, env.spec.max_episode_steps, args.n_rollouts
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "mlp_{}_{}".format(args.env_id, timestamp)
    tensorboard_log = "tf_logs/tf_logs_" + label
    checkpoint_folder = "tf_ckpt/tf_ckpt_" + label

    if args.gridworlds:
        inverse_dynamics = InverseDynamicsMDN(
            env,
            experience_replay,
            hidden_layer_size=args.layer_size,
            n_hidden_layers=args.n_layers,
            learning_rate=args.learning_rate,
            tensorboard_log=tensorboard_log,
            checkpoint_folder=checkpoint_folder,
            gauss_stdev=0.1,
            n_out_states=3,
        )
    else:
        inverse_dynamics = InverseDynamicsMLP(
            env,
            experience_replay,
            hidden_layer_size=args.layer_size,
            n_hidden_layers=args.n_layers,
            learning_rate=args.learning_rate,
            tensorboard_log=tensorboard_log,
            checkpoint_folder=checkpoint_folder,
        )

    inverse_dynamics.learn(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        print_evaluation=False,
        verbose=True,
    )

    # Evaluation

    if args.gridworlds:
        obs = env.reset()
        for _ in range(env.time_horizon):
            s = env.obs_to_s(obs)

            print("agent_pos", s.agent_pos)
            print("vase_states", s.vase_states)
            print(np.reshape(obs, env.obs_shape)[:, :, 0])

            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            obs_bwd = inverse_dynamics.step(obs, action)

            s = env.obs_to_s(obs_bwd)
            print("bwd: agent_pos", s.agent_pos)
            print("bwd: vase_states", s.vase_states)
            print(np.reshape(obs_bwd, env.obs_shape)[:, :, 0])

            print()
            print("action", action)
            print()
    else:
        obs = env.reset()
        for _ in range(env.spec.max_episode_steps):
            print("obs", obs)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            obs_bwd = inverse_dynamics.step(obs, action)
            with np.printoptions(suppress=True):
                print("action", action)
                print("\tbwd", obs_bwd)

        obs = env.reset()
        rgbs = []
        for _ in range(env.spec.max_episode_steps):
            obs = np.clip(obs, -30, 30)
            rgb = render_mujoco_from_obs(env, obs)
            rgbs.append(rgb)
            action = env.action_space.sample()
            obs = inverse_dynamics.step(obs, action, sample=True)
        save_video(rgbs, f"train_{args.env_id}.avi", 20)


if __name__ == "__main__":
    main()
