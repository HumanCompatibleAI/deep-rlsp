import argparse
import datetime

import gym
import numpy as np

from deep_rlsp.model import StateVAE, ExperienceReplay


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("--n_rollouts", default=100, type=int)
    parser.add_argument("--state_size", default=30, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--layer_size", default=512, type=int)
    parser.add_argument("--prior_stdev", default=1, type=float)
    parser.add_argument("--divergence_factor", default=0.001, type=float)
    return parser.parse_args()


def main():
    args = parse_args()

    env = gym.make(args.env_id)
    experience_replay = ExperienceReplay(None)
    experience_replay.add_random_rollouts(
        env, env.spec.max_episode_steps, args.n_rollouts
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "vae_{}_{}".format(args.env_id, timestamp)
    tensorboard_log = "tf_logs/tf_logs_" + label
    checkpoint_folder = "tf_ckpt/tf_ckpt_" + label

    vae = StateVAE(
        env.observation_space.shape[0],
        args.state_size,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        learning_rate=args.learning_rate,
        prior_stdev=args.prior_stdev,
        divergence_factor=args.divergence_factor,
        tensorboard_log=tensorboard_log,
        checkpoint_folder=checkpoint_folder,
    )
    # vae.checkpoint_folder = None

    vae.learn(experience_replay, args.n_epochs, args.batch_size, verbose=True)

    with np.printoptions(suppress=True):
        for _ in range(20):
            x = experience_replay.sample(1)[0][0]
            print("x", x)
            z = vae.encoder(x)
            print("z", z)
            x = vae.decoder(z)
            print("x2", x)


if __name__ == "__main__":
    main()
