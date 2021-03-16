"""
Evaluate an inferred reward function by using it to train a policy in the original env.
"""

import argparse
import datetime

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySac

from deep_rlsp.util.results import Artifact, FileExperimentResults
from deep_rlsp.model import StateVAE
from deep_rlsp.envs.reward_wrapper import LatentSpaceRewardWrapper
from deep_rlsp.util.video import render_mujoco_from_obs
from deep_rlsp.util.helper import get_trajectory, evaluate_policy
from deep_rlsp.model.mujoco_debug_models import MujocoDebugFeatures, PendulumDynamics
from deep_rlsp.solvers import get_sac

# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = config["experiment_folder"].strip("/").split("/")[-1]
        custom_id = "{}_{}".format(timestamp, label)
        return custom_id  # started_event returns the _run._id


ex = Experiment("mujoco-eval")
ex.observers = [SetID(), FileStorageObserver.create("results/mujoco/eval")]


def print_rollout(env, policy, latent_space, decode=False):
    state = env.reset()
    done = False
    while not done:
        a, _ = policy.predict(state, deterministic=False)
        state, reward, done, info = env.step(a)
        if decode:
            obs = latent_space.decoder(state)
        else:
            obs = state
        print("action", a)
        print("obs", obs)
        print("reward", reward)


@ex.config
def config():
    experiment_folder = None  # noqa:F841
    iteration = -1  # noqa:F841


@ex.automain
def main(_run, experiment_folder, iteration, seed):
    ex = FileExperimentResults(experiment_folder)
    env_id = ex.config["env_id"]
    env = gym.make(env_id)

    if env_id == "InvertedPendulum-v2":
        iterations = int(6e4)
    else:
        iterations = int(2e6)

    label = experiment_folder.strip("/").split("/")[-1]

    if ex.config["debug_handcoded_features"]:
        latent_space = MujocoDebugFeatures(env)
    else:
        graph_latent = tf.Graph()
        latent_model_path = ex.info["latent_model_checkpoint"]
        with graph_latent.as_default():
            latent_space = StateVAE.restore(latent_model_path)

    r_inferred = ex.info["inferred_rewards"][iteration]
    r_inferred /= np.linalg.norm(r_inferred)

    print("r_inferred")
    print(r_inferred)
    if env_id.startswith("Fetch"):
        env_has_task_reward = True
        inferred_weight = 0.1
    else:
        env_has_task_reward = False
        inferred_weight = None

    env_inferred = LatentSpaceRewardWrapper(
        env,
        latent_space,
        r_inferred,
        inferred_weight=inferred_weight,
        use_task_reward=env_has_task_reward,
    )

    policy_inferred = get_sac(env_inferred)
    policy_inferred.learn(total_timesteps=iterations, log_interval=10)
    with Artifact(f"policy.zip", None, _run) as f:
        policy_inferred.save(f)

    print_rollout(env_inferred, policy_inferred, latent_space)

    N = 10
    true_reward_obtained = evaluate_policy(env, policy_inferred, N)
    print("Inferred reward policy: true return", true_reward_obtained)
    if env_has_task_reward:
        env.use_penalty = False
        task_reward_obtained = evaluate_policy(env, policy_inferred, N)
        print("Inferred reward policy: task return", task_reward_obtained)
        env.use_penalty = True
    with Artifact(f"video.mp4", None, _run) as f:
        inferred_reward_obtained = evaluate_policy(
            env_inferred, policy_inferred, N, video_out=f
        )
    print("Inferred reward policy: inferred return", inferred_reward_obtained)

    good_policy_path = ex.config["good_policy_path"]
    if good_policy_path is not None:
        true_reward_policy = SAC.load(good_policy_path)
        good_policy_true_reward_obtained = evaluate_policy(env, true_reward_policy, N)
        print("True reward policy: true return", good_policy_true_reward_obtained)
        if env_has_task_reward:
            env.use_penalty = False
            good_policy_task_reward_obtained = evaluate_policy(
                env, true_reward_policy, N
            )
            print("True reward policy: task return", good_policy_task_reward_obtained)
            env.use_penalty = True
        good_policy_inferred_reward_obtained = evaluate_policy(
            env_inferred, true_reward_policy, N
        )
        print(
            "True reward policy: inferred return", good_policy_inferred_reward_obtained
        )

    random_policy = SAC(MlpPolicySac, env_inferred, verbose=1)
    random_policy_true_reward_obtained = evaluate_policy(env, random_policy, N)
    print("Random policy: true return", random_policy_true_reward_obtained)
    if env_has_task_reward:
        env.use_penalty = False
        random_policy_task_reward_obtained = evaluate_policy(env, random_policy, N)
        print("Random reward policy: task return", random_policy_task_reward_obtained)
        env.use_penalty = True
    random_policy_inferred_reward_obtained = evaluate_policy(
        env_inferred, random_policy, N
    )
    print("Random policy: inferred return", random_policy_inferred_reward_obtained)
    print()
