import datetime

import gym
import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from deep_rlsp.model import StateVAE
from deep_rlsp.util.results import Artifact, FileExperimentResults
from deep_rlsp.solvers import get_sac
from deep_rlsp.util.helper import evaluate_policy


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if config["result_folder"] is not None:
            result_folder = config["result_folder"].strip("/").split("/")[-1]
            custom_id = f"{timestamp}_ablation_waypoints_{result_folder}"
        else:
            custom_id = f"{timestamp}_ablation_waypoints"
        return custom_id  # started_event returns the _run._id


ex = Experiment("mujoco-ablation-waypoints")
ex.observers = [
    SetID(),
    FileStorageObserver.create("results/mujoco/ablation_waypoints"),
]


class LatentSpaceTargetStateRewardWrapper(gym.Wrapper):
    def __init__(self, env, latent_space, target_states):
        self.env = env
        self.latent_space = latent_space
        self.target_states = [ts / np.linalg.norm(ts) for ts in target_states]
        self.state = None
        self.timestep = 0
        super().__init__(env)

    def reset(self):
        obs = super().reset()
        self.state = self.latent_space.encoder(obs)
        self.timestep = 0
        return obs

    def step(self, action: int):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, true_reward, done, info = self.env.step(action)
        self.state = self.latent_space.encoder(obs)

        waypoints = [np.dot(ts, self.state) for ts in self.target_states]
        reward = max(waypoints)

        # remove termination criteria from mujoco environments
        self.timestep += 1
        done = self.timestep > self.env.spec.max_episode_steps

        return obs, reward, done, info


@ex.config
def config():
    result_folder = None  # noqa:F841


@ex.automain
def main(_run, result_folder, seed):
    # result_folder = "results/mujoco/20200706_153755_HalfCheetah-FW-v2_optimal_50"
    ex = FileExperimentResults(result_folder)
    env_id = ex.config["env_id"]
    latent_model_checkpoint = ex.info["latent_model_checkpoint"]
    current_states = ex.info["current_states"]

    if env_id == "InvertedPendulum-v2":
        iterations = int(6e4)
    else:
        iterations = int(2e6)

    env = gym.make(env_id)

    latent_space = StateVAE.restore(latent_model_checkpoint)

    target_states = [latent_space.encoder(obs) for obs in current_states]
    env_inferred = LatentSpaceTargetStateRewardWrapper(env, latent_space, target_states)

    solver = get_sac(env_inferred)
    solver.learn(iterations)

    with Artifact(f"policy.zip", None, _run) as f:
        solver.save(f)

    N = 10
    true_reward_obtained = evaluate_policy(env, solver, N)
    inferred_reward_obtained = evaluate_policy(env_inferred, solver, N)
    print("Policy: true return", true_reward_obtained)
    print("Policy: inferred return", inferred_reward_obtained)
