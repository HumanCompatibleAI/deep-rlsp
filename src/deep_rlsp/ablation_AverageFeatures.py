import datetime

import gym
import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from deep_rlsp.envs.reward_wrapper import LatentSpaceRewardWrapper
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
            custom_id = f"{timestamp}_ablation_average_features_{result_folder}"
        else:
            custom_id = f"{timestamp}_ablation_average_features"
        return custom_id  # started_event returns the _run._id


ex = Experiment("mujoco-ablation-average-features")
ex.observers = [
    SetID(),
    FileStorageObserver.create("results/mujoco/ablation_average_features"),
]


@ex.config
def config():
    result_folder = None  # noqa:F841


@ex.automain
def main(_run, result_folder, seed):
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

    r_vec = sum([latent_space.encoder(obs) for obs in current_states])
    r_vec /= np.linalg.norm(r_vec)
    env_inferred = LatentSpaceRewardWrapper(env, latent_space, r_vec)

    solver = get_sac(env_inferred)
    solver.learn(iterations)

    with Artifact(f"policy.zip", None, _run) as f:
        solver.save(f)

    N = 10
    true_reward_obtained = evaluate_policy(env, solver, N)
    inferred_reward_obtained = evaluate_policy(env_inferred, solver, N)
    print("Policy: true return", true_reward_obtained)
    print("Policy: inferred return", inferred_reward_obtained)
