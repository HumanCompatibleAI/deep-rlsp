import os
import pickle
import datetime

import gym
import numpy as np
import tensorflow as tf

from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySac

from deep_rlsp.latent_rlsp import latent_rlsp
from deep_rlsp.util.dist import NormalDistribution, LaplaceDistribution
from deep_rlsp.model import (
    LatentSpaceModel,
    InverseDynamicsMLP,
    StateVAE,
    ExperienceReplay,
)
from deep_rlsp.envs.reward_wrapper import LatentSpaceRewardWrapper
from deep_rlsp.util.parameter_checks import check_in, check_not_none
from deep_rlsp.util.linalg import get_cosine_similarity
from deep_rlsp.util.helper import load_data
from deep_rlsp.util.helper import (
    sample_obs_from_trajectory,
    get_trajectory,
    evaluate_policy,
)
from deep_rlsp.util.results import Artifact
from deep_rlsp.model.mujoco_debug_models import (
    MujocoDebugFeatures,
    PendulumDynamics,
    IdentityFeatures,
)


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_state_from = config["current_state_from"]
        if current_state_from == "file":
            base = os.path.basename(config["current_state_file"])
            current_state_from = os.path.splitext(base)[0]
        n_sample_states = config["n_sample_states"]
        env_id = config["env_id"]
        # random number to avoid having the same id when parallelizing
        randint = np.random.randint(0, 1_000_000)
        custom_id = (
            f"{timestamp}_{env_id}_{current_state_from}_{n_sample_states}_{randint}"
        )
        return custom_id  # started_event returns the _run._id


ex = Experiment("rlsp-mujoco")
# this works for sacred < 0.7.5 after upgrading have to use FileStorageObserver(...)
ex.observers = [SetID(), FileStorageObserver.create("results/mujoco")]


def get_r_prior(prior, reward_center, std):
    if prior is not None:
        check_in("prior", prior, ("gaussian", "laplace", "uniform"))
        if prior == "gaussian":
            return NormalDistribution(reward_center, std)
        elif prior == "laplace":
            return LaplaceDistribution(reward_center, std)
    return None


@ex.named_config
def test():
    env_id = "InvertedPendulum-v2"  # noqa:F841
    prior = "gaussian"  # noqa:F841
    horizon = 1000  # noqa:F841
    policy_horizon_factor = 1  # noqa:F841
    learning_rate = 0.01  # noqa:F841
    epochs = 100  # noqa:F841
    std = 0.5  # noqa:F841
    print_level = 1  # noqa:F841
    n_trajectories = 20  # noqa:F841
    solver_iterations = 10000  # noqa:F841
    reset_solver = False  # noqa:F841

    latent_model_checkpoint = None  # noqa:F841
    inverse_dynamics_model_checkpoint = None  # noqa:F841

    n_trajectories_forward_factor = 1  # noqa:F841
    trajectory_video_path = "mujoco/latent_rlsp"  # noqa:F841

    current_state_from = "optimal"  # noqa:F841
    good_policy_path = "sac_pendulum_1e6"  # noqa:F841
    current_state_file = None  # noqa:F841
    clip_mujoco_obs = True  # noqa:F841

    experience_replay_size = 100_000  # noqa:F841
    debug_train_with_true_dynamics = False  # noqa:F841
    debug_handcoded_features = False  # noqa:F841
    vae_latent_space = True  # noqa:F841
    identity_latent_space = False  # noqa:F841

    continue_training_latent_space = True  # noqa:F841
    continue_training_dynamics = True  # noqa:F841

    horizon_curriculum = True  # noqa:F841
    n_sample_states = 1  # noqa:F841
    n_rollouts_init = 100  # noqa:F841

    reweight_gradient = True  # noqa:F841
    threshold = 1e-2  # noqa:F841
    max_epochs_per_horizon = 20  # noqa:F841
    init_from_policy = None  # noqa:F841
    add_policy_rollouts_to_replay = False  # noqa:F841
    reward_action_norm_factor = 0  # noqa:F841


@ex.named_config
def base():
    prior = None  # noqa:F841
    horizon = 30  # noqa:F841
    policy_horizon_factor = 1  # noqa:F841
    learning_rate = 0.01  # noqa:F841
    epochs = 1000  # noqa:F841
    std = 0.5  # noqa:F841
    print_level = 1  # noqa:F841
    n_trajectories = 200  # noqa:F841
    n_trajectories_forward_factor = 1  # noqa:F841
    reset_solver = False  # noqa:F841
    latent_model_checkpoint = None  # noqa:F841
    continue_training_latent_space = False  # noqa:F841
    inverse_dynamics_model_checkpoint = None  # noqa:F841
    continue_training_dynamics = False  # noqa:F841
    current_state_file = None  # noqa:F841
    clip_mujoco_obs = True  # noqa:F841
    experience_replay_size = 100_000  # noqa:F841
    debug_train_with_true_dynamics = False  # noqa:F841
    debug_handcoded_features = False  # noqa:F841
    vae_latent_space = True  # noqa:F841
    identity_latent_space = False  # noqa:F841
    horizon_curriculum = True  # noqa:F841
    n_rollouts_init = 100  # noqa:F841
    add_policy_rollouts_to_replay = False  # noqa:F841
    reweight_gradient = True  # noqa:F841
    max_epochs_per_horizon = 10  # noqa:F841
    threshold = 2.0  # noqa:F841
    reward_action_norm_factor = 0  # noqa:F841
    inverse_model_parameters = {  # noqa:F841
        "model": {
            "hidden_layer_size": 1024,
            "n_hidden_layers": 5,
            "learning_rate": 1e-5,
        },
        "learn": {"n_epochs": 100, "batch_size": 500, "print_evaluation": False},
    }
    latent_space_model_parameters = {  # noqa:F841
        "model": {
            "state_size": 30,
            "n_layers": 3,
            "layer_size": 512,
            "learning_rate": 1e-5,
            "prior_stdev": 1,
            "divergence_factor": 0.001,
        },
        "learn": {"n_epochs": 100, "batch_size": 500},
    }
    inverse_policy_parameters = {  # noqa:F841
        "model": {
            "tensorboard_log": None,
            "learning_rate": 1e-4,
            "n_layers": 3,
            "layer_size": 512,
            "n_out": 5,
            "gauss_stdev": 0.05,
        },
        "learn": {"n_epochs": 1, "batch_size": 500, "reinitialize": True},
    }


@ex.named_config
def pendulum():
    env_id = "InvertedPendulum-v2"  # noqa:F841
    solver_iterations = 50000  # noqa:F841
    reset_solver = True  # noqa:F841
    trajectory_video_path = "mujoco/latent_rlsp/pendulum"  # noqa:F841
    current_state_from = "optimal"  # noqa:F841
    good_policy_path = "policies/sac_pendulum_6e4"  # noqa:F841
    n_sample_states = 50  # noqa:F841
    n_rollouts_init = 50  # noqa:F841
    add_policy_rollouts_to_replay = True  # noqa:F841
    policy_horizon_factor = 100  # noqa:F841


@ex.named_config
def cheetah_fw():
    env_id = "HalfCheetah-FW-v2"  # noqa:F841
    solver_iterations = 10000  # noqa:F841
    trajectory_video_path = "mujoco/latent_rlsp/cheetah_fw"  # noqa:F841
    current_state_from = "optimal"  # noqa:F841
    good_policy_path = "policies/sac_cheetah_fw_2e6"  # noqa:F841
    n_sample_states = 50  # noqa:F841


@ex.named_config
def cheetah_bw():
    env_id = "HalfCheetah-BW-v2"  # noqa:F841
    solver_iterations = 10000  # noqa:F841
    trajectory_video_path = "mujoco/latent_rlsp/cheetah_bw"  # noqa:F841
    current_state_from = "optimal"  # noqa:F841
    good_policy_path = "policies/sac_cheetah_bw_2e6"  # noqa:F841
    n_sample_states = 50  # noqa:F841


@ex.named_config
def cheetah_skill():
    env_id = "HalfCheetah-v2"  # noqa:F841
    solver_iterations = 10000  # noqa:F841
    trajectory_video_path = "mujoco/latent_rlsp/cheetah_skills"  # noqa:F841
    current_state_from = "trajectory_file"  # noqa:F841
    good_policy_path = None  # noqa:F841
    current_state_file = None  # noqa:F841
    n_sample_states = 50  # noqa:F841


@ex.named_config
def ant():
    env_id = "Ant-FW-v2"  # noqa:F841
    solver_iterations = 20000  # noqa:F841
    trajectory_video_path = "mujoco/latent_rlsp/ant"  # noqa:F841
    current_state_from = "optimal"  # noqa:F841
    good_policy_path = "policies/sac_ant_2e6"  # noqa:F841
    n_sample_states = 50  # noqa:F841


@ex.named_config
def hopper():
    env_id = "Hopper-FW-v2"  # noqa:F841
    solver_iterations = 20000  # noqa:F841
    trajectory_video_path = "mujoco/latent_rlsp/hopper"  # noqa:F841
    current_state_from = "optimal"  # noqa:F841
    good_policy_path = "policies/sac_hopper_2e6"  # noqa:F841
    n_sample_states = 50  # noqa:F841


@ex.named_config
def reach_features():
    env_id = "FetchReachStack-v1"  # noqa:F841
    solver_iterations = 1000  # noqa:F841
    trajectory_video_path = "mujoco/latent_rlsp/reach"  # noqa:F841
    current_state_from = "initial"  # noqa:F841
    n_sample_states = 1  # noqa:F841
    vae_latent_space = False  # noqa:F841
    debug_handcoded_features = True  # noqa:F841
    n_rollouts_init = 1000  # noqa:F841
    prior = "gaussian"  # noqa:F841
    good_policy_path = "policies/sac_reach_true_2e6"  # noqa:F841
    policy_horizon_factor = 5  # noqa:F841
    reward_action_norm_factor = 0.1  # noqa:F841
    # horizon_curriculum = False


@ex.named_config
def reach_vae():
    env_id = "FetchReachStack-v1"  # noqa:F841
    solver_iterations = 1000  # noqa:F841
    trajectory_video_path = "mujoco/latent_rlsp/reach"  # noqa:F841
    current_state_from = "initial"  # noqa:F841
    n_sample_states = 1  # noqa:F841
    vae_latent_space = True  # noqa:F841
    debug_handcoded_features = False  # noqa:F841
    n_rollouts_init = 1000  # noqa:F841
    prior = "gaussian"  # noqa:F841
    good_policy_path = "policies/sac_reach_true_2e6"  # noqa:F841
    policy_horizon_factor = 5  # noqa:F841
    reward_action_norm_factor = 0.1  # noqa:F841
    # horizon_curriculum = False


@ex.config
def config():
    # Gym id of the environment
    env_id = None  # noqa:F841
    # Prior on the inferred reward function: one of [gaussian, laplace, uniform].
    prior = "gaussian"  # noqa:F841
    # Number of timesteps we assume the human has been acting.
    horizon = 20  # noqa:F841
    policy_horizon_factor = 1  # noqa:F841
    # Learning rate for rlsp gradient descent
    learning_rate = 0.01  # noqa:F841
    # Number of gradient descent steps to take.
    epochs = 20  # noqa:F841
    # Standard deviation for the prior
    std = 0.5  # noqa:F841
    # Level of verbosity. Shoud be 0 or 1.
    print_level = 1  # noqa:F841
    # Number of rollouts for approximating the rlsp gradient
    n_trajectories = 50  # noqa:F841
    # Number of iterations to use for ppo
    solver_iterations = 10000  # noqa:F841
    # Wheather to continue to update the same policy or reinitialize it in every step
    reset_solver = False  # noqa:F841
    # Path to load latent space model from (default is to train a new one)
    latent_model_checkpoint = None  # noqa:F841
    # Path to load inverse dynamics model from (default is to train a new one)
    inverse_dynamics_model_checkpoint = None  # noqa:F841
    # Number of forward trajectories sampled per backward trajectory
    # (1 corresponds to sampling the same number of forward and backward trajectories)
    n_trajectories_forward_factor = 1  # noqa:F841
    # Folder to write videos of the trajectories sampled during Deep RLSP to
    trajectory_video_path = None  # noqa:F841
    # Path to load a poicy trained on the true reward from
    good_policy_path = None  # noqa:F841

    # Where to get the input state from. Allowed values are:
    #  - "optimal": sampled from a policy trained on the true reward
    #  - "initial": states returned by env.reset()
    #  - "trajectory_file": from a pickle file that contains a trajectory to sample from
    #  - "file": from a pickle file that only contains the input states
    current_state_from = None  # noqa:F841
    # If `current_state_from` is set to "trajctory_file" or "file", the corresponding
    # file is provided here
    current_state_file = None  # noqa:F841

    # Size of the experience replay used to train the inverse policy
    experience_replay_size = 100_000  # noqa:F841
    # Whether to continue training the inverse dynamics model during Deep RLSP
    continue_training_dynamics = False  # noqa:F841
    # Whether to continue training the latent space model during Deep RLSP
    continue_training_latent_space = False  # noqa:F841
    # If set to false use RSSM instead of VAE
    vae_latent_space = True  # noqa:F841
    # If true, use the mujoco observations instead of a learned representation
    identity_latent_space = False  # noqa:F841
    # If true, clip mujoco observations that are outside a reasonable range
    clip_mujoco_obs = False  # noqa:F841
    # Number of rollouts to use for evaluating policies
    n_eval_rollouts = 10  # noqa:F841
    # Number of input states
    n_sample_states = 1  # noqa:F841
    # Number of random rollouts to initialize the experience replay with
    n_rollouts_init = 100  # noqa:F841

    # If true the gradient terms are reweighted by their distance to the initial
    # state of the environmnent
    reweight_gradient = False  # noqa:F841
    # If true, start with horizon 1 and increase it incrementally
    horizon_curriculum = False  # noqa:F841
    # Threshold for the gradient norm to advance the curriculum
    threshold = 1e-2  # noqa:F841
    # Number of epochs after which to always advance the curriculum
    max_epochs_per_horizon = 20  # noqa:F841
    add_policy_rollouts_to_replay = False  # noqa:F841
    reward_action_norm_factor = False  # noqa:F841

    # Parameters of latent space and inverse dynamics model and policy
    inverse_model_parameters = {"model": dict(), "learn": dict()}  # noqa:F841
    latent_space_model_parameters = {"model": dict(), "learn": dict()}  # noqa:F841
    inverse_policy_parameters = {"model": dict(), "learn": dict()}  # noqa:F841

    # For debugging
    debug_train_with_true_dynamics = False  # noqa:F841
    debug_handcoded_features = False  # noqa:F841
    init_from_policy = None  # noqa:F841


@ex.automain
def main(
    _run,
    env_id,
    prior,
    horizon,
    policy_horizon_factor,
    learning_rate,
    epochs,
    std,
    print_level,
    n_trajectories,
    solver_iterations,
    reset_solver,
    latent_model_checkpoint,
    inverse_dynamics_model_checkpoint,
    n_trajectories_forward_factor,
    trajectory_video_path,
    current_state_from,
    good_policy_path,
    current_state_file,
    continue_training_dynamics,
    continue_training_latent_space,
    debug_train_with_true_dynamics,
    debug_handcoded_features,
    vae_latent_space,
    identity_latent_space,
    clip_mujoco_obs,
    horizon_curriculum,
    n_eval_rollouts,
    inverse_model_parameters,
    latent_space_model_parameters,
    inverse_policy_parameters,
    experience_replay_size,
    n_sample_states,
    n_rollouts_init,
    add_policy_rollouts_to_replay,
    reweight_gradient,
    threshold,
    max_epochs_per_horizon,
    init_from_policy,
    reward_action_norm_factor,
    seed,
):
    print("--------")
    for key, val in locals().items():
        print(key, val)
    print("--------")

    # Check the parameters so that we fail fast
    check_in(
        "env_id",
        env_id,
        [
            "InvertedPendulum-v2",
            "HalfCheetah-v2",
            "HalfCheetah-FW-v2",
            "HalfCheetah-BW-v2",
            "Ant-v2",
            "Ant-FW-v2",
            "Hopper-v2",
            "Hopper-FW-v2",
            "FetchReachStack-v1",
        ],
    )
    if prior is not None:
        check_in("prior", prior, ["gaussian", "laplace", "uniform"])
    check_in(
        "current_state_from",
        current_state_from,
        ["optimal", "initial", "trajectory_file", "file"],
    )
    assert sum([debug_handcoded_features, vae_latent_space, identity_latent_space]) <= 1
    if current_state_from == "optimal":
        check_not_none("good_policy_path", good_policy_path)
    if not (debug_handcoded_features or vae_latent_space):
        check_not_none("latent_model_checkpoint", latent_model_checkpoint)

    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    env = gym.make(env_id)

    if good_policy_path is None:
        true_reward_policy = None
    else:
        true_reward_policy = SAC.load(good_policy_path)

    random_policy = SAC(MlpPolicySac, env, verbose=0)

    # Sample input states
    if current_state_from == "file":
        with open(current_state_file, "rb") as f:
            current_states = list(pickle.load(f))
    elif current_state_from == "trajectory_file":
        rollout = load_data(current_state_file)["observations"][0]
        current_states = sample_obs_from_trajectory(rollout, n_sample_states)
    else:
        current_states = []
        for _ in range(n_sample_states):
            if current_state_from == "initial":
                state = env.reset()
            elif current_state_from == "optimal":
                observations, _, total_reward = get_trajectory(
                    env, true_reward_policy, True, False, True
                )
                print("Sample policy return:", total_reward)
                state = sample_obs_from_trajectory(observations, 1)[0]
            else:
                raise NotImplementedError()
            current_states.append(state)

    _run.info["current_states"] = current_states

    experience_replay = ExperienceReplay(experience_replay_size)
    if add_policy_rollouts_to_replay:
        experience_replay.add_random_rollouts(
            env, env.spec.max_episode_steps, int(0.25 * n_rollouts_init)
        )
        experience_replay.add_policy_rollouts(
            env,
            true_reward_policy,
            int(0.25 * n_rollouts_init),
            env.spec.max_episode_steps,
            eps_greedy=0,
        )
        experience_replay.add_policy_rollouts(
            env,
            true_reward_policy,
            int(0.25 * n_rollouts_init),
            env.spec.max_episode_steps,
            eps_greedy=0.12,
        )
        experience_replay.add_policy_rollouts(
            env,
            true_reward_policy,
            int(0.25 * n_rollouts_init),
            env.spec.max_episode_steps,
            eps_greedy=0.3,
        )
    else:
        experience_replay.add_random_rollouts(
            env, env.spec.max_episode_steps, n_rollouts_init
        )

    # Train / load transition models
    graph_latent, graph_bwd = tf.Graph(), tf.Graph()
    os.makedirs("tf_ckpt", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if debug_handcoded_features:
        latent_space = MujocoDebugFeatures(env)
    elif identity_latent_space:
        latent_space = IdentityFeatures(env)
    elif vae_latent_space:
        with graph_latent.as_default():
            if latent_model_checkpoint is not None:
                latent_space = StateVAE.restore(latent_model_checkpoint)
                latent_space.checkpoint_folder = None  # don't continue saving model
            else:
                label = "vae_{}_{}".format(env_id, timestamp)
                latent_model_checkpoint = "tf_ckpt/tf_ckpt_" + label

                latent_space = StateVAE(
                    env.observation_space.shape[0],
                    checkpoint_folder=latent_model_checkpoint,
                    **latent_space_model_parameters["model"],
                )
                initial_loss, final_loss = latent_space.learn(
                    experience_replay,
                    return_initial_loss=True,
                    verbose=True,
                    **latent_space_model_parameters["learn"],
                )
                print(
                    "Initialized latent space VAE: {} --> {}".format(
                        initial_loss, final_loss
                    )
                )
    else:
        if latent_model_checkpoint is not None:
            with graph_latent.as_default():
                latent_space = LatentSpaceModel.restore(env, latent_model_checkpoint)
            latent_space.checkpoint_folder = None  # don't continue saving model
        else:
            raise NotImplementedError()

    with graph_bwd.as_default():
        if debug_train_with_true_dynamics:
            inverse_transition_model = PendulumDynamics(latent_space, backward=True)
        elif inverse_dynamics_model_checkpoint is not None:
            inverse_transition_model = InverseDynamicsMLP.restore(
                env, experience_replay, inverse_dynamics_model_checkpoint
            )
            inverse_transition_model.checkpoint_folder = None
        else:
            label = "mlp_{}_{}".format(env_id, timestamp)
            inverse_dynamics_model_checkpoint = "tf_ckpt/tf_ckpt_" + label

            inverse_transition_model = InverseDynamicsMLP(
                env,
                experience_replay,
                checkpoint_folder=inverse_dynamics_model_checkpoint,
                **inverse_model_parameters["model"],
            )
            initial_loss, final_loss = inverse_transition_model.learn(
                return_initial_loss=True,
                verbose=True,
                **inverse_model_parameters["learn"],
            )
            print(
                "Initialized backward model: {} --> {}".format(initial_loss, final_loss)
            )

    _run.info["inverse_dynamics_model_checkpoint"] = inverse_dynamics_model_checkpoint
    _run.info["latent_model_checkpoint"] = latent_model_checkpoint

    tf_graphs = {"latent": graph_latent, "inverse": graph_bwd}

    reward_center = np.zeros(latent_space.state_size)
    r_prior = get_r_prior(prior, reward_center, std)

    last_n = 5
    feature_counts_forward_last_n = None
    feature_counts_backward_last_n = None
    inferred_reward_last_n = None

    _run.info["inferred_rewards"] = []

    def log_metrics(loc, glob):
        del glob
        global feature_counts_forward_last_n
        global feature_counts_backward_last_n
        global inferred_reward_last_n
        step = loc["epoch"]
        _run.log_scalar("inverse_policy_error", loc["inverse_policy_final_loss"], step)
        _run.log_scalar(
            "inverse_policy_initial_loss", loc["inverse_policy_initial_loss"], step
        )
        _run.log_scalar("grad_norm", loc["grad_norm"], step)
        _run.log_scalar("last_n_grad_norm", loc["last_n_grad_norm"], step)

        # feature counts
        feature_counts_forward = loc["feature_counts_forward"]
        feature_counts_backward = loc["feature_counts_backward"]

        r_inferred = loc["r_vec"]
        _run.info["inferred_rewards"].append(r_inferred)

        if step == 1:
            feature_counts_forward_last_n = [feature_counts_forward] * last_n
            feature_counts_backward_last_n = [feature_counts_backward] * last_n
            inferred_reward_last_n = [r_inferred] * last_n

        # magnitude
        fw_mag = np.linalg.norm(feature_counts_forward)
        bw_mag = np.linalg.norm(feature_counts_backward)
        rew_mag = np.linalg.norm(r_inferred)
        _run.log_scalar("feature_counts_forward_magnitude", fw_mag, step)
        _run.log_scalar("feature_counts_backward_magnitude", bw_mag, step)
        _run.log_scalar("inferred_reward_magnitude", rew_mag, step)
        # direction
        fw_cos_last = get_cosine_similarity(
            feature_counts_forward, feature_counts_forward_last_n[-1]
        )
        bw_cos_last = get_cosine_similarity(
            feature_counts_backward, feature_counts_backward_last_n[-1]
        )
        rew_cos_last = get_cosine_similarity(r_inferred, inferred_reward_last_n[-1])
        _run.log_scalar("feature_counts_forward_cos_last", fw_cos_last, step)
        _run.log_scalar("feature_counts_backward_cos_last", bw_cos_last, step)
        _run.log_scalar("inferred_reward_cos_last", rew_cos_last, step)

        fw_cos_last_n = get_cosine_similarity(
            feature_counts_forward, feature_counts_forward_last_n[0]
        )
        bw_cos_last_n = get_cosine_similarity(
            feature_counts_backward, feature_counts_backward_last_n[0]
        )
        rew_cos_last_n = get_cosine_similarity(r_inferred, inferred_reward_last_n[0])
        _run.log_scalar(
            "feature_counts_forward_cos_last_{}".format(last_n), fw_cos_last_n, step
        )
        _run.log_scalar(
            "feature_counts_backward_cos_last_{}".format(last_n), bw_cos_last_n, step
        )
        _run.log_scalar(
            "inferred_reward_cos_last_{}".format(last_n), rew_cos_last_n, step
        )

        _run.log_scalar("horizon", loc["horizon"], step)
        _run.log_scalar("threshold", loc["threshold"], step)

        feature_counts_forward_last_n.append(feature_counts_forward)
        feature_counts_backward_last_n.append(feature_counts_backward)
        inferred_reward_last_n.append(r_inferred)
        feature_counts_forward_last_n.pop(0)
        feature_counts_backward_last_n.pop(0)
        inferred_reward_last_n.pop(0)

        r_inferred_normalized = r_inferred / np.linalg.norm(r_inferred)

        env_inferred = LatentSpaceRewardWrapper(
            env, latent_space, r_inferred_normalized
        )
        if true_reward_policy is not None:
            good_policy_true_reward_obtained = evaluate_policy(
                env, true_reward_policy, n_eval_rollouts
            )
            good_policy_inferred_reward_obtained = evaluate_policy(
                env_inferred, true_reward_policy, n_eval_rollouts
            )

            _run.log_scalar(
                "good_policy_true_reward_obtained",
                good_policy_true_reward_obtained,
                step,
            )
            _run.log_scalar(
                "good_policy_inferred_reward_obtained",
                good_policy_inferred_reward_obtained,
                step,
            )

        random_policy_true_reward_obtained = evaluate_policy(
            env, random_policy, n_eval_rollouts
        )
        random_policy_inferred_reward_obtained = evaluate_policy(
            env_inferred, random_policy, n_eval_rollouts
        )

        _run.log_scalar(
            "random_policy_true_reward_obtained",
            random_policy_true_reward_obtained,
            step,
        )
        _run.log_scalar(
            "random_policy_inferred_reward_obtained",
            random_policy_inferred_reward_obtained,
            step,
        )

        rlsp_policy = loc["solver"]
        with Artifact(f"rlsp_policy_{step}.zip", None, _run) as f:
            rlsp_policy.save(f)

        rlsp_policy_true_reward_obtained = evaluate_policy(
            env, rlsp_policy, n_eval_rollouts
        )
        rlsp_policy_inferred_reward_obtained = evaluate_policy(
            env_inferred, rlsp_policy, n_eval_rollouts
        )

        _run.log_scalar(
            "rlsp_policy_true_reward_obtained", rlsp_policy_true_reward_obtained, step,
        )
        _run.log_scalar(
            "rlsp_policy_inferred_reward_obtained",
            rlsp_policy_inferred_reward_obtained,
            step,
        )

        if true_reward_policy is not None:
            print("True reward policy: true return", good_policy_true_reward_obtained)
            print(
                "True reward policy: inferred return",
                good_policy_inferred_reward_obtained,
            )
        print("RLSP policy: true return", rlsp_policy_true_reward_obtained)
        print("RLSP policy: inferred return", rlsp_policy_inferred_reward_obtained)
        print("Random policy: true return", random_policy_true_reward_obtained)
        print("Random policy: inferred return", random_policy_inferred_reward_obtained)

        _run.log_scalar("latent_initial_loss", loc["latent_initial_loss"], step)
        _run.log_scalar("latent_final_loss", loc["latent_final_loss"], step)
        _run.log_scalar("backward_initial_loss", loc["backward_initial_loss"], step)
        _run.log_scalar("backward_final_loss", loc["backward_final_loss"], step)

    r_inferred = latent_rlsp(
        _run,
        env,
        current_states,
        horizon,
        experience_replay,
        latent_space,
        inverse_transition_model,
        policy_horizon_factor=policy_horizon_factor,
        epochs=epochs,
        learning_rate=learning_rate,
        r_prior=r_prior,
        threshold=threshold,
        n_trajectories=n_trajectories,
        reset_solver=reset_solver,
        solver_iterations=solver_iterations,
        print_level=print_level,
        n_trajectories_forward_factor=n_trajectories_forward_factor,
        callback=log_metrics,
        trajectory_video_path=trajectory_video_path,
        continue_training_dynamics=continue_training_dynamics,
        continue_training_latent_space=continue_training_latent_space,
        tf_graphs=tf_graphs,
        clip_mujoco_obs=clip_mujoco_obs,
        horizon_curriculum=horizon_curriculum,
        inverse_model_parameters=inverse_model_parameters,
        latent_space_model_parameters=latent_space_model_parameters,
        inverse_policy_parameters=inverse_policy_parameters,
        reweight_gradient=reweight_gradient,
        max_epochs_per_horizon=max_epochs_per_horizon,
        init_from_policy=init_from_policy,
        reward_action_norm_factor=reward_action_norm_factor,
    )
