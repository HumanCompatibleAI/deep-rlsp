import os
import itertools

import numpy as np
import tensorflow as tf

from stable_baselines import SAC

from deep_rlsp.model import InversePolicyMDN
from deep_rlsp.model.exact_dynamics_mujoco import ExactDynamicsMujoco
from deep_rlsp.util.timer import Timer
from deep_rlsp.util.video import save_video, render_mujoco_from_obs
from deep_rlsp.util.mujoco import MujocoObsClipper
from deep_rlsp.util.helper import init_env_from_obs
from deep_rlsp.envs.reward_wrapper import LatentSpaceRewardWrapper

# from deep_rlsp.util.helper import memoize
from deep_rlsp.util.linalg import get_cosine_similarity
from deep_rlsp.solvers import get_sac, get_ppo


def _print_timing(timer):
    print("Timing:")
    print(
        "\tForward: {:.1f}s total,  {:.1f}s avg".format(
            timer.get_total_time("forward"), timer.get_average_time("forward")
        )
    )
    print(
        "\tBackward: {:.1f}s total,  {:.1f}s avg".format(
            timer.get_total_time("backward"), timer.get_average_time("backward")
        )
    )


# @memoize
# def similarity(a, b):
#     if np.all(a == b):
#         return 1
#     return np.clip(1 / np.sum(np.square(a - b)), 1e-10, 1e10)

similarity = get_cosine_similarity


def latent_rlsp(
    _run,
    env,
    current_obs,
    max_horizon,
    experience_replay,
    latent_space,
    inverse_transition_model,
    policy_horizon_factor=1,
    epochs=1,
    learning_rate=0.2,
    r_prior=None,
    r_vec=None,
    threshold=1e-2,
    n_trajectories=50,
    n_trajectories_forward_factor=1.0,
    print_level=0,
    callback=None,
    reset_solver=False,
    solver_iterations=1000,
    trajectory_video_path=None,
    continue_training_dynamics=False,
    continue_training_latent_space=False,
    tf_graphs=None,
    clip_mujoco_obs=False,
    horizon_curriculum=False,
    inverse_model_parameters=dict(),
    latent_space_model_parameters=dict(),
    inverse_policy_parameters=dict(),
    reweight_gradient=False,
    max_epochs_per_horizon=20,
    init_from_policy=None,
    solver_str="sac",
    reward_action_norm_factor=0,
):
    """
    Deep RLSP algorithm.
    """
    assert solver_str in ("sac", "ppo")

    def update_policy(r_vec, solver, horizon, obs_backward):
        print("Updating policy")
        obs_backward.extend(current_obs)
        wrapped_env = LatentSpaceRewardWrapper(
            env,
            latent_space,
            r_vec,
            inferred_weight=None,
            init_observations=obs_backward,
            time_horizon=max_horizon * policy_horizon_factor,
            use_task_reward=False,
            reward_action_norm_factor=reward_action_norm_factor,
        )

        if reset_solver:
            print("resetting solver")
            if solver_str == "sac":
                solver = get_sac(
                    wrapped_env, learning_starts=0, verbose=int(print_level >= 2)
                )
            else:
                solver = get_ppo(wrapped_env, verbose=int(print_level >= 2))
        else:
            solver.set_env(wrapped_env)
        solver.learn(total_timesteps=solver_iterations, log_interval=100)
        return solver

    def update_inverse_policy(solver):
        print("Updating inverse policy")
        # inverse_policy = InversePolicyMDN(
        #     env, solver, experience_replay, **inverse_policy_parameters["model"]
        # )
        # inverse_policy.solver = solver
        first_epoch_loss, last_epoch_loss = inverse_policy.learn(
            return_initial_loss=True,
            verbose=print_level >= 2,
            **inverse_policy_parameters["learn"],
        )
        print(
            "Inverse policy loss: {} --> {}".format(first_epoch_loss, last_epoch_loss)
        )
        return first_epoch_loss, last_epoch_loss

    def _get_transitions_from_rollout(
        observations, actions, dynamics, experience_replay
    ):
        for obs, action in zip(observations, actions):
            try:
                # action = env.action_space.sample()
                next_obs = dynamics.dynamics(obs, action)
                experience_replay.append(obs, action, next_obs)
            except Exception as e:
                print("_get_transitions_from_rollout", e)

    def update_experience_replay(
        traj_actions_backward,
        traj_actions_forward,
        traj_observations_backward,
        traj_observations_forward,
        solver,
        experience_replay,
    ):
        """
        Rolls out the action sequences observed in the true dynamics model and keeps
        training the dynamics models on these.

        This is currently only used for debugging.
        """
        dynamics = ExactDynamicsMujoco(
            env.unwrapped.spec.id, tolerance=1e-3, max_iters=100
        )

        for states, actions in zip(traj_observations_backward, traj_actions_backward):
            _get_transitions_from_rollout(states, actions, dynamics, experience_replay)

        for states, actions in zip(traj_observations_forward, traj_actions_forward):
            _get_transitions_from_rollout(states, actions, dynamics, experience_replay)

        # experience_replay.add_policy_rollouts(
        #     env, solver, 10, env.spec.max_episode_steps
        # )

    def compute_grad(r_vec, epoch, env, horizon):
        init_obs_list = []
        feature_counts_backward = np.zeros_like(r_vec)

        weight_sum_bwd = 0
        obs_counts_backward = np.zeros_like(current_obs[0])

        if trajectory_video_path is not None:
            trajectory_rgbs = []

        states_backward = []
        states_forward = []
        obs_backward = []
        traj_actions_backward = []
        traj_observations_backward = []

        np.random.shuffle(current_obs)
        current_obs_cycle = itertools.cycle(current_obs)

        print("Horizon", horizon)

        if print_level >= 2:
            print("Backward")

        for i_traj in range(n_trajectories):
            obs = np.copy(next(current_obs_cycle))
            if i_traj == 0 and trajectory_video_path is not None:
                rgb = render_mujoco_from_obs(env, obs)
                trajectory_rgbs.append(rgb)
            state = latent_space.encoder(obs)  # t = T

            feature_counts_backward_traj = np.copy(state)
            obs_counts_backward_traj = np.copy(obs)
            actions_backward = []
            observations_backward = []

            # simulate trajectory into the past
            # iterates for t = T, T-1, T-2, ..., 1
            for t in range(horizon, 0, -1):
                action = inverse_policy.step(obs)
                if print_level >= 2 and i_traj == 0:
                    print("inverse action", action)
                actions_backward.append(action)
                with timer.start("backward"):
                    obs = inverse_transition_model.step(obs, action, sample=True)
                    if clip_mujoco_obs:
                        obs = clipper.clip(obs)[0]
                    state = latent_space.encoder(obs)

                feature_counts_backward_traj += state
                states_backward.append(state)
                obs_backward.append(obs)

                # debugging
                observations_backward.append(obs)
                if print_level >= 2 and i_traj == 0:
                    with np.printoptions(suppress=True):
                        print(obs)
                obs_counts_backward_traj += obs
                if i_traj == 0 and trajectory_video_path is not None:
                    rgb = render_mujoco_from_obs(env, obs)
                    trajectory_rgbs.append(rgb)

            init_obs_list.append(obs)
            weight = similarity(obs, initial_obs) if reweight_gradient else 1
            weight_sum_bwd += weight
            if print_level >= 2:
                print("similarity weight", weight)

            feature_counts_backward += feature_counts_backward_traj * weight
            obs_counts_backward += obs_counts_backward_traj * weight
            traj_actions_backward.append(actions_backward)
            traj_observations_backward.append(observations_backward)

        if trajectory_video_path is not None:
            trajectory_rgbs.extend([np.zeros_like(rgb)] * 5)

        init_obs_cycle = itertools.cycle(init_obs_list)
        feature_counts_forward = np.zeros_like(r_vec)

        weight_sum_fwd = 0
        obs_counts_forward = np.zeros_like(current_obs[0])

        n_trajectories_forward = int(n_trajectories * n_trajectories_forward_factor)
        traj_actions_forward = []
        traj_observations_forward = []

        if print_level >= 2:
            print("Forward")
        for i_traj in range(n_trajectories_forward):
            obs = np.copy(next(init_obs_cycle))  # t = 0

            if i_traj == 0 and trajectory_video_path is not None:
                rgb = render_mujoco_from_obs(env, obs)
                trajectory_rgbs.append(rgb)

            weight = similarity(obs, initial_obs) if reweight_gradient else 1
            weight_sum_fwd += weight

            env = init_env_from_obs(env, obs)
            state = latent_space.encoder(obs)

            feature_counts_forward_traj = np.copy(state)
            obs_counts_forward_traj = np.copy(obs)
            actions_forward = []
            observations_forward = []
            failed = False

            # iterates for t = 0, ..., T-1
            for t in range(horizon):
                action = solver.predict(obs)[0]
                if print_level >= 2 and i_traj == 0:
                    print("forward action", action)
                actions_forward.append(action)
                with timer.start("forward"):
                    if not failed:
                        try:
                            new_obs = env.step(action)[0]
                            obs = new_obs
                        except Exception as e:
                            failed = True
                            print("compute_grad", e)

                        if clip_mujoco_obs:
                            obs, clipped = clipper.clip(obs)
                            if clipped:
                                env = init_env_from_obs(env, obs)
                    state = latent_space.encoder(obs)
                feature_counts_forward_traj += state
                states_forward.append(state)

                # debugging
                observations_forward.append(obs)
                if print_level >= 2 and i_traj == 0:
                    with np.printoptions(suppress=True):
                        print(obs)
                obs_counts_forward_traj += obs

                if i_traj == 0 and trajectory_video_path is not None:
                    rgb = env.render("rgb_array")
                    trajectory_rgbs.append(rgb)

            feature_counts_forward += weight * feature_counts_forward_traj
            obs_counts_forward += weight * obs_counts_forward_traj
            traj_actions_forward.append(actions_forward)
            traj_observations_forward.append(observations_forward)

        if trajectory_video_path is not None:
            video_path = os.path.join(
                trajectory_video_path, "epoch_{}_traj.avi".format(epoch)
            )
            save_video(trajectory_rgbs, video_path, fps=2.0)
            print("Saved video to", video_path)

        # Normalize the gradient per-action,
        # so that its magnitude is not sensitive to the horizon
        feature_counts_forward /= weight_sum_fwd * horizon
        feature_counts_backward /= weight_sum_bwd * horizon

        dL_dr_vec = feature_counts_backward - feature_counts_forward

        print()
        print("\tfeature_counts_backward", feature_counts_backward)
        print("\tfeature_counts_forward", feature_counts_forward)
        print("\tn_trajectories_forward", n_trajectories_forward)
        print("\tn_trajectories", n_trajectories)
        print("\tdL_dr_vec", dL_dr_vec)
        print()

        # debugging
        if env.unwrapped.spec.id == "InvertedPendulum-v2":
            obs_counts_forward /= weight_sum_fwd * horizon
            obs_counts_backward /= weight_sum_bwd * horizon
            obs_counts_backward_enc = latent_space.encoder(obs_counts_backward)
            obs_counts_forward_enc = latent_space.encoder(obs_counts_forward)
            obs_counts_backward_old_reward = np.dot(r_vec, obs_counts_backward_enc)
            obs_counts_forward_old_reward = np.dot(r_vec, obs_counts_forward_enc)
            r_vec_new = r_vec + learning_rate * dL_dr_vec
            obs_counts_backward_new_reward = np.dot(r_vec_new, obs_counts_backward_enc)
            obs_counts_forward_new_reward = np.dot(r_vec_new, obs_counts_forward_enc)
            print("\tdebugging info")
            print("\tweight_sum_bwd", weight_sum_bwd)
            print("\tobs_counts_backward", obs_counts_backward)
            print("\tweight_sum_fwd", weight_sum_fwd)
            print("\tobs_counts_forward", obs_counts_forward)
            print("\told reward")
            print("\tobs_counts_backward", obs_counts_backward_old_reward)
            print("\tobs_counts_forward", obs_counts_forward_old_reward)
            print("\tnew reward")
            print("\tobs_counts_backward", obs_counts_backward_new_reward)
            print("\tobs_counts_forward", obs_counts_forward_new_reward)

            _run.log_scalar("debug_forward_pos", obs_counts_forward[0], epoch)
            _run.log_scalar("debug_forward_angle", obs_counts_forward[1], epoch)
            _run.log_scalar("debug_forward_velocity", obs_counts_forward[2], epoch)
            _run.log_scalar(
                "debug_forward_angular_velocity", obs_counts_forward[3], epoch
            )
            _run.log_scalar("debug_backward_pos", obs_counts_backward[0], epoch)
            _run.log_scalar("debug_backward_angle", obs_counts_backward[1], epoch)
            _run.log_scalar("debug_backward_velocity", obs_counts_backward[2], epoch)
            _run.log_scalar(
                "debug_backward_angular_velocity", obs_counts_backward[3], epoch
            )
            init_obs_array = np.array(init_obs_list)
            _run.log_scalar("debug_init_state_pos", init_obs_array[:, 0].mean(), epoch)
            _run.log_scalar(
                "debug_init_state_angle", init_obs_array[:, 1].mean(), epoch
            )
            _run.log_scalar(
                "debug_init_state_velocity", init_obs_array[:, 2].mean(), epoch
            )
            _run.log_scalar(
                "debug_init_state_angular_velocity", init_obs_array[:, 3].mean(), epoch
            )

        # Gradient of the prior
        if r_prior is not None:
            dL_dr_vec += r_prior.logdistr_grad(r_vec)

        return (
            dL_dr_vec,
            feature_counts_forward,
            feature_counts_backward,
            traj_actions_backward,
            traj_actions_forward,
            traj_observations_backward,
            traj_observations_forward,
            states_forward,
            states_backward,
            obs_backward,
        )

    timer = Timer()
    clipper = MujocoObsClipper(env.unwrapped.spec.id)

    if trajectory_video_path is not None:
        os.makedirs(trajectory_video_path, exist_ok=True)

    current_state = [latent_space.encoder(obs) for obs in current_obs]
    initial_obs = env.reset()

    if r_vec is None:
        r_vec = sum(current_state)
        r_vec /= np.linalg.norm(r_vec)

    with np.printoptions(precision=4, suppress=True, threshold=10):
        print("Initial reward vector: {}".format(r_vec))

    dynamics = ExactDynamicsMujoco(env.unwrapped.spec.id, tolerance=1e-3, max_iters=100)

    wrapped_env = LatentSpaceRewardWrapper(env, latent_space, r_vec)

    if init_from_policy is not None:
        print(f"Loading policy from {init_from_policy}")
        solver = SAC.load(init_from_policy)
    else:
        if solver_str == "sac":
            solver = get_sac(
                wrapped_env, learning_starts=0, verbose=int(print_level >= 2)
            )
        else:
            solver = get_ppo(wrapped_env, verbose=int(print_level >= 2))

    inverse_policy_graph = tf.Graph()

    with inverse_policy_graph.as_default():
        inverse_policy = InversePolicyMDN(
            env, solver, experience_replay, **inverse_policy_parameters["model"]
        )

    gradients = []
    obs_backward = []

    solver = update_policy(r_vec, solver, 1, obs_backward)
    with inverse_policy_graph.as_default():
        (
            inverse_policy_initial_loss,
            inverse_policy_final_loss,
        ) = update_inverse_policy(solver)

    epoch = 0
    for horizon in range(1, max_horizon + 1):
        if not horizon_curriculum:
            horizon = max_horizon
            max_horizon = env.spec.max_episode_steps
            max_epochs_per_horizon = epochs
            threshold = -float("inf")

        last_n_grad_norm = float("inf")
        # initialize negatively in case we don't continue to train
        backward_final_loss = -float("inf")
        latent_final_loss = -float("inf")
        inverse_policy_final_loss = -float("inf")

        backward_threshold = float("inf")
        latent_threshold = float("inf")
        inverse_policy_threshold = float("inf")

        gradients = []
        print(f"New horizon: {horizon}")
        epochs_per_horizon = 0

        while epochs_per_horizon < max_epochs_per_horizon and (
            last_n_grad_norm > threshold
            or backward_final_loss > backward_threshold
            or latent_final_loss > latent_threshold
            or inverse_policy_final_loss > inverse_policy_threshold
        ):
            epochs_per_horizon += 1
            epoch += 1
            if epoch > epochs:
                print(f"Stopping after {epoch} epochs.")
                return r_vec

            (
                dL_dr_vec,
                feature_counts_forward,
                feature_counts_backward,
                traj_actions_backward,
                traj_actions_forward,
                traj_observations_backward,
                traj_observations_forward,
                states_forward,
                states_backward,
                obs_backward,
            ) = compute_grad(r_vec, epoch, env, horizon)

            print("threshold", threshold)

            if clip_mujoco_obs:
                print("clipper.counter", clipper.counter)
                clipper.counter = 0

            if print_level >= 1:
                _print_timing(timer)

            grad_mean_n = 10
            gradients.append(dL_dr_vec)
            last_n_grad_norm = np.linalg.norm(np.mean(gradients[-grad_mean_n:], axis=0))
            # Clip gradient by norm
            grad_norm = np.linalg.norm(gradients[-1])
            if grad_norm > 10:
                dL_dr_vec = 10 * dL_dr_vec / grad_norm

            # Gradient ascent
            r_vec = r_vec + learning_rate * dL_dr_vec

            with np.printoptions(precision=3, suppress=True, threshold=10):
                print(
                    f"Epoch {epoch}; Reward vector: {r_vec} ",
                    "(norm {:.3f}); grad_norm {:.3f}; last_n_grad_norm: {:.3f}".format(
                        np.linalg.norm(r_vec), grad_norm, last_n_grad_norm
                    ),
                )

            latent_initial_loss = None
            forward_initial_loss = None
            backward_initial_loss = None
            if continue_training_dynamics or continue_training_latent_space:
                assert tf_graphs is not None
                update_experience_replay(
                    traj_actions_backward,
                    traj_actions_forward,
                    traj_observations_backward,
                    traj_observations_forward,
                    solver,
                    experience_replay,
                )

                if continue_training_dynamics:
                    with tf_graphs["inverse"].as_default():
                        (
                            backward_initial_loss,
                            backward_final_loss,
                        ) = inverse_transition_model.learn(
                            return_initial_loss=True,
                            verbose=print_level >= 2,
                            **inverse_model_parameters["learn"],
                        )
                    print(
                        "Backward model loss:  {} --> {}".format(
                            backward_initial_loss, backward_final_loss
                        )
                    )

                if continue_training_latent_space:
                    with tf_graphs["latent"].as_default():
                        latent_initial_loss, latent_final_loss = latent_space.learn(
                            experience_replay,
                            return_initial_loss=True,
                            verbose=print_level >= 2,
                            **latent_space_model_parameters["learn"],
                        )
                    print(
                        "Latent space loss:  {} --> {}".format(
                            latent_initial_loss, latent_final_loss
                        )
                    )

            solver = update_policy(r_vec, solver, horizon, obs_backward)
            with inverse_policy_graph.as_default():
                (
                    inverse_policy_initial_loss,
                    inverse_policy_final_loss,
                ) = update_inverse_policy(solver)

            if callback:
                callback(locals(), globals())

    return r_vec
