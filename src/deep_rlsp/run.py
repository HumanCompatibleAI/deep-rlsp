import os
import datetime
import numpy as np
import tensorflow as tf

from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from deep_rlsp.envs.gridworlds import TOY_PROBLEMS, TOY_ENV_CLASSES
from deep_rlsp.envs.gridworlds.gym_envs import get_gym_gridworld
from deep_rlsp.relative_reachability import relative_reachability_penalty
from deep_rlsp.rlsp import rlsp
from deep_rlsp.latent_rlsp import latent_rlsp
from deep_rlsp.sampling import sample_from_posterior
from deep_rlsp.util.dist import NormalDistribution, LaplaceDistribution
from deep_rlsp.solvers.value_iter import value_iter, evaluate_policy
from deep_rlsp.model import (
    InverseDynamicsMDN,
    ExperienceReplay,
    StateVAE,
)
from deep_rlsp.model.gridworlds_feature_space import GridworldsFeatureSpace
from deep_rlsp.envs.reward_wrapper import LatentSpaceRewardWrapper
from deep_rlsp.util.parameter_checks import check_in, check_not_none


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_id = "{}_{}_{}_{}_{}".format(
            timestamp,
            config["env_name"],
            config["problem_spec"],
            config["inference_algorithm"],
            config["temperature"],
        )

        if config["inference_algorithm"] == "rlsp":
            custom_id += "_" + config["solver"]
            if config["solver"] == "ppo":
                custom_id += "_{}_{}".format(
                    config["solver_iterations"], config["reset_solver"]
                )

        return custom_id  # started_event returns the _run._id


ex = Experiment("rlsp")
ex.observers = [SetID(), FileStorageObserver.create("results")]


def get_all_rewards_from_latent_space(
    env, latent_space, r_inferred, r_task, inferred_weight
):
    all_rewards_inferred = np.zeros(env.nS)
    for state_id in range(env.nS):
        obs = env.s_to_obs(env.get_state_from_num(state_id))
        state = latent_space.encoder(obs)
        reward = np.dot(r_inferred, state)
        all_rewards_inferred[state_id] = reward
    return env.f_matrix @ r_task + inferred_weight * all_rewards_inferred


def print_rollout(env, start_state, policies, last_steps_printed, horizon):
    if last_steps_printed == 0:
        last_steps_printed = horizon

    env.reset()
    env.s = start_state
    print("Executing the policy from state:")
    print(env.render("ansi"))
    print()
    print("Last {} of the {} rolled out steps:".format(last_steps_printed, horizon))

    for i in range(horizon - 1):
        s_num = env.get_num_from_state(env.s)
        a = np.random.choice(env.nA, p=policies[i][s_num, :])
        # a = env.reverse_action(a)
        _, reward, _, _ = env.step(a)

        if i >= (horizon - last_steps_printed - 1):
            print(env.render("ansi"))
            print("features", env.s_to_f(env.s))
            print("reward", reward)
            print()


def print_rollout_latent(env, policy, N, random=False, latent_space_policy=False):
    print("Executing the policy from state:")
    print(env.render("ansi"))
    print()
    total_reward = 0

    for i in range(N):
        print("Trajectory {}".format(i))
        obs = env.reset()
        done = False
        while not done:
            if random:
                a = env.action_space.sample()
            else:
                if latent_space_policy:
                    a, _ = policy.predict(
                        np.expand_dims(env.state, 0), deterministic=False
                    )
                else:
                    a, _ = policy.predict(np.expand_dims(obs, 0), deterministic=False)
            obs, reward, done, info = env.step(a, return_true_reward=True)
            total_reward += reward
            print("Action", a)
            print(env.render("ansi"))
            print(
                "Reward: {} (task {}, inferred {})".format(
                    reward, info["task"], info["inferred"]
                )
            )
            print()
    return total_reward / N


def forward_rl_solve(
    env, r_inferred, r_task, r_true, inferred_weight, latent_space, iterations
):
    """
    Learns a policy using PPO for a gridworld with a reward defined in the latent space.
    """
    env_task = LatentSpaceRewardWrapper(env, latent_space, None, r_task, r_true, 0)
    ppo_task = PPO2(MlpPolicy, DummyVecEnv([lambda: env_task]), verbose=1)
    ppo_task.learn(total_timesteps=iterations, log_interval=10)

    env_inferred = LatentSpaceRewardWrapper(
        env, latent_space, r_inferred, None, r_true, 1
    )
    ppo_inferred = PPO2(MlpPolicy, DummyVecEnv([lambda: env_inferred]), verbose=1)
    ppo_inferred.learn(total_timesteps=iterations, log_interval=10)

    env = LatentSpaceRewardWrapper(
        env, latent_space, r_inferred, r_task, r_true, inferred_weight
    )
    ppo = PPO2(MlpPolicy, DummyVecEnv([lambda: env]), verbose=1)
    ppo.learn(total_timesteps=iterations, log_interval=10)

    N = 10
    print()
    print("Random policy:")
    ret_rand = print_rollout_latent(env, None, N, random=True)
    print("Avg. Return", ret_rand)
    print()
    print("Policy trained only on task:")
    ret_task = print_rollout_latent(env, ppo_task, N)
    print("Avg. Return", ret_task)
    print()
    print("Policy trained only on inferred:")
    ret_inferred = print_rollout_latent(env, ppo_inferred, N)
    print("Avg. Return", ret_inferred)
    print()
    print("Policy trained on task + inferred:")
    ret = print_rollout_latent(env, ppo, N)
    print("Avg. Return", ret)
    print()
    print("Random policy return", ret_rand)
    print("Task-only policy return", ret_task)
    print("Inferred-only policy return", ret_inferred)
    print("Task+Inferred policy return", ret)
    return ret


def forward_rl(
    env,
    r_planning,
    r_true,
    h=40,
    temp=0,
    last_steps_printed=0,
    current_s_num=None,
    weight=1,
    penalize_deviation=False,
    relative_reachability=False,
    print_level=1,
    all_rewards=None,
):
    """
    Given an env and R, runs soft VI for h steps and rolls out the resulting policy
    """
    if all_rewards is None:
        r_s = env.f_matrix @ r_planning
    else:
        r_s = all_rewards

    current_state = env.get_state_from_num(current_s_num)
    time_dependent_reward = False

    if penalize_deviation:
        diff = env.f_matrix - env.s_to_f(current_state).T
        r_s -= weight * np.linalg.norm(diff, axis=1)
    if relative_reachability:
        time_dependent_reward = True
        r_r = relative_reachability_penalty(env, h, current_s_num)
        r_s = np.expand_dims(r_s, 0) - weight * r_r

    # For evaluation, plan optimally instead of Boltzmann-rationally
    policies = value_iter(
        env, 1, r_s, h, temperature=temp, time_dependent_reward=time_dependent_reward
    )

    # For print level >= 1, print a rollout
    if print_level >= 1:
        print_rollout(env, current_state, policies, last_steps_printed, h)

    return evaluate_policy(env, policies, current_s_num, 1, env.f_matrix @ r_true, h)


def get_problem_parameters(env_name, problem_name):
    check_in("env_name", env_name, TOY_ENV_CLASSES)
    check_in("problem_name", problem_name, TOY_PROBLEMS[env_name])
    spec, cur_state, r_task, r_true = TOY_PROBLEMS[env_name][problem_name]
    env = TOY_ENV_CLASSES[env_name](spec)
    return env, env.get_num_from_state(cur_state), r_task, r_true


def get_r_prior(prior, reward_center, std):
    check_in("prior", prior, ("gaussian", "laplace", "uniform"))
    if prior == "gaussian":
        return NormalDistribution(reward_center, std)
    elif prior == "laplace":
        return LaplaceDistribution(reward_center, std)
    return None


def check_parameters(args):
    """Check the parameters so that we fail fast."""
    inference_algorithm = args["inference_algorithm"]
    combination_algorithm = args["combination_algorithm"]
    measures = args["measures"]
    prior = args["prior"]
    inverse_dynamics_model_checkpoint = args["inverse_dynamics_model_checkpoint"]

    check_in(
        "inference_algorithm",
        inference_algorithm,
        [
            "rlsp",
            "latent_rlsp",
            "latent_rlsp_ablation",
            "sampling",
            "deviation",
            "reachability",
            "spec",
        ],
    )
    check_in(
        "combination_algorithm",
        combination_algorithm,
        ("additive", "bayesian", "latent_vi", "latent_ppo"),
    )
    check_in("prior", prior, ["gaussian", "laplace", "uniform"])

    for i, measure in enumerate(measures):
        check_in(
            "measure {}".format(i),
            measure,
            ["inferred_reward", "true_reward", "final_reward", "model_training_error"],
        )

    if combination_algorithm == "bayesian":
        check_in("inference_algorithm", inference_algorithm, ["rlsp", "sampling"])

    if inference_algorithm == "latent_rlsp":
        check_not_none(
            "inverse_dynamics_model_checkpoint", inverse_dynamics_model_checkpoint
        )

    if (
        combination_algorithm.startswith("latent")
        and inference_algorithm != "latent_rlsp"
    ):
        raise ValueError(
            "combination_algorithm 'latent' should only be used with 'latent_rlsp'"
        )


@ex.named_config
def latent_rlsp_config():
    inference_algorithm = "latent_rlsp"  # noqa:F841
    combination_algorithm = "latent_vi"  # noqa:F841
    prior = "gaussian"  # noqa:F841
    evaluation_horizon = 20  # noqa:F841
    learning_rate = 0.1  # noqa:F841
    inferred_weight = 1  # noqa:F841
    epochs = 50  # noqa:F841
    uniform_prior = False  # noqa:F841
    measures = [  # noqa:F841
        "inferred_reward",
        "final_reward",
        "true_reward",
        "model_training_error",
    ]
    std = 0.5  # noqa:F841
    print_level = 1  # noqa:F841
    soft_forward_rl = False  # noqa:F841
    reward_constant = 1.0  # noqa:F841
    check_grads = False  # noqa:F841
    n_trajectories = 100  # noqa:F841
    solver = "ppo"  # noqa:F841
    solver_iterations = 1000  # noqa:F841
    reset_solver = True  # noqa:F841
    create_heatmaps = False  # noqa:F841
    n_trajectories_forward_factor = 1.0  # noqa:F841


@ex.named_config
def latent_rlsp_ablation_config():
    inference_algorithm = "latent_rlsp_ablation"  # noqa:F841
    combination_algorithm = "latent_vi"  # noqa:F841
    prior = "gaussian"  # noqa:F841
    evaluation_horizon = 20  # noqa:F841
    learning_rate = 0.1  # noqa:F841
    inferred_weight = 1  # noqa:F841
    epochs = 50  # noqa:F841
    uniform_prior = False  # noqa:F841
    measures = [  # noqa:F841
        "inferred_reward",
        "final_reward",
        "true_reward",
    ]
    std = 0.5  # noqa:F841
    print_level = 1  # noqa:F841
    soft_forward_rl = False  # noqa:F841
    reward_constant = 1.0  # noqa:F841
    check_grads = False  # noqa:F841
    n_trajectories = 100  # noqa:F841
    solver = "ppo"  # noqa:F841
    solver_iterations = 1000  # noqa:F841
    reset_solver = True  # noqa:F841
    create_heatmaps = False  # noqa:F841
    n_trajectories_forward_factor = 1.0  # noqa:F841


@ex.named_config
def room_default():
    env_name = "room"  # noqa:F841
    problem_spec = "default"  # noqa:F841
    inverse_dynamics_model_checkpoint = (  # noqa:F841
        "tf_ckpt/tf_ckpt_mlp_RoomDefault-v0_20200930_123339"
    )
    latent_model_checkpoint = None  # noqa:F841
    combination_algorithm = "additive"  # noqa:F841
    horizon = 7  # noqa:F841


@ex.named_config
def room_bad():
    env_name = "room"  # noqa:F841
    problem_spec = "bad"  # noqa:F841
    inverse_dynamics_model_checkpoint = None  # noqa:F841
    latent_model_checkpoint = None  # noqa:F841
    combination_algorithm = "additive"  # noqa:F841
    horizon = 5  # noqa:F841


@ex.named_config
def train_default():
    env_name = "train"  # noqa:F841
    problem_spec = "default"  # noqa:F841
    inverse_dynamics_model_checkpoint = None  # noqa:F841
    latent_model_checkpoint = None  # noqa:F841
    combination_algorithm = "additive"  # noqa:F841
    horizon = 8  # noqa:F841


@ex.named_config
def apples_default():
    env_name = "apples"  # noqa:F841
    problem_spec = "default"  # noqa:F841
    inverse_dynamics_model_checkpoint = None  # noqa:F841
    latent_model_checkpoint = None  # noqa:F841
    combination_algorithm = "additive"  # noqa:F841
    horizon = 11  # noqa:F841


@ex.named_config
def batteries_easy():
    env_name = "batteries"  # noqa:F841
    problem_spec = "easy"  # noqa:F841
    inverse_dynamics_model_checkpoint = None  # noqa:F841
    latent_model_checkpoint = None  # noqa:F841
    combination_algorithm = "additive"  # noqa:F841
    horizon = 11  # noqa:F841


@ex.named_config
def batteries_default():
    env_name = "batteries"  # noqa:F841
    problem_spec = "default"  # noqa:F841
    inverse_dynamics_model_checkpoint = None  # noqa:F841
    latent_model_checkpoint = None  # noqa:F841
    combination_algorithm = "additive"  # noqa:F841
    horizon = 11  # noqa:F841


@ex.named_config
def test():
    env_name = "room"  # noqa:F841
    problem_spec = "default"  # noqa:F841
    inference_algorithm = "latent_rlsp"  # noqa:F841
    combination_algorithm = "latent_vi"  # noqa:F841
    prior = "gaussian"  # noqa:F841
    horizon = 20  # noqa:F841
    evaluation_horizon = 0  # noqa:F841
    temperature = 1.0  # noqa:F841
    learning_rate = 0.1  # noqa:F841
    inferred_weight = 1  # noqa:F841
    epochs = 50  # noqa:F841
    uniform_prior = False  # noqa:F841
    measures = [  # noqa:F841
        "inferred_reward",
        "final_reward",
        "true_reward",
        "model_training_error",
    ]
    std = 0.5  # noqa:F841
    print_level = 1  # noqa:F841
    soft_forward_rl = False  # noqa:F841
    reward_constant = 1.0  # noqa:F841
    check_grads = False  # noqa:F841
    n_trajectories = 10  # noqa:F841
    solver = "ppo"  # noqa:F841
    solver_iterations = 1000  # noqa:F841
    reset_solver = True  # noqa:F841
    learn_inverse_policy = True  # noqa:F841
    create_heatmaps = False  # noqa:F841
    n_trajectories_forward_factor = 1.0  # noqa:F841

    latent_model_checkpoint = None  # noqa:F841
    inverse_dynamics_model_checkpoint = (  # noqa:F841
        "tf_ckpt/tf_ckpt_mlp_RoomDefault-v0_20200929_130545"
    )


@ex.config
def config():
    # Environment to run: one of [room, apples, train, batteries]
    env_name = "room"  # noqa:F841
    # The name of the problem specification to solve.
    problem_spec = "default"  # noqa:F841
    # Frame condition inference algorithm:
    inference_algorithm = "spec"  # noqa:F841
    # How to combine the task reward and inferred reward for forward RL:
    # one of [additive, bayesian]
    # bayesian only has an effect if algorithm is rlsp or sampling.
    combination_algorithm = "additive"  # noqa:F841
    # Prior on the inferred reward function: one of [gaussian, laplace, uniform].
    # Centered at zero if combination_algorithm is additive, and at the task reward
    # if combination_algorithm is bayesian.
    # Only has an effect if inference_algorithm is rlsp or sampling.
    prior = "gaussian"  # noqa:F841
    # Number of timesteps we assume the human has been acting.
    horizon = 20  # noqa:F841
    # Number of timesteps we act after inferring the reward.
    evaluation_horizon = 0  # noqa:F841
    # Boltzmann rationality constant for the human.
    # Note this is temperature, which is the inverse of beta.
    temperature = 1.0  # noqa:F841
    # Learning rate for gradient descent. Applies when inference_algorithm is rlsp.
    learning_rate = 0.1  # noqa:F841
    # Weight for the inferred reward when adding task and inferred rewards.
    # Applies if combination_algorithm is additive.
    inferred_weight = 1  # noqa:F841
    # Number of gradient descent steps to take.
    epochs = 50  # noqa:F841
    # Use a uniform prior?
    uniform_prior = False  # noqa:F841
    # Dependent variables to measure and report
    measures = [  # noqa:F841
        "inferred_reward",
        "final_reward",
        "true_reward",
        "model_training_error",
    ]
    # Number of samples to generate with MCMC
    n_samples = 10000  # noqa:F841
    # Number of samples to ignore at the start
    mcmc_burn_in = 1000  # noqa:F841
    # Step size for computing neighbor reward functions.
    # Only has an effect if inference_algorithm is sampling.
    step_size = 0.01  # noqa:F841
    # Standard deviation for the prior
    std = 0.5  # noqa:F841
    # Level of verbosity.
    print_level = 1  # noqa:F841
    # False
    soft_forward_rl = False  # noqa:F841
    # Living reward provided when evaluating performance.
    reward_constant = 1.0  # noqa:F841
    # Whether to check gradients with scipy
    check_grads = False  # noqa:F841
    # Number of rollouts for approximating the rlsp gradient
    n_trajectories = 10  # noqa:F841
    # Which solver to use in the RLSP algorithm ('value_iter' or 'ppo')
    solver = "value_iter"  # noqa:F841
    # Number of iterations to use for ppo
    solver_iterations = 1000  # noqa:F841
    # Wheather to continue to update the same policy or reinitialize it in every step
    # (currently only used for PPO)
    reset_solver = False  # noqa:F841
    # Which transition model to use
    transition_model = "tabular"  # noqa:F841
    # Wheather to learn an inverse policy (instead of fully inverting a tabular policy)
    learn_inverse_policy = False  # noqa:F841
    latent_model_checkpoint = None  # noqa:F841
    inverse_dynamics_model_checkpoint = None  # noqa:F841
    forward_dynamics_model_checkpoint = None  # noqa:F841
    create_heatmaps = False  # noqa:F841
    n_trajectories_forward_factor = 1.0  # noqa:F841


@ex.automain
def main(
    _run,
    env_name,
    problem_spec,
    inference_algorithm,
    combination_algorithm,
    prior,
    horizon,
    evaluation_horizon,
    temperature,
    learning_rate,
    inferred_weight,
    epochs,
    uniform_prior,
    measures,
    n_samples,
    mcmc_burn_in,
    step_size,
    seed,
    std,
    print_level,
    soft_forward_rl,
    reward_constant,
    check_grads,
    n_trajectories,
    solver,
    solver_iterations,
    reset_solver,
    transition_model,
    learn_inverse_policy,
    latent_model_checkpoint,
    inverse_dynamics_model_checkpoint,
    forward_dynamics_model_checkpoint,
    create_heatmaps,
    n_trajectories_forward_factor,
):
    args = locals()
    print("--------")
    for key, val in args.items():
        print(key, val)
    print("--------")

    check_parameters(args)

    if evaluation_horizon == 0:
        evaluation_horizon = horizon

    np.random.seed(seed)
    env, s_current, r_task, r_true = get_problem_parameters(env_name, problem_spec)
    env.time_horizon = horizon

    print("Number of states: {}   ;   Number of actions: {}".format(env.nS, env.nA))
    print("r_true:", r_true)
    print("r_task:", r_task)

    if print_level >= 1:
        print("Initial state:")
        print(env.s_to_ansi(env.init_state))
        print("features", env.s_to_f(env.init_state))
        print()

    if print_level >= 1:
        print("Current state (for inference):")
        print(env.s_to_ansi(env.get_state_from_num(s_current)))
        print("features", env.s_to_f(env.get_state_from_num(s_current)))
        print()

    if inference_algorithm.startswith("latent_rlsp"):
        graph_latent, graph_bwd = tf.Graph(), tf.Graph()
        tf_graphs = {"latent": graph_latent, "inverse": graph_bwd}
        if latent_model_checkpoint is None:
            latent_space = GridworldsFeatureSpace(env)
        else:
            with graph_latent.as_default():
                latent_space = StateVAE.restore(latent_model_checkpoint)

    p_0 = env.get_initial_state_distribution(known_initial_state=not uniform_prior)

    deviation = inference_algorithm == "deviation"
    reachability = inference_algorithm == "reachability"
    if combination_algorithm == "bayesian":
        reward_center = r_task
    elif combination_algorithm.startswith("latent"):
        reward_center = np.zeros(latent_space.state_size)
    else:
        reward_center = np.zeros(env.num_features)

    r_prior = get_r_prior(prior, reward_center, std)
    model_training_error = dict()

    # Infer reward by observing the world state
    if inference_algorithm == "rlsp":
        r_inferred = rlsp(
            _run,
            env,
            s_current,
            p_0,
            horizon,
            temperature,
            epochs,
            learning_rate,
            r_prior,
            solver=solver,
            reset_solver=reset_solver,
            solver_iterations=solver_iterations,
        )
    elif inference_algorithm == "latent_rlsp":
        env = get_gym_gridworld(env_name, problem_spec)

        experience_replay = ExperienceReplay(10000)
        experience_replay.add_random_rollouts(env, env.spec.max_episode_steps, 1000)

        with graph_bwd.as_default():
            inverse_transition_model = InverseDynamicsMDN.restore(
                env, experience_replay, None, inverse_dynamics_model_checkpoint
            )

        def print_debugging_info(loc, glob):
            del glob
            step = loc["epoch"]
            _run.log_scalar("single_grad_norm", np.linalg.norm(loc["dL_dr_vec"]), step)
            _run.log_scalar("mean_grad_norm", loc["grad_norm"], step)

        inverse_model_parameters = {"model": dict(), "learn": dict()}  # noqa:F841
        latent_space_model_parameters = {"model": dict(), "learn": dict()}  # noqa:F841
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

        reweight_gradient = False

        r_inferred = latent_rlsp(
            _run,
            env,
            [env.s_to_obs(env.get_state_from_num(s_current))],
            # the old RLSP code seems to consider trajectories (s1, ..., sT) (length T)
            # the approximation uses (s0, ..., sT) which is why we use T-1 here
            horizon - 1,
            experience_replay,
            latent_space,
            inverse_transition_model,
            epochs=epochs,
            learning_rate=learning_rate,
            r_prior=r_prior,
            # r_vec=np.zeros(latent_space.state_size),
            n_trajectories=n_trajectories,
            reset_solver=reset_solver,
            solver_iterations=solver_iterations,
            print_level=print_level,
            n_trajectories_forward_factor=n_trajectories_forward_factor,
            callback=print_debugging_info,
            continue_training_dynamics=False,
            continue_training_latent_space=False,
            tf_graphs=tf_graphs,
            clip_mujoco_obs=False,
            horizon_curriculum=False,
            inverse_model_parameters=inverse_model_parameters,
            latent_space_model_parameters=latent_space_model_parameters,
            inverse_policy_parameters=inverse_policy_parameters,
            reweight_gradient=reweight_gradient,
            init_from_policy=None,
            solver_str=solver,
        )

        r_inferred /= np.linalg.norm(r_inferred)
    elif inference_algorithm == "latent_rlsp_ablation":
        r_inferred = latent_space.encoder(
            env.s_to_obs(env.get_state_from_num(s_current))
        )
        r_inferred /= np.linalg.norm(r_inferred)
    elif inference_algorithm == "sampling":
        r_samples = sample_from_posterior(
            env,
            s_current,
            p_0,
            horizon,
            temperature,
            n_samples,
            step_size,
            r_prior,
            gamma=1,
            print_level=print_level,
        )
        r_inferred = np.mean(r_samples[mcmc_burn_in::], axis=0)
    else:
        r_inferred = None

    if print_level >= 1 and r_inferred is not None:
        with np.printoptions(precision=4, suppress=True):
            print()
            print("Inferred reward vector: ", r_inferred)

    # Run forward RL to evaluate
    def evaluate(forward_rl_temp):
        if combination_algorithm == "additive":
            r_final = r_task
            if r_inferred is not None:
                r_final = r_task + inferred_weight * r_inferred
            true_reward_obtained = forward_rl(
                env,
                r_final,
                r_true,
                temp=forward_rl_temp,
                h=evaluation_horizon,
                current_s_num=s_current,
                weight=inferred_weight,
                penalize_deviation=deviation,
                relative_reachability=reachability,
                print_level=print_level,
            )
        elif combination_algorithm == "bayesian":
            assert r_inferred is not None
            assert (not deviation) and (not reachability)
            r_final = r_inferred
            true_reward_obtained = forward_rl(
                env,
                r_final,
                r_true,
                temp=forward_rl_temp,
                h=evaluation_horizon,
                current_s_num=s_current,
                penalize_deviation=False,
                relative_reachability=False,
                print_level=print_level,
            )
        elif combination_algorithm == "latent_ppo":
            assert r_inferred is not None
            r_final = r_inferred
            true_reward_obtained = forward_rl_solve(
                env,
                r_inferred,
                r_task,
                r_true,
                inferred_weight,
                latent_space,
                iterations=1000,
            )
        elif combination_algorithm == "latent_vi":
            assert r_inferred is not None
            r_final = r_inferred
            all_rewards = get_all_rewards_from_latent_space(
                env, latent_space, r_inferred, r_task, inferred_weight
            )
            print("mean reward", np.mean(all_rewards))
            if create_heatmaps:
                heatmaps_folder = "results/heatmaps_{}_{}/".format(
                    env_name, problem_spec
                )
                os.makedirs(heatmaps_folder, exist_ok=True)
                env.make_reward_heatmaps(all_rewards, heatmaps_folder + "heatmap_")
            true_reward_obtained = forward_rl(
                env,
                None,
                r_true,
                temp=forward_rl_temp,
                h=evaluation_horizon,
                current_s_num=s_current,
                penalize_deviation=False,
                relative_reachability=False,
                print_level=print_level,
                all_rewards=all_rewards,
            )

        best_possible_reward = forward_rl(
            env,
            r_true,
            r_true,
            temp=forward_rl_temp,
            h=evaluation_horizon,
            current_s_num=s_current,
            print_level=0,
        )

        # Add the reward constant in
        true_reward_obtained += reward_constant * evaluation_horizon
        best_possible_reward += reward_constant * evaluation_horizon

        def get_measure(measure):
            if measure == "inferred_reward":
                return list(r_inferred) if r_inferred is not None else None
            elif measure == "final_reward":
                return list(r_final)
            elif measure == "true_reward":
                return true_reward_obtained * 1.0 / best_possible_reward
            return model_training_error

        return [get_measure(measure) for measure in measures]

    if soft_forward_rl:
        temperatures = [0, 0.1, 0.5, 1, 5, 10]
        evaluation = [evaluate(temp) for temp in [0, 0.1, 0.5, 1, 5, 10]]
    else:
        temperatures = [0]
        evaluation = [evaluate(0.0)]

    experiment_result = dict([(mes, []) for mes in measures])
    for temp in temperatures:
        results = evaluate(temp)
        for mes, res in zip(measures, results):
            experiment_result[mes].append(res)

    print()
    print("Results:")
    print("--------------")
    with np.printoptions(precision=4, suppress=True, threshold=10):
        for mes, result in experiment_result.items():
            print(mes, result)
    print("--------------")

    return experiment_result
