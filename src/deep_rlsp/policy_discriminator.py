import pickle
import datetime

import gym

import numpy as np
import tensorflow as tf

from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from deep_rlsp.util.parameter_checks import check_less_equal
from deep_rlsp.util.train import (
    get_learning_rate,
    tensorboard_log_gradients,
    get_tf_session,
)
from deep_rlsp.util.helper import load_data


class PolicyDiscriminator:
    def __init__(
        self,
        input_size,
        hidden_layer_size=512,
        n_hidden_layers=3,
        learning_rate=3e-4,
        tensorboard_log=None,
    ):
        self.input_size = input_size

        self._hidden_layer_size = hidden_layer_size
        self._n_hidden_layers = n_hidden_layers

        self._define_input_placeholders()
        self._define_model()

        self.loss = self._define_loss()
        self.learning_rate, self.global_step = get_learning_rate(learning_rate, None, 1)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients = self.optimizer.compute_gradients(loss=self.loss)
        self.optimization_op = self.optimizer.apply_gradients(
            self.gradients, global_step=self.global_step
        )

        self.tensorboard_log = tensorboard_log
        if self.tensorboard_log is not None:
            self._define_tensorboard_metrics()

        self.sess = None

    def _define_input_placeholders(self):
        self.in_trajectory = tf.placeholder(
            tf.float32, (None, self.input_size), name="trajectory"
        )
        self.in_policy_label = tf.placeholder(
            tf.float32, (None, 2), name="policy_label"
        )

    def _define_model(self):
        activation = tf.nn.relu

        x = self.in_trajectory
        for i in range(self._n_hidden_layers):
            x = tf.keras.layers.Dense(
                self._hidden_layer_size,
                activation=activation,
                name="hidden_{}".format(i + 1),
            )(x)

        self.out_policy_logits = tf.keras.layers.Dense(
            2, activation=None, name="out_policy_logits"
        )(x)

    def _define_loss(self):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.in_policy_label, logits=self.out_policy_logits
            )
        )

    def _define_tensorboard_metrics(self):
        tf.summary.scalar("loss/loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tensorboard_log_gradients(self.gradients)

    def learn(self, trajectories, labels, n_epochs=1, batch_size=16, verbose=True):
        """
        Main training loop
        """
        n_samples = len(labels)
        check_less_equal("batch_size", batch_size, n_samples)
        n_batches = n_samples // batch_size

        if self.sess is None:
            self.sess = get_tf_session()
            self.sess.run(tf.global_variables_initializer())

        if self.tensorboard_log is not None:
            summaries_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(
                self.tensorboard_log, self.sess.graph
            )
        else:
            summaries_op = tf.no_op()

        losses = []
        for epoch in range(n_epochs):
            for batch in range(n_batches):
                batch_trajectory = trajectories[
                    batch * batch_size : (batch + 1) * batch_size
                ]
                batch_policy_label = labels[
                    batch * batch_size : (batch + 1) * batch_size
                ]

                (batch_loss, _, batch_lr, summary, step) = self.sess.run(
                    [
                        self.loss,
                        self.optimization_op,
                        self.learning_rate,
                        summaries_op,
                        self.global_step,
                    ],
                    feed_dict={
                        self.in_trajectory: batch_trajectory,
                        self.in_policy_label: batch_policy_label,
                    },
                )

                if self.tensorboard_log is not None:
                    summary_writer.add_summary(summary, step)

                losses.append(batch_loss)

                if verbose:
                    print(
                        "Epoch: {}/{}...".format(epoch + 1, n_epochs),
                        "Batch: {}/{}...".format(batch + 1, n_batches),
                        "Training loss: {:.4f}  ".format(batch_loss),
                        "(learning_rate = {:.6f})".format(batch_lr),
                    )
        return losses


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return timestamp  # started_event returns the _run._id


ex = Experiment("policy-discriminator")
ex.observers = [SetID(), FileStorageObserver.create("results/policy_discriminator")]


def get_trajectories(env, policy_path, policy_type, n_rollouts, time_horizon):
    if policy_type == "sac":
        from stable_baselines import SAC

        model = SAC.load(policy_path)

        def get_action(obs):
            return model.predict(obs, deterministic=True)[0]

    elif policy_type == "gail":
        from imitation.policies import serialize
        from stable_baselines3.common.vec_env import DummyVecEnv

        venv = DummyVecEnv([lambda: env])
        model = serialize.load_policy("ppo", policy_path, venv)

        def get_action(obs):
            return model.predict(obs)[0]

    elif policy_type == "dads":
        data = load_data(policy_path)
        return data["observations"]
    else:
        raise NotImplementedError()

    trajectories = []

    for _ in range(n_rollouts):
        trajectory = []
        obs = env.reset()
        trajectory.append(list(obs))
        for t in range(time_horizon - 1):
            action = get_action(obs)
            # trajectory.extend(list(action))
            obs, reward, done, info = env.step(action)
            trajectory.append(list(obs))
        trajectories.append(trajectory)

    return trajectories


def get_samples_from_trajectories(trajectories, sample_length):
    samples = []
    obs_len = len(trajectories[0][0])
    for trajectory in trajectories:
        sample = []
        for obs in trajectory:
            sample.extend(obs)
            if len(sample) == sample_length * obs_len:
                samples.append(list(sample))
                sample = []
    return samples


@ex.config
def config():
    policy1_path = None  # noqa:F841
    policy1_type = None  # noqa:F841
    policy2_paths = None  # noqa:F841
    policy2_type = None  # noqa:F841
    loss_out_file = None  # noqa:F841
    env_label = None  # noqa:F841

    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841
    n_rollouts = 300  # noqa:F841
    n_epochs = 50  # noqa:F841
    batch_size = 50  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    time_horizon = 1000  # noqa:F841
    sample_length = 50  # noqa:F841
    n_seeds = 10  # noqa:F841


@ex.named_config
def hopper_gail_10():
    env_label = "Hopper-FW-v2"  # noqa:F841
    policy1_path = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/sac_hopper_2e6.zip"
    )
    policy1_type = "sac"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/gail_envs/gail_hopper_len_10_demoseed_647574168",  # noqa:E501
        "/home/david/tmp/git/deep-rlsp/policies/gail_envs/gail_hopper_len_10_demoseed_700423463",  # noqa:E501
        "/home/david/tmp/git/deep-rlsp/policies/gail_envs/gail_hopper_len_10_demoseed_750917369",  # noqa:E501
    )
    policy2_type = "gail"  # noqa:F841
    loss_out_file = "hopper_gail_10.pkl"  # noqa:F841


@ex.named_config
def hopper_deep_rlsp_10():
    env_label = "Hopper-FW-v2"  # noqa:F841
    policy1_path = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/sac_hopper_2e6.zip"
    )
    policy1_type = "sac"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/deep_rlsp/hopper/eval/20201001_034850_20200929_102643_Hopper-FW-v2_optimal_10_243986/policy.zip",  # noqa:E501
        "/home/david/tmp/git/deep-rlsp/iclr_results/deep_rlsp/hopper/eval/20201001_034851_20200929_102650_Hopper-FW-v2_optimal_10_608888/policy.zip",  # noqa:E501
        "/home/david/tmp/git/deep-rlsp/iclr_results/deep_rlsp/hopper/eval/20201001_034850_20200929_102638_Hopper-FW-v2_optimal_10_928453/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "hopper_deep_rlsp_10.pkl"  # noqa:F841


@ex.named_config
def hopper_ablation_average_features_10():
    env_label = "Hopper-FW-v2"  # noqa:F841
    policy1_path = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/sac_hopper_2e6.zip"
    )
    policy1_type = "sac"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/hopper/ablation_average_features/20200930_051245_ablation_average_features_20200929_102643_Hopper-FW-v2_optimal_10_243986/policy.zip",  # noqa:E501
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/hopper/ablation_average_features/20200930_051241_ablation_average_features_20200929_102650_Hopper-FW-v2_optimal_10_608888/policy.zip",  # noqa:E501
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/hopper/ablation_average_features/20200930_051250_ablation_average_features_20200929_102638_Hopper-FW-v2_optimal_10_928453/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "hopper_ablation_average_features_10.pkl"  # noqa:F841


@ex.named_config
def hopper_ablation_waypoints_10():
    env_label = "Hopper-FW-v2"  # noqa:F841
    policy1_path = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/sac_hopper_2e6.zip"
    )
    policy1_type = "sac"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/hopper/ablation_waypoints/20200930_143907_ablation_waypoints_20200929_102643_Hopper-FW-v2_optimal_10_243986/policy.zip",  # noqa:E501
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/hopper/ablation_waypoints/20200930_143901_ablation_waypoints_20200929_102650_Hopper-FW-v2_optimal_10_608888/policy.zip",  # noqa:E501
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/hopper/ablation_waypoints/20200930_143910_ablation_waypoints_20200929_102638_Hopper-FW-v2_optimal_10_928453/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "hopper_ablation_waypoints_10.pkl"  # noqa:F841


# CHEETAH BALANCING
@ex.named_config
def cheetah_balancing_1_gail():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/gail_skills/gail_balancing_len_1",
    )
    policy2_type = "gail"  # noqa:F841

    loss_out_file = "discriminator_cheetah_balancing_1_gail.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_1_rlsp():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/deep_rlsp/cheetah_balancing/20201001_175259_HalfCheetah-v2_trajectory_file_1_89302/rlsp_policy_300.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_balancing_1_rlsp.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_1_waypoints():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_balancing/ablation_waypoints/20201001_202954_ablation_waypoints_20201001_175259_HalfCheetah-v2_trajectory_file_1_89302/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_balancing_1_waypoints.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_1_average_features():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_balancing/ablation_average_features/20201001_202934_ablation_average_features_20201001_175259_HalfCheetah-v2_trajectory_file_1_89302/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = (  # noqa:F841
        "discriminator_cheetah_balancing_1_average_features.pkl"
    )
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_10_gail():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/gail_skills/gail_balancing_len_10",
    )
    policy2_type = "gail"  # noqa:F841

    loss_out_file = "discriminator_cheetah_balancing_10_gail.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_10_rlsp():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/deep_rlsp/cheetah_balancing/20201001_175257_HalfCheetah-v2_trajectory_file_10_36270/rlsp_policy_296.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_balancing_10_rlsp.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_10_waypoints():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_balancing/ablation_waypoints/20201001_203036_ablation_waypoints_20201001_175257_HalfCheetah-v2_trajectory_file_10_36270/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_balancing_10_waypoints.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_10_average_features():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_balancing/ablation_average_features/20201001_203009_ablation_average_features_20201001_175257_HalfCheetah-v2_trajectory_file_10_36270/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = (  # noqa:F841
        "discriminator_cheetah_balancing_10_average_features.pkl"
    )
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_50_gail():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/gail_skills/gail_balancing_len_50",
    )
    policy2_type = "gail"  # noqa:F841

    loss_out_file = "discriminator_cheetah_balancing_50_gail.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_50_rlsp():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/deep_rlsp/cheetah_balancing/20201001_175305_HalfCheetah-v2_trajectory_file_50_703486/rlsp_policy_293.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_balancing_50_rlsp.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_50_waypoints():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_balancing/ablation_waypoints/20201001_203104_ablation_waypoints_20201001_175305_HalfCheetah-v2_trajectory_file_50_703486/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_balancing_50_waypoints.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_balancing_50_average_features():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/one_leg_moving_0_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_balancing/ablation_average_features/20201001_203050_ablation_average_features_20201001_175305_HalfCheetah-v2_trajectory_file_50_703486/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = (  # noqa:F841
        "discriminator_cheetah_balancing_50_average_features.pkl"
    )
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


# CHEETAH JUMPING
@ex.named_config
def cheetah_jumping_1_gail():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/gail_skills/gail_jumping_len_1",
    )
    policy2_type = "gail"  # noqa:F841

    loss_out_file = "discriminator_cheetah_jumping_1_gail.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_1_rlsp():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/deep_rlsp/cheetah_jumping/20201001_175249_HalfCheetah-v2_trajectory_file_1_718930/rlsp_policy_300.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_jumping_1_rlsp.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_1_waypoints():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_jumping/ablation_waypoints/20201001_195000_ablation_waypoints_20201001_175249_HalfCheetah-v2_trajectory_file_1_718930/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_jumping_1_waypoints.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_1_average_features():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_jumping/ablation_average_features/20201001_194816_ablation_average_features_20201001_175249_HalfCheetah-v2_trajectory_file_1_718930/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_jumping_1_average_features.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_10_gail():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/gail_skills/gail_jumping_len_10",
    )
    policy2_type = "gail"  # noqa:F841

    loss_out_file = "discriminator_cheetah_jumping_10_gail.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_10_rlsp():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/deep_rlsp/cheetah_jumping/20201001_175251_HalfCheetah-v2_trajectory_file_10_725505/rlsp_policy_291.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_jumping_10_rlsp.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_10_waypoints():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_jumping/ablation_waypoints/20201001_195034_ablation_waypoints_20201001_175251_HalfCheetah-v2_trajectory_file_10_725505/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_jumping_10_waypoints.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_10_average_features():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_jumping/ablation_average_features/20201001_194920_ablation_average_features_20201001_175251_HalfCheetah-v2_trajectory_file_10_725505/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_jumping_10_average_features.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_50_gail():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/policies/gail_skills/gail_jumping_len_50",
    )
    policy2_type = "gail"  # noqa:F841

    loss_out_file = "discriminator_cheetah_jumping_50_gail.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_50_rlsp():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/deep_rlsp/cheetah_jumping/20201001_175256_HalfCheetah-v2_trajectory_file_50_948364/rlsp_policy_300.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_jumping_50_rlsp.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_50_waypoints():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_jumping/ablation_waypoints/20201001_195053_ablation_waypoints_20201001_175256_HalfCheetah-v2_trajectory_file_50_948364/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_jumping_50_waypoints.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.named_config
def cheetah_jumping_50_average_features():
    env_label = "HalfCheetah-v2"  # noqa:F841
    policy1_path = "/home/david/tmp/git/deep-rlsp/play_data/halfcheetah_skills/jumping_4_0.pkl"  # noqa:F841,E501
    policy1_type = "dads"  # noqa:F841

    policy2_paths = (  # noqa:F841
        "/home/david/tmp/git/deep-rlsp/iclr_results/ablations/cheetah_jumping/ablation_average_features/20201001_194932_ablation_average_features_20201001_175256_HalfCheetah-v2_trajectory_file_50_948364/policy.zip",  # noqa:E501
    )
    policy2_type = "sac"  # noqa:F841
    loss_out_file = "discriminator_cheetah_jumping_50_average_features.pkl"  # noqa:F841
    n_rollouts = 1  # noqa:F841
    time_horizon = 100  # noqa:F841
    sample_length = 5  # noqa:F841
    batch_size = 40  # noqa:F841
    n_epochs = 1000  # noqa:F841
    learning_rate = 1e-4  # noqa:F841
    n_hidden_layers = 1  # noqa:F841
    hidden_layer_size = 10  # noqa:F841


@ex.automain
def main(
    _run,
    policy1_path,
    policy1_type,
    policy2_paths,
    policy2_type,
    env_label,
    n_hidden_layers,
    hidden_layer_size,
    n_rollouts,
    n_epochs,
    batch_size,
    learning_rate,
    time_horizon,
    sample_length,
    loss_out_file,
    n_seeds,
    seed,
):

    env = gym.make(env_label)

    avg_losses = None

    for policy2_path in policy2_paths:
        for _ in range(n_seeds):
            trajectories_1 = get_trajectories(
                env, policy1_path, policy1_type, n_rollouts, time_horizon
            )
            trajectories_2 = get_trajectories(
                env, policy2_path, policy2_type, n_rollouts, time_horizon
            )
            trajectories_1 = get_samples_from_trajectories(
                trajectories_1, sample_length
            )
            trajectories_2 = get_samples_from_trajectories(
                trajectories_2, sample_length
            )

            trajectories = trajectories_1 + trajectories_2
            labels = [[1, 0]] * len(trajectories_1) + [[0, 1]] * len(trajectories_2)
            idx = np.arange(len(trajectories))
            np.random.shuffle(idx)
            trajectories = np.array(trajectories)[idx]
            labels = np.array(labels)[idx]

            input_size = len(trajectories[0])
            policy_discriminator = PolicyDiscriminator(
                input_size,
                hidden_layer_size=hidden_layer_size,
                n_hidden_layers=n_hidden_layers,
                learning_rate=learning_rate,
                tensorboard_log=None,
            )

            losses = policy_discriminator.learn(
                trajectories,
                labels,
                n_epochs=n_epochs,
                batch_size=batch_size,
                verbose=True,
            )

            if avg_losses is None:
                avg_losses = np.array(losses)
            else:
                avg_losses += np.array(losses)

    avg_losses /= n_seeds * len(policy2_paths)
    with open(loss_out_file, "wb") as f:
        pickle.dump(avg_losses, f)
