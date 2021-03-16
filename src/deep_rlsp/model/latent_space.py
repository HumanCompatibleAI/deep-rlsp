import os
import json

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from deep_rlsp.model.rssm import RSSM
from deep_rlsp.envs.gridworlds.env import action_id_to_string, Direction
from deep_rlsp.util.parameter_checks import (
    check_in,
    check_greater_equal,
    check_between,
    check_less_equal,
)
from deep_rlsp.util.train import (
    get_learning_rate,
    tensorboard_log_gradients,
    get_tf_session,
)
from deep_rlsp.util.video import render_mujoco_from_obs, save_video


def get_batch(data, batch, batch_size, T):
    observations, actions = data
    batch_observations, batch_actions = [], []
    for i in range(batch * batch_size, (batch + 1) * batch_size):
        start = np.random.randint(0, len(observations[i]) - T)
        batch_observations.append(observations[i][start : start + T])
        batch_actions.append(actions[i][start : start + T])
    batch_actions = np.array(batch_actions)
    batch_observations = np.array(batch_observations)
    return batch_observations, batch_actions


def extract_play_data(play_data, T=0):
    observations, actions = play_data["observations"], play_data["actions"]
    lengths = map(len, observations)
    if min(lengths) < T:
        raise ValueError(
            "Dataset must only contain trajectories with length > {}".format(T)
        )
    # shuffle trajectories
    n_traj = len(observations)
    assert len(actions) == n_traj
    shuffled = np.arange(n_traj)
    np.random.shuffle(shuffled)
    return np.array(observations)[shuffled], np.array(actions)[shuffled]


class LatentSpaceModel:
    """
    Learns a latent state space and a transition model, similar to [1].

    [1] Hafner, Danijar, et al. "Learning latent dynamics for planning from pixels."
        arXiv preprint arXiv:1811.04551 (2018).
    """

    def __init__(
        self,
        env,
        tensorboard_log=None,
        checkpoint_folder=None,
        observation_dist="gaussian",
        pixel_observations=False,
        hidden_layer_size=200,
        n_hidden_layers=2,
        rnn_state_size=30,
        learning_rate=3e-4,
        obs_stddev=0.01,
        likelihood_scale=1,
        mujoco_video_out_path="mujoco",
        timesteps_training=20,
        fixed_latent_stddev=None,
    ):
        assert isinstance(env.action_space, gym.spaces.Box)
        self.env = env
        self.action_space_shape = list(self.env.action_space.shape)

        self.pixel_observations = pixel_observations
        self.observation_dist = observation_dist

        if self.pixel_observations:
            self.data_shape = [64, 64, 3]
            self._obs_stddev = 1
        else:
            self.data_shape = list(self.env.observation_space.shape)

        self._obs_stddev = obs_stddev
        self._rnn_state_size = rnn_state_size
        self._num_layers = n_hidden_layers
        self._hidden_layer_size = hidden_layer_size
        self._min_stddev = 0.01
        self.likelihood_scale = 1  # 1e-2
        self._fixed_latent_stddev = fixed_latent_stddev

        self.state_size = 3 * self._rnn_state_size

        self.mujoco_video_out_path = mujoco_video_out_path
        self.timesteps = timesteps_training
        self.N_samples = 3
        self.N_samples_for_gradient = 1
        check_greater_equal("N_samples", self.N_samples, self.N_samples_for_gradient)

        if self.pixel_observations and self.observation_dist != "gaussian":
            raise ValueError(
                'Pixel observations require the observation model to be "gaussian"'
            )

        self.time_axis = tf.convert_to_tensor([1])

        self._define_input_placeholders()
        self._define_model()

        self.loss = self._define_loss()
        self.learning_rate, self.global_step = get_learning_rate(learning_rate, None, 1)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-4)
        self.gradients = self.optimizer.compute_gradients(loss=self.loss)
        self.gradients = [
            (tf.clip_by_norm(grad, 1000), var) for grad, var in self.gradients
        ]
        self.optimization_op = self.optimizer.apply_gradients(
            self.gradients, global_step=self.global_step
        )

        self.tensorboard_log = tensorboard_log
        if self.tensorboard_log is not None:
            self._define_tensorboard_metrics()

        self.checkpoint_folder = checkpoint_folder
        if self.checkpoint_folder is not None:
            self.saver = tf.train.Saver()

        self.policy = None
        self.inverse_policy = None
        self.sess = None

    def models_observations(self):
        return True

    def update_policy(self, policy, inverse_policy):
        self.policy = policy
        self.inverse_policy = inverse_policy

    def latent_step(self, state, action, sample=False):
        if sample:
            next_state_tensor = self.next_state_from_transition_sample
        else:
            next_state_tensor = self.next_state_from_transition_mean
        state = np.expand_dims(state, 0)
        action = np.expand_dims(action, 0)
        (next_state,) = self.sess.run(
            [next_state_tensor],
            feed_dict={
                self.state_for_transition: state,
                self.action_for_transition: action,
            },
        )
        return next_state[0]

    def encoder(self, obs):
        obs = np.expand_dims(obs, 0)
        batch_observations = np.reshape(obs, [-1, 1] + list(obs.shape[1:]))
        batch = batch_observations.shape[0]
        batch_actions = np.zeros([batch, 1] + self.action_space_shape)
        # batch_actions[0, 0, 4] = 1  # stay at t=0

        (encoded_state,) = self.sess.run(
            [self.encoded_state],
            feed_dict={
                self.in_obs_seq: batch_observations,
                self.in_actions: batch_actions,
                self.horizon: 1,
            },
        )
        state = encoded_state[:, 0, :]
        assert state.shape[-1] == self.state_size
        return state[0]

    def decoder(self, state):
        state = np.expand_dims(state, 0)
        batch_states = np.reshape(state, (-1, 1, self.state_size))
        (obs,) = self.sess.run(
            [self.state_decoded], feed_dict={self.state_to_decode: batch_states}
        )
        return obs[0, 0, :]

    def _collect_data(self, n_rollouts, debug_only_stay=False):
        observations, actions = [], []
        for _ in range(n_rollouts):
            traj_len = 0
            # ensure trajectories are longer than self.timesteps
            while traj_len < self.timesteps:
                obs = self.env.reset()
                traj_act = []
                traj_obs = [obs]
                done = False
                while not done:
                    if debug_only_stay:
                        action = Direction.get_number_from_direction(Direction.STAY)
                    else:
                        action = self.env.action_space.sample()
                    obs, _, done, _ = self.env.step(action)
                    traj_obs.append(obs)
                    traj_act.append(action)
                traj_len = len(traj_obs)
            traj_act.append(np.zeros(self.action_space_shape))
            observations.append(traj_obs)
            actions.append(traj_act)
        return observations, actions

    def _define_input_placeholders(self):
        self.in_obs_seq = tf.placeholder(
            tf.float32, [None, None] + self.data_shape, name="state"
        )
        self.in_actions = tf.placeholder(
            tf.float32, [None, None] + self.action_space_shape, name="action"
        )
        self.horizon = tf.placeholder(tf.int32, (), name="horizon")
        self.state_to_decode = tf.placeholder(
            tf.float32, (None, None, self.state_size), name="state_to_decode"
        )
        self.state_for_transition = tf.placeholder(
            tf.float32, (1, self.state_size), name="state_for_transition"
        )
        self.action_for_transition = tf.placeholder(
            tf.float32, [1] + self.action_space_shape, name="action_for_transition"
        )

    def _define_model(self):
        """
        Defines the model architecture.
        """

        def encoder(obs):
            """Extract deterministic features from an observation [1]."""
            timesteps = tf.shape(obs)[1]
            if self.pixel_observations:
                # pixel encoder based on encoder [1] for deepmind lab envs
                kwargs = dict(strides=2, activation=tf.nn.relu)
                hidden = tf.reshape(obs, [-1] + obs.shape[2:].as_list())
                hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs)
                hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs)
                hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs)
                hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs)
                hidden = tf.layers.flatten(hidden)
                hidden = tf.reshape(
                    hidden, [-1, timesteps] + [np.prod(hidden.shape[1:].as_list())]
                )
            else:
                # encoder for gridworlds / structured observations
                hidden = tf.reshape(obs, [-1, timesteps, np.prod(self.data_shape)])
                for _ in range(3):
                    hidden = tf.layers.dense(
                        hidden, self._hidden_layer_size, tf.nn.relu
                    )
            return hidden

        def decoder(state_sample, observation_dist="gaussian"):
            """Compute the data distribution of an observation from its state [1]."""
            check_in(
                "observation_dist",
                observation_dist,
                ("gaussian", "laplace", "bernoulli", "multinomial"),
            )

            timesteps = tf.shape(state_sample)[1]

            if self.pixel_observations:
                # original decoder from [1] for deepmind lab envs
                hidden = tf.layers.dense(state_sample, 1024, None)
                kwargs = dict(strides=2, activation=tf.nn.relu)
                hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1]])
                # 1 x 1
                hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs)
                # 5 x 5 x 128
                hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
                # 13 x 13 x 64
                hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
                # 30 x 30 x 32
                mean = 255 * tf.layers.conv2d_transpose(
                    hidden, 3, 6, strides=2, activation=tf.nn.sigmoid
                )
                # 64 x 64 x 3
                assert mean.shape[1:].as_list() == [64, 64, 3], mean.shape
            else:
                # decoder for gridworlds / structured observations
                hidden = state_sample
                d = self._hidden_layer_size
                for _ in range(4):
                    hidden = tf.layers.dense(hidden, d, tf.nn.relu)
                mean = tf.layers.dense(hidden, np.prod(self.data_shape), None)

            mean = tf.reshape(mean, [-1, timesteps] + list(self.data_shape))

            check_in(
                "observation_dist",
                observation_dist,
                ("gaussian", "laplace", "bernoulli", "multinomial"),
            )
            if observation_dist == "gaussian":
                dist = tfd.Normal(mean, self._obs_stddev)
            elif observation_dist == "laplace":
                dist = tfd.Laplace(mean, self._obs_stddev / np.sqrt(2))
            elif observation_dist == "bernoulli":
                dist = tfd.Bernoulli(probs=mean)
            else:
                mean = tf.reshape(
                    mean, [-1, timesteps] + [np.prod(list(self.data_shape))]
                )
                dist = tfd.Multinomial(total_count=1, probs=mean)
                reshape = tfp.bijectors.Reshape(event_shape_out=list(self.data_shape))
                dist = reshape(dist)
                return dist

            dist = tfd.Independent(dist, len(self.data_shape))
            return dist

        self._encoder = tf.make_template("encoder", encoder)
        self._decoder = tf.make_template("decoder", decoder)

        embedded = self._encoder(self.in_obs_seq)

        # create mask `use_obs` that is 1 for t < horizon and 0 else
        shape = tf.shape(embedded)
        timesteps = tf.range(shape[1])
        t_idx = tf.reshape(timesteps, (1, -1))
        t_idx = tf.tile(t_idx, [shape[0], 1])
        t_idx = tf.expand_dims(t_idx, -1)
        use_obs = t_idx < self.horizon

        # Note DL: tf.keras.layers.RNN does not seem to work with the state being a dict
        # which is why we use the deprecated dynamic_rnn here

        inputs = (embedded, self.in_actions, use_obs)
        with tf.variable_scope("rssm"):
            self.rssm_cell = RSSM(
                self._rnn_state_size,
                self._rnn_state_size,
                self._rnn_state_size,
                min_stddev=self._min_stddev,
                fixed_latent_stddev=self._fixed_latent_stddev,
                mean_only=True,
            )
            (self.prior, self.posterior), _ = tf.nn.dynamic_rnn(
                self.rssm_cell, inputs, dtype=tf.float32
            )

        self.encoded_state = tf.concat(
            [
                self.posterior["rnn_state"],
                self.posterior["mean"],
                self.posterior["belief"],
            ],
            -1,
        )

        # reconstruction for training
        posterior_dist = self.rssm_cell.dist_from_state(self.posterior)
        self.reconstruction_dist = []
        belief = self.posterior["belief"]
        for _ in range(self.N_samples):
            sample = posterior_dist.sample()
            features = tf.concat([sample, belief], -1)
            self.reconstruction_dist.append(
                self._decoder(features, observation_dist=self.observation_dist)
            )
        self.out_samples = [dist.sample() for dist in self.reconstruction_dist]

        # reconstruction for decoder
        _, features = tf.split(
            self.state_to_decode, [self._rnn_state_size, 2 * self._rnn_state_size], -1
        )
        self.state_decoded = self._decoder(
            features, observation_dist=self.observation_dist
        ).distribution.loc

        # latent transition model
        rnn_state, mean, belief = tf.split(
            self.state_for_transition,
            [self._rnn_state_size, self._rnn_state_size, self._rnn_state_size],
            -1,
        )
        prev_state = {
            "mean": mean,
            "stddev": None,
            "sample": mean,
            "belief": belief,
            "rnn_state": rnn_state,
        }
        next_state = self.rssm_cell._transition_tpl(
            prev_state, self.action_for_transition, None
        )
        self.next_state_from_transition_mean = tf.concat(
            [next_state["rnn_state"], next_state["mean"], next_state["belief"]], -1
        )
        self.next_state_from_transition_sample = tf.concat(
            [next_state["rnn_state"], next_state["sample"], next_state["belief"]], -1
        )

    def _define_loss(self):
        # self.likelihood = self.reconstruction_dist.log_prob(self.in_obs_seq)
        log_probs = [
            dist.log_prob(self.in_obs_seq) for dist in self.reconstruction_dist
        ]
        self.likelihood = (
            sum(log_probs[: self.N_samples_for_gradient]) / self.N_samples_for_gradient
        )
        self.divergence = self.rssm_cell.divergence_from_states(
            self.prior, self.posterior
        )
        self.divergence = tf.maximum(self.divergence, 3)
        self.elbo = tf.reduce_mean(
            self.likelihood_scale * self.likelihood - self.divergence
        )
        return -self.elbo

    def _get_learning_rate(
        self, initial_learning_rate=3e-4, decay_steps=100, decay_rate=1
    ):
        global_step = tf.Variable(0, trainable=False)
        if decay_rate == 1:
            learning_rate = tf.convert_to_tensor(initial_learning_rate)
        else:
            check_between("decay_rate", decay_rate, 0, 1)
            check_greater_equal("decay_steps", 1, 0, 1)
            learning_rate = tf.train.exponential_decay(
                initial_learning_rate,
                global_step,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
            )
        return learning_rate, global_step

    def _define_tensorboard_metrics(self):
        tensorboard_log_gradients(self.gradients)

        tf.summary.scalar("loss/loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)

        tf.summary.scalar("loss/likelihood", tf.reduce_mean(self.likelihood))
        tf.summary.scalar("loss/divergence", tf.reduce_mean(self.divergence))
        tf.summary.scalar("loss/elbo", self.elbo)

    def learn(
        self,
        n_rollouts=10000,
        n_epochs=15,
        batch_size=1,
        print_evaluation=False,
        play_data=None,
        return_initial_loss=False,
    ):
        """
        Main training loop
        """
        check_greater_equal("n_epochs", n_epochs, 1)
        check_greater_equal("batch_size", batch_size, 1)
        check_greater_equal("n_rollouts", n_rollouts, batch_size)

        use_play_data = play_data is not None
        if use_play_data:
            data = extract_play_data(play_data, self.timesteps)
        else:
            data = self._collect_data(n_rollouts)
        n_samples = len(data[0])
        assert n_samples == len(data[1])

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

        first_epoch_losses = []
        last_epoch_losses = []

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                batch_observations, batch_actions = get_batch(
                    data, batch, batch_size, self.timesteps
                )

                batch_loss, batch_elbo, _, batch_lr, summary, step = self.sess.run(
                    [
                        self.loss,
                        self.elbo,
                        self.optimization_op,
                        self.learning_rate,
                        summaries_op,
                        self.global_step,
                    ],
                    feed_dict={
                        self.in_obs_seq: batch_observations,
                        self.in_actions: batch_actions,
                        self.horizon: self.timesteps + 1,
                    },
                )

                if epoch == 0:
                    first_epoch_losses.append(batch_loss)
                if epoch == n_epochs - 1:
                    last_epoch_losses.append(batch_loss)

                if self.tensorboard_log is not None:
                    if (step - 1) % 110 == 0:
                        if use_play_data:
                            self._create_mujoco_vids(
                                self.env, batch_observations, batch_actions, step
                            )
                    summary_writer.add_summary(summary, step)

                print(
                    "Epoch: {}/{}...".format(epoch + 1, n_epochs),
                    "Batch: {}/{}...".format(batch + 1, n_batches),
                    "Training loss: {:.4f}   (learning_rate = {:.6f})".format(
                        batch_loss, batch_lr
                    ),
                    flush=True,
                )

            if self.checkpoint_folder is not None:
                params = {
                    "observation_dist": self.observation_dist,
                    "pixel_observations": self.pixel_observations,
                    "hidden_layer_size": int(self._hidden_layer_size),
                    "n_hidden_layers": int(self._num_layers),
                    "rnn_state_size": int(self._rnn_state_size),
                    "learning_rate": float(self.learning_rate.eval(session=self.sess)),
                    "obs_stddev": float(self._obs_stddev),
                    "likelihood_scale": float(self.likelihood_scale),
                    "mujoco_video_out_path": self.mujoco_video_out_path,
                    "timesteps_training": int(self.timesteps),
                    "fixed_latent_stddev": float(self._fixed_latent_stddev)
                    if self._fixed_latent_stddev is not None
                    else None,
                }
                with open("_".join([self.checkpoint_folder, "params.json"]), "w") as f:
                    json.dump(params, f)
                self.saver.save(self.sess, self.checkpoint_folder)

        if print_evaluation:
            if use_play_data:
                print("Warning: print_evaluation is only supported for gridworlds.")
            else:
                self._print_evaluation()

        if return_initial_loss:
            return np.mean(first_epoch_losses), np.mean(last_epoch_losses)
        return np.mean(last_epoch_losses)

    @classmethod
    def restore(cls, env, checkpoint_folder):
        """
        Restore the model from a checkpoint.
        """
        with open("_".join([checkpoint_folder, "params.json"]), "r") as f:
            params = json.load(f)
        model = cls(
            env, tensorboard_log=None, checkpoint_folder=checkpoint_folder, **params
        )
        model.sess = get_tf_session()
        model.saver.restore(model.sess, checkpoint_folder)
        return model

    def _create_mujoco_vids(self, env, batch_observations, batch_actions, step):
        """
        Creates videos to evaluate the training process.

        Takes the first trajectory from the current batch and outputs two videos:
            - the actual trajectory
            - the model prediction starting from only the first observation
        """
        if self.mujoco_video_out_path is None:
            return

        os.makedirs(self.mujoco_video_out_path, exist_ok=True)

        i = np.random.randint(0, batch_observations.shape[0])
        observations, actions = batch_observations[i], batch_actions[i]
        rgb_arrays_truth = [render_mujoco_from_obs(env, obs) for obs in observations]
        save_video(
            rgb_arrays_truth,
            os.path.join(
                self.mujoco_video_out_path, "mujoco_{}_truth.avi".format(step)
            ),
            fps=20.0,
        )

        for horizon in (1, 5, 50):
            (mean,) = self.sess.run(
                [self.reconstruction_dist[0].distribution.loc],
                feed_dict={
                    self.in_obs_seq: np.expand_dims(observations, 0),
                    self.in_actions: np.expand_dims(actions, 0),
                    self.horizon: horizon,
                },
            )
            out_observations = mean[0]
            rgb_arrays_model = [
                render_mujoco_from_obs(env, obs) for obs in out_observations
            ]
            save_video(
                rgb_arrays_model,
                os.path.join(
                    self.mujoco_video_out_path,
                    "mujoco_{}_model_horizon_{}.avi".format(step, horizon),
                ),
                fps=20.0,
            )

        obs = observations[0]
        rgb = render_mujoco_from_obs(env, obs)
        rgb_arrays_latent_step = [rgb]
        state = self.encoder(obs)
        for action in actions:
            state = self.latent_step(state, action)
            obs = self.decoder(state)
            rgb = render_mujoco_from_obs(env, obs)
            rgb_arrays_latent_step.append(rgb)
        save_video(
            rgb_arrays_model,
            os.path.join(
                self.mujoco_video_out_path,
                "mujoco_{}_model_latent_step.avi".format(step),
            ),
            fps=20.0,
        )

    def _print_evaluation(self, data=None):
        if data is None:
            data = self._collect_data(100)
        batch_observations, batch_actions = get_batch(data, 0, 16, self.timesteps)
        print("WITH HISTORY")
        self._print_evaluation_traj(batch_observations, batch_actions)
        print()
        print("RECONSTRUCTION")
        self._print_reconstruction_eval()

    def _print_evaluation_traj(self, batch_observations, batch_actions):
        for traj_i in range(len(batch_observations)):
            horizon = int(self.timesteps // 4)
            # horizon = self.timesteps
            obs = np.expand_dims(batch_observations[traj_i], 0)
            # obs[0, horizon:] = 0
            (out_samples,) = self.sess.run(
                [self.out_samples],
                feed_dict={
                    self.in_obs_seq: obs,
                    self.in_actions: np.expand_dims(batch_actions[traj_i], 0),
                    self.horizon: horizon,
                },
            )

            with np.printoptions(precision=2, suppress=True):
                for i in range(self.timesteps):
                    if i == horizon:
                        print("Horizon", i)
                    action_id = np.argmax(batch_actions[traj_i][i])
                    print("Action:", action_id_to_string(action_id))
                    print("Obs:")
                    print(batch_observations[traj_i][i].transpose((2, 0, 1)))
                    for j in range(3):
                        out_sample_i = out_samples[j][0, i].transpose((2, 0, 1))
                        print("Model Out {}:".format(j + 1))
                        print(out_sample_i)

    def _print_reconstruction_eval(self):
        with np.printoptions(precision=2, suppress=True):
            obs = self.env.reset()

            for j in range(5):
                print("Original:")
                print(obs.transpose((2, 0, 1)))

                for i in range(3):
                    print("Reconstruction {}".format(i + 1))
                    obs2 = self.decoder(self.encoder(np.expand_dims(obs, 0)))
                    print(obs2.transpose((0, 3, 1, 2)))
                obs, _, _, _ = self.env.step(self.env.action_space.sample())
