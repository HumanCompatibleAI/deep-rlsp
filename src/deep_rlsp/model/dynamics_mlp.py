import json

import numpy as np
import tensorflow as tf

from deep_rlsp.util.parameter_checks import check_less_equal
from deep_rlsp.util.train import (
    get_learning_rate,
    tensorboard_log_gradients,
    get_tf_session,
)


class InverseDynamicsMLP:
    def __init__(
        self,
        env,
        experience_replay,
        hidden_layer_size=512,
        n_hidden_layers=3,
        learning_rate=3e-4,
        tensorboard_log=None,
        checkpoint_folder=None,
        obs_mean=0,
        obs_std=1,
        act_mean=0,
        act_std=1,
        delta_mean=0,
        delta_std=1,
        delta_min=-float("inf"),
        delta_max=float("inf"),
    ):
        assert experience_replay is not None
        self.env = env
        self.experience_replay = experience_replay

        self.action_space_shape = list(self.env.action_space.shape)

        self._hidden_layer_size = hidden_layer_size
        self._n_hidden_layers = n_hidden_layers

        assert len(self.env.observation_space.shape) == 1
        self.state_size = self.env.observation_space.shape[0]

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

        self.checkpoint_folder = checkpoint_folder
        if self.checkpoint_folder is not None:
            self.saver = tf.train.Saver()

        self.sess = None

        self.obs_mean = np.array(obs_mean)
        self.obs_std = np.array(obs_std)
        self.act_mean = np.array(act_mean)
        self.act_std = np.array(act_std)
        self.delta_mean = np.array(delta_mean)
        self.delta_std = np.array(delta_std)
        self.delta_min = np.array(delta_min)
        self.delta_max = np.array(delta_max)

    def step(self, in_state, in_action, sample=True):
        in_state_norm = (in_state - self.obs_mean) / self.obs_std
        in_action_norm = (in_action - self.act_mean) / self.act_std

        batch_in_states = np.expand_dims(in_state_norm, 0)
        batch_actions = np.expand_dims(in_action_norm, 0)

        (delta,) = self.sess.run(
            [self.predict_deltas],
            feed_dict={self.in_action: batch_actions, self.in_state: batch_in_states},
        )
        delta = delta[0]
        delta = delta * self.delta_std + self.delta_mean
        delta = np.clip(delta, self.delta_min, self.delta_max)
        out_state = in_state + delta
        return out_state

    def _update_mean_std(self):
        self.obs_mean = self.experience_replay.obs_normalizer.mean
        self.obs_std = self.experience_replay.obs_normalizer.std
        self.act_mean = self.experience_replay.act_normalizer.mean
        self.act_std = self.experience_replay.act_normalizer.std
        self.delta_mean = self.experience_replay.delta_normalizer.mean
        self.delta_std = self.experience_replay.delta_normalizer.std
        self.delta_min = self.experience_replay.min_delta
        self.delta_max = self.experience_replay.max_delta

    def _define_input_placeholders(self):
        self.deltas = tf.placeholder(tf.float32, (None, self.state_size), name="state")
        self.in_action = tf.placeholder(
            tf.float32, [None] + self.action_space_shape, name="action"
        )
        self.in_state = tf.placeholder(
            tf.float32, (None, self.state_size), name="next_state"
        )

    def _define_model(self):
        activation = tf.nn.relu

        x = tf.concat([self.in_state, self.in_action], axis=-1)

        for i in range(self._n_hidden_layers):
            x = tf.keras.layers.Dense(
                self._hidden_layer_size,
                activation=activation,
                name="hidden_{}".format(i + 1),
            )(x)

        self.predict_deltas = tf.keras.layers.Dense(
            self.state_size, activation=None, name="output"
        )(x)

    def _define_loss(self):
        return tf.reduce_mean(tf.square(self.predict_deltas - self.deltas))

    def _define_tensorboard_metrics(self):
        tf.summary.scalar("loss/loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tensorboard_log_gradients(self.gradients)

    def learn(
        self,
        n_rollouts=None,
        n_epochs=1,
        batch_size=16,
        print_evaluation=False,
        data=None,
        return_initial_loss=False,
        verbose=True,
    ):
        """
        Main training loop
        """
        self._update_mean_std()
        n_samples = len(self.experience_replay)

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
                (
                    batch_states,
                    batch_actions,
                    batch_next_states,
                ) = self.experience_replay.sample(batch_size, normalize=False)

                batch_deltas = batch_states - batch_next_states
                batch_deltas = self.experience_replay.normalize_delta(batch_deltas)
                batch_actions = self.experience_replay.normalize_act(batch_actions)
                batch_next_states = self.experience_replay.normalize_obs(
                    batch_next_states
                )

                sigma = 0.001
                batch_deltas += sigma * np.random.randn(*batch_deltas.shape)
                batch_actions += sigma * np.random.randn(*batch_actions.shape)
                batch_next_states += sigma * np.random.randn(*batch_next_states.shape)

                (batch_loss, _, batch_lr, summary, step) = self.sess.run(
                    [
                        self.loss,
                        self.optimization_op,
                        self.learning_rate,
                        summaries_op,
                        self.global_step,
                    ],
                    feed_dict={
                        self.in_state: batch_next_states,
                        self.in_action: batch_actions,
                        self.deltas: batch_deltas,
                    },
                )

                if epoch == 0:
                    first_epoch_losses.append(batch_loss)
                if epoch == n_epochs - 1:
                    last_epoch_losses.append(batch_loss)

                if self.tensorboard_log is not None:
                    summary_writer.add_summary(summary, step)

                if verbose:
                    print(
                        "Epoch: {}/{}...".format(epoch + 1, n_epochs),
                        "Batch: {}/{}...".format(batch + 1, n_batches),
                        "Training loss: {:.4f}  ".format(batch_loss),
                        "(learning_rate = {:.6f})".format(batch_lr),
                    )

            if self.checkpoint_folder is not None:
                params = {
                    "hidden_layer_size": self._hidden_layer_size,
                    "n_hidden_layers": self._n_hidden_layers,
                    "learning_rate": float(self.learning_rate.eval(session=self.sess)),
                    "obs_mean": self.obs_mean.tolist(),
                    "obs_std": self.obs_std.tolist(),
                    "act_mean": self.act_mean.tolist(),
                    "act_std": self.act_std.tolist(),
                    "delta_mean": self.delta_mean.tolist(),
                    "delta_std": self.delta_std.tolist(),
                    "delta_min": self.delta_min.tolist(),
                    "delta_max": self.delta_max.tolist(),
                }
                with open("_".join([self.checkpoint_folder, "params.json"]), "w") as f:
                    json.dump(params, f)

                self.saver.save(self.sess, self.checkpoint_folder)

        if print_evaluation:
            self._print_evaluation(data)

        if return_initial_loss:
            return np.mean(first_epoch_losses), np.mean(last_epoch_losses)
        return np.mean(last_epoch_losses)

    @classmethod
    def restore(cls, env, experience_replay, checkpoint_folder):
        """
        Restore the model from a checkpoint.
        """
        with open("_".join([checkpoint_folder, "params.json"]), "r") as f:
            params = json.load(f)
        model = cls(
            env,
            experience_replay,
            tensorboard_log=None,
            checkpoint_folder=checkpoint_folder,
            **params
        )
        model.sess = get_tf_session()
        model.saver.restore(model.sess, checkpoint_folder)
        return model
