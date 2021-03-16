import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from deep_rlsp.util.train import (
    get_learning_rate,
    tensorboard_log_gradients,
    get_tf_session,
)
from deep_rlsp.model.exact_dynamics_mujoco import ExactDynamicsMujoco


class InversePolicyMDN:
    """
    InversePolicyMDN
    """

    def __init__(
        self,
        env,
        solver,
        experience_replay,
        tensorboard_log=None,
        learning_rate=3e-4,
        n_layers=10,
        layer_size=256,
        n_out=1,
        gauss_stdev=None,
    ):
        self.env = env
        self.solver = solver
        self.experience_replay = experience_replay
        self.dynamics = ExactDynamicsMujoco(
            env.unwrapped.spec.id, tolerance=1e-3, max_iters=100
        )

        assert len(self.env.action_space.shape) == 1
        self.action_dim = self.env.action_space.shape[0]
        self.observation_shape = list(self.env.observation_space.shape)
        self.n_out = n_out
        self.gauss_stdev = gauss_stdev
        self.layer_size = layer_size
        self.n_layers = n_layers

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

    def step(self, in_state, sample=True):
        batch_in_states = np.expand_dims(in_state, 0)

        if sample:
            (out,) = self.sess.run(
                [self.out_sample], feed_dict={self.in_state: batch_in_states}
            )
            out = out[0]
        else:
            factors, means, = self.sess.run(
                [self.mixture_factors, [c.loc for c in self.components]],
                feed_dict={self.in_state: batch_in_states},
            )
            out = means[np.argmax(factors[0])][0]
        out = np.clip(out, self.env.action_space.low, self.env.action_space.high)
        return out

    def _define_input_placeholders(self):
        self.true_action = tf.placeholder(
            tf.float32, [None, self.action_dim], name="action"
        )
        self.in_state = tf.placeholder(
            tf.float32, [None] + self.observation_shape, name="in_state"
        )

    def _define_model(self):
        activation = tf.nn.relu

        x = tf.reshape(self.in_state, [-1, np.prod(self.observation_shape)])
        x_1 = x
        for i in range(self.n_layers):
            x = tf.keras.layers.Dense(
                self.layer_size, activation=activation, name="hidden_{}".format(i + 1)
            )(x)

        means = x
        means = tf.keras.layers.Dense(
            self.n_out * self.action_dim, activation=None, name="output"
        )(means)
        # note: this requires an 1d action space
        self.mixture_means = tf.split(means, [self.action_dim] * self.n_out, -1)

        if self.gauss_stdev is None:
            stddevs = x
            stddevs = tf.keras.layers.Dense(self.n_out, activation=None)(stddevs)
            stddevs = tf.exp(stddevs)
            stddevs = tf.clip_by_value(
                stddevs, clip_value_min=1e-10, clip_value_max=1e10
            )
            # same stddev for all components
            # self.mixture_stddevs = tf.reshape(
            #     stddevs, (-1, self.n_out, self.state_size)
            # )
            self.mixture_stddevs = tf.split(stddevs, [1] * self.n_out, -1)
        else:
            self.mixture_stddevs = None

        factors = tf.concat([x, x_1], -1)
        # factors = tf.keras.layers.Dense(
        #     self.layer_size // 2, activation=activation, name="factors_hidden"
        # )(factors)
        factors = tf.keras.layers.Dense(
            self.n_out, activation=None, name="factors_out"
        )(factors)
        # if we don't clip these, we get NANs in the loss computation
        factors = tf.clip_by_value(factors, clip_value_min=0.1, clip_value_max=10)
        self.mixture_factors = tf.nn.softmax(factors)

        if self.gauss_stdev is None:
            self.components = [
                tfd.MultivariateNormalDiag(
                    loc=mean,
                    scale_diag=tf.tile(stddev, [1, self.action_dim]),
                    validate_args=True,
                    allow_nan_stats=False,
                )
                for mean, stddev in zip(self.mixture_means, self.mixture_stddevs)
            ]
        else:
            self.components = [
                tfd.MultivariateNormalDiag(
                    loc=mean,
                    scale_diag=tf.fill(tf.shape(mean), self.gauss_stdev),
                    validate_args=True,
                    allow_nan_stats=False,
                )
                for mean in self.mixture_means
            ]

        self.mixture_distribution = tfd.Categorical(
            probs=self.mixture_factors, allow_nan_stats=False
        )
        self.out_distribution = tfd.Mixture(
            cat=self.mixture_distribution,
            components=self.components,
            validate_args=True,
            allow_nan_stats=False,
        )
        self.out_sample = self.out_distribution.sample()

    def _define_loss(self):
        neg_logprob = -tf.reduce_mean(self.out_distribution.log_prob(self.true_action))
        self.logprobs = [
            tf.reduce_mean(dist.log_prob(self.true_action)) for dist in self.components
        ]
        self.mixture_entropy = tf.reduce_mean(self.mixture_distribution.entropy())
        return neg_logprob

    def _define_tensorboard_metrics(self):
        tf.summary.scalar("loss/loss", self.loss)
        tf.summary.scalar("loss/mixture_entropy", self.mixture_entropy)
        tf.summary.scalar("learning_rate", self.learning_rate)
        for i, lp in enumerate(self.logprobs):
            tf.summary.scalar("loss/logprob_{}".format(i + 1), lp)
        tensorboard_log_gradients(self.gradients)

    def _apply_policy(self, observations):
        next_states, actions = [], []
        for obs in observations:
            action = self.solver.predict(obs)[0]
            try:
                obs = self.dynamics.dynamics(obs, action)
                next_states.append(np.copy(obs))
                actions.append(np.copy(action))
            except Exception as e:
                print("_apply_policy", e)
        return next_states, actions

    def learn(
        self,
        n_epochs=1,
        batch_size=16,
        return_initial_loss=False,
        verbose=True,
        reinitialize=False,
    ):
        """
        Main training loop
        """
        if self.sess is None:
            self.sess = get_tf_session()
            reinitialize = True
        if reinitialize:
            self.sess.run(tf.global_variables_initializer())

        if self.tensorboard_log is not None:
            summaries_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(
                self.tensorboard_log, self.sess.graph
            )
        else:
            summaries_op = tf.no_op()

        n_batches = len(self.experience_replay) // batch_size

        first_epoch_losses = []
        last_epoch_losses = []

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                obs, _, _ = self.experience_replay.sample(batch_size, normalize=False)
                batch_next_states, batch_actions = self._apply_policy(obs)

                (
                    batch_loss,
                    _,
                    batch_lr,
                    summary,
                    step,
                    mixture_entropy,
                ) = self.sess.run(
                    [
                        self.loss,
                        self.optimization_op,
                        self.learning_rate,
                        summaries_op,
                        self.global_step,
                        self.mixture_entropy,
                    ],
                    feed_dict={
                        self.in_state: batch_next_states,
                        self.true_action: batch_actions,
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
                        "Training loss: {:.4f}   (ent {:.4f})   ".format(
                            batch_loss, mixture_entropy
                        ),
                        "(learning_rate = {:.6f})".format(batch_lr),
                    )

        if return_initial_loss:
            return np.mean(first_epoch_losses), np.mean(last_epoch_losses)
        return np.mean(last_epoch_losses)
