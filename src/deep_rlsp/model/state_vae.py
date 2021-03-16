import json

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from deep_rlsp.util.train import (
    tensorboard_log_gradients,
    get_learning_rate,
    get_tf_session,
)


class StateVAE:
    def __init__(
        self,
        input_dim,
        state_size,
        n_layers=3,
        layer_size=256,
        learning_rate=1e-4,
        prior_stdev=1,
        divergence_factor=1,
        checkpoint_folder=None,
        tensorboard_log=None,
    ):
        self.input_dim = input_dim
        self.state_size = state_size
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.activation = tf.nn.leaky_relu
        self.prior_stdev = prior_stdev
        self.divergence_factor = divergence_factor

        self._define_model()
        self.sess = None

        self.learning_rate, self.global_step = get_learning_rate(learning_rate, None, 1)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients = self.optimizer.compute_gradients(loss=self.loss)
        self.optimization_op = self.optimizer.apply_gradients(
            self.gradients, global_step=self.global_step
        )

        self.checkpoint_folder = checkpoint_folder
        if self.checkpoint_folder is not None:
            self.saver = tf.train.Saver()

        self.tensorboard_log = tensorboard_log
        if self.tensorboard_log is not None:
            self._define_tensorboard_metrics()

    def _define_model(self):
        self.x = tf.placeholder(
            name="x", dtype=tf.float32, shape=[None, self.input_dim]
        )
        self.z_enc = tf.placeholder(
            name="z_enc", dtype=tf.float32, shape=[None, self.state_size]
        )

        def encoder(x):
            for i in range(self.n_layers - 1):
                x = tf.layers.dense(x, self.layer_size, self.activation)
            loc = tf.layers.dense(x, self.state_size, None)
            scale = tf.layers.dense(x, self.state_size, tf.nn.softplus)
            scale = tf.clip_by_value(scale, 1e-10, 1e10)
            return tfd.MultivariateNormalDiag(
                loc=loc, scale_diag=scale, validate_args=True
            )

        def decoder(z):
            x = z
            for i in range(self.n_layers - 1):
                x = tf.layers.dense(x, self.layer_size, self.activation)
            loc = tf.layers.dense(x, self.input_dim, None)
            # scale = tf.layers.dense(x, self.input_dim, tf.nn.softplus)
            return tfd.MultivariateNormalDiag(
                loc=loc, scale_diag=tf.ones(self.input_dim), validate_args=True
            )

        self._encoder = tf.make_template("encoder", encoder)
        self._decoder = tf.make_template("decoder", decoder)

        self.posterior = self._encoder(self.x)
        self.z = self.posterior.sample()

        self.decoded_dist = self._decoder(self.z)
        self.x_dec = self._decoder(self.z_enc).loc

        prior = tfd.MultivariateNormalDiag(
            tf.zeros(self.state_size), self.prior_stdev * tf.ones(self.state_size)
        )
        likelihood = self.decoded_dist.log_prob(self.x)
        divergence = self.divergence_factor * tfd.kl_divergence(self.posterior, prior)
        elbo = tf.reduce_mean(likelihood - divergence)
        self.loss = -elbo

    def _define_tensorboard_metrics(self):
        tf.summary.scalar("loss/loss", self.loss)
        tensorboard_log_gradients(self.gradients)

    def learn(
        self,
        experience_replay,
        n_epochs=1,
        batch_size=10,
        verbose=False,
        return_initial_loss=False,
    ):
        if self.sess is None:
            self.sess = get_tf_session()
            self.sess.run(tf.global_variables_initializer())

        n_samples = len(experience_replay)
        n_batches = n_samples // batch_size

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
                x, _, _ = experience_replay.sample(batch_size)
                loss, summary, step, _ = self.sess.run(
                    [self.loss, summaries_op, self.global_step, self.optimization_op],
                    feed_dict={self.x: x},
                )

                if epoch == 0:
                    first_epoch_losses.append(loss)
                if epoch == n_epochs - 1:
                    last_epoch_losses.append(loss)

                if self.tensorboard_log is not None:
                    summary_writer.add_summary(summary, step)

                if verbose:
                    print(
                        "Epoch: {}/{}...".format(epoch + 1, n_epochs),
                        "Batch: {}/{}...".format(batch + 1, n_batches),
                        "Training loss: " "{:.4f}".format(loss),
                    )

            if self.checkpoint_folder is not None:
                params = {
                    "input_dim": self.input_dim,
                    "state_size": self.state_size,
                    "learning_rate": float(self.learning_rate.eval(session=self.sess)),
                    "n_layers": self.n_layers,
                    "layer_size": self.layer_size,
                }
                with open("_".join([self.checkpoint_folder, "params.json"]), "w") as f:
                    json.dump(params, f)

                self.saver.save(self.sess, self.checkpoint_folder)

        if return_initial_loss:
            return np.mean(first_epoch_losses), np.mean(last_epoch_losses)
        return np.mean(last_epoch_losses)

    def decoder(self, z):
        z = np.expand_dims(z, 0)
        x_dec = self.sess.run(self.x_dec, feed_dict={self.z_enc: z})
        x_dec = x_dec[0]
        return x_dec

    def encoder(self, x):
        x = np.expand_dims(x, 0)
        z = self.sess.run(self.z, feed_dict={self.x: x})
        z = z[0]
        return z

    @classmethod
    def restore(cls, checkpoint_folder):
        """
        Restore the model from a checkpoint.
        """
        with open("_".join([checkpoint_folder, "params.json"]), "r") as f:
            params = json.load(f)
        model = cls(checkpoint_folder=checkpoint_folder, **params)
        model.sess = get_tf_session()
        model.saver.restore(model.sess, checkpoint_folder)
        return model
