import json

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from deep_rlsp.model import LatentSpaceModel
from deep_rlsp.util.parameter_checks import check_less_equal, check_greater_equal
from deep_rlsp.util.train import (
    get_learning_rate,
    tensorboard_log_gradients,
    get_batch,
    shuffle_data,
    get_tf_session,
)
from deep_rlsp.envs.gridworlds.env import Direction, action_id_to_string


def extract_play_data(latent_space, play_data):
    observations, play_actions = play_data["observations"], play_data["actions"]
    n_traj = len(observations)
    assert len(play_actions) == n_traj
    states, actions, next_states = [], [], []
    for i in range(n_traj):
        print("i {}/{}".format(i, n_traj))
        l_traj = len(observations[i])
        assert len(play_actions[i]) >= l_traj - 1
        for t in range(l_traj - 1):
            obs = observations[i][t]
            next_obs = observations[i][t + 1]
            state = latent_space.encoder(obs)
            next_state = latent_space.encoder(next_obs)
            action = play_actions[i][t]
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
    return [states, actions, next_states]


def encode_transition_data(latent_space, data):
    states, actions, next_states = data
    out_states = [latent_space.encoder(s) for s in states]
    out_next_states = [latent_space.encoder(s) for s in next_states]
    return out_states, actions, out_next_states


class GridworldSpace:
    def __init__(self, env):
        self.obs_shape = list(env.observation_space.shape)
        super().__init__(np.prod(self.obs_shape))

    def encoder(self, obs):
        return np.reshape(obs, [-1, self.state_size])

    def decoder(self, state):
        return np.reshape(state, [-1] + self.obs_shape)


class InverseDynamicsMDN:
    """
    Implements a mixture density model to learn dynamics.

    Can be used to learn forward and backward dynamics.

    The dynamics are represented as a mixture of Gaussians in the latent space.
    This means the conditional distribution of the next state given the current state
    and the last action is given by a mixture of Gaussians with parameters that are
    given as outputs of a feedforward neural network.
    """

    def __init__(
        self,
        env,
        experience_replay,
        backward=True,
        hidden_layer_size=1024,
        n_hidden_layers=3,
        learning_rate=3e-4,
        tensorboard_log=None,
        checkpoint_folder=None,
        latent_space=None,
        n_out_states=1,
        gauss_stdev=None,
        play_data_range=None,
    ):
        self.env = env
        self.experience_replay = experience_replay
        self.backward = backward

        self.action_space_shape = list(self.env.action_space.shape)
        self.n_out_states = n_out_states
        self.gauss_stdev = gauss_stdev
        # self.individual_likelihood_factor = 0.0001
        self._hidden_layer_size = hidden_layer_size
        self._n_hidden_layers = n_hidden_layers

        if latent_space == "gridworld":
            self.latent_space = GridworldSpace(env)
        else:
            self.latent_space = latent_space

        if self.latent_space is None:
            assert len(self.env.observation_space.shape) == 1
            self.state_size = self.env.observation_space.shape[0]
        else:
            self.state_size = self.latent_space.state_size

        self._define_input_placeholders()
        self._define_model()

        self.loss = self._define_loss()
        # self.learning_rate, self.global_step = get_learning_rate(1e-2, 20, 0.98)
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

        self.play_data_range = play_data_range
        self.sess = None

    def step(self, in_state, in_action, sample=True):
        if self.experience_replay is not None:
            in_state = self.experience_replay.normalize_obs(in_state)
            in_action = self.experience_replay.normalize_act(in_action)

        batch_actions = np.expand_dims(in_action, 0)
        batch_in_states = np.expand_dims(in_state, 0)
        if sample:
            (out_state,) = self.sess.run(
                [self.out_sample],
                feed_dict={
                    self.in_action: batch_actions,
                    self.in_state: batch_in_states,
                },
            )
            out_state = out_state[0]
        else:
            factors, means, = self.sess.run(
                [self.mixture_factors, [c.loc for c in self.components]],
                feed_dict={
                    self.in_action: batch_actions,
                    self.in_state: batch_in_states,
                },
            )
            out_state = means[np.argmax(factors[0])][0]

        if self.experience_replay is not None:
            out_state = self.experience_replay.unnormalize_obs(out_state)
        return out_state

    def _collect_data(self, n_rollouts):
        states, actions, next_states = [], [], []
        for _ in range(n_rollouts):
            obs1 = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                obs2, _, done, _ = self.env.step(action)
                if self.latent_space is None:
                    state1, state2 = obs1, obs2
                else:
                    # we apply the encoder to a single state, because the history is not
                    # available when using the inverse model (to simulate the history)
                    state1 = self.latent_space.encoder(obs1)
                    state2 = self.latent_space.encoder(obs2)
                states.append(state1)
                actions.append(action)
                next_states.append(state2)
                obs1 = obs2
        return [states, actions, next_states]

    def _define_input_placeholders(self):
        self.out_state = tf.placeholder(
            tf.float32, (None, self.state_size), name="state"
        )
        self.in_action = tf.placeholder(
            tf.float32, [None] + self.action_space_shape, name="action"
        )
        self.in_state = tf.placeholder(
            tf.float32, (None, self.state_size), name="next_state"
        )

    def _define_model(self):
        activation = tf.nn.leaky_relu
        # activation = tf.nn.relu

        x = tf.concat([self.in_state, self.in_action], axis=-1)
        x_1 = x
        for i in range(self._n_hidden_layers):
            x = tf.keras.layers.Dense(
                self._hidden_layer_size,
                activation=activation,
                name="hidden_{}".format(i + 1),
            )(x)

        # means = tf.concat([x, x_1], -1)  # TODO: remove?
        means = x
        # means = tf.keras.layers.Dense(self._hidden_layer_size, activation=activation)(
        #     means
        # )
        means = tf.keras.layers.Dense(
            self.n_out_states * self.state_size, activation=None, name="output"
        )(means)
        self.mixture_means = tf.split(means, [self.state_size] * self.n_out_states, -1)

        if self.gauss_stdev is None:
            stddevs = x
            stddevs = tf.keras.layers.Dense(self.n_out_states, activation=None)(stddevs)
            stddevs = tf.exp(stddevs)
            # same stddev for all components
            # self.mixture_stddevs = tf.reshape(
            #     stddevs, (-1, self.n_out_states, self.state_size)
            # )
            self.mixture_stddevs = tf.split(stddevs, [1] * self.n_out_states, -1)
        else:
            self.mixture_stddevs = None

        factors = tf.concat([x, x_1], -1)
        # factors = tf.keras.layers.Dense(
        #     self._hidden_layer_size // 2, activation=activation, name="factors_hidden"
        # )(factors)
        factors = tf.keras.layers.Dense(
            self.n_out_states, activation=None, name="factors_out"
        )(factors)
        # if we don't clip these, we get NANs in the loss computation
        factors = tf.clip_by_value(factors, clip_value_min=0.1, clip_value_max=10)
        self.mixture_factors = tf.nn.softmax(factors)

        # self.fixed_factors = [1 / 3, 1 / 3, 1 / 3]
        # self.fixed_factors = [0.5, 0.5, 0]
        # self.mixture_factors = tf.tile(
        #     tf.convert_to_tensor([self.fixed_factors]), [tf.shape(x)[0], 1]
        # )

        if self.gauss_stdev is None:
            self.components = [
                tfd.MultivariateNormalDiag(
                    loc=mean,
                    scale_diag=tf.tile(stddev, [1, self.state_size]),
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
        neg_logprob = -tf.reduce_mean(self.out_distribution.log_prob(self.out_state))
        self.logprobs = [
            tf.reduce_mean(dist.log_prob(self.out_state)) for dist in self.components
        ]
        self.mixture_entropy = tf.reduce_mean(self.mixture_distribution.entropy())
        return neg_logprob
        # return neg_logprob - self.individual_likelihood_factor * tf.reduce_sum(
        #     self.logprobs
        # )  # - 0.2 * self.mixture_entropy

    def _define_tensorboard_metrics(self):
        tf.summary.scalar("loss/loss", self.loss)
        tf.summary.scalar("loss/mixture_entropy", self.mixture_entropy)
        tf.summary.scalar("learning_rate", self.learning_rate)
        for i, lp in enumerate(self.logprobs):
            tf.summary.scalar("loss/logprob_{}".format(i + 1), lp)
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
        experience_replay = self.experience_replay

        check_greater_equal("n_epochs", n_epochs, 1)
        check_greater_equal("batch_size", batch_size, 1)
        if data is None and experience_replay is None:
            check_greater_equal("n_rollouts", n_rollouts, batch_size)

        assert data is None or experience_replay is None
        assert (not print_evaluation) or (data is not None)

        if data is not None:
            if "states" in data and "actions" in data and "next_states" in data:
                data = (data["states"], data["actions"], data["next_states"])
                data = encode_transition_data(self.latent_space, data)
            else:
                data = extract_play_data(self.latent_space, data)
                states, actions, next_states = data
                min_val = min(np.min(states), np.min(next_states))
                max_val = max(np.max(states), np.max(next_states))
                self.play_data_range = (float(min_val), float(max_val))
                print("Playdata range of features: [{}, {}]".format(min_val, max_val))
        elif experience_replay is None:
            data = self._collect_data(n_rollouts)

        if data is not None:
            data = shuffle_data(data)
            n_samples = len(data[0])
        else:
            assert experience_replay is not None
            n_samples = len(experience_replay)

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
                if data is not None:
                    batch_states, batch_actions, batch_next_states = get_batch(
                        data, batch, batch_size
                    )
                else:
                    (
                        batch_states,
                        batch_actions,
                        batch_next_states,
                    ) = experience_replay.sample(batch_size, normalize=True)

                if self.backward:
                    batch_in_states = batch_next_states
                    batch_out_states = batch_states
                else:
                    batch_in_states = batch_states
                    batch_out_states = batch_next_states

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
                        self.in_state: batch_in_states,
                        self.in_action: batch_actions,
                        self.out_state: batch_out_states,
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
                        "Training loss: {:.4f}   (ent {:.4f})  ".format(
                            batch_loss, mixture_entropy
                        ),
                        "(learning_rate = {:.6f})".format(batch_lr),
                    )

            if self.checkpoint_folder is not None:
                params = {
                    "hidden_layer_size": self._hidden_layer_size,
                    "n_hidden_layers": self._n_hidden_layers,
                    "learning_rate": float(self.learning_rate.eval(session=self.sess)),
                    "n_out_states": self.n_out_states,
                    "gauss_stdev": self.gauss_stdev,
                    "play_data_range": self.play_data_range,
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
    def restore(cls, env, experience_replay, latent_space, checkpoint_folder):
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
            latent_space=latent_space,
            **params
        )
        model.sess = get_tf_session()
        model.saver.restore(model.sess, checkpoint_folder)
        return model

    def _get_eval_trajectory(self):
        eval_actions = [
            Direction.get_onehot_from_direction(Direction.EAST),
            Direction.get_onehot_from_direction(Direction.EAST),
            Direction.get_onehot_from_direction(Direction.EAST),
            Direction.get_onehot_from_direction(Direction.NORTH),
            Direction.get_onehot_from_direction(Direction.NORTH),
            Direction.get_onehot_from_direction(Direction.NORTH),
            Direction.get_onehot_from_direction(Direction.WEST),
            Direction.get_onehot_from_direction(Direction.SOUTH),
            Direction.get_onehot_from_direction(Direction.WEST),
            Direction.get_onehot_from_direction(Direction.WEST),
            Direction.get_onehot_from_direction(Direction.WEST),
            Direction.get_onehot_from_direction(Direction.WEST),
            Direction.get_onehot_from_direction(Direction.SOUTH),
            Direction.get_onehot_from_direction(Direction.SOUTH),
        ]
        assert len(eval_actions) < self.env.time_horizon
        trajectory = []
        obs1 = self.env.reset()
        done = False
        for action in eval_actions:
            obs2, _, done, _ = self.env.step(action)
            state1 = self.latent_space.encoder(np.expand_dims(obs1, 0))
            state2 = self.latent_space.encoder(np.expand_dims(obs2, 0))
            trajectory.append((state1, action, state2))
            obs1 = obs2
        return trajectory

    def _print_one_step_eval(self, trajectory):
        print()
        print("One-step prediction")
        with np.printoptions(precision=4, suppress=True):
            # print a few example predictions
            print()
            print("---------------------")
            for s, a, ns in trajectory:
                batch_in_states = ns
                batch_out_states = s
                batch_actions = np.expand_dims(a, 0)
                mixture_means, mixture_factors, out_sample = self.sess.run(
                    [self.mixture_means, self.mixture_factors, self.out_sample],
                    feed_dict={
                        self.out_state: batch_out_states,
                        self.in_action: batch_actions,
                        self.in_state: batch_in_states,
                    },
                )
                action_id = np.argmax(a)
                print("Actual Transition:")
                print("In State")
                print(
                    self.latent_space.decoder(batch_in_states).transpose((0, 3, 1, 2))
                )
                print("Action:", action_id_to_string(action_id))
                print("Out State")
                print(
                    self.latent_space.decoder(batch_out_states).transpose((0, 3, 1, 2))
                )
                print("Mixture distribution (of out state):")
                for i in range(len(mixture_means)):
                    print("Component {}".format(i + 1))
                    print("Mean")
                    print(
                        np.array(
                            self.latent_space.decoder(
                                np.expand_dims(mixture_means[i][0], 0)
                            )
                        ).transpose((0, 3, 1, 2))
                    )
                    print("Mixture factor:", mixture_factors[0, i])
                print("Sample")
                print(
                    np.array(
                        self.latent_space.decoder(np.expand_dims(out_sample[0], 0))
                    ).transpose((0, 3, 1, 2))
                )

    def _print_multi_step_eval(self, trajectory):
        print()
        print()
        print("Multi-step prediction")
        print()
        with np.printoptions(precision=4, suppress=True):
            print("Actual trajectory:")
            for s, a, ns in trajectory:
                print("State")
                print(self.latent_space.decoder(s).transpose((0, 3, 1, 2)))
                print("Action")
                print(action_id_to_string(np.argmax(a)))

            print("Final State")
            state = trajectory[-1][2]
            print(self.latent_space.decoder(state).transpose((0, 3, 1, 2)))

            print("Sample")
            actions = [a for s, a, ns in trajectory]
            for a in actions[::-1]:
                print("Action")
                print(action_id_to_string(np.argmax(a)))
                print("State")
                state = np.expand_dims(self.step(state[0], a), 0)
                print(self.latent_space.decoder(state).transpose((0, 3, 1, 2)))

    def _print_evaluation(self, transitions):
        trajectory = self._get_eval_trajectory()
        self._print_one_step_eval(trajectory)
        self._print_multi_step_eval(trajectory)


def main():
    import datetime
    from deep_rlsp.run import get_problem_parameters
    from deep_rlsp.envs.one_hot_action_space_wrapper import OneHotActionSpaceWrapper

    # from deep_rlsp.envs.basic_room import BasicRoomEnv

    seed = 2
    hidden_layer_size = 512
    n_hidden_layers = 2
    learning_rate = 1e-6
    n_rollouts = 1000
    n_epochs = 1
    batch_size = 50

    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    env, _, _, _ = get_problem_parameters("room", "default")
    # env = BasicRoomEnv(1, pixel_observations)
    env = OneHotActionSpaceWrapper(env)

    latent_model_path = "tf_ckpt/tf_ckpt_latent_room_default_20200414_142642"
    backward = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    direction = "backward" if backward else "forward"
    checkpoint_folder = "tf_ckpt/tf_ckpt_{}_mdn_room_default_{}".format(
        direction, timestamp
    )

    g1, g2 = tf.Graph(), tf.Graph()

    with g1.as_default():
        latent_model = LatentSpaceModel.restore(env, latent_model_path)

    with g2.as_default():
        model = InverseDynamicsMDN(
            env,
            backward=backward,
            hidden_layer_size=hidden_layer_size,
            n_hidden_layers=n_hidden_layers,
            learning_rate=learning_rate,
            tensorboard_log="tf_logs/tf_logs_" + timestamp,
            latent_space=latent_model,
            checkpoint_folder=checkpoint_folder,
        )
        model.learn(
            n_rollouts=n_rollouts,
            n_epochs=n_epochs,
            batch_size=batch_size,
            print_evaluation=True,
        )
    print("Wrote model to checkpoint folder", checkpoint_folder)


if __name__ == "__main__":
    main()
