import numpy as np
import tensorflow as tf

from deep_rlsp.util.parameter_checks import check_between, check_greater_equal


def get_tf_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def get_learning_rate(initial_learning_rate, decay_steps, decay_rate):
    global_step = tf.Variable(0, trainable=False)
    if decay_rate == 1:
        learning_rate = tf.convert_to_tensor(initial_learning_rate)
    else:
        check_between("decay_rate", decay_rate, 0, 1)
        check_greater_equal("decay_steps", decay_steps, 1)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
        )
    return learning_rate, global_step


def tensorboard_log_gradients(gradients):
    for gradient, variable in gradients:
        tf.summary.scalar("gradients/" + variable.name, tf.norm(gradient, ord=2))
        tf.summary.scalar("variables/" + variable.name, tf.norm(variable, ord=2))


def get_batch(data, batch, batch_size):
    batches = []
    for dataset in data:
        batch_array = dataset[batch * batch_size : (batch + 1) * batch_size]
        batches.append(batch_array)
    return batches


def shuffle_data(data):
    n_states = len(data[0])
    shuffled = np.arange(n_states)
    np.random.shuffle(shuffled)
    shuffled_data = []
    for dataset in data:
        assert len(dataset) == n_states
        shuffled_data.append(np.array(dataset)[shuffled])
    return shuffled_data
