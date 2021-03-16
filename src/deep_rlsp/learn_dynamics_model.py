import os
import datetime
import numpy as np
import tensorflow as tf

from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from deep_rlsp.model import LatentSpaceModel, InverseDynamicsMDN
from deep_rlsp.run import get_problem_parameters


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_id = "{}_learn_dynamics_{}_{}".format(
            timestamp, config["env_name"], config["problem_spec"]
        )
        return custom_id  # started_event returns the _run._id


ex = Experiment("learn_dynamics_model")
ex.observers = [SetID(), FileStorageObserver.create("results")]


def _get_log_folders(checkpoint_base, tensorboard_base, label):
    checkpoint_folder = os.path.join(checkpoint_base, label)
    tensorboard_folder = os.path.join(tensorboard_base, label)
    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(tensorboard_folder, exist_ok=True)
    return checkpoint_folder, tensorboard_folder


def train_latent_space_model(
    env,
    hidden_layer_size,
    rnn_state_size,
    n_rollouts,
    n_epochs,
    batch_size,
    learning_rate,
    tensorboard_folder,
    checkpoint_folder,
):
    model = LatentSpaceModel(
        env,
        tensorboard_log=tensorboard_folder,
        checkpoint_folder=checkpoint_folder,
        learning_rate=learning_rate,
        hidden_layer_size=hidden_layer_size,
        rnn_state_size=rnn_state_size,
    )
    loss = model.learn(
        n_rollouts=n_rollouts,
        n_epochs=n_epochs,
        batch_size=batch_size,
        print_evaluation=True,
    )
    return model, loss


def train_inverse_dynamics_model(
    env,
    latent_model,
    hidden_layer_size,
    n_hidden_layers,
    n_rollouts,
    n_epochs,
    batch_size,
    learning_rate,
    tensorboard_folder,
    checkpoint_folder,
):
    model = InverseDynamicsMDN(
        env,
        hidden_layer_size=hidden_layer_size,
        n_hidden_layers=n_hidden_layers,
        learning_rate=learning_rate,
        tensorboard_log=tensorboard_folder,
        checkpoint_folder=checkpoint_folder,
        latent_space=latent_model,
    )
    loss = model.learn(
        n_rollouts=n_rollouts,
        n_epochs=n_epochs,
        batch_size=batch_size,
        print_evaluation=True,
    )
    return model, loss


@ex.config
def config():
    env_name = "room"  # noqa:F841
    problem_spec = "default"  # noqa:F841

    n_rollouts_latent = 100  # noqa:F841
    n_epochs_latent = 1  # noqa:F841
    batch_size_latent = 32  # noqa:F841
    learning_rate_latent = 1e-4  # noqa:F841
    hidden_layer_size_latent = 200  # noqa:F841
    rnn_state_size_latent = 30  # noqa:F841

    n_rollouts_backward = 100  # noqa:F841
    n_epochs_backward = 1  # noqa:F841
    batch_size_backward = 32  # noqa:F841
    learning_rate_backward = 1e-4  # noqa:F841
    hidden_layer_size_backward = 512  # noqa:F841
    n_hidden_layers_backward = 3  # noqa:F841

    checkpoint_folder = "tf_ckpt"  # noqa:F841
    tensorboard_folder = "tf_logs"  # noqa:F841
    label_latent = None  # noqa:F841
    label_backward = None  # noqa:F841


@ex.automain
def main(
    _run,
    seed,
    env_name,
    problem_spec,
    n_rollouts_latent,
    n_epochs_latent,
    batch_size_latent,
    learning_rate_latent,
    hidden_layer_size_latent,
    rnn_state_size_latent,
    n_rollouts_backward,
    n_epochs_backward,
    batch_size_backward,
    learning_rate_backward,
    hidden_layer_size_backward,
    n_hidden_layers_backward,
    checkpoint_folder,
    tensorboard_folder,
    label_latent,
    label_backward,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa:F841
    if label_latent is None:
        label_latent = "{}_{}_{}_latent".format(env_name, problem_spec, timestamp)
    if label_backward is None:
        label_backward = "{}_{}_{}_backward".format(env_name, problem_spec, timestamp)

    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    print("Loading environment:", env_name)
    env, _, _, _ = get_problem_parameters(env_name, problem_spec)

    g1, g2 = tf.Graph(), tf.Graph()

    print("Learning latent model")
    checkpoint_folder_latent, tensorboard_folder_latent = _get_log_folders(
        checkpoint_folder, tensorboard_folder, label_latent
    )
    with g1.as_default():
        latent_model, latent_loss = train_latent_space_model(
            env,
            hidden_layer_size_latent,
            rnn_state_size_latent,
            n_rollouts_latent,
            n_epochs_latent,
            batch_size_latent,
            learning_rate_latent,
            tensorboard_folder_latent,
            checkpoint_folder_latent,
        )

    print("Learning inverse model")
    checkpoint_folder_backward, tensorboard_folder_backward = _get_log_folders(
        checkpoint_folder, tensorboard_folder, label_backward
    )
    with g2.as_default():
        inverse_dynamics_model, inverse_dynamics_loss = train_inverse_dynamics_model(
            env,
            latent_model,
            hidden_layer_size_backward,
            n_hidden_layers_backward,
            n_rollouts_backward,
            n_epochs_backward,
            batch_size_backward,
            learning_rate_backward,
            tensorboard_folder_backward,
            checkpoint_folder_backward,
        )

    results = {
        "checkpoint_folder_latent": checkpoint_folder_latent,
        "tensorboard_folder_latent": tensorboard_folder_latent,
        "latent_loss": latent_loss,
        "checkpoint_folder_backward": checkpoint_folder_backward,
        "tensorboard_folder_backward": tensorboard_folder_backward,
        "inverse_dynamics_loss": inverse_dynamics_loss,
    }
    print()
    print("-------------------------")
    for key, val in results.items():
        print("{}: {}".format(key, val))
    print("-------------------------")
    print()
    return results
