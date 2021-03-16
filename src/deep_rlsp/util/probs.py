import numpy as np


def sample_observation(obs_probs_tuple):
    # inverse of zip(observations, probs)
    observations, probs = zip(*obs_probs_tuple)
    i = np.random.choice(len(probs), p=probs)
    return observations[i]


def get_out_probs_tuple(out_probs, dtype, shape):
    """
    Convert between a dictionary of state-probability pairs to a tuple.

    Returns a tuple consisting of an observation and a probability given a dictionary
    with string representations of the observations as keys and probabilities as values.

    Normalizes the probabilities in the process.
    """
    probs_sum = sum(prob for _, prob in out_probs.items())
    out_probs_tuple = (
        (np.fromstring(obs_str, dtype).reshape(shape), prob / probs_sum)
        for obs_str, prob in out_probs.items()
    )
    return out_probs_tuple


def add_obs_prob_to_dict(dictionary, obs, prob):
    """
    Updates a dictionary that tracks a probability distribution over observations.
    """
    obs_str = obs.tostring()
    if obs_str not in dictionary:
        dictionary[obs_str] = 0
    dictionary[obs_str] += prob
    return dictionary
