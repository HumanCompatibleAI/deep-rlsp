# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability import distributions as tfd


def flatten_(structure):
    """Combine all leaves of a nested structure into a tuple.

  The nested structure can consist of any combination of tuples, lists, and
  dicts. Dictionary keys will be discarded but values will ordered by the
  sorting of the keys.

  Args:
    structure: Nested structure.

  Returns:
    Flat tuple.
  """
    if isinstance(structure, dict):
        result = ()
        for key in sorted(list(structure.keys())):
            result += flatten_(structure[key])
        return result
    if isinstance(structure, (tuple, list)):
        result = ()
        for element in structure:
            result += flatten_(element)
        return result
    return (structure,)


def apply_mask(tensor, mask=None, length=None, value=0, debug=False):
    """Set padding elements of a batch of sequences to a constant.

  Useful for setting padding elements to zero before summing along the time
  dimension, or for preventing infinite results in padding elements. Either
  mask or length must be provided.

  Args:
    tensor: Tensor of sequences.
    mask: Boolean mask of valid indices.
    length: Batch of sequence lengths.
    value: Value to write into padding elemnts.
    debug: Test for infinite values; slows down performance.

  Raises:
    KeyError: If both or non of `mask` and `length` are provided.

  Returns:
    Masked sequences.
  """
    if len([x for x in (mask, length) if x is not None]) != 1:
        raise KeyError("Exactly one of mask and length must be provided.")
    with tf.name_scope("mask"):
        if mask is None:
            range_ = tf.range(tensor.shape[1].value)
            mask = range_[None, :] < length[:, None]
        batch_dims = mask.shape.ndims
        while tensor.shape.ndims > mask.shape.ndims:
            mask = mask[..., None]
        multiples = [1] * batch_dims + tensor.shape[batch_dims:].as_list()
        mask = tf.tile(mask, multiples)
        masked = tf.where(mask, tensor, value * tf.ones_like(tensor))
        if debug:
            masked = tf.check_numerics(masked, "masked")
        return masked


def nested_map(function, *structures, **kwargs):
    """Apply a function to every element in a nested structure.

  If multiple structures are provided as input, their structure must match and
  the function will be applied to corresponding groups of elements. The nested
  structure can consist of any combination of lists, tuples, and dicts.

  Args:
    function: The function to apply to the elements of the structure. Receives
        one argument for every structure that is provided.
    *structures: One of more nested structures.
    flatten: Whether to flatten the resulting structure into a tuple. Keys of
        dictionaries will be discarded.

  Returns:
    Nested structure.
  """
    # Named keyword arguments are not allowed after *args in Python 2.
    flatten = kwargs.pop("flatten", False)
    assert not kwargs, "map() got unexpected keyword arguments."

    def impl(function, *structures):
        if len(structures) == 0:
            return structures
        if all(isinstance(s, (tuple, list)) for s in structures):
            if len(set(len(x) for x in structures)) > 1:
                raise ValueError("Cannot merge tuples or lists of different length.")
            args = tuple((impl(function, *x) for x in zip(*structures)))
            if hasattr(structures[0], "_fields"):  # namedtuple
                return type(structures[0])(*args)
            else:  # tuple, list
                return type(structures[0])(args)
        if all(isinstance(s, dict) for s in structures):
            if len(set(frozenset(x.keys()) for x in structures)) > 1:
                raise ValueError("Cannot merge dicts with different keys.")
            merged = {
                k: impl(function, *(s[k] for s in structures)) for k in structures[0]
            }
            return type(structures[0])(merged)
        return function(*structures)

    result = impl(function, *structures)
    if flatten:
        result = flatten_(result)
    return result


class Base(tf.nn.rnn_cell.RNNCell):
    def __init__(self, transition_tpl, posterior_tpl, reuse=None):
        super(Base, self).__init__(_reuse=reuse)
        self._posterior_tpl = posterior_tpl
        self._transition_tpl = transition_tpl
        self._debug = False

    @property
    def state_size(self):
        raise NotImplementedError

    @property
    def updates(self):
        return []

    @property
    def losses(self):
        return []

    @property
    def output_size(self):
        return (self.state_size, self.state_size)

    def zero_state(self, batch_size, dtype):
        return nested_map(
            lambda size: tf.zeros([batch_size, size], dtype), self.state_size
        )

    def call(self, inputs, prev_state):
        obs, prev_action, use_obs = inputs
        if self._debug:
            with tf.control_dependencies([tf.assert_equal(use_obs, use_obs[0, 0])]):
                use_obs = tf.identity(use_obs)
        use_obs = use_obs[0, 0]
        zero_obs = nested_map(tf.zeros_like, obs)
        prior = self._transition_tpl(prev_state, prev_action, zero_obs)
        posterior = tf.cond(
            use_obs,
            lambda: self._posterior_tpl(prev_state, prev_action, obs),
            lambda: prior,
        )
        return (prior, posterior), posterior


class RSSM(Base):
    """Deterministic and stochastic state model.

  The stochastic latent is computed from the hidden state at the same time
  step. If an observation is present, the posterior latent is compute from both
  the hidden state and the observation.

  Prior:    Posterior:

  (a)       (a)
     \\         \\
      v         v
  [h]->[h]  [h]->[h]
      ^ |       ^ :
     /  v      /  v
  (s)  (s)  (s)  (s)
                  ^
                  :
                 (o)
  """

    def __init__(
        self,
        state_size,
        belief_size,
        embed_size,
        future_rnn=True,
        mean_only=False,
        min_stddev=0.1,
        fixed_latent_stddev=None,
        activation=tf.nn.elu,
        num_layers=1,
    ):
        self._state_size = state_size
        self._belief_size = belief_size
        self._embed_size = embed_size
        self._future_rnn = future_rnn
        self._cell = tf.contrib.rnn.GRUBlockCell(self._belief_size)
        self._kwargs = dict(units=self._embed_size, activation=activation)
        self._mean_only = mean_only
        self._min_stddev = min_stddev
        self._fixed_latent_stddev = fixed_latent_stddev
        self._num_layers = num_layers
        super(RSSM, self).__init__(
            tf.make_template("transition", self._transition),
            tf.make_template("posterior", self._posterior),
        )

    @property
    def state_size(self):
        return {
            "mean": self._state_size,
            "stddev": self._state_size,
            "sample": self._state_size,
            "belief": self._belief_size,
            "rnn_state": self._belief_size,
        }

    def dist_from_state(self, state, mask=None):
        """Extract the latent distribution from a prior or posterior state."""
        if mask is not None:
            stddev = apply_mask(state["stddev"], mask, value=1)
        else:
            stddev = state["stddev"]
        dist = tfd.MultivariateNormalDiag(state["mean"], stddev)
        return dist

    def features_from_state(self, state):
        """Extract features for the decoder network from a prior or posterior."""
        return tf.concat([state["sample"], state["belief"]], -1)

    def divergence_from_states(self, lhs, rhs, mask=None):
        """Compute the divergence measure between two states."""
        lhs = self.dist_from_state(lhs, mask)
        rhs = self.dist_from_state(rhs, mask)
        divergence = tfd.kl_divergence(lhs, rhs)
        if mask is not None:
            divergence = apply_mask(divergence, mask)
        return divergence

    def _transition(self, prev_state, prev_action, zero_obs):
        """Compute prior next state by applying the transition dynamics."""
        hidden = tf.concat([prev_state["sample"], prev_action], -1)
        for _ in range(self._num_layers):
            hidden = tf.layers.dense(hidden, **self._kwargs)
        rnn_state = prev_state["rnn_state"]
        belief, rnn_state = self._cell(hidden, rnn_state)
        if self._future_rnn:
            hidden = belief
        for _ in range(self._num_layers):
            hidden = tf.layers.dense(hidden, **self._kwargs)
        mean = tf.layers.dense(hidden, self._state_size, None)
        if self._fixed_latent_stddev is not None:
            stddev = tf.fill(
                (tf.shape(hidden)[0], self._state_size), self._fixed_latent_stddev
            )
        else:
            stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
            stddev += self._min_stddev
        if self._mean_only:
            sample = mean
        else:
            sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "belief": belief,
            "rnn_state": rnn_state,
        }

    def _posterior(self, prev_state, prev_action, obs):
        """Compute posterior state from previous state and current observation."""
        prior = self._transition_tpl(prev_state, prev_action, tf.zeros_like(obs))
        hidden = tf.concat([prior["belief"], obs], -1)
        for _ in range(self._num_layers):
            hidden = tf.layers.dense(hidden, **self._kwargs)
        mean = tf.layers.dense(hidden, self._state_size, None)
        if self._fixed_latent_stddev is not None:
            stddev = tf.fill(
                (tf.shape(hidden)[0], self._state_size), self._fixed_latent_stddev
            )
        else:
            stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
            stddev += self._min_stddev
        if self._mean_only:
            sample = mean
        else:
            sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "belief": prior["belief"],
            "rnn_state": prior["rnn_state"],
        }
