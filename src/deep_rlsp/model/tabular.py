import numpy as np
import scipy.sparse as sp

from deep_rlsp.model.base import TransitionModel


class TabularTransitionModel(TransitionModel):
    """
    Simulates a learned transition model based on the known MDP dynamics and
    a given policy.

    Can perform forward steps
        P(s' | s, \\pi)
    and backward steps
        P(s | s', \\pi)
    """

    def __init__(self, env):
        self.env = env
        self.T_policy_matrix = None
        self.policy = None
        self.inverse_policy = None
        self.inverse_dynamics_by_time = dict()
        self.update_initial_state_distribution()

    def models_observations(self):
        return False

    def update_initial_state_distribution(self, initial_state_distribution=None):
        """
        Sets the initial state distribution to a specific distribution.

        If no argument is given, the distribution provided by the environment is used.

        Can be called to use a specific distribution over initial states.
        """
        if initial_state_distribution is None:
            self.initial_state_distribution = self.env.get_initial_state_distribution()
        else:
            self.initial_state_distribution = initial_state_distribution

    def update_policy(self, policy, inverse_policy=None):
        """
        Updates the dynamics based on a new policy.
        """
        self.make_T_policy_matrix(policy)
        self.inverse_dynamics_by_time = dict()
        self.policy = policy
        self.inverse_policy = inverse_policy

    def make_T_policy_matrix(self, policy):
        """
        Creates a list of `nA` sparse matrices of the shape `nS` x `nS`
        that represents P(s', a | s) = P(s' | s, a) * pi(a | s).
        """
        # Note DL: this could probably be done faster if we kept the combined
        # state-action index
        self.T_policy_matrix = [
            sp.lil_matrix((self.env.nS, self.env.nS)) for _ in range(self.env.nA)
        ]
        for a in range(self.env.nA):
            for s in range(self.env.nS):
                state_action_index = s * self.env.nA + a
                self.T_policy_matrix[a][s, :] = (
                    self.env.T_matrix[state_action_index, :] * policy[s, a]
                )
            self.T_policy_matrix[a] = self.T_policy_matrix[a].tocsr()

    def _get_inverse_dynamics(self, t):
        """
        Returns the inverse dynamics model (at time t).

        The model is seperated into inverse environment dynamics `T_inv_t` and an
        inverse policy `policy_inv_t`.
            - `T_inv_t` corresponds to $\\tilde{T}_t(s | a, s')$
            - `policy_inv_t` corresponds to $\\tilde{\\pi}_t(a | s')$

        """

        # Note DL: The computation of the inverse dynamics is not optimized to be
        # maximally efficient. At the moment this is fine, because we only use this
        # with very small envs. If performance becomes an issue, the following points
        # could be changed:
        # - we still do some unnecessary converting between numpy and
        #   scipy.sparse matrices, which negatively impacts performance
        # - it is very likely that the computations could be simplified
        # - instead of computing the inverse dynamics lazily for each timestep, we could
        #   compute them eagerly, because we use all of the timesteps anyway.
        #   This would improve readability and performance, because we would not have
        #   to exponentiate `T_s_next_s` every time. (h/t Rohin)

        if t not in self.inverse_dynamics_by_time:
            if self.T_policy_matrix is None:
                raise Exception(
                    "Have to call `make_T_policy_matrix` before `get_inverse_dynamics`"
                )

            # T_s_next_s = P(s' | s, a ~ \pi)
            T_s_next_s = sum(self.T_policy_matrix[a] for a in range(self.env.nA))

            # P_t_s = P_t(s)
            # note that scipy.sparse implements * as matrix multiplication
            P_0_s = self.initial_state_distribution
            P_t_s = P_0_s * T_s_next_s ** t

            # P_t_s_a_next_s = P_t(s, a, s') = P(s' | s, a) * pi(a | s) * P_t(s)
            P_t_s_a_next_s = [
                sp.csr_matrix.multiply(self.T_policy_matrix[a], P_t_s.reshape((-1, 1)))
                for a in range(self.env.nA)
            ]

            # P_a_next_s = P_t(a, s') = \sum_s P_t(s, a, s')
            P_t_a_next_s = np.vstack(
                [
                    P_t_s_a_next_s[a].sum(axis=0).reshape((1, -1))
                    for a in range(self.env.nA)
                ]
            )

            # ignore 0/0 errors (filled with 0 in next step)
            with np.errstate(invalid="ignore"):
                # T_inv_t = \tilde{T}_t(s | a, s') = P_t(s, a, s') / P_t(a, s')
                T_inv_t = [
                    P_t_s_a_next_s[a] / P_t_a_next_s[a, :].reshape((1, -1))
                    for a in range(self.env.nA)
                ]

                # policy_inv_t = \tilde{\pi}_t(a | s') = P_t(a, s') / \sum_a P_t(a, s')
                policy_inv_t = P_t_a_next_s / P_t_a_next_s.sum(axis=0)

            # fill nans with 0
            T_inv_t = np.nan_to_num(T_inv_t)
            policy_inv_t = np.nan_to_num(policy_inv_t)

            T_inv_t = [sp.csr_matrix(T_inv_t[a]) for a in range(self.env.nA)]
            policy_inv_t = sp.csr_matrix(policy_inv_t)

            self.inverse_dynamics_by_time[t] = (T_inv_t, policy_inv_t)
        return self.inverse_dynamics_by_time[t]

    def _sample_state(self, probs):
        assert probs.shape == (self.env.nS,)
        state_id = np.random.choice(np.arange(self.env.nS), p=probs)
        return self.env.get_state_from_num(state_id)

    def _get_states(self, probs):
        return [
            (self.env.get_state_from_num(next_state_id), prob)
            for next_state_id, prob in enumerate(probs)
            if prob > 0
        ]

    def forward_sample(self, state):
        """
        Samples from P(s' | s, \\pi).
        """
        probs = self.forward_step(state, return_probs_vector=True)
        return self._sample_state(probs)

    def backward_sample(self, next_state, t):
        """
        Samples from P(s | s', \\pi) at timestep t.
        """
        probs = self.backward_step(next_state, t, return_probs_vector=True)
        return self._sample_state(probs)

    def forward_step(self, state, return_probs_vector=False):
        """
        Computes P(s' | s, \\pi) for a specific state s.

        Returns a vector of probabilities if `return_probs_vector` is `True`
        and a list of state-probability pairs otherwise.
        """
        state_id = self.env.get_num_from_state(state)
        probs = (
            sum(
                self.env.T_matrix[state_id * self.env.nA + a, :]
                * self.policy[state_id, a]
                for a in range(self.env.nA)
            )
            .toarray()
            .squeeze()
        )
        if return_probs_vector:
            return probs
        else:
            return self._get_states(probs)

    def backward_step(self, state, t, return_probs_vector=False):
        """
        Computes P(s | s', \\pi) at timestep t for a specific state s.

        Returns a vector of probabilities if `return_probs_vector` is `True`
        and a list of state-probability pairs otherwise.

        Makes a step from t to t-1 and raises an exception if there is no valid
        transition that could have lead to `state` at timestep `t-1`.
        """
        if t <= 0:
            raise ValueError("timestep t must be greater than 0")
        state_id = self.env.get_num_from_state(state)
        T_inv_t, policy_inv_t = self._get_inverse_dynamics(t - 1)
        if self.inverse_policy is not None:
            policy_inv_t = self.inverse_policy
        probs = sum(
            T_inv_t[a][:, state_id] * policy_inv_t[a, state_id]
            for a in range(self.env.nA)
        )
        probs = probs.toarray().squeeze()

        if self.inverse_policy is not None:
            probs /= probs.sum()

        if not np.isclose(np.sum(probs), 1):
            raise Exception(
                "Transition model can not do consistent backward step. "
                "Double check the MDP transitions and the consistency of the initial "
                "and the current state."
            )
        if return_probs_vector:
            return probs
        else:
            return self._get_states(probs)
