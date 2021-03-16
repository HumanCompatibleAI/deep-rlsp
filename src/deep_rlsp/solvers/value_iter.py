import numpy as np


def value_iter(mdp, gamma, r, horizon, temperature=1, time_dependent_reward=False):
    """
    Performs (soft) value iteration to find a (Boltzman-)optimal policy.

    Finds the optimal state and state-action value functions via value iteration with
    the "soft" max-ent Bellman backup:

    $$
        Q_{sa} = r_s + gamma * \\sum_{s'} p(s'|s,a)V_{s'}
        V'_s = temperature * log(\\sum_a exp(Q_{sa}/temperature))
    $$

    Then, computes the Boltzmann rational policy

    $$
        \\pi_{s,a} = exp((Q_{s,a} - V_s)/temperature).
    $$

    Args
    ----------
    mdp (Env): MDP describing the environment.
    gamma (float): Discount factor; 0 <= gamma <= 1.
    r (np.ndarray): Reward vector of length mdp.nS.
    horizon (int): Time-horizon for the finite-horizon value iteration.
    temperature (float): Rationality constant for the soft value iteration equation.

    Returns (list of np.ndarray):
        Arrays of shape (mdp.nS, mdp.nA), each value p[t][s,a] is the probability of
        taking action a in state s at time t.
    """
    nS, nA = mdp.nS, mdp.nA

    if not time_dependent_reward:
        r = [r] * horizon  # Fast, since we aren't making copies

    policies = []
    V = np.copy(r[horizon - 1])
    for t in range(horizon - 2, -1, -1):
        future_values = mdp.T_matrix.dot(V).reshape((nS, nA))
        Q = np.expand_dims(r[t], axis=1) + gamma * future_values

        if temperature == 0:
            V = Q.max(axis=1)
            # Argmax to find the action number, then index into np.eye to
            # one hot encode. Note this will deterministically break ties
            # towards the smaller action.
            policy = np.eye(nA)[np.argmax(Q, axis=1)]
        else:
            # ∀ s: V_s = temperature * log(\\sum_a exp(Q_sa/temperature))
            # ∀ s,a: policy_{s,a} = exp((Q_{s,a} - V_s)/t)
            V = softmax(Q, temperature)
            V = np.expand_dims(V, axis=1)
            policy = np.exp((Q - V) / temperature)

        policies.append(policy)

        if gamma == 1:
            # When \\gamma=1, the backup operator is equivariant under adding
            # a constant to all entries of V, so we can translate min(V)
            # to be 0 at each step of the softmax value iteration without
            # changing the policy it converges to, and this fixes the problem
            # where log(nA) keep getting added at each iteration.
            V = V - np.amin(V)

    return policies[::-1]


def evaluate_policy(mdp, policy, start, gamma, r, horizon):
    """Expected reward from the policy."""
    policy = np.array(policy)
    if len(policy.shape) == 2:
        policy = [policy] * horizon
    V = r
    for t in range(horizon - 2, -1, -1):
        future_values = mdp.T_matrix.dot(V).reshape((mdp.nS, mdp.nA))
        Q = np.expand_dims(r, axis=1) + gamma * future_values
        V = np.sum(policy[t] * Q, axis=1)
    return V[start]


def softmax(x, t=1):
    """
    Numerically stable computation of t*log(\\sum_j^n exp(x_j / t))

    If the input is a 1D numpy array, computes it's softmax:
        output = t*log(\\sum_j^n exp(x_j / t)).
    If the input is a 2D numpy array, computes the softmax of each of the rows:
        output_i = t*log(\\sum_j^n exp(x_{ij} / t))

    Parameters
    ----------
    x : 1D or 2D numpy array

    Returns
    -------
    1D numpy array
        shape = (n,), where:
            n = 1 if x was 1D, or
            n is the number of rows (=x.shape[0]) if x was 2D.
    """
    assert t >= 0
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    if t == 0:
        return np.amax(x, axis=1)
    if x.shape[1] == 1:
        return x

    def softmax_2_arg(x1, x2, t):
        """
        Numerically stable computation of t*log(exp(x1/t) + exp(x2/t))

        Parameters
        ----------
        x1 : numpy array of shape (n,1)
        x2 : numpy array of shape (n,1)

        Returns
        -------
        numpy array of shape (n,1)
            Each output_i = t*log(exp(x1_i / t) + exp(x2_i / t))
        """

        def tlog(x):
            return t * np.log(x)

        def expt(x):
            return np.exp(x / t)

        max_x = np.amax((x1, x2), axis=0)
        min_x = np.amin((x1, x2), axis=0)
        return max_x + tlog(1 + expt((min_x - max_x)))

    sm = softmax_2_arg(x[:, 0], x[:, 1], t)
    # Use the following property of softmax_2_arg:
    # softmax_2_arg(softmax_2_arg(x1,x2),x3) = log(exp(x1) + exp(x2) + exp(x3))
    # which is true since
    # log(exp(log(exp(x1) + exp(x2))) + exp(x3)) = log(exp(x1) + exp(x2) + exp(x3))
    for (i, x_i) in enumerate(x.T):
        if i > 1:
            sm = softmax_2_arg(sm, x_i, t)
    return sm
