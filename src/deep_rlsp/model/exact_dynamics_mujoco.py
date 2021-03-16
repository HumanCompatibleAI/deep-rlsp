import gym
import math
import numpy as np

from deep_rlsp.util.helper import init_env_from_obs

GOLDEN_RATIO = (math.sqrt(5) + 1) / 2


def gss(f, a, b, tol=1e-12):
    """Golden section search to find the minimum of f on [a,b].

    This implementation does not reuse function evaluations and assumes the minimum is c
    or d (not on the edges at a or b).
    Source: https://en.wikipedia.org/wiki/Golden_section_search

    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678
    """
    c = b - (b - a) / GOLDEN_RATIO
    d = a + (b - a) / GOLDEN_RATIO
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        # We recompute both c and d here to avoid loss of precision
        # which may lead to incorrect results or infinite loop
        c = b - (b - a) / GOLDEN_RATIO
        d = a + (b - a) / GOLDEN_RATIO
    return (b + a) / 2


def loss(y1, y2):
    return np.square(y1 - y2).mean()


def invert(f, y, tolerance=1e-12, max_iters=1000):
    """
    f: function to invert (for expensive f, make sure to memoize)
    y: output to invert
    tolerance: Acceptible numerical error
    max_iters: Maximum iterations to try
    Returns: x' such that f(x') = y up to tolerance or up to amount achieved
    in max_iters time.
    """
    x = y
    for i in range(max_iters):
        dx = np.random.normal(size=x.shape)

        def line_fn(fac):
            return loss(f(x + fac * dx), y)

        factor = gss(line_fn, -10.0, 10.0)
        x = x + factor * dx
        if loss(f(x), y) < tolerance:
            # print(f"Took {i} iterations")
            return x
    print("Max it reached, loss value is {}".format(loss(f(x), y)))
    return x


class ExactDynamicsMujoco:
    def __init__(self, env_id, tolerance=1e-12, max_iters=1000):
        self.env = gym.make(env_id)
        self.env.reset()
        self.tolerance = tolerance
        self.max_iters = max_iters

        # @memoize
        def dynamics(s, a):
            self.env = init_env_from_obs(self.env, s)
            s2, _, _, _ = self.env.step(a)
            return s2

        self.dynamics = dynamics

    def inverse_dynamics(self, s2, a):
        def fn(s):
            return self.dynamics(s, a)

        return invert(fn, s2, tolerance=self.tolerance, max_iters=self.max_iters)


def main():
    dynamics = ExactDynamicsMujoco("InvertedPendulum-v2")
    s = dynamics.env.reset()
    for _ in range(5):
        a = dynamics.env.action_space.sample()
        s2 = dynamics.dynamics(s, a)
        inverted_s = dynamics.inverse_dynamics(s2, a)
        print(f"s2: {s2}\na:  {a}\ns:        {s}")
        print(f"Inverted: {inverted_s}")
        print("RMSE: {}\n".format(np.sqrt(loss(s, inverted_s))))
        s = s2


if __name__ == "__main__":
    main()
