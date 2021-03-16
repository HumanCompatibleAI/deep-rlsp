import numpy as np
from scipy.stats import norm, laplace


class NormalDistribution(object):
    def __init__(self, mu, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.distribution = norm(loc=mu, scale=sigma)

    def rvs(self):
        """sample"""
        return self.distribution.rvs()

    def pdf(self, x):
        return self.distribution.pdf(x)

    def logpdf(self, x):
        return self.distribution.logpdf(x)

    def logdistr_grad(self, x):
        return (self.mu - x) / (self.sigma ** 2)


class LaplaceDistribution(object):
    def __init__(self, mu, b=1):
        self.mu = mu
        self.b = b
        self.distribution = laplace(loc=mu, scale=b)

    def rvs(self):
        """sample"""
        return self.distribution.rvs()

    def pdf(self, x):
        return self.distribution.pdf(x)

    def logpdf(self, x):
        return self.distribution.logpdf(x)

    def logdistr_grad(self, x):
        return (self.mu - x) / (np.fabs(x - self.mu) * self.b)
