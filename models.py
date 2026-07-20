from scipy.stats import norm
import numpy as np


class BlackScholesModel:
    def __init__(self, volatility, sigma=None):
        if volatility < 0:
            raise ValueError("volatility must be non-negative")
        self.volatility = volatility
        self.sigma = sigma if sigma is not None else volatility

    def transition_cdf(self, z, dt):
        d_plus = (np.log(z) + 0.5 * self.sigma**2 * dt) / (self.sigma * np.sqrt(dt))
        return norm.cdf(d_plus)

    def partial_expectation(self, z, dt):
        d_moins = (np.log(z) - 0.5 * self.sigma**2 * dt) / (self.sigma * np.sqrt(dt))
        return norm.cdf(d_moins)