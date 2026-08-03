
import scipy
from scipy.stats import norm, poisson
import numpy as np


class BlackScholesModel:
    def __init__(self, volatility, sigma=None):
        if volatility < 0:
            raise ValueError("volatility must be non-negative")
        self.volatility = volatility
        self.sigma = sigma if sigma is not None else volatility

    def transition_cdf(self, z, dt):
        z = np.maximum(z, np.finfo(float).tiny)
        d_plus = (np.log(z) + 0.5 * self.sigma**2 * dt) / (self.sigma * np.sqrt(dt))
        return norm.cdf(d_plus)

    def partial_expectation(self, z, dt):
        z = np.maximum(z, np.finfo(float).tiny)
        d_moins = (np.log(z) - 0.5 * self.sigma**2 * dt) / (self.sigma * np.sqrt(dt))
        return norm.cdf(d_moins)

class KouModel:
    def __init__(self, sigma, lambda_pos,lambda_neg, eta_pos, eta_neg):
        if sigma < 0 or lambda_pos < 0 or lambda_neg < 0  or eta_pos <=  0  or eta_neg <= 0 or eta_pos > 1:
            raise ValueError("Invalid parameters for Kou model")
        self.sigma = sigma
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.eta_pos = eta_pos
        self.eta_neg = eta_neg
        self.gamma = -self.sigma**2/2 - self.lambda_pos*self.eta_pos/(1 - self.eta_pos) + self.lambda_neg*self.eta_neg/(1 + self.eta_neg)

    
    def compute_cdf(self, z, dt, gamma, sigma, lam_p, lam_n, eta_p, eta_n):
        z = np.atleast_1d(z)
        log_z = np.log(np.maximum(z, 1e-300))
        result = np.zeros_like(log_z)
    
        n_max = 10
        for np_ in range(n_max + 1):
            pp = poisson.pmf(np_, lam_p * dt)
            if pp < 1e-15:
                continue
            for nm in range(n_max + 1):
                pm = poisson.pmf(nm, lam_n * dt)
                if pm < 1e-15:
                    continue
                
                mu = gamma * dt + np_ * eta_p - nm * eta_n
                var = sigma**2 * dt + np_ * eta_p**2 + nm * eta_n**2
                std = np.sqrt(var)
                
                # Vectorisé : calcule pour tous les z en une seule ligne
                result += pp * pm * norm.cdf((log_z - mu) / std)
    
        return result if len(result) > 1 else result[0]

    def transition_cdf(self, z, dt):
        return self.compute_cdf(z, dt, self.gamma, self.sigma, self.lambda_pos, self.lambda_neg, self.eta_pos, self.eta_neg)
    def partial_expectation(self, z, dt):
        gamma_prime = self.gamma + self.sigma**2
        lambda_pos_prime = self.lambda_pos / (1 - self.eta_pos)
        lambda_neg_prime = self.lambda_neg / (1 + self.eta_neg)
        eta_pos_prime = self.eta_pos / (1 - self.eta_pos)
        eta_neg_prime = self.eta_neg / (1 + self.eta_neg)
    
        return self.compute_cdf(z, dt, gamma_prime, self.sigma,
                              lambda_pos_prime, lambda_neg_prime,
                              eta_pos_prime, eta_neg_prime)



