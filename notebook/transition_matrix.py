from grid import Grid
from scipy import stats as scipy_stats
from scipy.stats import norm
from models import BlackScholesModel
import numpy as np

class TransitionMatrix:
    def __init__(self, grid: Grid, model: BlackScholesModel, interest=0.005,dt = 1/52):
        self.grid = grid
        self.model = model
        self.interest = interest
        self.dt = dt
        self.w = self.calculate_w()
        self.matrix = self.compute_transition_matrix()
    def calculate_w(self):
        W = np.zeros(self.grid.N)
        for i in range(self.grid.N):
            if self.grid.grid_values[i] <= 0:
                W[i] = 0
            else:
                W[i] = max(0, self.grid.m*(self.grid.grid_values[i] - 1)/self.grid.grid_values[i])
        return W
    def compute_transition_matrix(self):
        N = self.grid.N
        g = self.grid.grid_values
        M = np.zeros((N, N))

        for i in range(N):
            if self.w[i] == 0:
                M[i, i] = 1.0
                continue

            # L pour tous les points d'un coup
            L = 1 + (g - g[i]) / (self.w[i] * g[i])
            L = np.maximum(L, 1e-15)

            # UN appel vectorisé au lieu de N appels scalaires
            cdf_vals = self.model.transition_cdf(L, self.dt)
            pe_vals = self.model.partial_expectation(L, self.dt)

            # Queue gauche → g[0]
            M[i, 0] += cdf_vals[0]

            # Queue droite → g[N-1]
            M[i, N-1] += 1.0 - cdf_vals[N-1]

            # Q et Q1 vectorisés (taille N-1)
            Q = cdf_vals[1:] - cdf_vals[:-1]
            Q1 = (self.w[i] * g[i] * (pe_vals[1:] - pe_vals[:-1])
                + (1 - self.w[i]) * g[i] * Q)

            dg = g[1:] - g[:-1]

            # Évite division par zéro
            dg_safe = np.where(np.abs(dg) < 1e-15, 1e-15, dg)

            m_plus = (g[1:] * Q - Q1) / dg_safe
            m_minus = (Q1 - g[:-1] * Q) / dg_safe

            M[i, :-1] += m_plus
            M[i, 1:] += m_minus

        self.matrix = M
        return M
