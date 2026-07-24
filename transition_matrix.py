from grid import Grid
from scipy import stats as scipy_stats
from scipy.stats import norm
from models import BlackScholesModel
import numpy as np

class TransitionMatrix:
    def __init__(self, grid: Grid, model: BlackScholesModel, interest=0.025,dt = 0.019):
        self.grid = grid
        self.model = model
        self.interest = interest
        self.dt = dt
        self.w = self.calculate_w()
        self.Qzero,self.Qone,self.M_minus,self.M_plus = self.compute_transition_matrix()
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
        transition_matrix = np.zeros((N, N))
        partial_expectation_matrix = np.zeros((N, N))
        separation_grid = self.grid.separation_grid()
        L = np.zeros((self.grid.N, self.grid.N+1))
        M_plus = np.zeros((self.grid.N, self.grid.N))
        M_minus = np.zeros((self.grid.N, self.grid.N))
        for i in range(N):
            if self.w[i] == 0:
                transition_matrix[i, i] = 1
                partial_expectation_matrix[i, i] = self.grid.grid_values[i]
                continue
            for j in range(N + 1):
                if self.w[i] == 0:
                    L[i, j] = 0
                else:
                    L[i, j] = 1 + (separation_grid[j]-self.grid.grid_values[i])/(self.w[i]*self.grid.grid_values[i])
            for j in range(N):    
                lower_bound = L[i, j]
                upper_bound = L[i,j+1] if j < N - 1 else np.inf
                transition_matrix[i, j] = (
                    self.model.transition_cdf(upper_bound, dt=self.dt)
                    - self.model.transition_cdf(lower_bound, dt=self.dt)
                )
                partial_expectation_matrix[i,j] = self.w[i] * self.grid.grid_values[i] * (
                    self.model.partial_expectation(upper_bound, dt=self.dt)
                    - self.model.partial_expectation(lower_bound, dt=self.dt)
                ) + (1 - self.w[i]) * self.grid.grid_values[i] * transition_matrix[i, j]*np.exp(self.interest * self.dt)
            for j in range(N-1):
                if j == 0:
                    M_plus[i, j] = (self.grid.grid_values[j+1]*transition_matrix[i, j] - partial_expectation_matrix[i, j])/(self.grid.grid_values[j+1]-self.grid.grid_values[j])
                    M_minus[i, j] = 0
                else:
                    M_plus[i, j] = (self.grid.grid_values[j+1]*transition_matrix[i, j] - partial_expectation_matrix[i, j])/(self.grid.grid_values[j+1]-self.grid.grid_values[j])
                    M_minus[i, j] = (partial_expectation_matrix[i, j-1] - self.grid.grid_values[j-1]*transition_matrix[i, j-1])/(self.grid.grid_values[j]-self.grid.grid_values[j-1])
        return transition_matrix,partial_expectation_matrix,M_minus,M_plus
