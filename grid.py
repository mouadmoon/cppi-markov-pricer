import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
class Grid:
    def __init__(self, N, vol, T, epsilon, m,tanh_strength=3):
        self.N = N
        self.vol = vol
        self.T = T
        self.epsilon = epsilon
        self.m = m
        self.tanh_strength = tanh_strength
        self.grid_values = self.compute_grid()

    def compute_grid(self):
        upper_bound = 1 + np.exp(
            np.sqrt(self.T) * self.vol * self.m * scipy_stats.norm.ppf(1 - self.epsilon)
            - (self.vol*self.m) ** 2 * self.T / 2
        )
        lower_bound = (1 - self.m) * upper_bound


        # Build the auxiliary grid so that u=0 is always included.
        n_left = self.N // 2
        n_right = self.N - n_left
        left_u = np.linspace(-1, 0, n_left + 1)[:-1]
        right_u = np.linspace(0, 1, n_right)
        u = np.concatenate((left_u, right_u))

        strength = self.tanh_strength
        # Map u=0 to 1 and compress points near that center.
        distance_from_1 = (
            1 - np.tanh(strength * (1 - np.abs(u))) / np.tanh(strength)
        )
        grid_values = np.where(
            u < 0,
            1 - distance_from_1 * (1 - lower_bound),
            1 + distance_from_1 * (upper_bound - 1),
        )
        return grid_values
    def separation_grid(self):
        grid_values = np.sort(self.grid_values)  # Ensure the grid values are sorted
        # Add extrapolated outer boundaries around the interior midpoints.
        left_boundary = grid_values[0] - (grid_values[1] - grid_values[0]) / 2
        right_boundary = grid_values[-1] + (grid_values[-1] - grid_values[-2]) / 2
        middle_boundaries = (grid_values[:-1] + grid_values[1:]) / 2
        return np.concatenate(([left_boundary], middle_boundaries, [right_boundary]))
    
    def find_index(self, value,):
        grid_values = self.separation_grid(self.grid_values)
        if value < grid_values[0] or value > grid_values[-1]:
            raise ValueError("Value is out of the bounds of the grid.")
        return np.searchsorted(grid_values, value, side='right') - 1

if __name__ == "__main__":
    grid_instance = Grid(
        m=4,
        vol=0.1321,
        T=10,
        N=501,
        epsilon=1e-10,
    )
    grid_values = grid_instance.compute_grid()

    print("Grid shape:", grid_values.shape)
    print("First values:", grid_values[:5])
    print("Center value:", grid_values[len(grid_values) // 2])
    print("Last values:", grid_values[-5:])
    print("Increasing:", np.all(np.diff(grid_values) > 0))

    center_index = np.argmin(np.abs(grid_values - 1))
    start = max(0, center_index - 5)
    end = min(len(grid_values), center_index + 6)
    values_near_one = grid_values[start:end]

    print("Values around 1:", values_near_one)
    print("Distances from 1:", np.abs(values_near_one - 1))