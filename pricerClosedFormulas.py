import numpy as np

class CPPIPricer:
    def __init__(self, matrix, grid_values, n_rebal, r, dt, X0):
        self.matrix = matrix
        self.g = grid_values
        self.n_rebal = n_rebal
        self.r = r
        self.dt = dt
        self.X0 = X0

    def _value_at_X0(self, V):
        j = np.searchsorted(self.g, self.X0) - 1
        j = max(0, min(j, len(self.g) - 2))
        alpha = (self.X0 - self.g[j]) / (self.g[j+1] - self.g[j])
        return (1 - alpha) * V[j] + alpha * V[j+1]

    def propagateActualise(self, payoff):
        V = payoff.copy()
        for _ in range(self.n_rebal):
            V = np.exp(-self.r * self.dt) * self.matrix @ V
        return V

    def propagateNoActualise(self, payoff):
        V = payoff.copy()
        for _ in range(self.n_rebal):
            V = self.matrix @ V
        return V

    def price_put(self, strike=1.0):
        payoff = np.maximum(strike - self.g, 0)
        V = self.propagateActualise(payoff)
        return self._value_at_X0(V)

    def gap_proportion(self):
        payoff = (self.g < 1.0).astype(float)
        V = self.propagateNoActualise(payoff)
        return self._value_at_X0(V)

    def expected_loss(self):
        payoff = np.maximum(1.0 - self.g, 0)
        V = self.propagateNoActualise(payoff)    # SANS actualisation
        return self._value_at_X0(V)

    def conditional_loss(self):
        gap = self.gap_proportion()
        if gap < 1e-15:
            return 0.0
        return self.expected_loss() / gap