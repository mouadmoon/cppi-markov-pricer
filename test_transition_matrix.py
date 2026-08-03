from matplotlib.pyplot import grid
import numpy as np
from transition_matrix import TransitionMatrix
from grid import Grid
from models import BlackScholesModel
import pytest
class TestTransitionMatrix:
    def test_plot(self):
        import io
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        grid = Grid(N=50, vol=0.2, T=1, epsilon=1e-10, m=4)
        model = BlackScholesModel(volatility=0.2)
        transition_matrix_instance = TransitionMatrix(grid, model)

        fig, ax = plt.subplots()
        image = ax.imshow(transition_matrix_instance.Qzero, cmap='viridis', interpolation='nearest')
        fig.colorbar(image, ax=ax)
        fig.canvas.draw()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        assert buffer.tell() > 0, "Plot did not produce a renderable PNG buffer."
        plt.close(fig)

    def test_sum_of_rows(self):
        grid = Grid(N=500, vol=0.2, T=10, epsilon=1e-10, m=4)
        model = BlackScholesModel(volatility=0.2,sigma=0.5)
        transition_matrix_instance = TransitionMatrix(grid, model)
        row_sums = transition_matrix_instance.Qzero.sum(axis=1)
        print("Row sums of the transition matrix:", row_sums)
        print("Values of w:", transition_matrix_instance.w)
        print(transition_matrix_instance.grid.grid_values)
        assert all(abs(row_sum - 1) < 1e-10 for row_sum in row_sums), "Not all rows sum to 1."
    def test_diagonal_when_w_is_0(self):
        grid = Grid(N=50, vol=0.2, T=1, epsilon=1e-10, m=4)
        model = BlackScholesModel(volatility=0.2)
        transition_matrix_instance = TransitionMatrix(grid, model)
        for i in range(grid.N):
            if transition_matrix_instance.w[i] == 0:
                assert transition_matrix_instance.Qzero[i, i] == 1, f"Diagonal element at index {i} is not 1 when w is 0."
    def test_martingale_property(self):
        grid = Grid(N=100, vol=0.2, T=10, epsilon=1e-10, m=4)
        model = BlackScholesModel(volatility=0.2)
        interest = 0.025
        dt = 0.019
    
        transition_matrix_instance = TransitionMatrix(grid, model, interest=interest, dt=dt)
    
        for i in range(grid.N):
        # 1. Calcul de l'espérance réelle via la somme de la ligne Qone
            expected_value = sum(transition_matrix_instance.Qone[i, j] for j in range(grid.N))
        
        # 2. Valeur théorique sous la mesure risque-neutre (avec capitalisation e^{r*dt})
            theoretical_value = grid.grid_values[i] * np.exp(interest * dt)
        
            print(f"Row {i}: Expected value = {expected_value:.6f}, Theoretical value = {theoretical_value:.6f}")
        
        # 3. Test de l'écart relatif (sauf pour les cas limites / monétisés à 0)
            if theoretical_value > 0:
                assert abs(expected_value - theoretical_value) < 1e-2 * theoretical_value, \
                    f"Martingale property violated at index {i}."
if __name__ == "__main__":
    test_instance = TestTransitionMatrix()
    test_instance.test_plot()
    test_instance.test_sum_of_rows()
    test_instance.test_diagonal_when_w_is_0()
    test_instance.test_martingale_property()
    print("All tests passed.")