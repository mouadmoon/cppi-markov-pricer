import numpy as np
from grid import Grid
from models import BlackScholesModel
from transition_matrix import TransitionMatrix
from scipy.stats import norm
from math import exp
from pricerClosedFormulas import CPPIPricer



m, sigma, r, T = 4, 0.1321, 0.025, 10
dt = 1/52
n_rebal = 520
X0 = np.exp(r * T)

model = BlackScholesModel(volatility=sigma)

m, sigma, r, T = 4, 0.30, 0.05, 10
dt = 1/52
n_rebal = 520
X0 = np.exp(r * T)
model = BlackScholesModel(volatility=sigma)

# Formule fermée
P_lsf = model.transition_cdf((m-1)/m, dt=dt)
B = (1-m)*P_lsf + m*model.partial_expectation((m-1)/m, dt=dt)
A = 1 - B
gap_closed = 1 - (1-P_lsf)**n_rebal
put_closed = (1-X0)*(1-A**n_rebal)
put_closed_act = put_closed * np.exp(-r*T)

print(f"Fermé: Gap={gap_closed:.6%}, Put={put_closed_act:.10f}")

# Markov
for N in [500, 1000, 2000]:
    grid_obj = Grid(N=N, vol=sigma, T=T, epsilon=1e-10, m=m, tanh_strength=18)
    tm = TransitionMatrix(grid=grid_obj, model=model, dt=dt)
    g = grid_obj.grid_values
    j0 = np.argmin(np.abs(g - X0))
    
    V = np.maximum(1.0 - g, 0).copy()
    for _ in range(n_rebal):
        V = np.exp(-r*dt) * tm.matrix @ V
    put_m = V[j0]
    
    V = (g < 1.0).astype(float).copy()
    for _ in range(n_rebal):
        V = tm.matrix @ V
    gap_m = V[j0]
    
    print(f"N={N}: Gap={gap_m:.6%}, Put={put_m:.10f}, "
          f"err_put={abs(put_m-put_closed_act)/max(abs(put_closed_act),1e-15):.2%}")