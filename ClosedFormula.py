import numpy as np
from grid import Grid
from models import BlackScholesModel
from transition_matrix import TransitionMatrix
from scipy.stats import norm
from math import exp
from pricerClosedFormulas import CPPIPricer

# Paramètres
m, sigma, r, T = 4, 0.50, 0.05, 10
dt = 1 / 52
n_rebal = 520
X0 = np.exp(r * T)

# Construction des objets
grid_obj = Grid(N=1000, vol=sigma, T=T, epsilon=1e-10, m=m)
model = BlackScholesModel(volatility=sigma)
tm = TransitionMatrix(grid=grid_obj, model=model, dt=dt)

# Pricer
pricer = CPPIPricer(
    matrix=tm.matrix,
    grid=grid_obj,
    n_rebal=n_rebal,
    r=r,
    dt=dt,
    X0=X0
)

# Résultats Markov
gap_markov = pricer.gap_proportion()
put_markov = pricer.price_put()
exp_loss_markov = pricer.expected_loss()

# Formule fermée
P_lsf = model.transition_cdf((m - 1) / m, dt=dt)
B = (1 - m) * P_lsf + m * model.partial_expectation((m - 1) / m, dt=dt)
A = 1 - B
gap_closed = 1 - (1 - P_lsf) ** n_rebal
put_closed_non_act = (1 - X0) * (1 - A ** n_rebal)

# Comparaison
print(f"{'Mesure':<25} {'Markov':<15} {'Fermé':<15}")
print("-" * 55)
print(f"{'Gap Proportion':<25} {gap_markov:<15.6f} {gap_closed:<15.6f}")
print(f"{'Expected Loss':<25} {exp_loss_markov:<15.6f}")
print(f"{'Put (actualisé)':<25} {put_markov:<15.6f} {put_closed_non_act * np.exp(-r*T):<15.6f}")

X0 = np.exp(r * T)
g = grid_obj.grid_values
j0 = np.argmin(np.abs(g - X0))
print(f"X0 = {X0:.4f}")
print(f"g[j0] = {g[j0]:.4f}")
print(f"Écart = {abs(g[j0] - X0):.4f}")
print(f"w[j0] = {tm.w[j0]:.6f}")

# Combien de points entre 1.0 et 2.0 ?
mask = (g > 1.0) & (g < 2.0)
print(f"Points entre 1.0 et 2.0: {mask.sum()}")
print(f"Valeurs: {g[mask]}")
grid_test = Grid(N=1000, vol=0.50, T=10, epsilon=1e-10, m=4, tanh_strength=18)
g = grid_test.grid_values

X0 = np.exp(0.05 * 10)
j0 = np.argmin(np.abs(g - X0))

print(f"X0 = {X0:.4f}, g[j0] = {g[j0]:.4f}, écart = {abs(g[j0]-X0):.4f}")
print(f"Points entre 0.8 et 1.2: {np.sum((g > 0.8) & (g < 1.2))}")
print(f"Points entre 1.0 et 2.0: {np.sum((g > 1.0) & (g < 2.0))}")
print(f"Espacement moyen autour de X=1: {np.mean(np.diff(g[(g > 0.8) & (g < 1.2)])):.4f}")


for N in [200, 500, 1000, 2000, 3000]:
    grid_obj = Grid(N=N, vol=sigma, T=T, epsilon=1e-10, m=m, tanh_strength=20)
    model = BlackScholesModel(volatility=sigma)
    tm = TransitionMatrix(grid=grid_obj, model=model, dt=dt)
    
    g = grid_obj.grid_values
    X0 = np.exp(r * T)
    j0 = np.argmin(np.abs(g - X0))
    
    # Gap proportion
    payoff_gap = (g < 1.0).astype(float)
    V = payoff_gap.copy()
    for _ in range(n_rebal):
        V = tm.matrix @ V
    gap_markov = V[j0]
    
    # Put actualisé
    payoff_put = np.maximum(1.0 - g, 0)
    V = payoff_put.copy()
    for _ in range(n_rebal):
        V = np.exp(-r * dt) * tm.matrix @ V
    put_markov = V[j0]
    
    err_gap = abs(gap_markov - gap_closed) / gap_closed
    err_put = abs(put_markov - put_closed_non_act*np.exp(-r*T)) / put_closed_non_act*np.exp(-r*T)
    
    print(f"N={N:>5}: Gap={gap_markov:.6f} (err={err_gap:.2%}), "
          f"Put={put_markov:.6f} (err={err_put:.2%})")