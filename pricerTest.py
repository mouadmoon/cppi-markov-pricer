import numpy as np
from models import BlackScholesModel, KouModel
from grid import Grid
from transition_matrix import TransitionMatrix
from pricerClosedFormulas import CPPIPricer
import time

# ============================================================
# TEST 1 — PARAMÈTRES PAPIER 2009
# ============================================================
print("=" * 60)
print("  TEST 1 — KOU PARAMÈTRES PAPIER 2009")
print("=" * 60)

m, r, T = 4, 0.05, 10
dt = 1/52
n_rebal = 520
X0 = np.exp(r * T)

kou_paper = KouModel(sigma=0.20, lambda_pos=0.1, lambda_neg=0.1,
                      eta_pos=0.05, eta_neg=0.10)

t0 = time.time()
grid1 = Grid(N=1000, vol=0.20, T=T, epsilon=1e-10, m=m, tanh_strength=18, r=r)
t1 = time.time()
tm1 = TransitionMatrix(grid=grid1, model=kou_paper, dt=dt)
t2 = time.time()
pricer1 = CPPIPricer(tm1.matrix, grid1.grid_values, n_rebal, r, dt, X0)
t3 = time.time()

print(f"Grille: {t1-t0:.1f}s, Matrice: {t2-t1:.1f}s, Pricer: {t3-t2:.1f}s")
print(f"X0 = {X0:.4f}")
print(f"g[j0] = {grid1.grid_values[np.argmin(np.abs(grid1.grid_values - X0))]:.4f}")
print()

gap1 = pricer1.gap_proportion()
exp_loss1 = pricer1.expected_loss()
cond_loss1 = pricer1.conditional_loss()
put1 = pricer1.price_put()

print(f"{'Mesure':<25} {'Markov':<15} {'Papier':<15}")
print("-" * 55)
print(f"{'Gap Proportion':<25} {gap1:<15.4%} {'5.71%':<15}")
print(f"{'Expected Loss':<25} {exp_loss1:<15.4%} {'1.052%':<15}")
print(f"{'Conditional Loss':<25} {cond_loss1:<15.4%} {'18.41%':<15}")
print(f"{'Put Price':<25} {put1:<15.6f}")

# ============================================================
# TEST 2 — KOU SUR MASI
# ============================================================
print()
print("=" * 60)
print("  TEST 2 — KOU SUR MASI")
print("=" * 60)

m, r, T = 4, 0.025, 10
dt = 1/52
n_rebal = 520
X0 = np.exp(r * T)

kou_masi = KouModel(sigma=0.1024, lambda_pos=1.97, lambda_neg=1.82,
                     eta_pos=0.0344, eta_neg=0.0402)

t0 = time.time()
grid2 = Grid(N=1000, vol=0.1024, T=T, epsilon=1e-10, m=m, tanh_strength=18, r=r)
tm2 = TransitionMatrix(grid=grid2, model=kou_masi, dt=dt)
pricer2 = CPPIPricer(tm2.matrix, grid2.grid_values, n_rebal, r, dt, X0)
t1 = time.time()
print(f"Temps total: {t1-t0:.1f}s")
print()

gap2 = pricer2.gap_proportion()
exp_loss2 = pricer2.expected_loss()
cond_loss2 = pricer2.conditional_loss()
put2 = pricer2.price_put()

print(f"Gap Proportion:   {gap2:.4%}")
print(f"Expected Loss:    {exp_loss2:.4%}")
print(f"Conditional Loss: {cond_loss2:.4%}")
print(f"Put Price:        {put2:.6f}")

# ============================================================
# TEST 3 — BS vs KOU SUR MASI
# ============================================================
print()
print("=" * 60)
print("  TEST 3 — COMPARAISON BS vs KOU SUR MASI")
print("=" * 60)

bs_masi = BlackScholesModel(volatility=0.1321)
grid3 = Grid(N=1000, vol=0.1321, T=T, epsilon=1e-10, m=m, tanh_strength=18, r=r)
tm3 = TransitionMatrix(grid=grid3, model=bs_masi, dt=dt)
pricer_bs = CPPIPricer(tm3.matrix, grid3.grid_values, n_rebal, r, dt, X0)

gap_bs = pricer_bs.gap_proportion()
put_bs = pricer_bs.price_put()

print(f"{'Mesure':<25} {'BS':<15} {'Kou':<15}")
print("-" * 55)
print(f"{'Gap Proportion':<25} {gap_bs:<15.4%} {gap2:<15.4%}")
print(f"{'Expected Loss':<25} {pricer_bs.expected_loss():<15.4%} {exp_loss2:<15.4%}")
print(f"{'Put Price':<25} {put_bs:<15.6f} {put2:<15.6f}")

# ============================================================
# TEST 4 — SENSIBILITÉ AU MULTIPLICATEUR
# ============================================================
print()
print("=" * 60)
print("  TEST 4 — SENSIBILITÉ AU MULTIPLICATEUR (KOU MASI)")
print("=" * 60)

print(f"{'m':<5} {'Gap (BS)':<15} {'Gap (Kou)':<15} {'ExpLoss (Kou)':<15} {'Put (Kou)':<15}")
print("-" * 65)

for m_val in [2, 3, 4, 5, 6]:
    X0 = np.exp(r * T)
    
    # Kou
    grid_k = Grid(N=1000, vol=0.1024, T=T, epsilon=1e-10, m=m_val, tanh_strength=18, r=r)
    tm_k = TransitionMatrix(grid=grid_k, model=kou_masi, dt=dt)
    pricer_k = CPPIPricer(tm_k.matrix, grid_k.grid_values, n_rebal, r, dt, X0)
    
    # BS
    grid_b = Grid(N=1000, vol=0.1321, T=T, epsilon=1e-10, m=m_val, tanh_strength=18, r=r)
    tm_b = TransitionMatrix(grid=grid_b, model=bs_masi, dt=dt)
    pricer_b = CPPIPricer(tm_b.matrix, grid_b.grid_values, n_rebal, r, dt, X0)
    
    gap_k = pricer_k.gap_proportion()
    gap_b = pricer_b.gap_proportion()
    exp_k = pricer_k.expected_loss()
    put_k = pricer_k.price_put()
    
    print(f"{m_val:<5} {gap_b:<15.4%} {gap_k:<15.4%} {exp_k:<15.4%} {put_k:<15.6f}")

# Sensibilité au cap d'exposition pour m=4 sur MASI
print("=== IMPACT DU CAP D'EXPOSITION ===")
print(f"{'Cap':<10} {'Gap':<15} {'ExpLoss':<15} {'Put':<15}")
print("-" * 55)

for w_max in [1.0, 1.5, 2.0, 3.0, 4.0]:
    grid_k = Grid(N=1000, vol=0.1024, T=10, epsilon=1e-10, m=4, tanh_strength=18, r=0.025)
    # Modifie w pour appliquer le cap
    tm_k = TransitionMatrix(grid=grid_k, model=kou_masi, dt=1/52)
    # Cap: w = min(w, w_max)
    tm_k.w = np.minimum(tm_k.w, w_max)
    tm_k.matrix = tm_k.compute_transition_matrix()
    
    pricer_k = CPPIPricer(tm_k.matrix, grid_k.grid_values, 520, 0.025, 1/52, np.exp(0.25))
    print(f"{w_max:<10.0%} {pricer_k.gap_proportion():<15.4%} {pricer_k.expected_loss():<15.4%} {pricer_k.price_put():<15.6f}")