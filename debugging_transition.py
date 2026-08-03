from models import BlackScholesModel, KouModel
import numpy as np
from scipy.stats import norm
from grid import Grid
from transition_matrix import TransitionMatrix
from pricerClosedFormulas import CPPIPricer


# Test 1 — Paramètres papier 2009


kou_paper = KouModel(sigma=0.20, lambda_pos=0.1, lambda_neg=0.1,
                      eta_pos=0.05, eta_neg=0.10)
# ... construis grille, matrice, pricer ...
print(f"Gap: {pricer.gap_proportion():.4%}")
print(f"Expected Loss: {pricer.expected_loss():.4%}")
print(f"Conditional Loss: {pricer.conditional_loss():.4%}")

# Test 2 — MASI
kou_masi = KouModel(sigma=0.1024, lambda_pos=1.97, lambda_neg=1.82,
                     eta_pos=0.0344, eta_neg=0.0402)
# ... construis grille, matrice, pricer ...
print(f"Gap: {pricer.gap_proportion():.4%}")
print(f"Expected Loss: {pricer.expected_loss():.4%}")
print(f"Put Price: {pricer.price_put():.6f}")