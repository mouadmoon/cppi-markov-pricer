import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_excel("MASI_HISTO.xls")

price_series = pd.to_numeric(df["VALEUR RÉF"], errors="coerce").dropna()

returns = price_series.pct_change().dropna()
def compute_stats(returns, annualization_factor=252):
    n = len(returns)

    # Tendance centrale
    mean_daily = returns.mean()
    mean_annual = mean_daily * annualization_factor
    median_daily = returns.median()
    vol_daily = returns.std(ddof=1)
    vol_annual = vol_daily * np.sqrt(annualization_factor)

    # Forme de la distribution
    skewness = scipy_stats.skew(returns)
    kurt_excess = scipy_stats.kurtosis(returns)  # excess kurtosis (normal = 0)

    # Extremes
    min_return = returns.min()
    max_return = returns.max()

    # Test de normalité
    jb_stat, jb_pvalue = scipy_stats.jarque_bera(returns)

    results = {
        "Nombre d'observations": n,
        "Rendement moyen quotidien": mean_daily,
        "Rendement annualisé": mean_annual,
        "Médiane quotidienne": median_daily,
        "Volatilité quotidienne": vol_daily,
        "Volatilité annualisée": vol_annual,
        "Skewness": skewness,
        "Kurtosis": kurt_excess,
        "Rendement min": min_return,
        "Rendement max": max_return,
        "Jarque-Bera statistique": jb_stat,
        "Jarque-Bera p-value": jb_pvalue,
        "Distribution normale ?": "Oui" if jb_pvalue > 0.05 else "Non",
    }
    return results
def plot_histogram(returns, bins=30):
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=bins, edgecolor="black", alpha=0.7)
    plt.title("Histogramme des rendements quotidiens")
    plt.xlabel("Rendement")
    plt.ylabel("Fréquence")
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig("returns_histogram.png", dpi=200)
    plt.show()
    plt.close()

stats_results = compute_stats(returns)
print(stats_results)
plot_histogram(returns)
