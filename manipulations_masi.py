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

def cppi_gap_analysis(prices, multiplier=4, rebal_days=5, name="Actif"):
   
    # Rendements sur la fréquence de rebalancement
    rebal_prices = prices.iloc[::rebal_days]
    rebal_returns = np.log(rebal_prices / rebal_prices.shift(1)).dropna()
    
    gap_threshold = -1.0 / multiplier  # ex: -25% pour m=4
    n_periods = len(rebal_returns)
    n_gaps = (rebal_returns < gap_threshold).sum()
    
    p_lsf = n_gaps / n_periods  # probabilité de gap locale
    
    # Probabilité de gap sur différents horizons
    horizons = {'5 ans': int(5 * 252 / rebal_days),
                '8 ans': int(8 * 252 / rebal_days),
                '10 ans': int(10 * 252 / rebal_days)}
    
    freq_label = f"{rebal_days}j"
    if rebal_days == 5:
        freq_label = "hebdomadaire"
    elif rebal_days == 21:
        freq_label = "mensuel"
    
    print(f"{'='*60}")
    print(f"  ANALYSE GAP RISK CPPI — {name}")
    print(f"{'='*60}")
    print(f"  Multiplicateur         : m = {multiplier}")
    print(f"  Seuil de gap           : {gap_threshold:.2%}")
    print(f"  Rebalancement          : {freq_label}")
    print(f"  Périodes observées      : {n_periods}")
    print(f"  Périodes avec gap      : {n_gaps}")
    print(f"  P(gap locale)          : {p_lsf:.4%}")
    print()
    
    for label, n in horizons.items():
        p_gap_global = 1 - (1 - p_lsf) ** n
        print(f"  P(gap sur {label})     : {p_gap_global:.4%}  ({n} périodes)")
    
    # Plus grande perte sur la fréquence de rebalancement
    worst = rebal_returns.min()
    print(f"\n  Pire perte ({freq_label}) : {worst:.2%}")
    print(f"  Seuil de gap              : {gap_threshold:.2%}")
    print(f"  Le gap se serait produit  : {'OUI' if worst < gap_threshold else 'NON'}")
    print(f"{'='*60}\n")
    
    return {
        'p_lsf': p_lsf,
        'n_gaps': n_gaps,
        'n_periods': n_periods,
        'worst_return': worst,
        'gap_threshold': gap_threshold
    }


if __name__ == "__main__":
    stats_results = compute_stats(returns)
    print(stats_results)
    plot_histogram(returns)

    test_cases = [
        {"multiplier": 2, "rebal_days": 1, "name": "MASI - quotidien - m=2"},
        {"multiplier": 4, "rebal_days": 5, "name": "MASI - hebdomadaire - m=4"},
        {"multiplier": 6, "rebal_days": 21, "name": "MASI - mensuel - m=6"},
    ]

    results = []
    for case in test_cases:
        result = cppi_gap_analysis(
            prices=price_series,
            multiplier=case["multiplier"],
            rebal_days=case["rebal_days"],
            name=case["name"],
        )
        results.append({
            "cas": case["name"],
            **result,
        })

    results_df = pd.DataFrame(results)
    results_df["p_lsf"] = results_df["p_lsf"].map(lambda value: f"{value:.4%}")
    results_df["worst_return"] = results_df["worst_return"].map(lambda value: f"{value:.2%}")
    results_df["gap_threshold"] = results_df["gap_threshold"].map(lambda value: f"{value:.2%}")

    print("\nTABLEAU RECAPITULATIF")
    print(results_df.to_string(index=False))
 
