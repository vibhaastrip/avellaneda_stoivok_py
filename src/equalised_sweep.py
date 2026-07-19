
import argparse
from dataclasses import replace
 
import numpy as np
import pandas as pd
from scipy import stats
 
from src.config import SimulationConfig
from src.simulation import run_simulation, Strategy
from src.analysis import calculate_metrics
 
 
def as_average_spread(gamma: float, sigma: float, k: float, T: float) -> float:
    order_flow_term = (2.0 / gamma) * np.log(1.0 + gamma / k)
    risk_term_at_0 = gamma * sigma ** 2 * T
    return risk_term_at_0 / 2.0 + order_flow_term
 
 
def compare_at_gamma(gamma: float, n_runs: int, base_config: SimulationConfig) -> dict:
    matched_spread = as_average_spread(gamma, base_config.sigma, base_config.k, base_config.T)
 
    cfg = replace(base_config, gamma=gamma, naive_spread=matched_spread)
 
    naive_sharpes, as_sharpes = [], []
    naive_pnls, as_pnls = [], []
 
    for seed in range(n_runs):
        naive_result = run_simulation(cfg, Strategy.NAIVE, rng=np.random.default_rng(seed))
        as_result = run_simulation(cfg, Strategy.AVELLANEDA_STOIKOV, rng=np.random.default_rng(seed))
 
        naive_metrics = calculate_metrics(naive_result, cfg.sharpe_annualization_factor)
        as_metrics = calculate_metrics(as_result, cfg.sharpe_annualization_factor)
 
        naive_sharpes.append(naive_metrics.sharpe)
        as_sharpes.append(as_metrics.sharpe)
        naive_pnls.append(naive_metrics.net_pnl)
        as_pnls.append(as_metrics.net_pnl)
 
    _, sharpe_p = stats.ttest_rel(as_sharpes, naive_sharpes)
    _, pnl_p = stats.ttest_rel(as_pnls, naive_pnls)
 
    return {
        "gamma": gamma,
        "matched_spread": matched_spread,
        "mean_sharpe_diff": np.mean(as_sharpes) - np.mean(naive_sharpes),
        "sharpe_p_value": sharpe_p,
        "mean_pnl_diff": np.mean(as_pnls) - np.mean(naive_pnls),
        "pnl_p_value": pnl_p,
    }
 
 
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=300)
    parser.add_argument("--gamma-values", default="1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0")
    return parser.parse_args()
 
 
def main():
    args = _parse_args()
    gamma_values = [float(g.strip()) for g in args.gamma_values.split(",") if g.strip()]
    base_config = SimulationConfig()
 
    rows = [compare_at_gamma(g, args.n_runs, base_config) for g in gamma_values]
    results = pd.DataFrame(rows).set_index("gamma")
 
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(results)
    print()
    print("Rows where sharpe_p_value < 0.05 show a genuine (confound-free)")
    print("inventory-skewing benefit at that gamma, once spread width is matched.")
 
 
if __name__ == "__main__":
    main()