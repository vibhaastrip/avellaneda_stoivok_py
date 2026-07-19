import pandas as pd
from scipy import stats

from src.storage.db import engine
def load_runs():
    try:
        return pd.read_sql("SELECT * FROM runs", engine)
    except Exception:
        return pd.DataFrame()
    
def summary_stats():
    runs = load_runs()
    if runs.empty:
        return pd.DataFrame()
    
    return runs.groupby("strategy").agg(
        mean_pnl=("net_pnl", "mean"),
        pnl_std=("net_pnl", "std"),
        mean_sharpe=("sharpe", "mean"),
        mean_sortino=("sortino", "mean"),
        mean_max_drawdown=("max_drawdown", "mean"),
        mean_max_abs_inventory=("max_abs_inventory", "mean"),
        n_runs=("seed", "count"),
    )

def paired_comp(strategy_a: str = 'avellaneda_stoikiov', strategy_b: str = 'naive'):
    runs = load_runs()
    if runs.empty:
        runs = pd.DataFrame()

    paired = runs[runs["strategy"].isin([strategy_a,strategy_b])].pivot_table(index='seed', columns='strategy',values=['net_pnl','sharpe'])

    if strategy_a not in paired["net_pnl"].columns or strategy_b not in paired["net_pnl"].columns:
        return pd.DataFrame
    
    paired = paired.dropna()

    results= []
    for metric in ['net_pnl','sharpe']:
        a_values = paired[metric][strategy_a]
        b_values = paired[metric][strategy_b]
        mean_diff = (a_values-b_values).mean()
        _ , p_values = stats.ttest_rel(a_values,b_values)
        results.append({"metric": metric, "mean_diff": mean_diff, "p_value": p_values})


    return pd.DataFrame(results).set_index('metric')

def sweep_summary():
    runs = load_runs()
    if runs.empty:
        runs = pd.DataFrame()
    
    as_runs = runs[runs["strategy"] == "avellaneda_stoikov"]
    if as_runs.empty:
        return pd.DataFrame()
    
    summary = as_runs.groupby("gamma").agg(
        mean_pnl=("net_pnl", "mean"),
        pnl_std=("net_pnl", "std"),
        mean_sharpe=("sharpe", "mean"),
        mean_max_drawdown=("max_drawdown", "mean"),
        n_runs=("seed", "count")
    )


    return summary.sort_values("mean_sharpe", ascending = False)

def gamma_paired_comp():
    gamma_summary = sweep_summary()
    if gamma_summary.empty:
        return pd.DataFrame()

    best_gamma = gamma_summary.index[0]

    runs = load_runs()
    as_best = runs[(runs["strategy"] == "avellaneda_stoikov") & (runs["gamma"] == best_gamma)]
    naive = runs[runs["strategy"] == "naive"]

    combined = pd.concat([as_best, naive])
    paired = combined.pivot_table(index="seed", columns="strategy", values=["net_pnl", "sharpe"]).dropna()

    if "avellaneda_stoikov" not in paired["net_pnl"].columns or "naive" not in paired["net_pnl"].columns:
        return pd.DataFrame()

    results = []
    for metric in ["net_pnl", "sharpe"]:
        a_values = paired[metric]["avellaneda_stoikov"]
        b_values = paired[metric]["naive"]
        mean_diff = (a_values - b_values).mean()
        _, p_value = stats.ttest_rel(a_values, b_values)
        results.append({"metric": metric, "mean_diff": mean_diff, "p_value": p_value})

    return pd.DataFrame(results).set_index("metric")
