from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation import SimulationResult

@dataclass
class SimulationMetrics:
    net_pnl: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_abs_inventory: float
    fees_paid: float

    @property
    def total_pnl(self):
        return self.net_pnl
    
def _max_drawdown(wealth_history: list[float]):
    if not wealth_history:
        return 0.0
    
    running_peak = wealth_history[0]
    max_drawdown = 0.0

    for wealth in wealth_history:
        running_peak = max(running_peak,wealth)
        drawdown = running_peak-wealth
        max_drawdown = max(max_drawdown,drawdown)
    
    return max_drawdown

def calculate_metrics(
        result: SimulationResult,
        sharpe_annualisation_factor: float,
):
    net_pnl = (result.wealth_history[-1]-result.wealth_history[0] if result.wealth_history else 0.0)
    max_abs_inventory = (
        float(max(abs(inventory)for inventory in result.inventory_history))
        if result.wealth_history
        else 0.0
    )
    max_drawdown = _max_drawdown(result.wealth_history)

    if len(result.wealth_history) < 2:
        return SimulationMetrics(
            net_pnl = net_pnl,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=max_drawdown,
            max_abs_inventory= max_abs_inventory,
            fees_paid=result.fees_paid
        )
    
    series = pd.Series(result.wealth_history)
    returns = series.pct_change().replace([np.inf,-np.inf], np.nan).dropna()

    if len(returns) < 2:
        return SimulationMetrics(
            net_pnl=net_pnl,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=max_drawdown,
            max_abs_inventory=max_abs_inventory,
            fees_paid=result.fees_paid,
        )
    
    std_dev = returns.std()
    downside_returns = returns[returns<0]
    downside_dev = np.sqrt((downside_returns**2).mean())

    sharpe = 0.0
    if pd.isna(std_dev) or std_dev ==0:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / std_dev) * np.sqrt(sharpe_annualisation_factor)

    sortino = 0.0
    if not pd.isna(downside_dev) and downside_dev != 0:
        sortino = (
            returns.mean() / downside_dev * np.sqrt(sharpe_annualisation_factor)

        )
    
    return SimulationMetrics(
        net_pnl=net_pnl,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        max_abs_inventory=max_abs_inventory,
        fees_paid=result.fees_paid,
    )


def plot_comparison(
    naive_result: SimulationResult,
    as_result: SimulationResult,
    naive_metrics: SimulationMetrics,
    as_metrics: SimulationMetrics,
):
    _, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

    axes[0].plot(as_result.price_history, color="black", label="Market Price", alpha=0.5)
    axes[0].set_title("1. The Market (Geometric Brownian Motion)")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(naive_result.inventory_history, label="Naive", color="red", alpha=0.6)
    axes[1].plot(as_result.inventory_history, label="Avellaneda-Stoikov", color="blue")
    axes[1].set_title("2. Inventory Control")
    axes[1].set_ylabel("Inventory")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(
        naive_result.wealth_history,
        label=f"Naive (Sharpe: {naive_metrics.sharpe:.2f})",
        color="red",
        alpha=0.6,
    )
    axes[2].plot(
        as_result.wealth_history,
        label=f"AS (Sharpe: {as_metrics.sharpe:.2f})",
        color="blue",
    )
    axes[2].set_title("3. Cumulative Wealth (PnL with Fees)")
    axes[2].set_ylabel("PnL ($)")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.show()



