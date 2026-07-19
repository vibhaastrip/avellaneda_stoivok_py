import os
from pathlib import Path

import matplotlib

if os.environ.get("MPLBACKEND"):
    matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt  # noqa: E402

from src.storage.db import DB_PATH
from src.storage.queries import load_runs, paired_comp, summary_stats, sweep_summary


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
BATCH_DISTRIBUTIONS_PATH = REPORTS_DIR / "batch_distributions.png"
STRATEGY_DIAGNOSTICS_PATH = REPORTS_DIR / "strategy_diagnostics.png"


def _runs_for_as_naive_pairing(runs):
    gamma_summary = sweep_summary()
    if gamma_summary.empty:
        return runs

    best_gamma = gamma_summary.index[0]
    return runs[
        (runs["strategy"] == "naive")
        | (
            (runs["strategy"] == "avellaneda_stoikov")
            & (runs["gamma"] == best_gamma)
        )
    ]


def print_batch_report():
    if not Path(DB_PATH).exists():
        print("No results.db found. Run a batch first.")
        return

    stats = summary_stats()
    if stats.empty:
        print("results.db has no run data.")
        return

    print("Summary stats:")
    print(stats)
    print()
    print("Paired comparison:")
    print(paired_comp())


def plot_batch_distributions():
    if not Path(DB_PATH).exists():
        print("No results.db found; skipping batch distribution plot.")
        return None

    runs = load_runs()
    if runs.empty:
        print("No run data found; skipping batch distribution plot.")
        return None

    strategy_groups = list(runs.groupby("strategy"))
    if not strategy_groups:
        print("No strategy groups found; skipping batch distribution plot.")
        return None

    paired = _runs_for_as_naive_pairing(runs).pivot_table(
        index="seed",
        columns="strategy",
        values="net_pnl",
        aggfunc="first",
    ).dropna()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for strategy, strategy_runs in strategy_groups:
        axes[0, 0].hist(strategy_runs["net_pnl"], bins=40, alpha=0.45, label=strategy)
    axes[0, 0].set_title("PnL Distribution")
    axes[0, 0].set_xlabel("Net PnL")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    for strategy, strategy_runs in strategy_groups:
        axes[0, 1].hist(strategy_runs["sharpe"], bins=40, alpha=0.45, label=strategy)
    axes[0, 1].set_title("Sharpe Distribution")
    axes[0, 1].set_xlabel("Sharpe")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    for strategy, strategy_runs in strategy_groups:
        axes[1, 0].hist(
            strategy_runs["max_drawdown"],
            bins=40,
            alpha=0.45,
            label=strategy,
        )
    axes[1, 0].set_title("Max Drawdown Distribution")
    axes[1, 0].set_xlabel("Max Drawdown")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()

    if {"avellaneda_stoikov", "naive"}.issubset(paired.columns):
        pnl_diff = paired["avellaneda_stoikov"] - paired["naive"]
    else:
        pnl_diff = []
    axes[1, 1].hist(pnl_diff, bins=40, alpha=0.75, color="purple")
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Paired PnL Difference")
    axes[1, 1].set_xlabel("AS - Naive Net PnL")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    fig.savefig(BATCH_DISTRIBUTIONS_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved batch distribution plot to {BATCH_DISTRIBUTIONS_PATH}")
    return BATCH_DISTRIBUTIONS_PATH


def plot_strategy_diagnostics():
    if not Path(DB_PATH).exists():
        print("No results.db found; skipping strategy diagnostics plot.")
        return None

    runs = load_runs()
    if runs.empty:
        print("No run data found; skipping strategy diagnostics plot.")
        return None

    strategy_groups = list(runs.groupby("strategy"))
    if not strategy_groups:
        print("No strategy groups found; skipping strategy diagnostics plot.")
        return None

    paired = _runs_for_as_naive_pairing(runs).pivot_table(
        index="seed",
        columns="strategy",
        values=["net_pnl", "sharpe"],
        aggfunc="first",
    ).dropna()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for strategy, strategy_runs in strategy_groups:
        axes[0, 0].scatter(
            strategy_runs["max_drawdown"],
            strategy_runs["net_pnl"],
            alpha=0.45,
            label=strategy,
            s=18,
        )
    axes[0, 0].set_title("PnL vs Max Drawdown")
    axes[0, 0].set_xlabel("Max Drawdown")
    axes[0, 0].set_ylabel("Net PnL")
    axes[0, 0].legend()

    axes[0, 1].boxplot(
        [strategy_runs["max_drawdown"] for _, strategy_runs in strategy_groups],
        tick_labels=[strategy for strategy, _ in strategy_groups],
    )
    axes[0, 1].set_title("Drawdown by Strategy")
    axes[0, 1].set_ylabel("Max Drawdown")

    axes[1, 0].boxplot(
        [strategy_runs["max_abs_inventory"] for _, strategy_runs in strategy_groups],
        tick_labels=[strategy for strategy, _ in strategy_groups],
    )
    axes[1, 0].set_title("Inventory Exposure by Strategy")
    axes[1, 0].set_ylabel("Max Absolute Inventory")

    if {
        ("sharpe", "avellaneda_stoikov"),
        ("sharpe", "naive"),
    }.issubset(set(paired.columns)):
        sharpe_diff = paired[("sharpe", "avellaneda_stoikov")] - paired[
            ("sharpe", "naive")
        ]
    else:
        sharpe_diff = []
    axes[1, 1].hist(sharpe_diff, bins=40, alpha=0.75, color="green")
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Paired Sharpe Difference")
    axes[1, 1].set_xlabel("AS - Naive Sharpe")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    fig.savefig(STRATEGY_DIAGNOSTICS_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved strategy diagnostics plot to {STRATEGY_DIAGNOSTICS_PATH}")
    return STRATEGY_DIAGNOSTICS_PATH
