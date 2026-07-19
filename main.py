import argparse

from src.analysis import calculate_metrics, plot_comparison
from src.analysis_report import (
    plot_batch_distributions,
    plot_strategy_diagnostics,
    print_batch_report,
)
from src.config import SimulationConfig
from src.simulation import Strategy, run_simulation
from src.storage.queries import gamma_paired_comp, sweep_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        summary = sweep_summary()
        if summary.empty:
            print("No AS sweep data found.")
        else:
            print(summary)
            print()
            print(gamma_paired_comp())
        return

    if args.batch:
        print_batch_report()
        if args.plot:
            plot_batch_distributions()
            plot_strategy_diagnostics()
        return

    config = SimulationConfig()

    naive_result = run_simulation(config, Strategy.NAIVE)
    as_result = run_simulation(config, Strategy.AVELLANEDA_STOIKOV)

    naive_metrics = calculate_metrics(
        naive_result,
        config.sharpe_annualization_factor,
    )
    as_metrics = calculate_metrics(
        as_result,
        config.sharpe_annualization_factor,
    )

    print(
        f"Naive Strategy: PnL = ${naive_metrics.net_pnl:.2f} | "
        f"Sharpe = {naive_metrics.sharpe:.2f}"
    )
    print(
        f"AS Strategy:    PnL = ${as_metrics.net_pnl:.2f}   | "
        f"Sharpe = {as_metrics.sharpe:.2f}"
    )

    plot_comparison(naive_result, as_result, naive_metrics, as_metrics)


if __name__ == "__main__":
    main()
