import argparse

import numpy as np
from sqlalchemy import select

from src.analysis import calculate_metrics
from src.config import SimulationConfig
from src.simulation import Strategy, run_simulation
from src.storage.db import SessionLocal, create_tables
from src.storage.models import Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=1000)
    parser.add_argument(
        "--strategy",
        choices=[
            "naive",
            "inventory_skewed",
            "volatility_scaled",
            "avellaneda_stoikov",
            "enhanced_avellaneda_stoikov",
            "both",
        ],
        default="both",
    )
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--A", type=float)
    parser.add_argument("--inventory-limit", type=int)
    parser.add_argument("--min-quote-spread", type=float)
    parser.add_argument("--max-quote-distance", type=float)
    parser.add_argument("--volatility-window", type=int)
    parser.add_argument("--volatility-spread-multiplier", type=float)
    parser.add_argument("--inventory-skew", type=float)
    parser.add_argument("--adverse-selection-strength", type=float)
    return parser.parse_args()


def _build_config(args) -> SimulationConfig:
    default_config = SimulationConfig()
    return SimulationConfig(
        gamma=args.gamma if args.gamma is not None else default_config.gamma,
        sigma=args.sigma if args.sigma is not None else default_config.sigma,
        k=args.k if args.k is not None else default_config.k,
        A=args.A if args.A is not None else default_config.A,
        inventory_limit=(
            args.inventory_limit
            if args.inventory_limit is not None
            else default_config.inventory_limit
        ),
        min_quote_spread=(
            args.min_quote_spread
            if args.min_quote_spread is not None
            else default_config.min_quote_spread
        ),
        max_quote_distance=(
            args.max_quote_distance
            if args.max_quote_distance is not None
            else default_config.max_quote_distance
        ),
        volatility_window=(
            args.volatility_window
            if args.volatility_window is not None
            else default_config.volatility_window
        ),
        volatility_spread_multiplier=(
            args.volatility_spread_multiplier
            if args.volatility_spread_multiplier is not None
            else default_config.volatility_spread_multiplier
        ),
        inventory_skew=(
            args.inventory_skew
            if args.inventory_skew is not None
            else default_config.inventory_skew
        ),
        adverse_selection_strength=(
            args.adverse_selection_strength
            if args.adverse_selection_strength is not None
            else default_config.adverse_selection_strength
        ),
    )

def selected_strategies(strategy_arg):
    if strategy_arg=='both':
        return[Strategy.NAIVE, Strategy.AVELLANEDA_STOIKOV]
    
def uses_gamma(strategy):
    return strategy in{
        Strategy.AVELLANEDA_STOIKOV,
        Strategy.ENHANCED_AVELLANEDA_STOIKOV
    }

def run_row(config: SimulationConfig, strategy: Strategy, seed):
    rng = np.random.default_rng(seed)
    result = run_simulation(config, strategy, rng=rng)
    metrics = calculate_metrics(result, config.sharpe_annualization_factor)

    return Run(
        strategy=strategy.value,
        seed=seed,
        gamma=config.gamma if uses_gamma(strategy) else None,
        sigma=config.sigma,
        k=config.k,
        A=config.A,
        inventory_limit=config.inventory_limit,
        min_quote_spread=config.min_quote_spread,
        max_quote_distance=config.max_quote_distance,
        volatility_spread_multiplier=config.volatility_spread_multiplier,
        inventory_skew=config.inventory_skew,
        adverse_selection_strength=config.adverse_selection_strength,
        net_pnl=metrics.net_pnl,
        sharpe=metrics.sharpe,
        sortino=metrics.sortino,
        max_drawdown=metrics.max_drawdown,
        max_abs_inventory=metrics.max_abs_inventory,
        fees_paid=metrics.fees_paid,

    )
def existing_run_keys(
        session,
        config: SimulationConfig,
        strategies: list[Strategy],
        n_runs
):
    strategy_values = [strategy.value for strategy in strategies]
    statement = select(
        Run.strategy,
        Run.seed,
        Run.gamma,
        Run.sigma,
        Run.k,
        Run.A,
        Run.inventory_limit,
        Run.min_quote_spread,
        Run.max_quote_distance,
        Run.volatility_spread_multiplier,
        Run.inventory_skew,
        Run.adverse_selection_strength,
    ).where(
        Run.strategy.in_(strategy_values),
        Run.seed.in_(range(n_runs)),
        Run.sigma == config.sigma,
        Run.k == config.k,
        Run.A == config.A,
        Run.inventory_limit == config.inventory_limit,
        Run.min_quote_spread == config.min_quote_spread,
        Run.max_quote_distance == config.max_quote_distance,
        Run.volatility_spread_multiplier == config.volatility_spread_multiplier,
        Run.inventory_skew == config.inventory_skew,
        Run.adverse_selection_strength == config.adverse_selection_strength,
    )

    gamma_strategies = [strategy.value for strategy in strategies if uses_gamma(strategy)]
    non_gamma_strategies = [
        strategy.value for strategy in strategies if not uses_gamma(strategy)
    ]

    if gamma_strategies and non_gamma_strategies:
        statement = statement.where(
            (Run.strategy.in_(non_gamma_strategies) & Run.gamma.is_(None))
            | (Run.strategy.in_(gamma_strategies) & (Run.gamma == config.gamma))
        )
    elif gamma_strategies:
        statement = statement.where(Run.gamma == config.gamma)
    else:
        statement = statement.where(Run.gamma.is_(None))

    return session.execute(statement).all()


def run_batch(config: SimulationConfig, strategies: list[Strategy], n_runs: int) -> int:
    create_tables()

    completed_runs = 0
    with SessionLocal() as session:
        existing_keys = existing_run_keys(session, config, strategies, n_runs)
        if existing_keys:
            sample = ", ".join(
                (
                    f"({strategy}, seed={seed}, gamma={gamma}, "
                    f"sigma={sigma}, k={k}, A={arrival_intensity}, "
                    f"limit={inventory_limit}, min_spread={min_quote_spread}, "
                    f"max_distance={max_quote_distance}, "
                    f"vol_mult={volatility_spread_multiplier}, "
                    f"skew={inventory_skew}, "
                    f"adverse={adverse_selection_strength})"
                )
                for (
                    strategy,
                    seed,
                    gamma,
                    sigma,
                    k,
                    arrival_intensity,
                    inventory_limit,
                    min_quote_spread,
                    max_quote_distance,
                    volatility_spread_multiplier,
                    inventory_skew,
                    adverse_selection_strength,
                ) in existing_keys[:5]
            )
            raise SystemExit(
                "Existing batch rows found for requested strategy/seed/parameter "
                "combinations. Clear results.db before re-running the same batch, "
                f"or choose a different parameter grid. Sample: {sample}"
            )

        for seed in range(n_runs):
            for strategy in strategies:
                session.add(run_row(config, strategy, seed))
                completed_runs += 1

                if completed_runs % 100 == 0:
                    session.commit()
                    print(f"Completed {completed_runs} runs")

        session.commit()

    print(f"Completed {completed_runs} runs")
    return completed_runs


def main():
    args = parse_args()
    config = _build_config(args)
    strategies = selected_strategies(args.strategy)

    run_batch(config, strategies, args.n_runs)


if __name__ == "__main__":
    main()
