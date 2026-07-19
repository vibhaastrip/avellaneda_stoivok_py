import argparse
from dataclasses import replace

from src.config import SimulationConfig
from src.run_batch import run_batch
from src.simulation import Strategy

DEFAULT_GAMMA_VALUES = "0.01,0.05,0.1,0.25,0.5,1.0,2.0"


def _parse_gamma_values(raw_values: str) -> list[float]:
    return [float(value.strip()) for value in raw_values.split(",") if value.strip()]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=1000)
    parser.add_argument("--gamma-values", default=DEFAULT_GAMMA_VALUES)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--A", type=float)
    return parser.parse_args()


def _base_config(args) -> SimulationConfig:
    default_config = SimulationConfig()
    return SimulationConfig(
        sigma=args.sigma if args.sigma is not None else default_config.sigma,
        k=args.k if args.k is not None else default_config.k,
        A=args.A if args.A is not None else default_config.A,
    )


def main():
    args = _parse_args()
    base_config = _base_config(args)
    gamma_values = _parse_gamma_values(args.gamma_values)

    total_completed = 0
    for gamma in gamma_values:
        config = replace(base_config, gamma=gamma)
        print(f"Running gamma={gamma} with seeds 0..{args.n_runs - 1}")
        total_completed += run_batch(
            config=config,
            strategies=[Strategy.AVELLANEDA_STOIKOV],
            n_runs=args.n_runs,
        )

    print(f"Completed {total_completed} total sweep runs")


if __name__ == "__main__":
    main()
