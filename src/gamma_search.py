import argparse
import numpy as np
from dataclasses import replace

from src.config import SimulationConfig
from src.simulation import run_simulation, Strategy
from src.analysis import calculate_metrics


def mean_sharpe_over_seeds(config, seeds):
    sharpes = []
    for seed in seeds:
        result = run_simulation(config, Strategy.AVELLANEDA_STOIKOV, rng=np.random.default_rng(seed))
        metrics = calculate_metrics(result, config.sharpe_annualization_factor)
        sharpes.append(metrics.sharpe)
    return np.mean(sharpes)


def search_gamma(base_config, gamma_candidates, train_seeds, validation_seeds):
    train_results = []
    for gamma in gamma_candidates:
        cfg = replace(base_config, gamma=gamma)
        sharpe = mean_sharpe_over_seeds(cfg, train_seeds)
        train_results.append((gamma, sharpe))

    train_results.sort(key=lambda pair: -pair[1])

    print("Training results (best first):")
    for gamma, sharpe in train_results:
        print(f"  gamma={gamma:.2f}: mean_sharpe={sharpe:.4f}")

    best_gamma = train_results[0][0]

    validation_cfg = replace(base_config, gamma=best_gamma)
    validation_sharpe = mean_sharpe_over_seeds(validation_cfg, validation_seeds)

    print(f"\nBest on training: gamma={best_gamma:.2f}")
    print(f"Validation mean_sharpe (held-out seeds): {validation_sharpe:.4f}")

    return best_gamma, validation_sharpe


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma-min", type=float, default=1.0)
    parser.add_argument("--gamma-max", type=float, default=8.0)
    parser.add_argument("--gamma-step", type=float, default=0.5)
    parser.add_argument("--n-train-seeds", type=int, default=500)
    parser.add_argument("--n-validation-seeds", type=int, default=500)
    return parser.parse_args()


def main():
    args = _parse_args()
    base_config = SimulationConfig()

    gamma_candidates = np.arange(args.gamma_min, args.gamma_max + args.gamma_step, args.gamma_step)
    train_seeds = range(args.n_train_seeds)
    validation_seeds = range(args.n_train_seeds, args.n_train_seeds + args.n_validation_seeds)

    search_gamma(base_config, gamma_candidates, train_seeds, validation_seeds)


if __name__ == "__main__":
    main()