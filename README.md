# Digital Market Maker

A Python market-making simulator comparing Avellaneda-Stoikov stochastic
control quoting against fixed-spread and risk-aware baselines, under GBM
price dynamics and Poisson order flow.

## Overview

This project models a liquidity provider quoting bid/ask prices around a
stochastic mid-price, and compares five quoting strategies on PnL, Sharpe
ratio, Sortino ratio, drawdown, and inventory exposure using paired-seed
Monte Carlo batch testing.

**Headline finding:** calibrated Avellaneda-Stoikov quoting produces a
statistically significant improvement in risk-adjusted return (Sharpe
ratio) over a naive fixed-spread baseline, without a statistically
significant improvement in raw PnL. The benefit comes from materially
lower PnL variance and drawdown, not higher average profit.

## Strategies

- **naive** — fixed spread around the mid-price
- **avellaneda_stoikov** — reservation-price and optimal-spread quoting
  from the AS stochastic control approximation
- **enhanced_avellaneda_stoikov** — AS quoting using realised volatility
  (estimated from recent price history) instead of a fixed sigma input
- **inventory_skewed** — fixed spread with inventory-dependent quote skew
- **volatility_scaled** — fixed spread widened by realised short-window
  volatility

## Model

Mid-price follows geometric Brownian motion. Order arrivals follow a
Poisson process with intensity:

```
lambda(delta) = A * exp(-k * delta)
```

The Avellaneda-Stoikov reservation price and optimal total spread:

```
r      = s - q * gamma * sigma^2 * (T - t)
spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
```

Fill probability at distance `delta` from mid, over one timestep, is
computed exactly as `1 - exp(-lambda(delta) * dt)`, rather than the
small-dt linear approximation `lambda(delta) * dt` — see *Bugs Found* below
for why this matters.

## Results (1,000 paired Monte Carlo runs, gamma=3.0)

| Strategy | Mean PnL | PnL Std | Mean Sharpe | Mean Max Drawdown |
|---|---|---|---|---|
| Avellaneda-Stoikov | 44.59 | 66.11 | **0.332** | 56.90 |
| Enhanced AS | ~equal to AS | ~equal | ~equal | ~equal |
| Volatility-scaled | 38.87 | 172.0 | 0.155 | 156.4 |
| Inventory-skewed | 35.32 | 173.5 | 0.160 | 174.9 |
| Naive | 32.85 | 346.5 | 0.182 | 346.96 |

Paired comparison, Avellaneda-Stoikov vs naive (gamma=3.0, 1,000 seeds):

| Metric | Mean Difference | p-value |
|---|---|---|
| Net PnL | +11.74 | 0.26 (not significant) |
| Sharpe | +0.149 | 0.0015 (highly significant) |

**gamma was chosen empirically**, not assumed — see *Gamma Selection*
below.

## Gamma Selection

Rather than assuming a "best" gamma, this project selects it empirically
via a train/validation split (`src/gamma_search.py`):

- Gamma candidates are swept on a training set of seeds
- The best candidate is re-evaluated on a completely separate, held-out
  set of seeds, to avoid selecting a gamma that only looks good due to
  sampling noise on one particular seed set

This produced **gamma ≈ 2.5–3.0** as the best-validated setting for this
project's parameters — notably different from the gamma=5 reported in a
similar project with different underlying parameters, which is expected:
optimal gamma depends on the specific volatility, order-flow, and spread
calibration used, not a universal constant.

## Isolating the Inventory-Skewing Effect (`equalized_sweep.py`)

AS's spread is naturally wider than naive's fixed spread by default —
meaning some of AS's apparent Sharpe advantage could simply be "it trades
less often" rather than genuine inventory-risk management. This is a
confound: two things (spread width AND inventory-skewing behaviour) differ
between the strategies at once.

`equalized_sweep.py` controls for this by setting naive's spread to match
AS's *time-averaged* spread at each gamma value, then re-running the
paired comparison — isolating whatever benefit remains to the
inventory-skewing mechanism itself.

**Finding (1,000 seeds):** even after equalizing spread width, AS retains
a statistically significant, confound-free Sharpe advantage at gamma ≈ 3,
6, and 7 (p < 0.01 in each case). At gamma=8, this benefit weakens
(p=0.083) while AS begins sacrificing a statistically significant amount
of raw PnL (p<0.0001) relative to naive — indicating a "sweet spot" range
of risk-aversion beyond which the strategy becomes over-cautious without
a compensating risk-adjusted payoff.

## Bugs Found and Fixed During Development

Two of these were caught through active testing and are worth being able
to discuss:

1. **Fill-probability approximation error.** The linear approximation
   `lambda * dt` overestimated fill probability by 26-39% at realistic
   quote distances (verified numerically). Fixed by using the exact
   Poisson probability `1 - exp(-lambda * dt)`.
2. **Silent drawdown bug.** The original `_max_drawdown` computed a
   *relative* (%) drawdown and silently returned 0.0 whenever the wealth
   path hadn't yet gone positive — meaning a strategy losing money
   continuously from the start could report zero drawdown despite a real
   loss. Fixed by switching to absolute (dollar) drawdown, which also
   matches this project's own reporting convention.
3. Several smaller implementation bugs (a truthy-check inversion in the
   realised-volatility estimator, mismatched method names between base
   and subclasses, a missing control-flow branch, matplotlib API version
   drift) were found via systematic end-to-end testing of each module
   before trusting its output.

## Usage

Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run a single simulation:
```bash
python main.py
```

Run a paired Monte Carlo batch:
```bash
python -m src.run_batch --n-runs 1000 --strategy both
```

Run all five strategies:
```bash
python -m src.run_batch --n-runs 1000 --strategy naive
python -m src.run_batch --n-runs 1000 --strategy avellaneda_stoikov
python -m src.run_batch --n-runs 1000 --strategy inventory_skewed
python -m src.run_batch --n-runs 1000 --strategy volatility_scaled
python -m src.run_batch --n-runs 1000 --strategy enhanced_avellaneda_stoikov
```

Generate the report summary and plots:
```bash
python main.py --batch --plot
```

Run a gamma sweep:
```bash
python -m src.sweep --n-runs 1000 --gamma-values 1,2,3,4,5,6,7,8
python main.py --sweep
```

Run the validated gamma search (train/validation split):
```bash
python -m src.gamma_search --gamma-min 1.0 --gamma-max 8.0 --gamma-step 0.5
```

Run the equalized-spread sweep (isolates the inventory-skewing effect):
```bash
python -m src.equalized_sweep --n-runs 1000 --gamma-values 1,2,3,4,5,6,7,8
```

The SQLite database is stored as `results.db`; report images are written
to `reports/`.

## Project Structure

```
src/
  config.py              -- simulation parameters
  environment.py          -- GBM price dynamics, Poisson fill probability
  market_maker.py           -- strategy implementations
  simulation.py              -- single-run simulation loop
  analysis.py                 -- Sharpe, Sortino, max drawdown
  analysis_report.py            -- batch report + plots
  run_batch.py                   -- paired Monte Carlo batch runner
  sweep.py                        -- gamma sweep
  gamma_search.py                  -- validated (train/test split) gamma search
  equalized_sweep.py                -- confound-controlled gamma comparison
  storage/
    db.py                            -- SQLite connection setup
    models.py                          -- Run table schema
    queries.py                          -- summary stats, paired t-tests
main.py                                  -- CLI entry point
```

## Conclusion

Calibrated Avellaneda-Stoikov quoting is the strongest strategy tested in
this project. It does not reliably increase average PnL versus naive
quoting, but produces a statistically significant improvement in Sharpe
ratio, and substantially reduces drawdown and PnL variance — and this
benefit survives even after controlling for the confound of AS simply
quoting a wider spread, within a specific, empirically-identified range
of the risk-aversion parameter.
