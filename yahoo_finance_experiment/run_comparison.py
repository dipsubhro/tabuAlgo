"""
Normal vs Optimized Tabu Search — Comparison Runner
====================================================

Runs BOTH the Standard (Normal) Tabu Search and the Optimized
Reactive Tabu Search (with Lévy Flights, Reactive Tenure,
Strategic Oscillation, Multi-Objective Aspiration) on the same
dataset and seeds, then writes a side-by-side comparison to
a text file.

Uses 30 S&P 500 stocks to create a harder optimization problem
where the optimized enhancements actually matter.

Outputs:
  - outputs/tabu_normal_vs_optimized.txt  (full comparison report)
"""

import sys
import os
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate
import time

warnings.filterwarnings('ignore')

# Ensure project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from yahoo_finance_experiment.rts_portfolio import (
    SingleReactiveTabuSearch,
    StandardTabuSearch,
)
from yahoo_finance_experiment.rts_portfolio.fitness import (
    repair_weights, calc_annual_return, calc_annual_risk,
    calc_all_metrics,
)
import yahoo_finance_experiment.config as cfg


# ═══════════════════════════════════════════════════════════════════════════
# USER CONFIGURABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# General Setup
NUM_RUNS = 3
SEED_POOL_SEED = 786683

# [STS] Normal Tabu Search Parameters
STS_ITER = 1000
STS_NEIGHBORS = 50
STS_TENURE = 10
STS_STEP = 0.50

# [RTS] Optimized Tabu Search Parameters
RTS_ITER = 1000
RTS_NEIGHBORS = 50
RTS_TENURE = 10
RTS_WEIGHT_CAP = 1.0
RTS_OSC_CAP = 1.0

# ═══════════════════════════════════════════════════════════════════════════
# DATA — 30 S&P 500 stocks for a challenging search space
# ═══════════════════════════════════════════════════════════════════════════

def load_stock_data():
    """
    Load daily returns for the configured stock universe.
    Source: data/combined_prices.csv  — unified close prices.
    """
    import pandas as pd

    combined_path = cfg.COMBINED_PRICES_CSV

    if os.path.exists(combined_path):
        print(f"  Loading combined close prices: {combined_path}")
        close = pd.read_csv(combined_path, index_col=0, parse_dates=True)
        print(f"  ✓ Close prices : {close.shape[1]} tickers, {close.shape[0]} rows")
        print(f"  ✓ Date range   : {close.index[0].date()} → {close.index[-1].date()}")

        returns       = close.pct_change().dropna()
        stock_names   = list(returns.columns)
        returns_array = returns.values

        print(f"  ✓ Daily returns: {returns.shape[0]} trading days")
    else:
        try:
            import yfinance as yf
        except ImportError:
            print("  [!] yfinance not installed. Run: uv add yfinance")
            sys.exit(1)

        print("  combined_prices.csv not found — downloading from Yahoo Finance...")
        raw = yf.download(
            cfg.TICKERS,
            start=cfg.DATA_START,
            end=cfg.DATA_END,
            auto_adjust=True,
            progress=False,
        )

        close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = close.dropna(axis=1, thresh=int(cfg.DATA_COVERAGE_THRESHOLD * len(close)))
        close = close.dropna()

        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        close.to_csv(combined_path)

        returns       = close.pct_change().dropna()
        stock_names   = list(returns.columns)
        returns_array = returns.values

        print(f"  ✓ Downloaded : {len(stock_names)} stocks, {len(returns_array)} trading days")

    return stock_names, returns_array


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE RUN WRAPPERS (for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════

def _run_standard(args):
    """Run a single Standard Tabu Search instance."""
    (returns_data, cov_matrix, n_assets, seed, rf,
     max_iter, neighbors_size, tenure) = args

    sts = StandardTabuSearch(
        returns_data=returns_data,
        cov_matrix=cov_matrix,
        n_assets=n_assets,
        rf=rf,
        max_iter=max_iter,
        neighbors_size=neighbors_size,
        tenure=tenure,
        step_scale=STS_STEP,
        seed=seed,
    )
    return sts.run()


def _run_optimized(args):
    """Run a single Optimized RTS instance."""
    (returns_data, cov_matrix, n_assets, seed, rf,
     max_iter, neighbors_size, tenure, weight_cap, osc_cap) = args

    rts = SingleReactiveTabuSearch(
        returns_data=returns_data,
        cov_matrix=cov_matrix,
        n_assets=n_assets,
        rf=rf,
        max_iter=max_iter,
        neighbors_size=neighbors_size,
        initial_tenure=tenure,
        weight_cap=weight_cap,
        oscillation_cap=osc_cap,
        seed=seed,
    )
    return rts.run()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  NORMAL vs OPTIMIZED TABU SEARCH — Comparison")
    print("  Dataset: S&P 500, Jan 2013 – Jan 2023")
    print("=" * 70)

    RF = 0.02

    # ── Step 1: Get Data ──────────────────────────────────────────
    print("\n  [1] Fetching real market data...")
    stock_names, returns_data = load_stock_data()
    n_assets = len(stock_names)

    cov_daily = np.cov(returns_data.T)

    # ── Step 2: Configure runs ────────────────────────────────────
    seed_rng = np.random.default_rng(SEED_POOL_SEED)
    SEEDS = [
        int(seed)
        for seed in seed_rng.choice(1_000_000_000, size=NUM_RUNS, replace=False)
    ]

    max_workers = max(1, os.cpu_count() - 2)

    print(f"\n  [2] Running NORMAL Tabu Search ({NUM_RUNS} runs, {n_assets} assets)...")
    print(f"      Iterations={STS_ITER}, Neighbors={STS_NEIGHBORS}, Fixed Tenure={STS_TENURE}")
    print(f"      Fixed step={STS_STEP}, No Lévy, No Reactive Tenure, No Oscillation")
    print(f"      No aspiration override, Random restart from scratch")

    std_args = [
        (returns_data, cov_daily, n_assets, seed, RF,
         STS_ITER, STS_NEIGHBORS, STS_TENURE)
        for seed in SEEDS
    ]

    t0 = time.time()
    std_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(_run_standard, std_args)):
            std_results.append(result)
            print(f"    [Normal] Run {i+1:>2}/{NUM_RUNS} | "
                  f"Sharpe={result['best_sharpe']:.4f} | "
                  f"Return={result['best_return']*100:.2f}% | "
                  f"Risk={result['best_risk']*100:.2f}%")
    std_time = time.time() - t0

    # ── Step 4: Run Optimized (Reactive) Tabu Search ──────────────
    print(f"\n  [3] Running OPTIMIZED Tabu Search ({NUM_RUNS} runs, {n_assets} assets)...")
    print(f"      Iterations={RTS_ITER}, Neighbors={RTS_NEIGHBORS}, Initial Tenure={RTS_TENURE}")
    print(f"      + Lévy Flight (β={cfg.RTS_LEVY_BETA}) + Reactive Tenure (dynamic)")
    print(f"      + Strategic Oscillation (cap cycling {RTS_WEIGHT_CAP*100:.0f}%/{RTS_OSC_CAP*100:.0f}%)")
    print(f"      + Multi-Objective Aspiration + Pareto Repository (100)")

    opt_args = [
        (returns_data, cov_daily, n_assets, seed, RF,
         RTS_ITER, RTS_NEIGHBORS, RTS_TENURE, RTS_WEIGHT_CAP, RTS_OSC_CAP)
        for seed in SEEDS
    ]

    t0 = time.time()
    opt_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(_run_optimized, opt_args)):
            opt_results.append(result)
            print(f"    [Optimized] Run {i+1:>2}/{NUM_RUNS} | "
                  f"Sharpe={result['best_sharpe']:.4f} | "
                  f"Return={result['best_return']*100:.2f}% | "
                  f"Risk={result['best_risk']*100:.2f}%")
    opt_time = time.time() - t0

    # ── Step 5: Aggregate & Compare ───────────────────────────────
    print(f"\n  [4] Building comparison report...")

    std_sharpes = [r['best_sharpe'] for r in std_results]
    opt_sharpes = [r['best_sharpe'] for r in opt_results]
    std_returns = [r['best_return'] for r in std_results]
    opt_returns = [r['best_return'] for r in opt_results]
    std_risks = [r['best_risk'] for r in std_results]
    opt_risks = [r['best_risk'] for r in opt_results]

    std_best_idx = int(np.argmax(std_sharpes))
    opt_best_idx = int(np.argmax(opt_sharpes))

    std_best = std_results[std_best_idx]
    opt_best = opt_results[opt_best_idx]

    std_metrics = calc_all_metrics(std_best['best_weights'], returns_data, cov_daily, RF)
    opt_metrics = calc_all_metrics(opt_best['best_weights'], returns_data, cov_daily, RF)

    # ── Build output text ─────────────────────────────────────────────
    output_path = os.path.join(cfg.OUTPUTS_DIR, "rts_sts_comparison.txt")
    lines = []

    lines.append("================================================================================")
    lines.append("  NORMAL vs OPTIMIZED TABU SEARCH — Comparison Report")
    lines.append("================================================================================")
    lines.append("")
    lines.append("  --- ALGORITHM PARAMETERS ---")
    lines.append("  [STS (Normal Tabu Search)]")
    lines.append(f"    - Iterations      : {STS_ITER}")
    lines.append(f"    - Neighbors       : {STS_NEIGHBORS}")
    lines.append(f"    - Tenure          : {STS_TENURE}")
    lines.append(f"    - Step Scale      : {STS_STEP}")
    lines.append("")
    lines.append("  [RTS (Optimized Tabu Search)]")
    lines.append(f"    - Iterations      : {RTS_ITER}")
    lines.append(f"    - Neighbors       : {RTS_NEIGHBORS}")
    lines.append(f"    - Initial Tenure  : {RTS_TENURE}")
    lines.append(f"    - Lévy Flight (β) : {cfg.RTS_LEVY_BETA}")
    lines.append(f"    - Weight Cap      : {RTS_WEIGHT_CAP}")
    lines.append(f"    - Oscillation Cap : {RTS_OSC_CAP}")
    lines.append("")
    lines.append("  --- COMPARISON & WINS/LOSSES ---")
    
    # ── Win/Loss Summary ──────────────────────────────────────────
    opt_wins = sum(1 for i in range(NUM_RUNS) if opt_sharpes[i] > std_sharpes[i] + 0.001)
    std_wins = sum(1 for i in range(NUM_RUNS) if std_sharpes[i] > opt_sharpes[i] + 0.001)
    ties = NUM_RUNS - opt_wins - std_wins

    run_rows = []
    for i in range(NUM_RUNS):
        diff = opt_sharpes[i] - std_sharpes[i]
        winner = "Optimized ✓" if opt_sharpes[i] > std_sharpes[i] + 0.001 else (
            "Normal" if std_sharpes[i] > opt_sharpes[i] + 0.001 else "Tie"
        )
        run_rows.append([
            i + 1,
            f"{std_sharpes[i]:.4f}",
            f"{opt_sharpes[i]:.4f}",
            winner,
        ])
    
    lines.append(tabulate(
        run_rows,
        headers=["Run", "STS Sharpe", "RTS Sharpe", "Winner"],
        tablefmt="fancy_grid",
    ))
    lines.append("")
    lines.append(f"  RTS Total Wins : {opt_wins}/{NUM_RUNS}")
    lines.append(f"  STS Total Wins : {std_wins}/{NUM_RUNS}")
    lines.append(f"  Ties           : {ties}/{NUM_RUNS}")
    lines.append("================================================================================")

    # ── Write to file ─────────────────────────────────────────────
    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\n  [✓] Comparison report saved: {output_path}")


if __name__ == "__main__":
    main()
