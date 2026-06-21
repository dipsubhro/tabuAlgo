import sys
import os
import json
import numpy as np
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate

warnings.filterwarnings('ignore')

# Ensure project root is on the path so IDEs and runtime both resolve correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from yahoo_finance_experiment.rts_portfolio import SingleReactiveTabuSearch
from yahoo_finance_experiment.rts_portfolio import StandardTabuSearch
from yahoo_finance_experiment.rts_portfolio.fitness import (
    repair_weights, calc_annual_return, calc_annual_risk,
    calc_all_metrics,
)
import yahoo_finance_experiment.config as cfg

# ── ALGORITHM SELECTOR: edit ALGORITHM in config.py ──
#    "RTS" = Reactive Tabu Search  (Lévy + Reactive Tenure + Oscillation)
#    "STS" = Standard Tabu Search  (baseline, fixed tenure, Gaussian steps)
ALGORITHM = cfg.ALGORITHM


# ═══════════════════════════════════════════════════════════════════════════
# DATA — Reuse the RMPSO reference download function
# ═══════════════════════════════════════════════════════════════════════════

def load_stock_data():
    """
    Load daily returns for the configured stock universe.

    Source: data/combined_prices.csv  — unified close prices (all positive),
            built by combining the Close column from each per-ticker CSV.

    The function reads the close prices and computes pct_change() internally
    so the algorithm always works from the cleanest source data.

    Falls back to downloading from Yahoo Finance if combined_prices.csv is
    missing and no individual ticker CSVs are present.

    Returns
    -------
    stock_names   : list[str]
    returns_array : np.ndarray  shape (trading_days, n_assets)
    """
    import pandas as pd

    combined_path = cfg.COMBINED_PRICES_CSV

    if os.path.exists(combined_path):
        # ── Load unified close prices & compute returns ────────────
        print(f"  Loading combined close prices: {combined_path}")
        close = pd.read_csv(combined_path, index_col=0, parse_dates=True)
        print(f"  ✓ Close prices : {close.shape[1]} tickers, {close.shape[0]} rows")
        print(f"  ✓ Date range   : {close.index[0].date()} → {close.index[-1].date()}")

        # Compute daily returns from close prices
        returns       = close.pct_change().dropna()
        stock_names   = list(returns.columns)
        returns_array = returns.values

        print(f"  ✓ Daily returns: {returns.shape[0]} trading days")

    else:
        # ── Fallback: download from Yahoo Finance ──────────────────
        try:
            import yfinance as yf
        except ImportError:
            print("  [!] yfinance not installed. Run: uv add yfinance")
            sys.exit(1)

        print("  combined_prices.csv not found — downloading from Yahoo Finance...")
        print(f"  Tickers : {cfg.TICKERS}")
        print(f"  Period  : {cfg.DATA_START} → {cfg.DATA_END}")

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

        # Save as combined_prices.csv for future runs
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        close.to_csv(combined_path)
        print(f"  ✓ Saved combined_prices.csv: {combined_path}")

        returns       = close.pct_change().dropna()
        stock_names   = list(returns.columns)
        returns_array = returns.values

        print(f"  ✓ Downloaded : {len(stock_names)} stocks, {len(returns_array)} trading days")

    return stock_names, returns_array



# ═══════════════════════════════════════════════════════════════════════════
# SINGLE RUN WRAPPER (for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════

def _run_single(args):
    """Run one search instance. Works for both RTS and STS."""
    (returns_data, cov_matrix, n_assets, seed, rf,
     max_iter, neighbors_size, algo_type) = args

    if algo_type == "STS":
        rts = StandardTabuSearch(
            returns_data=returns_data,
            cov_matrix=cov_matrix,
            n_assets=n_assets,
            rf=rf,
            max_iter=max_iter,
            neighbors_size=neighbors_size,
            tenure=cfg.STS_TENURE,
            step_scale=cfg.STS_STEP_SCALE,
            seed=seed,
        )
    else:  # "RTS"
        rts = SingleReactiveTabuSearch(
            returns_data=returns_data,
            cov_matrix=cov_matrix,
            n_assets=n_assets,
            rf=rf,
            max_iter=max_iter,
            neighbors_size=neighbors_size,
            initial_tenure=cfg.RTS_INITIAL_TENURE,
            weight_cap=cfg.RTS_WEIGHT_CAP,
            oscillation_cap=cfg.RTS_OSC_CAP,
            beta=cfg.RTS_LEVY_BETA,
            cycle_threshold=cfg.RTS_CYCLE_THRESHOLD,
            seed=seed,
        )
    return rts.run()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    if ALGORITHM == "STS":
        print("  STANDARD TABU SEARCH (STS) — S&P 500 Portfolio Optimization")
        print("  Baseline: fixed tenure, Gaussian neighbourhood, no enhancements")
    else:
        print("  REACTIVE TABU SEARCH (RTS) — S&P 500 Portfolio Optimization")
        print("  Modules: Lévy Flight | Reactive Tenure | Strategic Oscillation")
        print("           | Multi-Objective Aspiration")
    print("  Dataset: S&P 500, Jan 2013 – Jan 2023")
    print("=" * 70)

    RF = cfg.RF

    # ── Step 1: Get Data ──────────────────────────────────────────
    print("\n  [1] Fetching real market data...")
    stock_names, returns_data = load_stock_data()
    n_assets = len(stock_names)

    cov_daily = np.cov(returns_data.T)
    # NOTE: calc_annual_risk() multiplies by sqrt(252) internally,
    # so we pass the DAILY covariance matrix — not pre-annualised.

    # Show stock statistics
    mean_returns = np.mean(returns_data, axis=0) * 252
    std_returns = np.std(returns_data, axis=0) * np.sqrt(252)

    print("\n  Individual Stock Statistics (Annualised):")
    stock_table = [
        [name, f"{r*100:.2f}%", f"{v*100:.2f}%", f"{(r-RF)/v:.3f}"]
        for name, r, v in zip(stock_names, mean_returns, std_returns)
    ]
    print(tabulate(stock_table,
                   headers=["Stock", "Annual Return",
                            "Annual Volatility", "Individual Sharpe"],
                   tablefmt="fancy_grid"))

    # ── Step 2: Run RTS ───────────────────────────────────────────
    NUM_RUNS       = cfg.NUM_RUNS
    SEED_POOL_SEED = cfg.SEED_POOL_SEED
    MAX_ITER       = cfg.MAX_ITER
    NEIGHBORS      = cfg.NEIGHBORS

    print(f"\n  [2] Running {ALGORITHM} ({NUM_RUNS} runs)...")
    print(f"      Iterations={MAX_ITER}, Neighbors={NEIGHBORS}")
    if ALGORITHM == "RTS":
        print(f"      Tenure={cfg.RTS_INITIAL_TENURE}, Weight cap={cfg.RTS_WEIGHT_CAP*100:.0f}%, Oscillation cap={cfg.RTS_OSC_CAP*100:.0f}%")
    else:
        print(f"      Tenure={cfg.STS_TENURE}, Step scale={cfg.STS_STEP_SCALE:.2f}")
    print(f"      Seed pool seed={SEED_POOL_SEED}")

    seed_rng = np.random.default_rng(SEED_POOL_SEED)
    SEEDS = [int(s) for s in seed_rng.choice(1_000_000_000, size=NUM_RUNS, replace=False)]

    run_args = [
        (returns_data, cov_daily, n_assets, seed, RF,
         MAX_ITER, NEIGHBORS, ALGORITHM)
        for seed in SEEDS
    ]

    # Run in parallel
    max_workers = max(1, os.cpu_count() - 2)
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(_run_single, run_args)):
            all_results.append(result)
            print(f"  Run {i+1:>2}/{NUM_RUNS} | "
                  f"Sharpe={result['best_sharpe']:.4f} | "
                  f"Return={result['best_return']*100:.2f}% | "
                  f"Risk={result['best_risk']*100:.2f}%")

    print(f"\n  All {NUM_RUNS} runs complete!")

    # ── Cross-run statistics ───────────────────────────────────────
    all_returns = np.array([r['best_return'] for r in all_results])
    all_risks   = np.array([r['best_risk']   for r in all_results])
    all_sharpes_arr = np.array([r['best_sharpe'] for r in all_results])

    # ── Step 3: Aggregate Results ─────────────────────────────────
    all_sharpes = [r['best_sharpe'] for r in all_results]
    best_idx = int(np.argmax(all_sharpes))
    best_result = all_results[best_idx]

    rts_metrics = calc_all_metrics(
        best_result['best_weights'], returns_data, cov_daily, RF
    )

    # ── All-Runs Summary Table ────────────────────────────────────
    print("\n  " + "=" * 66)
    print("  ALL RUNS — INDIVIDUAL RESULTS")
    print("  " + "=" * 66)
    run_rows = [
        [
            f"{i + 1} (best)" if i == best_idx else i + 1,
            SEEDS[i],
            f"{r['best_return']*100:.2f}%",
            f"{r['best_risk']*100:.2f}%",
            f"{r['best_sharpe']:.4f}",
        ]
        for i, r in enumerate(all_results)
    ]
    print(tabulate(
        run_rows,
        headers=["Run #", "Seed", "Annual Return", "Annual Risk", "Sharpe"],
        tablefmt="fancy_grid",
    ))

    print("\n  " + "=" * 66)
    print(f"  RUN STATISTICS ({NUM_RUNS} runs)")
    print("  " + "=" * 66)
    metric_stats = [
        ["Sharpe Ratio",
         f"{np.mean(all_sharpes_arr):.4f}",
         f"{np.min(all_sharpes_arr):.4f}",
         f"{np.max(all_sharpes_arr):.4f}",
         f"{np.std(all_sharpes_arr):.4f}"],
    ]
    print(tabulate(
        metric_stats,
        headers=["Metric", "Mean", "Min", "Max", "Std Dev"],
        tablefmt="fancy_grid",
    ))

    print(f"\n  Best Run #{best_idx + 1}:")
    print(f"    Sharpe Ratio : {best_result['best_sharpe']:.4f}")
    print(f"    Annual Return: {best_result['best_return']*100:.2f}%")
    print(f"    Annual Risk  : {best_result['best_risk']*100:.2f}%")
    print(f"    Max Drawdown : {rts_metrics['max_drawdown']*100:.2f}%")

    # ── Step 4: Print per-stock allocation detail ───────────────────
    # ── Per-stock allocation detail table ──────────────────────────
    print("\n  " + "=" * 70)
    print("  BEST PORTFOLIO — STOCK-LEVEL ALLOCATION DETAIL")
    print("  " + "=" * 70)

    STOCK_FULL_NAMES = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com, Inc.',
        'JPM': 'JPMorgan Chase & Co.',
        'JNJ': 'Johnson & Johnson',
        'V': 'Visa Inc.',
        'PG': 'The Procter & Gamble Company',
        'XOM': 'Exxon Mobil Corporation',
        'NVDA': 'NVIDIA Corporation',
    }

    alloc_rows = []
    for name, w, r_ann, v_ann in zip(
            stock_names, best_result['best_weights'], mean_returns, std_returns):
        ind_sharpe = (r_ann - RF) / v_ann if v_ann > 0 else float('nan')
        full_name = STOCK_FULL_NAMES.get(name, name)
        alloc_rows.append({
            'ticker': name,
            'name': full_name,
            'alloc_val': w,
            'alloc_str': f"{w*100:.2f}%",
            'ret_str': f"{r_ann*100:.2f}%",
            'risk_str': f"{v_ann*100:.2f}%",
            'sharpe_val': round(ind_sharpe, 4),
        })

    # Sort by allocation descending
    alloc_rows.sort(key=lambda x: x['alloc_val'], reverse=True)

    final_alloc_rows = [
        [r['ticker'], r['name'], r['alloc_str']]
        for r in alloc_rows
    ]

    alloc_table_str = tabulate(
        final_alloc_rows,
        headers=["Ticker", "Stock Name", "Allocation"],
        tablefmt="pipe",
    )
    print(alloc_table_str)

    # ── Step 5: OUTPUT 1 — Convergence Table ──────────────────────
    print("\n  [3] Generating convergence table...")
    best_conv = best_result['convergence_log']

    conv_rows = []
    for entry in best_conv:
        it = entry[0]
        b_sharpe = entry[1]
        c_sharpe = entry[2]
        
        if ALGORITHM == "RTS":
            ten = entry[3] if len(entry) > 3 else cfg.RTS_INITIAL_TENURE
            phase = entry[4] if len(entry) > 4 else "Normal"
            conv_rows.append([it, f"{b_sharpe:.4f}", f"{c_sharpe:.4f}", ten, phase])
        else:
            conv_rows.append([it, f"{b_sharpe:.4f}", f"{c_sharpe:.4f}"])

    headers = ["Iteration", "Best Sharpe", "Current Sharpe"]
    if ALGORITHM == "RTS":
        headers.extend(["Tenure", "Phase"])

    conv_table = tabulate(
        conv_rows,
        headers=headers,
        tablefmt="grid",
    )

    with open(cfg.OUT_CONVERGENCE_TXT, "w") as f:
        f.write(f"{ALGORITHM} — Convergence Log\n")
        f.write(f"Best Run (Seed {SEEDS[best_idx]})\n")
        f.write("=" * 70 + "\n\n")
        f.write(conv_table)
        f.write("\n")
    print(f"  [✓] Convergence table saved: {cfg.OUT_CONVERGENCE_TXT}")

    # ── Step 6: OUTPUT 2 — Final Metrics ─────────────────────────
    print("\n  [4] Building summary table...")

    # Statistics across all configured runs
    stats_rows = [
        ["Mean Sharpe", f"{np.mean(all_sharpes):.4f}"],
        ["Median Sharpe", f"{np.median(all_sharpes):.4f}"],
        ["Best Sharpe", f"{np.max(all_sharpes):.4f}"],
        ["Worst Sharpe", f"{np.min(all_sharpes):.4f}"],
        ["Std Sharpe", f"{np.std(all_sharpes):.4f}"],
    ]
    stats_table = tabulate(stats_rows, headers=["Metric", "Value"],
                           tablefmt="grid")

    # Full per-run table for the saved file
    run_rows_file = [
        [f"{i + 1} (best)" if i == best_idx else i + 1, SEEDS[i],
         f"{r['best_return']*100:.2f}%",
         f"{r['best_risk']*100:.2f}%",
         f"{r['best_sharpe']:.4f}"]
        for i, r in enumerate(all_results)
    ]
    all_runs_table = tabulate(
        run_rows_file,
        headers=["Run #", "Seed", "Annual Return", "Annual Risk", "Sharpe"],
        tablefmt="grid",
    )

    # Extended metric stats (mean/min/max/std for all three metrics)
    full_metric_stats = [
        ["Sharpe Ratio",
         f"{np.mean(all_sharpes_arr):.4f}",
         f"{np.min(all_sharpes_arr):.4f}",
         f"{np.max(all_sharpes_arr):.4f}",
         f"{np.std(all_sharpes_arr):.4f}"],
    ]
    full_stats_table = tabulate(
        full_metric_stats,
        headers=["Metric", "Mean", "Min", "Max", "Std Dev"],
        tablefmt="grid",
    )

    with open(cfg.OUT_COMPARISON_TXT, "w") as f:
        f.write(f"{ALGORITHM} — Final Metrics\n")
        f.write("Dataset: S&P 500 Top 10 Stocks | "
                "Period: 2013–2023 | RF=2%\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Individual Run Results ({NUM_RUNS} runs)\n")
        f.write("-" * 60 + "\n")
        f.write(all_runs_table)
        f.write("\n\n")
        f.write(f"{ALGORITHM} Run Statistics ({NUM_RUNS} runs) — All Metrics\n")
        f.write("-" * 60 + "\n")
        f.write(full_stats_table)
        f.write("\n\n")
        f.write("Best Portfolio — Stock-Level Allocation Detail\n")
        f.write("-" * 60 + "\n")
        f.write(alloc_table_str)
        f.write("\n")

    print(f"\n  [✓] Summary table saved: {cfg.OUT_COMPARISON_TXT}")

    # ── Step 7: Save run data for graph generation ────────────────────
    print("\n  [5] Saving run data for graph generation...")
    os.makedirs(cfg.OUTPUTS_DIR, exist_ok=True)
    
    run_data = {
        "algorithm":    ALGORITHM,
        "num_runs":     NUM_RUNS,
        "best_idx":     best_idx,
        "seeds":        SEEDS,
        "stock_names":  stock_names,
        "best_result": {
            "best_sharpe":  best_result["best_sharpe"],
            "best_return":  best_result["best_return"],
            "best_risk":    best_result["best_risk"],
            "best_weights": best_result["best_weights"].tolist(),
        },
        "all_results": [
            {
                "best_sharpe":      r["best_sharpe"],
                "best_return":      r["best_return"],
                "best_risk":        r["best_risk"],
                "convergence_log":  [list(e) for e in r["convergence_log"]],
            }
            for r in all_results
        ],
    }
    with open(cfg.OUT_RUN_DATA_JSON, "w") as f:
        json.dump(run_data, f)
    print(f"  [✓] Run data saved: {cfg.OUT_RUN_DATA_JSON}")
    print(f"  →  Run  python generate_graphs.py  to regenerate the graphs.")

    # ── Final Summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {ALGORITHM} — COMPLETE!")
    print(f"  {ALGORITHM} Best Sharpe   = {best_result['best_sharpe']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
