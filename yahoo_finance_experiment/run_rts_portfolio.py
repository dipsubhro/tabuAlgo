"""
Reactive Tabu Search — S&P 500 Portfolio Optimization Runner
=============================================================

Uses the EXACT same dataset and calculations as the RMPSO reference script
(yfdataset.py) for a fair head-to-head comparison.

Produces three required outputs:
  1. Convergence Table   → rts_convergence.txt
  2. Comparative Verdict → rts_comparison.txt  (RTS vs RMPSO vs CSO)
  3. Pareto Front Plot   → rts_pareto_front.png
"""

import sys
import os
import numpy as np
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate

warnings.filterwarnings('ignore')

# Ensure project root is on the path so IDEs and runtime both resolve correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from yahoo_finance_experiment.rts_portfolio import SingleReactiveTabuSearch, SwarmReactiveTabuSearch

# ── ALGORITHM SELECTOR ──
ALGORITHM = "SINGLE"  # Options: "SINGLE" or "SWARM"

from yahoo_finance_experiment.rts_portfolio.fitness import (
    repair_weights, calc_annual_return, calc_annual_risk,
    calc_all_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# DATA — Reuse the RMPSO reference download function
# ═══════════════════════════════════════════════════════════════════════════

def download_stock_data():
    """
    Downloads 10 years of real S&P 500 stock data from Yahoo Finance.
    Same period as the target paper: Jan 2013 - Jan 2023.
    Identical to the function in yfdataset.py.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [!] yfinance not installed. Run: uv add yfinance")
        sys.exit(1)

    import pandas as pd

    tickers = [
        'AAPL',   # Apple         — Technology
        'MSFT',   # Microsoft     — Technology
        'GOOGL',  # Alphabet      — Technology
        'AMZN',   # Amazon        — Consumer
        'JPM',    # JPMorgan      — Finance
        'JNJ',    # Johnson&Johnson — Healthcare
        'V',      # Visa          — Finance
        'PG',     # Procter&Gamble — Consumer
        'XOM',    # ExxonMobil    — Energy
        'NVDA',   # NVIDIA        — Technology
    ]

    print("  Downloading data from Yahoo Finance...")
    print(f"  Tickers : {tickers}")
    print(f"  Period  : Jan 2013 – Jan 2023 (same as target paper)")

    raw = yf.download(
        tickers,
        start='2013-01-01',
        end='2023-01-01',
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw['Close']
    else:
        data = raw

    data = data.dropna(axis=1, thresh=int(0.9 * len(data)))
    data = data.dropna()

    returns = data.pct_change().dropna()
    stock_names = list(returns.columns)
    returns_array = returns.values

    print(f"  Downloaded : {len(stock_names)} stocks")
    print(f"  Trading days: {len(returns_array)}")
    print(f"  Date range  : {returns.index[0].date()} to {returns.index[-1].date()}")

    return stock_names, returns_array


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE RUN WRAPPER (for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════

def _run_single(args):
    """Run a single RTS instance. Designed for ProcessPoolExecutor."""
    (returns_data, cov_matrix, n_assets, seed, rf,
     max_iter, swarm_size, neighbors_size, initial_tenure,
     weight_cap, oscillation_cap, algo_type) = args

    if algo_type == "SWARM":
        rts = SwarmReactiveTabuSearch(
            returns_data=returns_data,
            cov_matrix=cov_matrix,
            n_assets=n_assets,
            rf=rf,
            max_iter=max_iter,
            swarm_size=swarm_size,
            neighbors_size=neighbors_size,
            initial_tenure=initial_tenure,
            weight_cap=weight_cap,
            oscillation_cap=oscillation_cap,
            seed=seed,
        )
    else:
        rts = SingleReactiveTabuSearch(
            returns_data=returns_data,
            cov_matrix=cov_matrix,
            n_assets=n_assets,
            rf=rf,
            max_iter=max_iter,
            neighbors_size=neighbors_size,
            initial_tenure=initial_tenure,
            weight_cap=weight_cap,
            oscillation_cap=oscillation_cap,
            seed=seed,
        )
    return rts.run()


# ═══════════════════════════════════════════════════════════════════════════
# PARETO FRONT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _build_pareto_front(points):
    """
    Extract non-dominated (risk, return) points.
    A point dominates another if it has higher return AND lower risk.
    """
    pts = np.array(points)  # (M, 3) — risk, return, sharpe
    # Sort by risk ascending
    sorted_idx = np.argsort(pts[:, 0])
    pts_sorted = pts[sorted_idx]

    front = []
    max_ret = -np.inf
    for risk, ret, sharpe in pts_sorted:
        if ret > max_ret:
            front.append((risk, ret, sharpe))
            max_ret = ret

    return front


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_pareto_front(all_explored, pareto_front, best_result,
                      rmpso_ref, cso_ref,
                      returns_data, cov_matrix, n_assets, rf,
                      output_path='rts_pareto_front.png'):
    """
    Generate the required Pareto Front plot showing risk-return trade-off
    of RTS results compared to a random-walk baseline.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    fig, ax = plt.subplots(figsize=(14, 9))

    # ── Random-walk baseline (1000 random portfolios) ──────────────
    np.random.seed(42)
    rand_risks, rand_rets = [], []
    for _ in range(2000):
        w = repair_weights(np.random.uniform(0, 1, n_assets))
        rand_rets.append(calc_annual_return(w, returns_data) * 100)
        rand_risks.append(calc_annual_risk(w, cov_matrix) * 100)
    ax.scatter(rand_risks, rand_rets, alpha=0.12, s=8, color='#bdc3c7',
               label='Random Portfolios (baseline)', zorder=1)

    # ── All RTS explored points ────────────────────────────────────
    # Subsample for performance (plot at most 5000 points)
    explored = np.array(all_explored)
    if len(explored) > 5000:
        idx = np.random.choice(len(explored), 5000, replace=False)
        explored = explored[idx]
    ax.scatter(explored[:, 0] * 100, explored[:, 1] * 100,
               alpha=0.06, s=4, color='#3498db',
               label='RTS Explored Solutions', zorder=2)

    # ── Pareto front line ──────────────────────────────────────────
    pf = np.array(pareto_front)
    sort_idx = np.argsort(pf[:, 0])
    ax.plot(pf[sort_idx, 0] * 100, pf[sort_idx, 1] * 100,
            'o-', color='#e74c3c', markersize=5, linewidth=2.5,
            label='RTS Pareto Front', zorder=5)

    # ── Best RTS solution (max Sharpe) ─────────────────────────────
    ax.scatter(best_result['best_risk'] * 100,
               best_result['best_return'] * 100,
               marker='*', s=400, color='#e74c3c', edgecolors='black',
               linewidths=1.5, zorder=7,
               label=f"★ RTS Best (Sharpe={best_result['best_sharpe']:.3f})")

    # ── RMPSO reference point ──────────────────────────────────────
    if rmpso_ref:
        ax.scatter(rmpso_ref['risk'] * 100, rmpso_ref['return'] * 100,
                   marker='D', s=180, color='#2ecc71', edgecolors='black',
                   linewidths=1.5, zorder=6,
                   label=f"RMPSO (Sharpe={rmpso_ref['sharpe']:.3f})")
        ax.annotate('RMPSO', (rmpso_ref['risk'] * 100, rmpso_ref['return'] * 100),
                    textcoords='offset points', xytext=(8, -12),
                    fontsize=9, fontweight='bold', color='#27ae60')

    # ── CSO reference point ────────────────────────────────────────
    if cso_ref:
        ax.scatter(cso_ref['risk'] * 100, cso_ref['return'] * 100,
                   marker='X', s=180, color='#9b59b6', edgecolors='black',
                   linewidths=1.5, zorder=6,
                   label=f"CSO (Sharpe={cso_ref['sharpe']:.3f})")
        ax.annotate('CSO', (cso_ref['risk'] * 100, cso_ref['return'] * 100),
                    textcoords='offset points', xytext=(8, 5),
                    fontsize=9, fontweight='bold', color='#8e44ad')

    # ── Annotate best RTS point ────────────────────────────────────
    ax.annotate(
        f"RTS Best ★\n"
        f"Return: {best_result['best_return']*100:.2f}%\n"
        f"Risk: {best_result['best_risk']*100:.2f}%\n"
        f"Sharpe: {best_result['best_sharpe']:.4f}",
        (best_result['best_risk'] * 100, best_result['best_return'] * 100),
        textcoords='offset points', xytext=(15, 15),
        fontsize=9, fontweight='bold', color='#c0392b',
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8',
                  edgecolor='#e74c3c', alpha=0.9),
    )

    # ── Style ──────────────────────────────────────────────────────
    ax.set_xlabel('Annual Risk / Volatility (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Expected Annual Return (%)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Reactive Tabu Search — Pareto Front (Risk vs Return)\n'
        'S&P 500 Portfolio | Jan 2013 – Jan 2023 | Rf = 2%',
        fontsize=15, fontweight='bold', pad=12,
    )
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [✓] Pareto front plot saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    if ALGORITHM == "SWARM":
        print("  MULTI-SOLUTION SWARM RTS — S&P 500 Portfolio Optimization")
        print("  Modules: Lévy Flight | Swarm Memory | Strategic Oscillation")
    else:
        print("  SINGLE-SOLUTION RTS — S&P 500 Portfolio Optimization")
        print("  Modules: Lévy Flight | Reactive Tenure | Strategic Oscillation")
    print("           | Multi-Objective Aspiration")
    print("  Dataset: S&P 500, Jan 2013 – Jan 2023")
    print("=" * 70)

    RF = 0.02

    # ── Step 1: Get Data ──────────────────────────────────────────
    print("\n  [1] Fetching real market data...")
    stock_names, returns_data = download_stock_data()
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
    NUM_RUNS = int(os.getenv("RTS_NUM_RUNS", "100"))
    SEED_POOL_SEED = int(os.getenv("RTS_SEED_POOL_SEED", "786683"))
    if ALGORITHM == "SWARM":
        MAX_ITER = 2000
        SWARM_SIZE = 10
    else:
        MAX_ITER = 5000
        SWARM_SIZE = 1
    NEIGHBORS = 50
    TENURE = 10
    WEIGHT_CAP = 0.10
    OSC_CAP = 0.15
    seed_rng = np.random.default_rng(SEED_POOL_SEED)
    SEEDS = [
        int(seed)
        for seed in seed_rng.choice(1_000_000_000, size=NUM_RUNS, replace=False)
    ]

    print(f"\n  [2] Running {ALGORITHM} RTS ({NUM_RUNS} runs)...")
    if ALGORITHM == "SWARM":
        print(f"      Iterations={MAX_ITER}, Swarm Size={SWARM_SIZE}, Neighbors={NEIGHBORS}, Tenure={TENURE}")
    else:
        print(f"      Iterations={MAX_ITER}, Neighbors={NEIGHBORS}, Tenure={TENURE}")
    print(f"      Weight cap={WEIGHT_CAP*100:.0f}%, Oscillation cap={OSC_CAP*100:.0f}%")
    print(f"      Seed sweep={NUM_RUNS} wide seeds, pool seed={SEED_POOL_SEED}")

    # Build argument tuples for parallel execution
    run_args = [
        (returns_data, cov_daily, n_assets, seed, RF,
         MAX_ITER, SWARM_SIZE, NEIGHBORS, TENURE, WEIGHT_CAP, OSC_CAP, ALGORITHM)
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

    # ── Step 3: Aggregate Results ─────────────────────────────────
    all_sharpes = [r['best_sharpe'] for r in all_results]
    best_idx = int(np.argmax(all_sharpes))
    best_result = all_results[best_idx]

    rts_metrics = calc_all_metrics(
        best_result['best_weights'], returns_data, cov_daily, RF
    )

    print(f"\n  Best Run #{best_idx + 1}:")
    print(f"    Sharpe Ratio : {best_result['best_sharpe']:.4f}")
    print(f"    Annual Return: {best_result['best_return']*100:.2f}%")
    print(f"    Annual Risk  : {best_result['best_risk']*100:.2f}%")
    print(f"    Max Drawdown : {rts_metrics['max_drawdown']*100:.2f}%")

    # ── Step 4: Print portfolio weights ───────────────────────────
    print("\n  " + "=" * 50)
    print("  RTS OPTIMAL PORTFOLIO WEIGHTS")
    print("  " + "=" * 50)
    weight_table = []
    for name, w in zip(stock_names, best_result['best_weights']):
        if w > 0.001:
            weight_table.append([name, f"{w*100:.2f}%"])
    weight_table.append(["── TOTAL", "100.00%"])
    print(tabulate(weight_table,
                   headers=["Stock", "Allocation"],
                   tablefmt="fancy_grid"))

    # ── Step 5: OUTPUT 1 — Convergence Table ──────────────────────
    print("\n  [3] Generating convergence table...")
    best_conv = best_result['convergence_log']

    conv_rows = []
    for (it, b_sharpe, c_sharpe, ten, phase) in best_conv:
        conv_rows.append([it, f"{b_sharpe:.4f}", f"{c_sharpe:.4f}",
                          ten, phase])

    conv_table = tabulate(
        conv_rows,
        headers=["Iteration", "Best Sharpe", "Current Sharpe",
                 "Tenure", "Phase"],
        tablefmt="grid",
    )

    with open("rts_convergence.txt", "w") as f:
        f.write("Reactive Tabu Search — Convergence Log\n")
        f.write(f"Best Run (Seed {SEEDS[best_idx]})\n")
        f.write("=" * 70 + "\n\n")
        f.write(conv_table)
        f.write("\n")
    print("  [✓] Convergence table saved: rts_convergence.txt")

    # ── Step 6: OUTPUT 2 — Comparative Verdict ────────────────────
    print("\n  [4] Building comparison table...")

    # Published reference values
    rmpso_ref = {'return': None, 'risk': None, 'sharpe': 1.159}
    cso_ref = {'return': 0.168, 'risk': 0.187, 'sharpe': 0.950}

    # Paper results (from the reference script)
    paper_results = {
        'BFO':       {'return': 0.142, 'risk': 0.198, 'sharpe': 0.617},
        'FWA':       {'return': 0.131, 'risk': 0.201, 'sharpe': 0.552},
        'CSO':       {'return': 0.168, 'risk': 0.187, 'sharpe': 0.950},
        'Bat':       {'return': 0.124, 'risk': 0.215, 'sharpe': 0.484},
        'mean-CVaR': {'return': 0.098, 'risk': 0.142, 'sharpe': 0.549},
    }

    comparison_rows = []

    # RMPSO reference (from user's own run — Sharpe 1.159)
    comparison_rows.append([
        "RMPSO (Reference)",
        "—", "—",
        f"{rmpso_ref['sharpe']:.4f}",
        "—",
        "Published / Own Run",
    ])

    # Paper methods
    for method, m in paper_results.items():
        comparison_rows.append([
            f"{method} (paper)",
            f"{m['return']*100:.2f}%",
            f"{m['risk']*100:.2f}%",
            f"{m['sharpe']:.4f}",
            "—",
            "Published",
        ])

    # Our RTS
    verdict = ""
    if rts_metrics['sharpe'] > rmpso_ref['sharpe']:
        verdict = "★ BEATS RMPSO"
    elif rts_metrics['sharpe'] > cso_ref['sharpe']:
        verdict = "Beats CSO"
    else:
        verdict = "—"

    comparison_rows.append([
        "★ RTS (Ours)",
        f"{rts_metrics['return']*100:.2f}%",
        f"{rts_metrics['risk']*100:.2f}%",
        f"{rts_metrics['sharpe']:.4f}",
        f"{rts_metrics['max_drawdown']*100:.2f}%",
        verdict,
    ])

    comp_table = tabulate(
        comparison_rows,
        headers=["Algorithm", "Annual Return", "Annual Risk",
                 "Sharpe Ratio", "Max Drawdown", "Verdict"],
        tablefmt="fancy_grid",
    )

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

    with open("rts_comparison.txt", "w") as f:
        f.write("Reactive Tabu Search — Comparative Verdict\n")
        f.write("Dataset: S&P 500 Top 10 Stocks | "
                "Period: 2013–2023 | RF=2%\n")
        f.write("=" * 70 + "\n\n")
        f.write(comp_table)
        f.write("\n\n")
        f.write(f"RTS Run Statistics ({NUM_RUNS} runs)\n")
        f.write("-" * 40 + "\n")
        f.write(stats_table)
        f.write("\n")

    print(comp_table)
    print(f"\n  [✓] Comparison table saved: rts_comparison.txt")

    # ── Step 7: OUTPUT 3 — Pareto Front Plot ──────────────────────
    print("\n  [5] Generating Pareto front plot...")

    # Merge all explored points from all runs
    all_explored = []
    for r in all_results:
        all_explored.extend(r['all_explored'])

    pareto_front = _build_pareto_front(all_explored)

    # For the reference points on the plot, use approximate values
    # (RMPSO may not have exact risk/return, estimate from Sharpe)
    rmpso_plot_ref = None
    # Try to estimate RMPSO risk/return from Sharpe assuming similar
    # return profile — but since we don't have exact values, skip if None
    if rmpso_ref.get('return') and rmpso_ref.get('risk'):
        rmpso_plot_ref = rmpso_ref

    cso_plot_ref = cso_ref

    plot_pareto_front(
        all_explored, pareto_front, best_result,
        rmpso_plot_ref, cso_plot_ref,
        returns_data, cov_daily, n_assets, RF,
    )

    # ── Final Summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REACTIVE TABU SEARCH — COMPLETE!")
    print(f"  RTS Best Sharpe   = {best_result['best_sharpe']:.4f}")
    print(f"  RMPSO Reference   = {rmpso_ref['sharpe']:.4f}")
    print(f"  CSO (Paper Best)  = {cso_ref['sharpe']:.4f}")

    improvement_rmpso = ((rts_metrics['sharpe'] - rmpso_ref['sharpe'])
                         / rmpso_ref['sharpe'] * 100)
    improvement_cso = ((rts_metrics['sharpe'] - cso_ref['sharpe'])
                       / cso_ref['sharpe'] * 100)

    if improvement_rmpso > 0:
        print(f"  vs RMPSO          : +{improvement_rmpso:.2f}% ✅")
    else:
        print(f"  vs RMPSO          : {improvement_rmpso:.2f}%")

    if improvement_cso > 0:
        print(f"  vs CSO            : +{improvement_cso:.2f}% ✅")
    else:
        print(f"  vs CSO            : {improvement_cso:.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()
