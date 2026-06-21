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
                      returns_data, cov_matrix, n_assets, rf,
                      output_path=None):
    """
    Generate the required Pareto Front plot showing risk-return trade-off
    of RTS results compared to a random-walk baseline.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    if output_path is None:
        output_path = cfg.OUT_PARETO_PNG

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
               label=f'{ALGORITHM} Explored Solutions', zorder=2)

    # ── Pareto front line ──────────────────────────────────────────
    pf = np.array(pareto_front)
    sort_idx = np.argsort(pf[:, 0])
    ax.plot(pf[sort_idx, 0] * 100, pf[sort_idx, 1] * 100,
            'o-', color='#e74c3c', markersize=5, linewidth=2.5,
            label=f'{ALGORITHM} Pareto Front', zorder=5)

    # ── Best RTS solution (max Sharpe) ─────────────────────────────
    ax.scatter(best_result['best_risk'] * 100,
               best_result['best_return'] * 100,
               marker='*', s=400, color='#e74c3c', edgecolors='black',
               linewidths=1.5, zorder=7,
               label=f"★ {ALGORITHM} Best (Sharpe={best_result['best_sharpe']:.3f})")

    # ── Annotate best point ────────────────────────────────────
    ax.annotate(
        f"{ALGORITHM} Best ★\n"
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
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    ax.set_xlabel('Annual Risk / Volatility (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Expected Annual Return (%)', fontsize=13, fontweight='bold')
    ax.set_title(
        f'{ALGORITHM} — Pareto Front (Risk vs Return)\n'
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
# CONVERGENCE GRAPHS
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence_graphs(all_results, best_idx, seeds, num_runs,
                            output_path=None):
    """
    Multi-panel convergence figure with:
      Panel 1 — All-runs Sharpe convergence (overlaid), best run highlighted
      Panel 2 — Best-run detail: Sharpe + tenure on dual-axis
      Panel 3 — Per-run final Sharpe bar chart (mean ± std band, zoomed y-axis)
      Panel 4 — Per-metric violin + scatter plots (one sub-panel per metric)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    if output_path is None:
        output_path = cfg.OUT_CONVERGENCE_GRAPH

    PALETTE = {
        'bg':        '#ffffff',
        'panel':     '#ffffff',
        'grid':      '#cccccc',
        'text':      '#000000',
        'muted':     '#333333',
        'accent':    '#4a90e2', # blue
        'best':      '#f39c12', # orange/gold
        'danger':    '#e74c3c', # red
        'success':   '#2ecc71', # green
    }

    all_sharpes_arr = np.array([r['best_sharpe'] for r in all_results])
    all_returns_arr = np.array([r['best_return'] * 100 for r in all_results])
    all_risks_arr   = np.array([r['best_risk']   * 100 for r in all_results])
    mean_s = np.mean(all_sharpes_arr)
    std_s  = np.std(all_sharpes_arr)

    fig = plt.figure(figsize=(20, 16), facecolor=PALETTE['bg'])
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.35,
                            left=0.07, right=0.97, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])  # all-run convergence
    ax2 = fig.add_subplot(gs[0, 1])  # best-run detail
    ax3 = fig.add_subplot(gs[1, 0])  # per-run Sharpe bars
    # gs[1, 1] is split into 3 sub-columns for per-metric violins

    def _style(ax, title, title_fontsize=12):
        ax.set_facecolor(PALETTE['panel'])
        ax.tick_params(colors=PALETTE['text'], labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        ax.grid(color=PALETTE['grid'], linestyle='--', linewidth=0.6, alpha=0.5, axis='y')
        ax.title.set_color(PALETTE['text'])
        ax.xaxis.label.set_color(PALETTE['muted'])
        ax.yaxis.label.set_color(PALETTE['muted'])
        ax.set_title(title, fontsize=title_fontsize + 2, fontweight='bold', pad=8)

    # ── Panel 1: All-run convergence ────────────────────────────────
    all_p1_sharpes = []
    for i, r in enumerate(all_results):
        log    = r['convergence_log']
        iters  = [e[0] for e in log]
        sharpe = [e[1] for e in log]
        all_p1_sharpes.extend(sharpe)
        if i == best_idx:
            continue   # draw best run on top
        ax1.plot(iters, sharpe, color=PALETTE['accent'], alpha=0.25, linewidth=0.8)

    best_log    = all_results[best_idx]['convergence_log']
    best_iters  = [e[0] for e in best_log]
    best_sharpe = [e[1] for e in best_log]
    ax1.plot(best_iters, best_sharpe, color=PALETTE['best'],
             linewidth=2.2, label=f'Best Run #{best_idx+1}', zorder=5)

    ax1.axhline(mean_s, color=PALETTE['success'], linestyle='--',
                linewidth=1.4, label=f'Mean = {mean_s:.3f}')
    ax1.axhspan(mean_s - std_s, mean_s + std_s,
                color=PALETTE['success'], alpha=0.12, label=f'±1σ ({std_s:.3f})')

    # Zoom y-axis to actual data range so differences are visible
    if all_p1_sharpes:
        y_lo, y_hi = min(all_p1_sharpes), max(all_p1_sharpes)
        y_pad = max((y_hi - y_lo) * 0.15, 0.002)
        ax1.set_ylim(y_lo - y_pad, y_hi + y_pad)

    _style(ax1, 'Convergence — All Runs')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Sharpe Ratio')
    ax1.legend(fontsize=8, facecolor=PALETTE['panel'],
               labelcolor=PALETTE['text'], framealpha=0.8, loc='lower right')

    # ── Panel 2: Best-run detail with tenure ────────────────────────
    iters_b  = [e[0] for e in best_log]
    sharpe_b = [e[1] for e in best_log]
    curr_b   = [e[2] for e in best_log]
    tenure_b = [e[3] if len(e) > 3 else (cfg.RTS_INITIAL_TENURE if ALGORITHM=="RTS" else cfg.STS_TENURE) for e in best_log]

    ax2.plot(iters_b, sharpe_b, color=PALETTE['best'],
             linewidth=2.0, label='Best Sharpe')
    ax2.plot(iters_b, curr_b,   color=PALETTE['accent'],
             linewidth=1.0, alpha=0.6, label='Current Sharpe')

    # Zoom y-axis for Panel 2
    all_p2 = sharpe_b + curr_b
    if all_p2:
        y2_lo, y2_hi = min(all_p2), max(all_p2)
        y2_pad = max((y2_hi - y2_lo) * 0.12, 0.002)
        ax2.set_ylim(y2_lo - y2_pad, y2_hi + y2_pad)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Sharpe Ratio', color=PALETTE['muted'])

    ax2r = ax2.twinx()
    ax2r.plot(iters_b, tenure_b, color=PALETTE['danger'],
              linewidth=1.2, linestyle=':', alpha=0.8, label='Tenure')
    ax2r.set_ylabel('Tenure', color=PALETTE['danger'])
    ax2r.tick_params(colors=PALETTE['danger'])
    ax2r.set_facecolor(PALETTE['panel'])
    for spine in ax2r.spines.values():
        spine.set_edgecolor(PALETTE['grid'])

    _style(ax2, f'Best Run #{best_idx+1} — Detail (Seed {seeds[best_idx]})')
    lines_a, labs_a = ax2.get_legend_handles_labels()
    lines_b, labs_b = ax2r.get_legend_handles_labels()
    ax2.legend(lines_a + lines_b, labs_a + labs_b, fontsize=8,
               facecolor=PALETTE['panel'], labelcolor=PALETTE['text'],
               framealpha=0.8, loc='lower right')

    # ── Panel 3: Per-run final Sharpe bar chart ──────────────────────
    x_pos      = np.arange(num_runs)
    colors_bar = [PALETTE['best'] if i == best_idx else PALETTE['accent']
                  for i in range(num_runs)]
    ax3.bar(x_pos, all_sharpes_arr, color=colors_bar, edgecolor='black', width=0.75, alpha=0.85)
    ax3.axhline(mean_s, color=PALETTE['success'], linestyle='--',
                linewidth=1.4, label=f'Mean = {mean_s:.4f}')
    ax3.axhspan(mean_s - std_s, mean_s + std_s,
                color=PALETTE['success'], alpha=0.12, label='±1σ')

    best_patch  = mpatches.Patch(color=PALETTE['best'],  label=f'Best Run #{best_idx+1}')
    other_patch = mpatches.Patch(color=PALETTE['accent'], label='Other Runs')
    ax3.legend(handles=[best_patch, other_patch], fontsize=8,
               facecolor=PALETTE['panel'], labelcolor=PALETTE['text'],
               framealpha=0.8)

    # Zoom y-axis so bar-height differences are clearly visible
    s_lo, s_hi = all_sharpes_arr.min(), all_sharpes_arr.max()
    s_pad = max((s_hi - s_lo) * 0.30, 0.002)
    ax3.set_ylim(max(0, s_lo - s_pad), s_hi + s_pad * 0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([str(i) for i in range(num_runs)],
                        fontsize=7, rotation=45 if num_runs > 15 else 0)

    _style(ax3, 'Final Sharpe per Run')
    ax3.set_xlabel('Run Index')
    ax3.set_ylabel('Sharpe Ratio')

    # ── Panel 4: One violin sub-panel per metric ─────────────────────
    # Split gs[1,1] into 3 equal columns
    gs4 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1, 1], wspace=0.40)

    metrics_def = [
        ('Sharpe',     all_sharpes_arr, PALETTE['best'],    '%.3f'),
        ('Return (%)', all_returns_arr, PALETTE['success'], '%.2f%%'),
        ('Risk (%)',   all_risks_arr,   PALETTE['danger'],  '%.2f%%'),
    ]

    rng = np.random.default_rng(42)
    for col_idx, (label, data, color, fmt) in enumerate(metrics_def):
        ax_m = fig.add_subplot(gs4[col_idx])
        ax_m.set_facecolor(PALETTE['panel'])
        for spine in ax_m.spines.values():
            spine.set_edgecolor('black')
        ax_m.grid(color=PALETTE['grid'], linestyle='--', linewidth=0.6, alpha=0.5, axis='y')
        ax_m.tick_params(colors=PALETTE['text'], labelsize=8)
        ax_m.xaxis.label.set_color(PALETTE['muted'])
        ax_m.yaxis.label.set_color(PALETTE['muted'])

        # Violin
        parts = ax_m.violinplot(data, positions=[0],
                                showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor(PALETTE['text'])
            pc.set_linewidth(0.8)
        for part_key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if part_key in parts:
                parts[part_key].set_edgecolor(PALETTE['text'])
                parts[part_key].set_linewidth(1.4)

        # Scatter individual run dots with jitter
        jitter = rng.uniform(-0.10, 0.10, len(data))
        ax_m.scatter(jitter, data, color=color, s=22, alpha=0.75,
                     edgecolors='white', linewidths=0.4, zorder=4)

        # Mean annotation
        mu = np.mean(data)
        ax_m.axhline(mu, color='white', linestyle='--', linewidth=1.1, alpha=0.7)
        ax_m.annotate(f'μ={fmt % mu}',
                      xy=(0.5, mu), xycoords=('axes fraction', 'data'),
                      fontsize=7, color='white', va='bottom', ha='center',
                      xytext=(0, 4), textcoords='offset points')

        ax_m.set_xlim(-0.50, 0.50)
        ax_m.set_xticks([])
        ax_m.set_title(label, fontsize=10, fontweight='bold',
                       color=PALETTE['text'], pad=6)
        if col_idx == 0:
            ax_m.set_ylabel('Value', color=PALETTE['muted'])

    # Section title for panel-4 area
    pos4 = gs[1, 1].get_position(fig)
    fig.text((pos4.x0 + pos4.x1) / 2, pos4.y1 + 0.005,
             f'Metric Distributions ({num_runs} runs)',
             ha='center', va='bottom', fontsize=12, fontweight='bold',
             color=PALETTE['text'])

    # ── Super title ─────────────────────────────────────────────────
    fig.suptitle(
        f'{ALGORITHM} — Convergence & Run Analysis\n'
        'S&P 500 Portfolio | Jan 2013 – Jan 2023',
        fontsize=16, fontweight='bold',
        color=PALETTE['text'], y=0.975,
    )

    plt.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor=PALETTE['bg'])
    plt.close()
    print(f"  [\u2713] Convergence graphs saved: {output_path}")


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

    # ── All-Runs Summary Table ────────────────────────────────────
    print("\n  " + "=" * 66)
    print("  ALL RUNS — INDIVIDUAL RESULTS")
    print("  " + "=" * 66)
    run_rows = [
        [
            i + 1,
            SEEDS[i],
            f"{r['best_sharpe']:.4f}",
            f"{r['best_return']*100:.2f}%",
            f"{r['best_risk']*100:.2f}%",
        ]
        for i, r in enumerate(all_results)
    ]
    print(tabulate(
        run_rows,
        headers=["Run #", "Seed", "Sharpe", "Annual Return", "Annual Risk"],
        tablefmt="fancy_grid",
    ))

    # ── Cross-run statistics ───────────────────────────────────────
    all_returns = np.array([r['best_return'] for r in all_results])
    all_risks   = np.array([r['best_risk']   for r in all_results])
    all_sharpes_arr = np.array([r['best_sharpe'] for r in all_results])

    print("\n  " + "=" * 66)
    print(f"  RUN STATISTICS ({NUM_RUNS} runs)")
    print("  " + "=" * 66)
    metric_stats = [
        ["Sharpe Ratio",
         f"{np.mean(all_sharpes_arr):.4f}",
         f"{np.min(all_sharpes_arr):.4f}",
         f"{np.max(all_sharpes_arr):.4f}",
         f"{np.std(all_sharpes_arr):.4f}"],
        ["Annual Return",
         f"{np.mean(all_returns)*100:.2f}%",
         f"{np.min(all_returns)*100:.2f}%",
         f"{np.max(all_returns)*100:.2f}%",
         f"{np.std(all_returns)*100:.2f}%"],
        ["Annual Risk",
         f"{np.mean(all_risks)*100:.2f}%",
         f"{np.min(all_risks)*100:.2f}%",
         f"{np.max(all_risks)*100:.2f}%",
         f"{np.std(all_risks)*100:.2f}%"],
    ]
    print(tabulate(
        metric_stats,
        headers=["Metric", "Mean", "Min", "Max", "Std Dev"],
        tablefmt="fancy_grid",
    ))

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

    # ── Step 4: Print portfolio weights + per-stock detail ───────────
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
        ten = entry[3] if len(entry) > 3 else (cfg.RTS_INITIAL_TENURE if ALGORITHM=="RTS" else cfg.STS_TENURE)
        phase = entry[4] if len(entry) > 4 else "Normal"
        
        conv_rows.append([it, f"{b_sharpe:.4f}", f"{c_sharpe:.4f}",
                          ten, phase])

    conv_table = tabulate(
        conv_rows,
        headers=["Iteration", "Best Sharpe", "Current Sharpe",
                 "Tenure", "Phase"],
        tablefmt="grid",
    )

    with open(cfg.OUT_CONVERGENCE_TXT, "w") as f:
        f.write(f"{ALGORITHM} — Convergence Log\n")
        f.write(f"Best Run (Seed {SEEDS[best_idx]})\n")
        f.write("=" * 70 + "\n\n")
        f.write(conv_table)
        f.write("\n")
    print(f"  [✓] Convergence table saved: {cfg.OUT_CONVERGENCE_TXT}")

    # ── Step 6: OUTPUT 2 — Comparative Verdict ────────────────────
    # ── Step 6: OUTPUT 2 — Final Metrics ────────────────────
    print("\n  [4] Building summary table...")

    comparison_rows = []
    comparison_rows.append([
        f"★ {ALGORITHM} (Ours)",
        f"{rts_metrics['return']*100:.2f}%",
        f"{rts_metrics['risk']*100:.2f}%",
        f"{rts_metrics['sharpe']:.4f}",
        f"{rts_metrics['max_drawdown']*100:.2f}%",
    ])

    comp_table = tabulate(
        comparison_rows,
        headers=["Algorithm", "Annual Return", "Annual Risk",
                 "Sharpe Ratio", "Max Drawdown"],
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

    # Full per-run table for the saved file
    run_rows_file = [
        [i + 1, SEEDS[i],
         f"{r['best_sharpe']:.4f}",
         f"{r['best_return']*100:.2f}%",
         f"{r['best_risk']*100:.2f}%"]
        for i, r in enumerate(all_results)
    ]
    all_runs_table = tabulate(
        run_rows_file,
        headers=["Run #", "Seed", "Sharpe", "Annual Return", "Annual Risk"],
        tablefmt="grid",
    )

    # Extended metric stats (mean/min/max/std for all three metrics)
    full_metric_stats = [
        ["Sharpe Ratio",
         f"{np.mean(all_sharpes_arr):.4f}",
         f"{np.min(all_sharpes_arr):.4f}",
         f"{np.max(all_sharpes_arr):.4f}",
         f"{np.std(all_sharpes_arr):.4f}"],
        ["Annual Return",
         f"{np.mean(all_returns)*100:.2f}%",
         f"{np.min(all_returns)*100:.2f}%",
         f"{np.max(all_returns)*100:.2f}%",
         f"{np.std(all_returns)*100:.2f}%"],
        ["Annual Risk",
         f"{np.mean(all_risks)*100:.2f}%",
         f"{np.min(all_risks)*100:.2f}%",
         f"{np.max(all_risks)*100:.2f}%",
         f"{np.std(all_risks)*100:.2f}%"],
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
        f.write("Best Portfolio — Overview\n")
        f.write("-" * 60 + "\n")
        f.write(comp_table)
        f.write("\n\n")
        f.write("Best Portfolio — Stock-Level Allocation Detail\n")
        f.write("-" * 60 + "\n")
        f.write(alloc_table_str)
        f.write("\n")

    print(comp_table)
    print(f"\n  [✓] Summary table saved: {cfg.OUT_COMPARISON_TXT}")

    # ── Step 7: OUTPUT 3 — Pareto Front Plot ──────────────────────
    print("\n  [5] Generating Pareto front plot...")

    # Merge all explored points from all runs
    all_explored = []
    for r in all_results:
        all_explored.extend(r['all_explored'])

    pareto_front = _build_pareto_front(all_explored)

    plot_pareto_front(
        all_explored, pareto_front, best_result,
        returns_data, cov_daily, n_assets, RF,
    )

    # ── Step 8: OUTPUT 4 — Convergence Graphs ─────────────────────
    print("\n  [6] Generating convergence graphs...")
    plot_convergence_graphs(all_results, best_idx, SEEDS, NUM_RUNS)

    # ── Final Summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {ALGORITHM} — COMPLETE!")
    print(f"  {ALGORITHM} Best Sharpe   = {best_result['best_sharpe']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
