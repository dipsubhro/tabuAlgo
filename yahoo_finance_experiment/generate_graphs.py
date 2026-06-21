import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import os
import sys

# Ensure project root is on the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config as cfg

def plot_convergence_graphs(run_data, output_path=None):
    """
    3-panel figure matching the preview layout:
      Panel 1 (left)   — Best-run Sharpe convergence line
      Panel 2 (centre) — Optimal portfolio weights bar chart
      Panel 3 (right)  — Final Sharpe per run bar chart
    """
    if output_path is None:
        output_path = cfg.OUT_CONVERGENCE_GRAPH

    algorithm = run_data['algorithm']
    num_runs = run_data['num_runs']
    best_idx = run_data['best_idx']
    seeds = run_data['seeds']
    stock_names = run_data['stock_names']
    best_result = run_data['best_result']
    all_results = run_data['all_results']

    all_sharpes_arr = np.array([r['best_sharpe'] for r in all_results])
    mean_s = np.mean(all_sharpes_arr)

    best_log = all_results[best_idx]['convergence_log']
    best_iters = [e[0] for e in best_log]
    best_sharpe = [e[1] for e in best_log]

    ret_pct = best_result['best_return'] * 100
    risk_pct = best_result['best_risk'] * 100
    sharpe_v = best_result['best_sharpe']
    weights = best_result['best_weights']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             facecolor='white',
                             gridspec_kw={'wspace': 0.35})
    fig.patch.set_facecolor('white')

    # ───────────────────────────────────────────────────────────────
    # Panel 1: Best-run convergence
    # ───────────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(best_iters, best_sharpe, color='#27ae60', linewidth=2.0,
             label='Best Run Optimization')

    y_lo, y_hi = min(best_sharpe), max(best_sharpe)
    y_pad = max((y_hi - y_lo) * 0.15, 0.002)
    ax1.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    ax1.set_facecolor('white')
    ax1.grid(True, linestyle='--', linewidth=0.6, alpha=0.5, color='#cccccc')
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.set_title(
        f'{algorithm} Convergence\n({num_runs} runs, {max(best_iters)} iterations)',
        fontsize=12, fontweight='bold'
    )
    ax1.legend(fontsize=9, framealpha=0.85, loc='lower right')

    # ───────────────────────────────────────────────────────────────
    # Panel 2: Portfolio weights bar chart
    # ───────────────────────────────────────────────────────────────
    ax2 = axes[1]

    # Build a distinct colour palette for each stock bar
    BAR_COLORS = [
        '#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6',
        '#1abc9c', '#f39c12', '#d35400', '#c0392b', '#8e44ad',
        '#16a085', '#2980b9', '#27ae60', '#f1c40f', '#7f8c8d',
        '#2c3e50', '#e91e63', '#00bcd4', '#ff5722', '#607d8b',
    ]

    w_pct = [w * 100 for w in weights]
    x_pos = np.arange(len(stock_names))
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(stock_names))]

    bars = ax2.bar(x_pos, w_pct, color=colors, edgecolor='white',
                   linewidth=0.5, width=0.7)

    # Label each bar with its percentage
    for bar, val in zip(bars, w_pct):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold'
        )

    ax2.set_facecolor('white')
    ax2.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.5, color='#cccccc')
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stock_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Allocation (%)', fontsize=11)
    ax2.set_ylim(0, max(w_pct) * 1.18)
    ax2.set_title(
        f'Optimal Portfolio Weights\n'
        f'Return={ret_pct:.2f}% | Risk={risk_pct:.2f}% | Sharpe={sharpe_v:.4f}',
        fontsize=12, fontweight='bold'
    )

    # ───────────────────────────────────────────────────────────────
    # Panel 3: Per-run Sharpe bar chart
    # ───────────────────────────────────────────────────────────────
    ax3 = axes[2]
    x3 = np.arange(num_runs)
    bar_colors3 = ['#f39c12' if i == best_idx else '#3498db' for i in range(num_runs)]
    ax3.bar(x3, all_sharpes_arr, color=bar_colors3, edgecolor='none', width=0.75)

    ax3.axhline(mean_s, color='#27ae60', linestyle='--', linewidth=1.4)

    s_lo, s_hi = all_sharpes_arr.min(), all_sharpes_arr.max()
    s_pad = max((s_hi - s_lo) * 0.30, 0.001)
    ax3.set_ylim(max(0, s_lo - s_pad), s_hi + s_pad * 0.5)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    ax3.set_facecolor('white')
    ax3.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.5, color='#cccccc')
    for spine in ax3.spines.values():
        spine.set_edgecolor('black')
    ax3.set_xticks(x3)
    run_labels = [str(i + 1) for i in range(num_runs)]
    ax3.set_xticklabels(run_labels,
                        fontsize=7 if num_runs > 15 else 9,
                        rotation=45 if num_runs > 20 else 0)
    ax3.set_xlabel('Run', fontsize=11)
    ax3.set_ylabel('Sharpe Ratio', fontsize=11)
    ax3.set_title('Final Sharpe per Run', fontsize=12, fontweight='bold')

    best_patch = mpatches.Patch(color='#f39c12', label=f'Best Run #{best_idx + 1}')
    other_patch = mpatches.Patch(color='#3498db', label='Other Runs')
    ax3.legend(handles=[best_patch, other_patch], fontsize=9,
               framealpha=0.85, loc='upper right')

    # ───────────────────────────────────────────────────────────────
    # Super title
    # ───────────────────────────────────────────────────────────────
    fig.suptitle(
        f'{algorithm} — Single Objective Portfolio Optimisation (Maximise Sharpe Ratio)\n'
        f'Data: S&P 500 Yahoo Finance {cfg.DATA_START} to {cfg.DATA_END}  |  '
        f'Seed: {seeds[best_idx]}  |  RF = {cfg.RF*100:.1f}%',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [✓] Convergence graphs saved: {output_path}")

def main():
    if not os.path.exists(cfg.OUT_RUN_DATA_JSON):
        print(f"Error: Run data not found at {cfg.OUT_RUN_DATA_JSON}")
        print("Please run python run_rts_portfolio.py first to generate the data.")
        sys.exit(1)

    print(f"Loading data from {cfg.OUT_RUN_DATA_JSON}...")
    with open(cfg.OUT_RUN_DATA_JSON, 'r') as f:
        run_data = json.load(f)

    print("Generating Convergence Graphs...")
    plot_convergence_graphs(run_data, cfg.OUT_CONVERGENCE_GRAPH)

    print("Graph generation complete!")

if __name__ == "__main__":
    main()
