"""
Visualize Tabu Search Results
Creates graphs and charts from experiment results.
"""

import matplotlib.pyplot as plt
import numpy as np


def create_convergence_curves(results):
    """
    Create a convergence curve graph similar to the PSO reference.
    Plots best cost (log scale) vs iteration for all functions.
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color palette - distinct colors for each function
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Yellow-green
        '#17becf',  # Cyan
        '#aec7e8',  # Light blue
        '#ffbb78',  # Light orange
        '#98df8a',  # Light green
        '#ff9896',  # Light red
        '#c5b0d5',  # Light purple
    ]
    
    # Extract and plot convergence history for each function
    max_iter = 0
    for idx, (name, result, *_) in enumerate(results):
        history = result.get('convergence_history', [])
        if len(history) > max_iter:
            max_iter = len(history)
        
        if len(history) > 0:
            iterations = range(len(history))
            # Add small epsilon to avoid log(0)
            history_safe = [max(h, 1e-150) for h in history]
            ax.plot(iterations, history_safe, 
                   color=colors[idx % len(colors)], 
                   linewidth=2,
                   label=name.replace('_', ' '))
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Add vertical dashed lines at 40% and 100% of max iterations (like the reference)
    if max_iter > 0:
        ax.axvline(x=int(max_iter * 0.4), color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=max_iter - 1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Cost (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Tabu Search Convergence Curves (D=5, N=40)', fontsize=14, fontweight='bold')
    
    # Legend outside on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=10, framealpha=0.95, borderaxespad=0)
    
    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.set_axisbelow(True)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    fig.subplots_adjust(right=0.78)
    
    # Save
    plt.savefig('convergence_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Convergence curves saved to convergence_curves.png")


def visualize_results(results):
    """
    Create and save visualization charts.
    """
    # Extract data
    names = [r[0] for r in results]
    best_f = [r[1]['best_f'] for r in results]
    avg_f = [r[1]['avg_f'] for r in results]
    max_f = [r[1]['max_f'] for r in results]
    
    # Set up figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Tabu Search Optimization Results', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # Plot 1: Best f(x) horizontal bar with values
    ax1 = axes[0, 0]
    bars = ax1.barh(names, best_f, color=colors)
    ax1.set_xlabel('Best f(x)')
    ax1.set_title('Best f(x) by Function')
    ax1.set_xscale('symlog', linthresh=1e-10)
    ax1.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, best_f):
        ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                f' {val:.2e}', va='center', fontsize=9)
    
    # Plot 2: Avg f(x) horizontal bar with values
    ax2 = axes[0, 1]
    bars = ax2.barh(names, avg_f, color=colors)
    ax2.set_xlabel('Avg f(x)')
    ax2.set_title('Average f(x) by Function')
    ax2.set_xscale('symlog', linthresh=1e-10)
    ax2.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, avg_f):
        ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                f' {val:.2e}', va='center', fontsize=9)
    
    # Plot 3: Best vs Avg vs Max comparison
    ax3 = axes[1, 0]
    x = np.arange(len(names))
    width = 0.25
    ax3.bar(x - width, best_f, width, label='Best', color='green')
    ax3.bar(x, avg_f, width, label='Avg', color='blue')
    ax3.bar(x + width, max_f, width, label='Max', color='red')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('f(x)')
    ax3.set_title('Best vs Avg vs Max')
    ax3.set_yscale('symlog', linthresh=1e-10)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Parameters
    ax4 = axes[1, 1]
    runs = [r[2] for r in results]
    neighbors = [r[3] for r in results]
    tenure = [r[4] for r in results]
    max_iter = [r[5] / 100 for r in results]
    
    x = np.arange(len(names))
    width = 0.2
    ax4.bar(x - 1.5*width, runs, width, label='Runs')
    ax4.bar(x - 0.5*width, neighbors, width, label='Neighbors')
    ax4.bar(x + 0.5*width, tenure, width, label='Tenure')
    ax4.bar(x + 1.5*width, max_iter, width, label='MaxIter/100')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.set_ylabel('Value')
    ax4.set_title('Parameters per Function')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.savefig('results_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Charts saved to results_chart.png")


def create_unigraph(results):
    """
    Create a unified single graph comparing all functions.
    Ranks functions by performance (best f(x)) and shows comprehensive stats.
    """
    # Extract data
    names = [r[0] for r in results]
    best_f = np.array([r[1]['best_f'] for r in results])
    avg_f = np.array([r[1]['avg_f'] for r in results])
    max_f = np.array([r[1]['max_f'] for r in results])
    std_f = np.array([r[1]['std_f'] for r in results])
    
    # Rank by best f(x) - lower is better
    rank_indices = np.argsort(best_f)
    
    # Reorder all data by ranking
    names = [names[i] for i in rank_indices]
    best_f = best_f[rank_indices]
    avg_f = avg_f[rank_indices]
    max_f = max_f[rank_indices]
    std_f = std_f[rank_indices]
    
    # Calculate performance score (normalized, lower best_f = higher score)
    # Using log scale for normalization due to wide range
    log_best = np.log10(best_f + 1e-15)
    perf_scores = 100 * (1 - (log_best - log_best.min()) / (log_best.max() - log_best.min() + 1e-10))
    
    # Set up figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 2, 1], 
                          hspace=0.35, wspace=0.3)
    
    # Main comparison chart (top left, spans 2 columns)
    ax_main = fig.add_subplot(gs[0, :2])
    
    x = np.arange(len(names))
    width = 0.28
    
    # Color gradient based on rank (green = best, red = worst)
    rank_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(names)))
    
    # Create grouped bars
    bars_best = ax_main.bar(x - width, best_f, width, label='Best f(x)', 
                            color='#10b981', edgecolor='#059669', linewidth=1.5, alpha=0.9)
    bars_avg = ax_main.bar(x, avg_f, width, label='Avg f(x) ± Std', 
                           color='#6366f1', edgecolor='#4f46e5', linewidth=1.5, alpha=0.9,
                           yerr=std_f, capsize=5, error_kw={'ecolor': '#f59e0b', 'linewidth': 2, 'capthick': 2})
    bars_max = ax_main.bar(x + width, max_f, width, label='Max f(x)', 
                           color='#ef4444', edgecolor='#dc2626', linewidth=1.5, alpha=0.9)
    
    # Style main chart
    ax_main.set_yscale('symlog', linthresh=1e-12)
    ax_main.set_xlabel('Function (Ranked by Best Performance →)', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('f(x) Value (Symmetric Log Scale)', fontsize=13, fontweight='bold')
    ax_main.set_title('Tabu Search Performance Ranking', fontsize=18, fontweight='bold', pad=15)
    
    # Add rank labels on x-axis
    rank_labels = [f"#{i+1} {name}" for i, name in enumerate(names)]
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(rank_labels, rotation=40, ha='right', fontsize=10, fontweight='medium')
    
    # Legend
    ax_main.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax_main.grid(axis='y', alpha=0.4, linestyle='-')
    
    # Performance score chart (top right)
    ax_score = fig.add_subplot(gs[0, 2])
    
    # Horizontal bar chart for performance scores
    bars_score = ax_score.barh(range(len(names)), perf_scores, color=rank_colors, edgecolor='white', linewidth=0.5)
    ax_score.set_yticks(range(len(names)))
    ax_score.set_yticklabels([f"#{i+1}" for i in range(len(names))], fontsize=10)
    ax_score.set_xlabel('Performance Score', fontsize=11, fontweight='bold')
    ax_score.set_title('Score (0-100)', fontsize=14, fontweight='bold')
    ax_score.set_xlim(0, 105)
    ax_score.invert_yaxis()
    
    # Add score values
    for i, (bar, score) in enumerate(zip(bars_score, perf_scores)):
        ax_score.text(score + 2, i, f'{score:.1f}', va='center', fontsize=9, fontweight='bold')
    
    # Statistics table (bottom left)
    ax_table = fig.add_subplot(gs[1, :2])
    ax_table.axis('off')
    
    # Create table data
    table_data = []
    for i, name in enumerate(names):
        table_data.append([
            f"#{i+1}",
            name,
            f"{best_f[i]:.2e}",
            f"{avg_f[i]:.2e}",
            f"{std_f[i]:.2e}",
            f"{max_f[i]:.2e}",
            f"{perf_scores[i]:.1f}"
        ])
    
    col_labels = ['Rank', 'Function', 'Best f(x)', 'Avg f(x)', 'Std Dev', 'Max f(x)', 'Score']
    
    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           loc='center', cellLoc='center',
                           colColours=['#e0e7ff'] * 7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style table header
    for i in range(len(col_labels)):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#4f46e5')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color code rows by rank
    for row in range(1, len(names) + 1):
        for col in range(len(col_labels)):
            if row <= 3:  # Top 3
                table[(row, col)].set_facecolor('#d1fae5')  # Light green
            elif row >= len(names) - 1:  # Bottom 2
                table[(row, col)].set_facecolor('#fee2e2')  # Light red
            else:
                table[(row, col)].set_facecolor('#f8fafc')  # Light gray
    
    ax_table.set_title('Detailed Statistics (Sorted by Performance)', fontsize=14, fontweight='bold', pad=10)
    
    # Summary insights (bottom right)
    ax_insights = fig.add_subplot(gs[1, 2])
    ax_insights.axis('off')
    
    insights_text = f"""
    [1st] Best Performer:
       {names[0]}
       Best f(x) = {best_f[0]:.2e}
    
    [2nd] Runner Up:
       {names[1]}
       Best f(x) = {best_f[1]:.2e}
    
    [!] Needs Improvement:
       {names[-1]}
       Best f(x) = {max_f[-1]:.2e}
    
    [+] Most Consistent:
       {names[np.argmin(std_f)]}
       Std = {std_f.min():.2e}
    """
    
    ax_insights.text(0.1, 0.95, insights_text, transform=ax_insights.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f9ff', edgecolor='#0284c7', alpha=0.9))
    ax_insights.set_title('Key Insights', fontsize=14, fontweight='bold')
    
    # Main title
    fig.suptitle('Unified Performance Analysis: Tabu Search Optimization', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig('unigraph.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.style.use('default')
    print("Unified graph saved to unigraph.png")
