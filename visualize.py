"""
Visualize Tabu Search Results
Creates graphs and charts from experiment results.
"""

import matplotlib.pyplot as plt
import numpy as np


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
