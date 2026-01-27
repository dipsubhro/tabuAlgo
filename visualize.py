"""
Visualize Tabu Search Results
Creates graphs and charts from experiment results.
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_results(results):
    """
    Create and save visualization charts.
    
    Args:
        results: list of tuples (name, result, num_runs, neighbors, tenure, max_iter, bounds, dims)
    """
    # Extract data for plotting
    names = [r[0] for r in results]
    best_f = [r[1]['best_f'] for r in results]
    avg_f = [r[1]['avg_f'] for r in results]
    median_f = [r[1]['median_f'] for r in results]
    max_f = [r[1]['max_f'] for r in results]
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tabu Search Optimization Results', fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
    
    # Plot 1: Best f(x) values
    ax1 = axes[0, 0]
    ax1.bar(names, best_f, color=colors)
    ax1.set_title('Best f(x) by Function')
    ax1.set_ylabel('Best f(x)')
    ax1.set_yscale('symlog')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Average f(x) values
    ax2 = axes[0, 1]
    ax2.bar(names, avg_f, color=colors)
    ax2.set_title('Average f(x) by Function')
    ax2.set_ylabel('Average f(x)')
    ax2.set_yscale('symlog')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Comparison (Best vs Avg vs Median vs Max)
    ax3 = axes[1, 0]
    x = np.arange(len(names))
    width = 0.2
    ax3.bar(x - 1.5*width, best_f, width, label='Best', color='green')
    ax3.bar(x - 0.5*width, avg_f, width, label='Avg', color='blue')
    ax3.bar(x + 0.5*width, median_f, width, label='Median', color='orange')
    ax3.bar(x + 1.5*width, max_f, width, label='Max', color='red')
    ax3.set_title('Comparison: Best vs Avg vs Median vs Max')
    ax3.set_ylabel('f(x)')
    ax3.set_yscale('symlog')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Parameters visualization
    ax4 = axes[1, 1]
    runs = [r[2] for r in results]
    neighbors = [r[3] for r in results]
    tenure = [r[4] for r in results]
    max_iter = [r[5] for r in results]
    
    x = np.arange(len(names))
    width = 0.2
    ax4.bar(x - 1.5*width, runs, width, label='Runs', color='purple')
    ax4.bar(x - 0.5*width, neighbors, width, label='Neighbors', color='cyan')
    ax4.bar(x + 0.5*width, tenure, width, label='Tenure', color='magenta')
    ax4.bar(x + 1.5*width, [m/100 for m in max_iter], width, label='MaxIter/100', color='yellow')
    ax4.set_title('Parameters per Function')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Charts saved to results_chart.png")
