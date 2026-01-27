"""Tabu Search Runner Function"""

from tabu import tabu_search
import numpy as np


def run_tabu(fn, num_runs=25, neighbors=10, tenure=5, max_iter=1000, bounds=(-5, 5), dims=5):
    """
    Run tabu search on a function multiple times.
    
    Args:
        fn: The objective function to minimize
        num_runs: Number of runs
        neighbors: Neighbors per iteration
        tenure: Tabu tenure
        max_iter: Max iterations
        bounds: (min, max) tuple
        dims: Number of dimensions
    
    Returns:
        dict with best_x, best_f, avg_f, median_f, max_f
    """
    best_f = float('inf')
    best_x = None
    all_f = []
    
    for _ in range(num_runs):
        x0 = np.random.uniform(bounds[0], bounds[1], size=dims)
        x, f, _, _, _ = tabu_search(fn, x0, tenure=tenure, max_iter=max_iter, 
                                     bounds=bounds, neighbors_size=neighbors)
        all_f.append(f)
        if f < best_f:
            best_f = f
            best_x = x
    
    return {
        "best_x": best_x,
        "best_f": best_f,
        "avg_f": np.mean(all_f),
        "median_f": np.median(all_f),
        "max_f": np.max(all_f)
    }