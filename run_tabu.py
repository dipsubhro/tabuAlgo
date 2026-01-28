"""Tabu Search Runner Function"""

from tabu import tabu_search
import numpy as np


def run_tabu(fn, num_runs=25, neighbors=10, tenure=5, max_iter=1000, bounds=(-5, 5), dims=5):
    """
    Run tabu search multiple times with deterministic seed policy.
    Random initial seed, then doubles after each run.
    """
    best_f = float('inf')
    best_x = None
    all_f = []
    
    # Generate random initial seed (1 to 1000), then double it each run
    initial_seed = np.random.randint(1, 1001)
    seed = initial_seed
    
    for run in range(num_runs):
        # Set seed for reproducibility - seed doubles each run
        np.random.seed(seed)
        
        x0 = np.random.uniform(bounds[0], bounds[1], size=dims)
        x, f, _, _, _ = tabu_search(fn, x0, tenure=tenure, max_iter=max_iter, 
                                     bounds=bounds, neighbors_size=neighbors)
        all_f.append(f)
        if f < best_f:
            best_f = f
            best_x = x
        
        # Double the seed for next run
        seed *= 2
    
    return {
        "best_x": best_x,
        "best_f": best_f,
        "avg_f": np.mean(all_f),
        "median_f": np.median(all_f),
        "max_f": np.max(all_f),
        "std_f": np.std(all_f)
    }