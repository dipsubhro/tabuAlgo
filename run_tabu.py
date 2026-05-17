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
    all_histories = []  # Track convergence history from all runs
    
    # Generate random initial seed within valid 32-bit range
    initial_seed = np.random.randint(1, 1001)
    seed = 954777839  # Valid seed within 32-bit range
    
    for run in range(num_runs):
        # Set seed for reproducibility
        np.random.seed(seed)
        
        x0 = np.random.uniform(bounds[0], bounds[1], size=dims)
        x, f, _, _, _, history = tabu_search(fn, x0, tenure=tenure, max_iter=max_iter, 
                                             bounds=bounds, neighbors_size=neighbors)
        all_f.append(f)
        all_histories.append(history)
        if f < best_f:
            best_f = f
            best_x = x
        
        # Increment seed for next run (avoids overflow)
        seed += 12345
    
    # Compute average convergence history across all runs
    # Pad histories to same length if needed
    max_len = max(len(h) for h in all_histories)
    padded_histories = []
    for h in all_histories:
        if len(h) < max_len:
            # Extend with the last value
            padded = h + [h[-1]] * (max_len - len(h))
        else:
            padded = h
        padded_histories.append(padded)
    
    avg_history = np.mean(padded_histories, axis=0).tolist()
    
    return {
        "best_x": best_x,
        "best_f": best_f,
        "avg_f": np.mean(all_f),
        "median_f": np.median(all_f),
        "max_f": np.max(all_f),
        "std_f": np.std(all_f),
        "convergence_history": avg_history
    }