"""
Main experiment file - each function has its own parameters.
Runs all functions concurrently for faster execution.
"""

from run_tabu import run_tabu
from func import (
    sphere, sum_of_squares, sum_of_different_powers, step, brown,
    zakharov, dixon_price, schumer_steiglitz, csendes, sixth_power,
    powell, quartic, rotated_hyper_ellipsoid, discus, exponential
)
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate
from visualize import visualize_results, create_unigraph, create_convergence_curves


def run_experiment(args):
    """Run a single experiment (called by each worker)."""
    name, fn, neighbors, tenure, max_iter, bounds, dims = args
    print(f"Running {name}...")
    result = run_tabu(fn, NUM_RUNS, neighbors, tenure, max_iter, bounds, dims)
    return (name, result, NUM_RUNS, neighbors, tenure, max_iter, bounds, dims)


# Define all experiments: (name, fn, neighbors, tenure, max_iter, bounds, dims)
# FIXED: num_runs = 25 for all
# FIXED: bounds = function-defined (standard domains)
# FIXED: dims = 5 for general functions
# VARIABLE: Only neighbors, tenure, max_iter are tuned per function
NUM_RUNS = 25
STANDARD_DIMS = 5  # Standard dimension for fair comparison

experiments = [
    # Sphere - simple unimodal (increased iters for fine tuning)
    ("Sphere", sphere, 40, 8, 2000, (-5.12, 5.12), STANDARD_DIMS),
    
    # Sum_of_Squares - unimodal (wider search space needs more exploration)
    ("Sum_of_Squares", sum_of_squares, 40, 8, 2000, (-10, 10), STANDARD_DIMS),
    
    # Sum_of_Different_Powers - tight bounds, needs fine control
    ("Sum_of_Different_Powers", sum_of_different_powers, 40, 8, 2000, (-1, 1), STANDARD_DIMS),
    
    # Step - discontinuous (more neighbors for plateau navigation)
    ("Step", step, 60, 12, 2000, (-5.12, 5.12), STANDARD_DIMS),
    
    # Brown - smooth unimodal
    ("Brown", brown, 40, 8, 2000, (-1, 4), STANDARD_DIMS),
    
    # Zakharov - unimodal, bowl-shaped (harder function)
    ("Zakharov", zakharov, 50, 10, 2500, (-5, 10), STANDARD_DIMS),
    
    # Dixon-Price - non-separable (challenging, needs more resources)
    ("Dixon_Price", dixon_price, 60, 12, 3000, (-10, 10), STANDARD_DIMS),
    
    # Schumer_Steiglitz - sum of fourth powers
    ("Schumer_Steiglitz", schumer_steiglitz, 40, 8, 2000, (-10, 10), STANDARD_DIMS),
    
    # Csendes - smooth function (tight bounds)
    ("Csendes", csendes, 40, 8, 2000, (-1, 1), STANDARD_DIMS),
    
    # Sixth Power - smooth like Csendes
    ("Sixth_Power", sixth_power, 40, 8, 2000, (-1, 1), STANDARD_DIMS),
    
    # Powell - non-separable (dims=4, complex landscape)
    ("Powell", powell, 50, 10, 2500, (-4, 5), 4),
    
    # Quartic - smooth unimodal
    ("Quartic", quartic, 35, 7, 1500, (-1.28, 1.28), STANDARD_DIMS),
    
    # Rotated Hyper-Ellipsoid - non-separable (tighter bounds for better focus)
    ("Rotated_Hyper_Ellipsoid", rotated_hyper_ellipsoid, 50, 10, 2500, (-30, 30), STANDARD_DIMS),
    
    # Discus - ill-conditioned (needs more exploration)
    ("Discus", discus, 60, 12, 3000, (-10, 10), STANDARD_DIMS),
    
    # Exponential - smooth unimodal
    ("Exponential", exponential, 35, 7, 1500, (-1, 1), STANDARD_DIMS),
]


if __name__ == "__main__":
    # Run all experiments concurrently (leave 2 cores free)
    import os
    max_workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_experiment, experiments))
    
    # Sort results by best_f to assign ranks
    sorted_results = sorted(results, key=lambda x: x[1]['best_f'])
    rank_map = {name: rank + 1 for rank, (name, *_) in enumerate(sorted_results)}
    
    # Build table rows (sorted by rank)
    table_rows = []
    for name, result, num_runs, neighbors, tenure, max_iter, bounds, dims in sorted_results:
        best_x_str = "[" + ", ".join(f"{x:.4f}" for x in result['best_x']) + "]"
        table_rows.append([
            rank_map[name],
            name,
            num_runs,
            neighbors,
            tenure,
            max_iter,
            f"{bounds}",
            dims,
            f"{result['best_f']:.4e}",
            f"{result['avg_f']:.4e}",
            f"{result['median_f']:.4e}",
            f"{result['max_f']:.4e}",
            f"{result['std_f']:.4e}",
            best_x_str
        ])
    
    headers = ["Rank", "Function", "Runs", "Neighbors", "Tenure", "MaxIter", "Bounds", "Dims", 
               "Best f", "Avg f", "Median f", "Max f", "Std f", "Best x"]
    
    table = tabulate(table_rows, headers=headers, tablefmt="grid")
    
    # Write to file
    with open("output.txt", "w") as f:
        f.write("Tabu Search Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(table)
        f.write("\n")
    
    print("Done! Results saved to output.txt")
    
    # Generate visualization
    visualize_results(results)
    
    # Generate unified comparison graph
    create_unigraph(results)
    
    # Generate convergence curves like the PSO reference
    create_convergence_curves(results)
