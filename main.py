"""
Main experiment file - each function has its own parameters.
Runs all functions concurrently for faster execution.
"""

from run_tabu import run_tabu
from func import (
    sphere, sum_of_squares, zakharov, matyas, booth,
    step, dixon_price, powell, bent_cigar, quartic
)
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate
from visualize import visualize_results, create_unigraph


def run_experiment(args):
    """Run a single experiment (called by each worker)."""
    name, fn, neighbors, tenure, max_iter, bounds, dims = args
    print(f"Running {name}...")
    result = run_tabu(fn, NUM_RUNS, neighbors, tenure, max_iter, bounds, dims)
    return (name, result, NUM_RUNS, neighbors, tenure, max_iter, bounds, dims)


# Define all experiments: (name, fn, neighbors, tenure, max_iter, bounds, dims)
# FIXED: num_runs = 25 for all
# FIXED: bounds = function-defined (standard domains)
# FIXED: dims = 5 for general functions, 2 for 2D-only functions, 4 for Powell (min valid)
# VARIABLE: Only neighbors, tenure, max_iter are tuned per function
NUM_RUNS = 25
STANDARD_DIMS = 5  # Standard dimension for fair comparison

experiments = [
    # Sphere - simple unimodal, converges fast
    ("Sphere", sphere, 15, 5, 1500, (-5.12, 5.12), STANDARD_DIMS),
    
    # Sum_of_Squares - unimodal, similar to sphere
    ("Sum_of_Squares", sum_of_squares, 15, 5, 1000, (-10, 10), STANDARD_DIMS),
    
    # Zakharov - unimodal but harder, needs more exploration
    ("Zakharov", zakharov, 20, 6, 1500, (-5, 10), STANDARD_DIMS),
    
    # Matyas - INHERENTLY 2D function (only defined for 2 vars)
    ("Matyas", matyas, 10, 4, 800, (-10, 10), 2),
    
    # Booth - INHERENTLY 2D function (only defined for 2 vars)
    ("Booth", booth, 10, 4, 800, (-10, 10), 2),
    
    # Step - plateaus, converges well
    ("Step", step, 10, 3, 500, (-5.12, 5.12), STANDARD_DIMS),
    
    # Dixon_Price - harder, needs more iterations
    ("Dixon_Price", dixon_price, 20, 7, 2000, (-10, 10), STANDARD_DIMS),
    
    # Powell - requires dims to be multiple of 4 (using minimum: 4)
    ("Powell", powell, 25, 8, 2000, (-4, 5), 4),
    
    # Bent_Cigar - very ill-conditioned
    ("Bent_Cigar", bent_cigar, 30, 5, 2500, (-100, 100), STANDARD_DIMS),
    
    # Quartic - performs well, small bounds
    ("Quartic", quartic, 15, 4, 800, (-1.28, 1.28), STANDARD_DIMS),
]


if __name__ == "__main__":
    # Run all experiments concurrently
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_experiment, experiments))
    
    # Build table rows
    table_rows = []
    for name, result, num_runs, neighbors, tenure, max_iter, bounds, dims in results:
        best_x_str = "[" + ", ".join(f"{x:.4f}" for x in result['best_x']) + "]"
        table_rows.append([
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
    
    headers = ["Function", "Runs", "Neighbors", "Tenure", "MaxIter", "Bounds", "Dims", 
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
