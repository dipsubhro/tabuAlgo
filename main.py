"""
Main experiment file - each function has its own parameters.
Runs all functions concurrently for faster execution.
"""

from run_tabu import run_tabu
from func import (
    sphere, sum_of_squares, schwefel_222, step, rosenbrock,
    rastrigin, ackley, griewank, levy, zakharov,
    dixon_price, bent_cigar, high_conditioned_elliptic, alpine, salomon
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
# FIXED: dims = 5 for general functions
# VARIABLE: Only neighbors, tenure, max_iter are tuned per function
NUM_RUNS = 25
STANDARD_DIMS = 5  # Standard dimension for fair comparison

experiments = [
    # Sphere - simple unimodal, converges fast
    ("Sphere", sphere, 15, 5, 1500, (-5.12, 5.12), STANDARD_DIMS),
    
    # Sum_of_Squares - unimodal, similar to sphere
    ("Sum_of_Squares", sum_of_squares, 15, 5, 1000, (-10, 10), STANDARD_DIMS),
    
    # Schwefel 2.22 - unimodal
    ("Schwefel_2.22", schwefel_222, 20, 6, 1500, (-10, 10), STANDARD_DIMS),
    
    # Step - plateaus, converges well
    ("Step", step, 10, 3, 500, (-100, 100), STANDARD_DIMS),
    
    # Rosenbrock - valley-shaped, classic
    ("Rosenbrock", rosenbrock, 25, 7, 2000, (-5, 10), STANDARD_DIMS),
    
    # Rastrigin - highly multimodal
    ("Rastrigin", rastrigin, 30, 8, 2500, (-5.12, 5.12), STANDARD_DIMS),
    
    # Ackley - multimodal with many local minima
    ("Ackley", ackley, 25, 7, 2000, (-32, 32), STANDARD_DIMS),
    
    # Griewank - multimodal
    ("Griewank", griewank, 25, 6, 2000, (-600, 600), STANDARD_DIMS),
    
    # Lévy - CEC competition function
    ("Levy", levy, 20, 6, 1500, (-10, 10), STANDARD_DIMS),
    
    # Zakharov - unimodal, bowl-shaped
    ("Zakharov", zakharov, 20, 6, 1500, (-5, 10), STANDARD_DIMS),
    
    # Dixon-Price - non-separable
    ("Dixon_Price", dixon_price, 20, 7, 2000, (-10, 10), STANDARD_DIMS),
    
    # Bent_Cigar - very ill-conditioned
    ("Bent_Cigar", bent_cigar, 30, 5, 2500, (-100, 100), STANDARD_DIMS),
    
    # High-Conditioned Elliptic - ill-conditioned
    ("High_Conditioned_Elliptic", high_conditioned_elliptic, 30, 6, 2500, (-100, 100), STANDARD_DIMS),
    
    # Alpine - multimodal
    ("Alpine", alpine, 20, 5, 1500, (-10, 10), STANDARD_DIMS),
    
    # Salomon - multimodal
    ("Salomon", salomon, 25, 6, 2000, (-100, 100), STANDARD_DIMS),
]


if __name__ == "__main__":
    # Run all experiments concurrently
    with ProcessPoolExecutor() as executor:
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
