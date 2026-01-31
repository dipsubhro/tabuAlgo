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
    ("Sphere", sphere, 20, 5, 2000, (-5.12, 5.12), STANDARD_DIMS),
    
    # Sum_of_Squares - unimodal, similar to sphere
    ("Sum_of_Squares", sum_of_squares, 20, 5, 1500, (-10, 10), STANDARD_DIMS),
    
    # Schwefel 2.22 - unimodal
    ("Schwefel_2.22", schwefel_222, 25, 7, 2000, (-10, 10), STANDARD_DIMS),
    
    # Step - HARD: discontinuous, needs many neighbors and high tenure
    ("Step", step, 50, 15, 5000, (-5.12, 5.12), STANDARD_DIMS),  # Reduced bounds, more exploration
    
    # Rosenbrock - valley-shaped, classic
    ("Rosenbrock", rosenbrock, 30, 8, 3000, (-5, 10), STANDARD_DIMS),
    
    # Rastrigin - HARD: highly multimodal, needs aggressive exploration
    ("Rastrigin", rastrigin, 50, 12, 5000, (-5.12, 5.12), STANDARD_DIMS),
    
    # Ackley - HARD: many local minima, reduced bounds help
    ("Ackley", ackley, 50, 10, 5000, (-5, 5), STANDARD_DIMS),  # Reduced bounds from (-32, 32)
    
    # Griewank - HARD: multimodal, reduced bounds significantly
    ("Griewank", griewank, 50, 12, 5000, (-100, 100), STANDARD_DIMS),  # Reduced from (-600, 600)
    
    # Lévy - HARD: deceptive, needs more exploration
    ("Levy", levy, 40, 10, 4000, (-10, 10), STANDARD_DIMS),
    
    # Zakharov - unimodal, bowl-shaped
    ("Zakharov", zakharov, 25, 7, 2000, (-5, 10), STANDARD_DIMS),
    
    # Dixon-Price - non-separable
    ("Dixon_Price", dixon_price, 30, 8, 3000, (-10, 10), STANDARD_DIMS),
    
    # Bent_Cigar - very ill-conditioned
    ("Bent_Cigar", bent_cigar, 40, 8, 4000, (-100, 100), STANDARD_DIMS),
    
    # High-Conditioned Elliptic - ill-conditioned
    ("High_Conditioned_Elliptic", high_conditioned_elliptic, 40, 8, 4000, (-100, 100), STANDARD_DIMS),
    
    # Alpine - multimodal
    ("Alpine", alpine, 30, 8, 3000, (-10, 10), STANDARD_DIMS),
    
    # Salomon - HARD: circular ridges, reduced bounds
    ("Salomon", salomon, 50, 12, 5000, (-50, 50), STANDARD_DIMS),  # Reduced from (-100, 100)
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
