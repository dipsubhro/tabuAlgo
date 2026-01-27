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
from visualize import visualize_results


def run_experiment(args):
    """Run a single experiment (called by each worker)."""
    name, fn, num_runs, neighbors, tenure, max_iter, bounds, dims = args
    print(f"Running {name}...")
    result = run_tabu(fn, num_runs, neighbors, tenure, max_iter, bounds, dims)
    return (name, result, num_runs, neighbors, tenure, max_iter, bounds, dims)


# Define all experiments: (name, fn, num_runs, neighbors, tenure, max_iter, bounds, dims)
# Parameters tuned for each function's characteristics
experiments = [
    # Sphere - simple unimodal, converges fast
    ("Sphere", sphere, 25, 15, 5, 1500, (-5, 5), 5),
    
    # Sum_of_Squares - unimodal, similar to sphere
    ("Sum_of_Squares", sum_of_squares, 25, 15, 5, 1000, (-10, 10), 5),
    
    # Zakharov - unimodal but harder, needs more exploration
    ("Zakharov", zakharov, 25, 20, 6, 1500, (-5, 10), 5),
    
    # Matyas - 2D function, very easy
    ("Matyas", matyas, 20, 10, 4, 800, (-10, 10), 2),
    
    # Booth - 2D function, optimal at (1,3)
    ("Booth", booth, 20, 10, 4, 800, (-10, 10), 2),
    
    # Step - plateaus, already converges well
    ("Step", step, 20, 10, 3, 500, (-5, 5), 5),
    
    # Dixon_Price - harder, needs more iterations and exploration
    ("Dixon_Price", dixon_price, 25, 20, 7, 2000, (-10, 10), 5),
    
    # Powell - needs 4n dims, more neighbors for complex landscape
    ("Powell", powell, 25, 25, 8, 2000, (-4, 5), 8),
    
    # Bent_Cigar - very ill-conditioned, smaller bounds help
    ("Bent_Cigar", bent_cigar, 25, 30, 5, 2500, (-10, 10), 5),
    
    # Quartic - already performs well, small bounds
    ("Quartic", quartic, 20, 15, 4, 800, (-1.28, 1.28), 5),
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
            best_x_str
        ])
    
    headers = ["Function", "Runs", "Neighbors", "Tenure", "MaxIter", "Bounds", "Dims", 
               "Best f", "Avg f", "Median f", "Max f", "Best x"]
    
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
