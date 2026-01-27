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
experiments = [
    ("Sphere", sphere, 20, 10, 5, 1000, (-5, 5), 5),
    ("Sum_of_Squares", sum_of_squares, 21, 10, 5, 600, (-10, 10), 5),
    ("Zakharov", zakharov, 22, 10, 5, 800, (-5, 10), 5),
    ("Matyas", matyas, 20, 10, 4, 500, (-10, 10), 5),
    ("Booth", booth, 20, 10, 4, 500, (-10, 10), 5),
    ("Step", step, 20, 10, 3, 500, (-5, 5), 5),
    ("Dixon_Price", dixon_price, 23, 12, 5, 1000, (-10, 10), 5),
    ("Powell", powell, 24, 15, 6, 1200, (-4, 5), 8),
    ("Bent_Cigar", bent_cigar, 22, 15, 5, 1000, (-100, 100), 5),
    ("Quartic", quartic, 20, 10, 4, 500, (-1.28, 1.28), 5),
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
