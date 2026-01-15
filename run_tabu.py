from tabu import tabu_search
from func import sphere, rastrigin, ackley, rosenbrock, step
import numpy as np
from tabulate import tabulate
from concurrent.futures import ProcessPoolExecutor

bounds = (-5, 5)
NUM_RUNS = 100

funcs = {
    "Sphere": sphere,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "step": step
}

def run_experiment(args):
    """
    Runs the full experiment (NUM_RUNS iterations) for a specific function.
    args: tuple containing (name, fn, bounds, num_runs)
    """
    name, fn, bounds, num_runs = args
    
    # Re-seed to ensure independence for this process
    np.random.seed()
    
    print(f"Starting experiment for {name}...")
    
    best_f_overall = float('inf')
    best_x_overall = None
    all_best_fs = []

    # Run the tabu search num_runs times sequentially within this process
    for _ in range(num_runs):
        x0 = np.random.uniform(bounds[0], bounds[1], size=5)
        best_x, best_f, _, _, _ = tabu_search(fn, x0, tenure=5, max_iter=1000, bounds=bounds)
        all_best_fs.append(best_f)
        if best_f < best_f_overall:
            best_f_overall = best_f
            best_x_overall = best_x

    count_best = sum(1 for f in all_best_fs if np.isclose(f, best_f_overall))
    global_min_rate = count_best / num_runs

    avg_best_f = np.mean(all_best_fs)
    median_best_f = np.median(all_best_fs)
    max_best_f = np.max(all_best_fs)

    # Format the result row here
    row = [
        name, 
        "[" + ",\n".join(str(x) for x in best_x_overall) + "]", 
        str(best_f_overall), 
        str(avg_best_f), 
        str(median_best_f), 
        str(max_best_f),
        f"{global_min_rate:.2f}"
    ]
    
    print(f"Finished experiment for {name}.")
    return row

if __name__ == "__main__":
    # Prepare arguments for each function
    experiment_args = [(name, fn, bounds, NUM_RUNS) for name, fn in funcs.items()]
    
    # Run experiments in parallel
    # We use a max_workers equal to len(funcs) or let it default (usually num_cpus)
    # Since we have 5 functions, we can process them all at once if we have 5+ cores.
    with ProcessPoolExecutor() as executor:
        results_data = list(executor.map(run_experiment, experiment_args))

    with open("output.txt", "w") as f:
        f.write("Tabu Search Experiment Results\n")
        f.write("==============================\n\n")
        f.write(f"Number of runs per function: {NUM_RUNS}\n\n")
        
        headers = ["Function", "Best x", "Best f(x)", "Avg Best f(x)", "Median Best f(x)", "Max Best f(x)", "Global Min Rate"]
        table = tabulate(results_data, headers=headers, tablefmt="grid")
        
        f.write(table)

    print("Results saved to output.txt")