from tabu import tabu_search
from func import sphere, rastrigin, ackley, rosenbrock, step
import numpy as np
from tabulate import tabulate

bounds = (-2, 2)
NUM_RUNS = 100

funcs = {
    "Sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "step": step
}

results_data = []
for name, fn in funcs.items():
    print(f"Running experiment for {name}...")
    best_f_overall = float('inf')
    best_x_overall = None
    all_best_fs = []

    for _ in range(NUM_RUNS):
        x0 = np.random.uniform(bounds[0], bounds[1], size=5)
        best_x, best_f, _, _, _ = tabu_search(fn, x0, tenure=2, max_iter=100, bounds=bounds)
        all_best_fs.append(best_f)
        if best_f < best_f_overall:
            best_f_overall = best_f
            best_x_overall = best_x

    count_best = sum(1 for f in all_best_fs if np.isclose(f, best_f_overall))
    
    global_min_chance = (count_best / NUM_RUNS) * 100

    avg_best_f = np.mean(all_best_fs)
    median_best_f = np.median(all_best_fs)
    max_best_f = np.max(all_best_fs)

    results_data.append([
        name, 
        str(best_x_overall), 
        f"{best_f_overall:.6f}", 
        f"{avg_best_f:.6f}", 
        f"{median_best_f:.6f}", 
        f"{max_best_f:.6f}",
        f"{global_min_chance:.2f}%"
    ])

with open("output.txt", "w") as f:
    f.write("Tabu Search Experiment Results\n")
    f.write("==============================\n\n")
    f.write(f"Number of runs per function: {NUM_RUNS}\n\n")
    
    headers = ["Function", "Best x", "Best f(x)", "Avg Best f(x)", "Median Best f(x)", "Max Best f(x)", "Global Min Chance (%)"]
    table = tabulate(results_data, headers=headers, tablefmt="grid")
    
    f.write(table)

print("✅ Results saved to output.txt")