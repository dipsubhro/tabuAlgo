from tabu import tabu_search
# from func import sphere, rastrigin, ackley, rosenbrock, step
from func import sphere, rastrigin, ackley, rosenbrock, step
import numpy as np
from tabulate import tabulate

bounds = (-2, 2)
x0 = np.random.uniform(bounds[0], bounds[1], size=5)

funcs = {
    "Sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "step": step
}

results_data = []
for name, fn in funcs.items():
    best_x, best_f, avg_f, median_f, max_f = tabu_search(fn, x0, tenure=2, max_iter=1000, bounds=bounds)
    results_data.append([name, str(best_x), f"{best_f:.6f}", f"{avg_f:.6f}", f"{median_f:.6f}", f"{max_f:.6f}"])

with open("output.txt", "w") as f:
    f.write("Tabu Search Results\n")
    f.write("===================\n\n")
    f.write(f"Initial solution (x0): {x0}\n\n")
    
    headers = ["Function", "Best x", "Best f(x)", "Avg f(x)", "Median f(x)", "Max f(x)"]
    table = tabulate(results_data, headers=headers, tablefmt="grid")
    
    f.write(table)

print("✅ Results saved to output.txt")

