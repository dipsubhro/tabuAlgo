from tabu import tabu_search
from func import sphere, rastrigin, ackley, rosenbrock, step

x0 = [3, -2, 1, 0, 2]
bounds = (-5, 5)
funcs = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Rosenbrock": rosenbrock,
    "Step": step
}

with open("output.txt", "w") as f:
    f.write("Tabu Search Results\n")
    f.write("===================\n\n")
    for name, fn in funcs.items():
        best_x, best_f = tabu_search(fn, x0, tenure=2, max_iter=100, bounds=bounds)
        line = f"{name:<12} → Best x = {best_x},  f(x) = {best_f:.6f}\n"
        f.write(line)

print("✅ Results saved to output.txt")

