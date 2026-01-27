"""
Main experiment file - each function has its own parameters.
"""

from run_tabu import run_tabu
from func import *


def write_result(f, name, result, num_runs, neighbors, tenure, max_iter, bounds, dims):
    """Write result to file."""
    f.write(f"{name} (runs={num_runs}, neighbors={neighbors}, tenure={tenure}, ")
    f.write(f"max_iter={max_iter}, bounds={bounds}, dims={dims})\n")
    f.write(f"  Best x:   {result['best_x']}\n")
    f.write(f"  Best f:   {result['best_f']:.6e}\n")
    f.write(f"  Avg f:    {result['avg_f']:.6e}\n")
    f.write(f"  Median f: {result['median_f']:.6e}\n")
    f.write(f"  Max f:    {result['max_f']:.6e}\n\n")


with open("output.txt", "w") as f:
    f.write("Tabu Search Results\n")
    f.write("=" * 60 + "\n\n")
    
    # Sphere
    print("Running Sphere...")
    result = run_tabu(sphere, num_runs=20, neighbors=10, tenure=5, max_iter=1000, bounds=(-5, 5), dims=5)
    write_result(f, "Sphere", result, 20, 10, 5, 1000, (-5, 5), 5)
    
    # Rastrigin
    print("Running Rastrigin...")
    result = run_tabu(rastrigin, num_runs=25, neighbors=15, tenure=7, max_iter=1500, bounds=(-5, 5), dims=5)
    write_result(f, "Rastrigin", result, 25, 15, 7, 1500, (-5, 5), 5)
    
    # Ackley
    print("Running Ackley...")
    result = run_tabu(ackley, num_runs=22, neighbors=20, tenure=5, max_iter=1000, bounds=(-5, 5), dims=5)
    write_result(f, "Ackley", result, 22, 20, 5, 1000, (-5, 5), 5)
    
    # Rosenbrock
    print("Running Rosenbrock...")
    result = run_tabu(rosenbrock, num_runs=24, neighbors=15, tenure=10, max_iter=2000, bounds=(-5, 5), dims=5)
    write_result(f, "Rosenbrock", result, 24, 15, 10, 2000, (-5, 5), 5)
    
    # Step
    print("Running Step...")
    result = run_tabu(step, num_runs=20, neighbors=10, tenure=3, max_iter=500, bounds=(-5, 5), dims=5)
    write_result(f, "Step", result, 20, 10, 3, 500, (-5, 5), 5)
    
    # Schwefel
    print("Running Schwefel...")
    result = run_tabu(schwefel, num_runs=25, neighbors=20, tenure=8, max_iter=1500, bounds=(-500, 500), dims=5)
    write_result(f, "Schwefel", result, 25, 20, 8, 1500, (-500, 500), 5)
    
    # Griewank
    print("Running Griewank...")
    result = run_tabu(griewank, num_runs=23, neighbors=15, tenure=5, max_iter=1000, bounds=(-600, 600), dims=5)
    write_result(f, "Griewank", result, 23, 15, 5, 1000, (-600, 600), 5)
    
    # Levy
    print("Running Levy...")
    result = run_tabu(levy, num_runs=21, neighbors=12, tenure=6, max_iter=1000, bounds=(-10, 10), dims=5)
    write_result(f, "Levy", result, 21, 12, 6, 1000, (-10, 10), 5)
    
    # Zakharov
    print("Running Zakharov...")
    result = run_tabu(zakharov, num_runs=22, neighbors=10, tenure=5, max_iter=800, bounds=(-5, 10), dims=5)
    write_result(f, "Zakharov", result, 22, 10, 5, 800, (-5, 10), 5)
    
    # Bohachevsky
    print("Running Bohachevsky...")
    result = run_tabu(bohachevsky, num_runs=24, neighbors=15, tenure=5, max_iter=1000, bounds=(-100, 100), dims=5)
    write_result(f, "Bohachevsky", result, 24, 15, 5, 1000, (-100, 100), 5)
    
    # Schaffer_N2
    print("Running Schaffer_N2...")
    result = run_tabu(schaffer_n2, num_runs=25, neighbors=20, tenure=7, max_iter=1200, bounds=(-100, 100), dims=5)
    write_result(f, "Schaffer_N2", result, 25, 20, 7, 1200, (-100, 100), 5)
    
    # Matyas
    print("Running Matyas...")
    result = run_tabu(matyas, num_runs=20, neighbors=10, tenure=4, max_iter=500, bounds=(-10, 10), dims=5)
    write_result(f, "Matyas", result, 20, 10, 4, 500, (-10, 10), 5)
    
    # Sum_of_Squares
    print("Running Sum_of_Squares...")
    result = run_tabu(sum_of_squares, num_runs=21, neighbors=10, tenure=5, max_iter=600, bounds=(-10, 10), dims=5)
    write_result(f, "Sum_of_Squares", result, 21, 10, 5, 600, (-10, 10), 5)
    
    # Trid
    print("Running Trid...")
    result = run_tabu(trid, num_runs=23, neighbors=15, tenure=6, max_iter=1000, bounds=(-25, 25), dims=5)
    write_result(f, "Trid", result, 23, 15, 6, 1000, (-25, 25), 5)
    
    # Booth
    print("Running Booth...")
    result = run_tabu(booth, num_runs=20, neighbors=10, tenure=4, max_iter=500, bounds=(-10, 10), dims=5)
    write_result(f, "Booth", result, 20, 10, 4, 500, (-10, 10), 5)

print("Done! Results saved to output.txt")
