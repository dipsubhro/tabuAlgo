import numpy as np
from tabulate import tabulate

from portfolio import cvar_portfolio_objective
from run_tabu import run_tabu


def build_sample_returns_matrix(num_periods=252):
    rng = np.random.default_rng(42)
    mean_returns = np.array([0.0008, 0.0006, 0.0005, 0.0009, 0.0004], dtype=float)
    volatilities = np.array([0.012, 0.010, 0.008, 0.015, 0.007], dtype=float)
    return rng.normal(loc=mean_returns, scale=volatilities, size=(num_periods, len(mean_returns)))


def run_portfolio_experiment(args):
    name, returns_matrix, lambda_, alpha, neighbors, tenure, max_iter = args
    dims = returns_matrix.shape[1]
    objective = lambda w: cvar_portfolio_objective(w, returns_matrix, lambda_, alpha)

    result = run_tabu(
        objective,
        num_runs=NUM_RUNS,
        neighbors=neighbors,
        tenure=tenure,
        max_iter=max_iter,
        bounds=(0.0, 1.0),
        dims=dims,
    )

    return (name, result, lambda_, alpha, neighbors, tenure, max_iter, dims)


NUM_RUNS = 10
RETURNS_MATRIX = build_sample_returns_matrix()

experiments = [
    ("CVaR_Portfolio", RETURNS_MATRIX, 2.0, 0.05, 20, 8, 500),
]


if __name__ == "__main__":
    results = []
    for experiment in experiments:
        print(f"Running {experiment[0]}...")
        results.append(run_portfolio_experiment(experiment))

    sorted_results = sorted(results, key=lambda x: x[1]["best_f"])

    table_rows = []
    for name, result, lambda_, alpha, neighbors, tenure, max_iter, dims in sorted_results:
        best_x_str = "[" + ", ".join(f"{x:.4f}" for x in result["best_x"]) + "]"
        table_rows.append([
            name,
            NUM_RUNS,
            lambda_,
            alpha,
            neighbors,
            tenure,
            max_iter,
            dims,
            f"{result['best_f']:.6f}",
            f"{result['avg_f']:.6f}",
            f"{result['median_f']:.6f}",
            f"{result['max_f']:.6f}",
            f"{result['std_f']:.6f}",
            best_x_str,
        ])

    headers = [
        "Experiment",
        "Runs",
        "Lambda",
        "Alpha",
        "Neighbors",
        "Tenure",
        "MaxIter",
        "Assets",
        "Best f",
        "Avg f",
        "Median f",
        "Max f",
        "Std f",
        "Best w",
    ]

    table = tabulate(table_rows, headers=headers, tablefmt="grid")

    with open("portfolio_output.txt", "w") as f:
        f.write("CVaR Portfolio Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(table)
        f.write("\n")

    print("Done! Results saved to portfolio_output.txt")
    print(table)
