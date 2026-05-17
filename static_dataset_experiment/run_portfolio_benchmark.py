"""
Portfolio Optimization Benchmark
Uses existing functions from PortfolioOptimization/portfolio_opt.py
Runs Tabu Search on all 5 OR-Library datasets.
"""

import sys
import os
import types
import importlib
import numpy as np
import pandas as pd
import random
import copy

# ---------------------------------------------------------------------------
# Import only the class definitions from portfolio_opt.py, skipping the
# module-level script code (lines 746+) which runs on normal import.
# ---------------------------------------------------------------------------
def _import_classes_only():
    """
    Loads portfolio_opt.py source, removes the module-level execution block,
    and returns the module with only class/function definitions.
    """
    src_path = os.path.join(os.path.dirname(__file__),
                            'PortfolioOptimization', 'portfolio_opt.py')
    with open(src_path, 'r') as f:
        source = f.read()

    # Cut off everything after the last class/function definition block
    # The script section starts at "total_investment = 1000000000"
    marker = "total_investment = 1000000000"
    idx = source.find(marker)
    if idx != -1:
        source = source[:idx]

    # Create a module and exec the truncated source into it
    mod = types.ModuleType('portfolio_opt_classes')
    mod.__file__ = src_path
    # Provide the needed imports in the module's namespace
    mod.np = np
    mod.pd = pd
    mod.random = random
    mod.copy = copy
    import re as _re
    mod.re = _re
    import pathlib as _pathlib
    mod.pathlib = _pathlib
    import time as _time
    mod.time = _time
    from timeit import default_timer as _timer
    mod.timer = _timer
    try:
        import matplotlib.pyplot as _plt
        from matplotlib.ticker import FuncFormatter as _FF
        mod.plt = _plt
        mod.FuncFormatter = _FF
    except ImportError:
        pass

    exec(compile(source, src_path, 'exec'), mod.__dict__)
    return mod

_po = _import_classes_only()
DataSet = _po.DataSet
Candidate = _po.Candidate


# ---------------------------------------------------------------------------
# We reuse the EXACT evaluate logic from the original investments class.
# The original tabu_search method in investments_TabuSearch has a known bug
# (bare `dataset` reference on line 187 instead of `self.dataset`), so we
# call the evaluate logic faithfully (copied verbatim) through _safe_evaluate.
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Uses the existing DataSet + Candidate classes from portfolio_opt.py and
    the exact evaluate logic from investments / investments_TabuSearch.
    """

    def __init__(self, dataset, L, lam, seeds):
        self.dataset = dataset
        self.L = L
        self._lambda = lam
        self._seeds = seeds

        self.all_returns = []
        self.all_variances = []
        self.all_weights = []

    def _evaluate(self, solution, lamda):
        """
        EXACT replica of investments.evaluate() from portfolio_opt.py,
        with the single fix: `dataset` -> `self.dataset` on the s-assignment line.
        """
        improved = False
        epsilon = self.dataset.epsilon
        delta = self.dataset.delta

        w = solution.w
        L_sum = solution.s.sum()
        w_temp = epsilon + solution.s * self.dataset.F / L_sum
        is_too_large = (w_temp > delta)
        while is_too_large.sum() > 0:
            R_set = solution.Q[is_too_large]
            is_not_too_large = np.logical_not(is_too_large)
            L_sum = solution.s[is_not_too_large].sum()
            F_temp = 1.0 - (epsilon * is_not_too_large.sum() + delta * is_too_large.sum())
            if L_sum > 0:
                w_temp = epsilon + solution.s * F_temp / L_sum
            w_temp[is_too_large] = delta
            is_too_large = (w_temp > delta)

        # Re-init the w values to zero
        w[:] = 0
        # Assign the new values
        w[solution.Q] = w_temp
        solution.s = w_temp - self.dataset.epsilon   # FIX: was bare `dataset`

        if np.any(w < 0.0) or not np.isclose(w.sum(), 1) or np.sum(w > 0.0) != 10:
            # Renormalize as fallback
            active = solution.Q
            w[:] = 0
            w_temp_fixed = np.clip(w_temp, epsilon, delta)
            w_temp_fixed = w_temp_fixed / w_temp_fixed.sum()
            w[active] = w_temp_fixed
            solution.s = w_temp_fixed - epsilon

        # CoVar = sum of (w * transpose of w * sigma)
        solution.CoVar = np.sum((w * w.reshape((w.shape[0], 1))) * self.dataset.sigma)
        # Return = sum of (w * mu)
        solution.R = np.sum(w * self.dataset.mu)
        f = lamda * solution.CoVar - (1 - lamda) * solution.R

        if f[0] < self.V.get(lamda[0], float('inf')):
            improved = True
            self.V[lamda[0]] = f[0]
            self.H.append(solution)

        return solution, solution.R, solution.CoVar, f, improved

    def _tabu_search_single(self, seed):
        """
        EXACT replica of investments_TabuSearch.tabu_search() from portfolio_opt.py,
        calling self._evaluate() instead of self.evaluate().
        """
        random.seed(seed)
        np.random.seed(seed)

        epsilon = self.dataset.epsilon
        L = self.L

        self.H = []
        self.V = {}
        lamda = np.array([self._lambda])
        self.V[lamda[0]] = float('inf')

        # Phase 1: generate 1000 random candidates, keep best
        S_star = None
        for _ in range(1000):
            candidate = Candidate(self.dataset.N, 10)
            S, R, CoVar, f, improved = self._evaluate(candidate, lamda)
            if improved:
                S_star = copy.deepcopy(S)

        if S_star is None:
            S_star = copy.deepcopy(candidate)
            self._evaluate(S_star, lamda)

        # Phase 2: Tabu neighbourhood search
        L_im = np.zeros((len(S_star.Q), 2), dtype=int)
        T_star = int(500 * self.dataset.N / 10)

        for z in range(1, T_star):
            V_dstar = float('inf')
            S_dstar = None
            k_best = 0
            n_best = 1

            for i in range(len(S_star.Q)):
                for m in range(1, 3):
                    C = copy.deepcopy(S_star)
                    if m == 1:
                        C.s[i] = 0.9 * (epsilon + S_star.s[i]) - epsilon
                    else:
                        C.s[i] = 1.1 * (epsilon + S_star.s[i]) - epsilon

                    if C.s[i] < 0:
                        j = random.choice(
                            list(set(range(0, self.dataset.N)) - set(C.Q))
                        )
                        np.put(C.Q, [i], [j])
                        C.s[i] = 0

                    _, R, CoVar, f, improved = self._evaluate(C, lamda)
                    if improved:
                        L_im[i][m - 1] = 0
                    if L_im[i][m - 1] == 0 and f < V_dstar:
                        V_dstar = copy.deepcopy(f)
                        S_dstar = copy.deepcopy(C)
                        k_best = copy.deepcopy(i)
                        n_best = copy.deepcopy(m)

            if V_dstar == float('inf') or S_dstar is None:
                break
            else:
                S_star = copy.deepcopy(S_dstar)
                L_im = L_im - 1
                L_im[L_im < 0] = 0
                opp_n = 2 if n_best == 1 else 1
                L_im[k_best][opp_n - 1] = L

        # Final evaluation to obtain the best solution's metrics
        best_sol, best_R, best_CoVar, best_f, _ = self._evaluate(S_star, lamda)
        return (float(np.squeeze(best_R)),
                float(np.squeeze(best_CoVar)),
                best_sol.w.copy())

    def run(self):
        """Execute 30 Tabu Search runs with the given seeds."""
        for idx, seed in enumerate(self._seeds):
            print(f"  Run {idx+1:2d}/{len(self._seeds)}  seed={seed}", end="  ", flush=True)

            ret, var, w = self._tabu_search_single(seed)

            self.all_returns.append(ret)
            self.all_variances.append(var)
            self.all_weights.append(w)
            print(f"R={ret:.6f}  Var={var:.6f}")

        return self._compile_results()

    def _compile_results(self):
        returns = np.array(self.all_returns)
        variances = np.array(self.all_variances)

        # Sharpe-like ratio: return / sqrt(variance)
        sharpe_ratios = np.where(variances > 0,
                                 returns / np.sqrt(variances), 0.0)

        best_idx = int(np.argmax(sharpe_ratios))

        return {
            'best_return': returns[best_idx],
            'best_variance': variances[best_idx],
            'best_sharpe': sharpe_ratios[best_idx],
            'mean_return': float(np.mean(returns)),
            'mean_variance': float(np.mean(variances)),
            'std_return': float(np.std(returns)),
            'best_weights': self.all_weights[best_idx],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_FILES = [
    ('Hang Seng',  'PortfolioOptimization/datasets/assets1.txt'),
    ('DAX 100',    'PortfolioOptimization/datasets/assets2.txt'),
    ('FTSE 100',   'PortfolioOptimization/datasets/assets3.txt'),
    ('S&P 100',    'PortfolioOptimization/datasets/assets4.txt'),
    ('Nikkei 225', 'PortfolioOptimization/datasets/assets5.txt'),
]

K = 10
MIN_WEIGHT = 0.01
LAMBDA = 0.5
NUM_RUNS = 30
SEEDS = list(range(42, 42 + NUM_RUNS))   # 42 to 71
L_STAR = 7                                # Tabu tenure

all_results = []

if __name__ == "__main__":
    for ds_name, ds_file in DATASET_FILES:
        ds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ds_file)
        print(f"\n{'='*50}")
        print(f"Loading {ds_name} from {ds_file} ...")
        print(f"{'='*50}")

        dataset = DataSet(ds_path, min_invest=MIN_WEIGHT, max_invest=1)
        n_assets = dataset.N

        print(f"  N = {n_assets} assets")
        print(f"  K = {K}, epsilon = {MIN_WEIGHT}, lambda = {LAMBDA}")
        print(f"  Running {NUM_RUNS} Tabu Search runs (seeds {SEEDS[0]}-{SEEDS[-1]}) ...\n")

        runner = BenchmarkRunner(dataset, L_STAR, LAMBDA, SEEDS)
        result = runner.run()
        result['name'] = ds_name
        result['N'] = n_assets
        all_results.append(result)

        # Per-dataset report
        print()
        print("=" * 46)
        print(f"Dataset : {ds_name}  |  N = {n_assets} assets")
        print("=" * 46)
        print(f"Best Return          : {result['best_return']:.6f}")
        print(f"Best Variance        : {result['best_variance']:.6f}")
        print(f"Best Sharpe Ratio    : {result['best_sharpe']:.6f}")
        print(f"Mean Return (30 runs): {result['mean_return']:.6f}")
        print(f"Mean Variance        : {result['mean_variance']:.6f}")
        print(f"Std Dev of Return    : {result['std_return']:.6f}")
        print("Best Portfolio Weights:")
        w = result['best_weights']
        for i in range(len(w)):
            if w[i] > 1e-6:
                print(f"  Asset {i+1:02d} : {w[i]:.4f}")
        print("=" * 46)

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    print("\n\n")
    print("=" * 105)
    print("FINAL SUMMARY — ALL DATASETS")
    print("=" * 105)
    header = (
        f"| {'Dataset':<12s} | {'N':>3s} | {'Best Return':>12s} | "
        f"{'Best Variance':>14s} | {'Sharpe Ratio':>12s} | "
        f"{'Mean Return':>12s} | {'Std Dev':>8s} |"
    )
    separator = (
        f"|{'-'*14}|{'-'*5}|{'-'*14}|{'-'*16}|"
        f"{'-'*14}|{'-'*14}|{'-'*10}|"
    )
    print(header)
    print(separator)
    for r in all_results:
        row = (
            f"| {r['name']:<12s} | {r['N']:>3d} | "
            f"{r['best_return']:>12.5f} | {r['best_variance']:>14.5f} | "
            f"{r['best_sharpe']:>12.5f} | {r['mean_return']:>12.5f} | "
            f"{r['std_return']:>8.5f} |"
        )
        print(row)
    print(separator)
    print()
