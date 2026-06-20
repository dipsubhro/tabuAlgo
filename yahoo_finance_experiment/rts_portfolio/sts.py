"""
Standard Tabu Search (STS) for Portfolio Optimization
=====================================================

A textbook baseline Tabu Search with NO enhancements:
  - Gaussian perturbation only (no Lévy flights — no long jumps)
  - Fixed tabu tenure (no reactive/dynamic adjustment)
  - No strategic oscillation (no constraint boundary exploration)
  - Strict tabu — NO aspiration criteria at all
  - No Pareto repository
  - Completely random restart on stagnation (no intelligent guidance)
  - Fixed step size (no adaptive decay)

This represents a truly "vanilla" Tabu Search to highlight the
improvements from the Optimized Reactive Tabu Search (Modules A–D).

Author:  Built as a baseline comparison engine
Target:  Maximise Sharpe Ratio on S&P 500 (Jan 2013 – Jan 2023)
"""

import numpy as np
from collections import deque
from .fitness import (
    repair_weights, calc_annual_return, calc_annual_risk,
    sharpe_ratio, repair_weights_2d, sharpe_ratio_2d,
)

class StandardTabuSearch:
    # ── Initialization ──
    def __init__(self, returns_data, cov_matrix, n_assets, rf=0.02, max_iter=5000,
                 neighbors_size=50, tenure=10, step_scale=0.10, seed=None):
        self.returns_data, self.cov_matrix, self.n_assets, self.rf = returns_data, cov_matrix, n_assets, rf
        self.max_iter, self.neighbors_size, self.tenure, self.step_scale = max_iter, neighbors_size, tenure, step_scale
        self.seed = seed

    def _sharpe(self, weights):
        return sharpe_ratio(weights, self.returns_data, self.cov_matrix, self.rf)

    @staticmethod
    def _hash(weights, precision=3):
        return hash(tuple(np.round(weights, precision)))

    # ── Neighbor Generation (Gaussian Only) ──
    def _generate_neighbors_2d(self, current):
        noise = np.random.normal(0, self.step_scale, (self.neighbors_size, self.n_assets))
        return repair_weights_2d(current + noise)

    # ── Main Search Loop ──
    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        # Initialize tracking and tabu list
        current = repair_weights(np.random.uniform(0, 1, self.n_assets))
        current_sharpe = self._sharpe(current)
        best_weights, best_sharpe = current.copy(), current_sharpe
        tabu_list = deque(maxlen=self.tenure)

        convergence_log, all_explored = [], []
        iters_without_improvement, diversification_count = 0, 0
        stagnation_threshold, max_diversifications = max(200, self.max_iter // 8), 5

        for iteration in range(self.max_iter):
            # Generate and evaluate candidates
            candidates_2d = self._generate_neighbors_2d(current)
            cand_sharpes, cand_rets, cand_risks = sharpe_ratio_2d(
                candidates_2d, self.returns_data, self.cov_matrix, self.rf
            )

            neighbors = []
            for i in range(self.neighbors_size):
                cand_w, cand_s = candidates_2d[i], cand_sharpes[i]
                cand_h = self._hash(cand_w)
                all_explored.append((cand_risks[i], cand_rets[i], cand_s))
                neighbors.append((cand_w, cand_s, cand_h))

            np.random.shuffle(neighbors)
            neighbors.sort(key=lambda t: t[1], reverse=True)

            # Strict Acceptance: take best non-tabu neighbor (No Aspiration)
            accepted = next((n for n in neighbors if n[2] not in tabu_list), neighbors[0])
            new_weights, new_sharpe, new_hash = accepted

            # Update state and tabu list
            current, current_sharpe = new_weights, new_sharpe
            tabu_list.append(new_hash)

            if new_sharpe > best_sharpe:
                best_weights, best_sharpe = new_weights.copy(), new_sharpe
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1

            # Complete random restart on stagnation
            if iters_without_improvement >= stagnation_threshold and diversification_count < max_diversifications:
                current = repair_weights(np.random.uniform(0, 1, self.n_assets))
                current_sharpe = self._sharpe(current)
                tabu_list.clear()
                iters_without_improvement = 0
                diversification_count += 1

            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                convergence_log.append((iteration, best_sharpe, current_sharpe))

        best_weights = repair_weights(best_weights)
        return {
            'best_weights': best_weights,
            'best_sharpe': self._sharpe(best_weights),
            'best_return': calc_annual_return(best_weights, self.returns_data),
            'best_risk': calc_annual_risk(best_weights, self.cov_matrix),
            'convergence_log': convergence_log,
            'all_explored': all_explored,
        }
