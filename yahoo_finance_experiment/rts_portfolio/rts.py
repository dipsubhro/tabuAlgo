"""
Reactive Tabu Search (RTS) for Portfolio Optimization
======================================================

Integrates four enhancement modules into a standard Tabu Search cycle:

  Module A  — Lévy Flight Neighborhood Generator
  Module B  — Reactive Tabu Tenure (Dynamic Memory)
  Module C  — Strategic Oscillation (Constraint Relaxation)
  Module D  — Multi-Objective Aspiration Criteria

Author:  Built on top of the existing adaptive tabu framework in tabu.py
Target:  Maximise Sharpe Ratio on S&P 500 (Jan 2013 – Jan 2023)
"""

import numpy as np
from collections import defaultdict, deque
from scipy.special import gamma as _gamma
import math

from .fitness import (
    repair_weights, repair_weights_capped, calc_annual_return, calc_annual_risk,
    sharpe_ratio, repair_weights_2d, repair_weights_capped_2d, sharpe_ratio_2d,
)
from .repository import best_by_sharpe, sample_repository_weights, update_repository

# ── Module A: Lévy Flight Neighborhood Generator ──
def levy_flight(size, beta=1.5):
    sigma_u = (_gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
               (_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, 1, size)
    return u / (np.abs(v) ** (1.0 / beta))

class SingleReactiveTabuSearch:
    # ── Initialization ──
    def __init__(self, returns_data, cov_matrix, n_assets, rf=0.02, max_iter=5000,
                 neighbors_size=50, initial_tenure=10, weight_cap=0.10,
                 oscillation_cap=0.15, beta=1.5, cycle_threshold=3,
                 osc_period=None, repository_size=100, seed=None):
        self.returns_data, self.cov_matrix, self.n_assets, self.rf = returns_data, cov_matrix, n_assets, rf
        self.max_iter, self.neighbors_size, self.initial_tenure = max_iter, neighbors_size, initial_tenure
        self.weight_cap, self.oscillation_cap, self.beta = weight_cap, oscillation_cap, beta
        self.cycle_threshold, self.repository_size, self.seed = cycle_threshold, repository_size, seed
        self.min_tenure, self.max_tenure = max(3, initial_tenure // 2), initial_tenure * 4
        self.osc_period = osc_period if osc_period is not None else max(50, max_iter // 20)

    def _sharpe(self, weights):
        return sharpe_ratio(weights, self.returns_data, self.cov_matrix, self.rf)

    @staticmethod
    def _hash(weights, precision=3):
        return hash(tuple(np.round(weights, precision)))

    # ── Neighbor Generation (Mixed Gaussian & Module A: Lévy) ──
    def _generate_neighbors_2d(self, current, step_scale, use_cap=False):
        n_levy = max(1, int(0.30 * self.neighbors_size))
        n_gauss = self.neighbors_size - n_levy

        cands_g = current + np.random.normal(0, step_scale, (n_gauss, self.n_assets))
        cands_l = current + step_scale * np.array([levy_flight(self.n_assets, self.beta) for _ in range(n_levy)])
        
        candidates = np.vstack([cands_g, cands_l])
        return repair_weights_capped_2d(candidates, self.oscillation_cap) if use_cap else repair_weights_2d(candidates)

    # ── Main Search Loop ──
    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        # Initialize tracking, tabu list, and repository
        current = repair_weights(np.random.uniform(0, 1, self.n_assets))
        current_sharpe = self._sharpe(current)
        best_weights, best_sharpe = current.copy(), current_sharpe
        repository = update_repository([], current, calc_annual_return(current, self.returns_data),
                                       calc_annual_risk(current, self.cov_matrix), current_sharpe, self.repository_size)[0]

        tabu_list = deque(maxlen=self.initial_tenure)
        visit_count = defaultdict(int)
        tenure = self.initial_tenure

        rng_jitter = np.random.default_rng(self.seed if self.seed is not None else 0)
        step_scale = np.clip(0.15 + float(rng_jitter.normal(0, 0.03)), 0.08, 0.25)
        step_decay, min_step_scale = 0.9999, 0.005

        convergence_log, all_explored = [], []
        iters_without_improvement, diversification_count = 0, 0
        stagnation_threshold, max_diversifications = max(200, self.max_iter // 8), 12

        for iteration in range(self.max_iter):
            # ── Module C: Strategic Oscillation (Toggle constraints) ──
            oscillation_active = ((iteration // self.osc_period) % 5 == 4)
            phase = "Oscillate" if oscillation_active else ("Diversify" if iters_without_improvement > stagnation_threshold // 2 else ("Intensify" if iters_without_improvement == 0 and iteration > 0 else "Feasible"))

            # Generate and evaluate candidates
            candidates_2d = self._generate_neighbors_2d(current, step_scale, use_cap=oscillation_active)
            cand_sharpes, cand_rets, cand_risks = sharpe_ratio_2d(candidates_2d, self.returns_data, self.cov_matrix, self.rf)

            neighbors = []
            for i in range(self.neighbors_size):
                cand_w, cand_s = candidates_2d[i], cand_sharpes[i]
                cand_h = self._hash(cand_w)
                repository, repo_added = update_repository(repository, cand_w, cand_rets[i], cand_risks[i], cand_s, max_size=self.repository_size)
                all_explored.append((cand_risks[i], cand_rets[i], cand_s))
                neighbors.append((cand_w, cand_s, cand_h, repo_added))

            np.random.shuffle(neighbors)
            neighbors.sort(key=lambda t: t[1], reverse=True)

            # ── Module D: Multi-Objective Aspiration (Override tabu if globally best) ──
            accepted = next((n[:3] for n in neighbors if n[1] > best_sharpe or n[2] not in tabu_list), neighbors[0][:3])
            new_weights, new_sharpe, new_hash = accepted

            # Update state and tabu list
            current, current_sharpe = new_weights, new_sharpe
            tabu_list.append(new_hash)
            visit_count[new_hash] += 1

            # ── Module B: Reactive Tabu Tenure ──
            if new_sharpe > best_sharpe:
                best_weights, best_sharpe = new_weights.copy(), new_sharpe
                iters_without_improvement = 0
                tenure = max(self.min_tenure, tenure - 1)
                tabu_list = deque(tabu_list, maxlen=tenure)
            else:
                iters_without_improvement += 1
                if visit_count[new_hash] >= self.cycle_threshold:
                    tenure = min(self.max_tenure, int(tenure * 1.5))
                    tabu_list = deque(tabu_list, maxlen=tenure)
                elif iters_without_improvement > stagnation_threshold // 2:
                    tenure = min(self.max_tenure, tenure + 1)
                    tabu_list = deque(tabu_list, maxlen=tenure)

            # Repository-guided restart on stagnation
            if iters_without_improvement >= stagnation_threshold and diversification_count < max_diversifications:
                restart_anchor = sample_repository_weights(repository)
                if restart_anchor is None: restart_anchor = best_weights
                current = repair_weights(restart_anchor + np.random.normal(0, 0.10, self.n_assets))
                current_sharpe = self._sharpe(current)
                step_scale = 0.10

                for _ in range(len(tabu_list) // 2): tabu_list.popleft()
                visit_count.clear()
                iters_without_improvement, diversification_count = 0, diversification_count + 1

            step_scale = max(min_step_scale, step_scale * step_decay)
            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                convergence_log.append((iteration, best_sharpe, current_sharpe, tenure, phase))

        repo_best = best_by_sharpe(repository)
        if repo_best and repo_best["sharpe"] > best_sharpe:
            best_weights, best_sharpe = repo_best["weights"].copy(), repo_best["sharpe"]

        best_weights = repair_weights(best_weights)
        return {
            'best_weights': best_weights,
            'best_sharpe': self._sharpe(best_weights),
            'best_return': calc_annual_return(best_weights, self.returns_data),
            'best_risk': calc_annual_risk(best_weights, self.cov_matrix),
            'convergence_log': convergence_log,
            'all_explored': all_explored,
            'repository': repository,
        }
