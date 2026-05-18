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
    repair_weights,
    repair_weights_capped,
    calc_annual_return,
    calc_annual_risk,
    sharpe_ratio,
    calc_all_metrics,
    repair_weights_2d,
    repair_weights_capped_2d,
    sharpe_ratio_2d,
)
from .repository import (
    best_by_sharpe,
    sample_repository_weights,
    update_repository,
)


# ═══════════════════════════════════════════════════════════════════════════
# MODULE A — Lévy Flight Neighborhood Generator
# ═══════════════════════════════════════════════════════════════════════════

def levy_flight(size, beta=1.5):
    """
    Generate Lévy flight step sizes via Mantegna's algorithm.

    L(s) ~ |s|^{-1-β}  where 0 < β ≤ 2

    Provides the "long-jump" capability of population-based swarms,
    enabling escape from deep local optima that Gaussian perturbation
    cannot reach.
    """
    sigma_u = (
        _gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, 1, size)
    step = u / (np.abs(v) ** (1.0 / beta))
    return step


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RTS CLASS
# ═══════════════════════════════════════════════════════════════════════════

class SwarmReactiveTabuSearch:
    """
    Reactive Tabu Search for portfolio weight optimisation.

    Parameters
    ----------
    returns_data : ndarray (T, N)
        Daily returns matrix.
    cov_matrix : ndarray (N, N)
        Annualised covariance matrix.
    n_assets : int
        Number of assets.
    rf : float
        Risk-free rate (annualised).
    max_iter : int
        Maximum number of search iterations.
    neighbors_size : int
        Number of candidate neighbors generated per iteration.
    initial_tenure : int
        Starting tabu tenure.
    weight_cap : float
        Hard per-asset weight cap (e.g. 0.10 for 10 %).
    oscillation_cap : float
        Relaxed cap during strategic oscillation (Module C).
    beta : float
        Lévy exponent for Module A.
    cycle_threshold : int
        How many revisits trigger diversification (Module B).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        returns_data,
        cov_matrix,
        n_assets,
        rf=0.02,
        max_iter=5000,
        swarm_size=10,
        neighbors_size=50,
        initial_tenure=10,
        weight_cap=0.10,
        oscillation_cap=0.15,
        beta=1.5,
        cycle_threshold=3,
        osc_period=None,
        repository_size=100,
        seed=None,
    ):
        self.returns_data = returns_data
        self.cov_matrix = cov_matrix
        self.n_assets = n_assets
        self.rf = rf
        self.max_iter = max_iter
        self.swarm_size = swarm_size
        self.neighbors_size = neighbors_size
        self.initial_tenure = initial_tenure
        self.weight_cap = weight_cap
        self.oscillation_cap = oscillation_cap
        self.beta = beta
        self.cycle_threshold = cycle_threshold
        self.repository_size = repository_size
        self.seed = seed

        # ── Reactive tenure bounds (Module B) ──
        self.min_tenure = max(3, initial_tenure // 2)
        self.max_tenure = initial_tenure * 4

        # ── Strategic oscillation schedule (Module C) ──
        # Oscillation toggles every `osc_period` iterations
        self.osc_period = osc_period if osc_period is not None else max(50, max_iter // 20)

    # -------------------------------------------------------------------
    # Evaluate a candidate
    # -------------------------------------------------------------------
    def _sharpe(self, weights):
        """Compute Sharpe Ratio for a weight vector (after repair)."""
        return sharpe_ratio(
            weights, self.returns_data, self.cov_matrix, self.rf
        )

    # -------------------------------------------------------------------
    # Hash a solution for tabu checking / cycle detection
    # -------------------------------------------------------------------
    @staticmethod
    def _hash(weights, precision=3):
        return hash(tuple(np.round(weights, precision)))

    # -------------------------------------------------------------------
    # Generate neighbor via Lévy Flight (Module A)
    # -------------------------------------------------------------------
    def _generate_neighbor(self, current, step_scale, use_cap=False):
        """
        Perturb `current` weights using a Lévy flight step,
        then repair to satisfy constraints.
        """
        steps = levy_flight(self.n_assets, self.beta)
        candidate = current + step_scale * steps
        if use_cap:
            return repair_weights_capped(candidate, self.oscillation_cap)
        return repair_weights(candidate)

    def _generate_neighbors_2d(self, current, step_scale, use_cap=False):
        """
        Vectorized version to generate all neighbors at once.
        """
        steps = levy_flight((self.neighbors_size, self.n_assets), self.beta)
        candidates = current + step_scale * steps
        if use_cap:
            return repair_weights_capped_2d(candidates, self.oscillation_cap)
        return repair_weights_2d(candidates)

    # -------------------------------------------------------------------
    # Main search loop
    # -------------------------------------------------------------------
    def run(self):
        """
        Execute the Multi-Solution Swarm Reactive Tabu Search.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # ── Initialise Swarm ──────────────────────────────────────────────
        current_swarm = np.random.uniform(0, 1, (self.swarm_size, self.n_assets))
        current_swarm = repair_weights_2d(current_swarm)
        
        init_s, init_rets, init_risks = sharpe_ratio_2d(
            current_swarm, self.returns_data, self.cov_matrix, self.rf
        )

        best_idx = np.argmax(init_s)
        best_weights = current_swarm[best_idx].copy()
        best_sharpe = init_s[best_idx]
        repository = []

        # Global Memory
        tabu_list = {}
        visit_count = defaultdict(int)
        tenure = self.initial_tenure

        step_scale = 0.15
        step_decay = 0.9995

        convergence_log = []
        all_explored = []
        for i in range(self.swarm_size):
            all_explored.append((init_risks[i], init_rets[i], init_s[i]))
            repository, _ = update_repository(
                repository,
                current_swarm[i],
                init_rets[i],
                init_risks[i],
                init_s[i],
                max_size=self.repository_size,
            )

        iters_without_improvement = 0
        stagnation_threshold = max(100, self.max_iter // 10)

        for iteration in range(self.max_iter):
            cycle_pos = (iteration // self.osc_period) % 2
            oscillation_active = (cycle_pos == 1)
            
            if iters_without_improvement > stagnation_threshold:
                # Global Diversification Restart
                step_scale = 0.5
                keys_to_remove = list(tabu_list.keys())[: len(tabu_list) // 2]
                for k in keys_to_remove:
                    del tabu_list[k]
                for agent_idx in range(self.swarm_size):
                    restart_anchor = sample_repository_weights(repository)
                    if restart_anchor is not None:
                        current_swarm[agent_idx] = self._generate_neighbor(
                            restart_anchor, step_scale, use_cap=oscillation_active
                        )
                iters_without_improvement = 0

            # ── Mass Vectorized Generation for the ENTIRE Swarm ──
            current_repeated = np.repeat(current_swarm, self.neighbors_size, axis=0)
            
            total_candidates = self.swarm_size * self.neighbors_size
            steps = levy_flight((total_candidates, self.n_assets), self.beta)
            candidates_2d = current_repeated + step_scale * steps
            
            if oscillation_active:
                candidates_2d = repair_weights_capped_2d(candidates_2d, self.oscillation_cap)
            else:
                candidates_2d = repair_weights_2d(candidates_2d)
                
            cand_sharpes, cand_rets, cand_risks = sharpe_ratio_2d(
                candidates_2d, self.returns_data, self.cov_matrix, self.rf
            )

            improved_this_iter = False

            # ── Agent-by-Agent Update ──
            for agent_idx in range(self.swarm_size):
                start_idx = agent_idx * self.neighbors_size
                end_idx = start_idx + self.neighbors_size
                
                agent_w = candidates_2d[start_idx:end_idx]
                agent_s = cand_sharpes[start_idx:end_idx]
                agent_rets = cand_rets[start_idx:end_idx]
                agent_risks = cand_risks[start_idx:end_idx]
                
                indices = np.arange(self.neighbors_size)
                np.random.shuffle(indices)
                indices = sorted(indices, key=lambda idx: agent_s[idx], reverse=True)
                
                best_neighbor_found = False
                
                for idx in indices:
                    cand_w_single = agent_w[idx]
                    cand_s_single = agent_s[idx]
                    cand_h = self._hash(cand_w_single)
                    
                    all_explored.append((agent_risks[idx], agent_rets[idx], cand_s_single))
                    repository, repo_added = update_repository(
                        repository,
                        cand_w_single,
                        agent_rets[idx],
                        agent_risks[idx],
                        cand_s_single,
                        max_size=self.repository_size,
                    )
                    
                    is_tabu = cand_h in tabu_list and tabu_list[cand_h] >= iteration
                    
                    # Multi-Objective Aspiration
                    if is_tabu and (cand_s_single > best_sharpe or repo_added):
                        is_tabu = False
                        
                    if not is_tabu:
                        current_swarm[agent_idx] = cand_w_single
                        visit_count[cand_h] += 1
                        
                        if visit_count[cand_h] >= self.cycle_threshold:
                            tenure = min(self.max_tenure, int(tenure * 1.5))
                            visit_count[cand_h] = 0
                        else:
                            tenure = max(self.min_tenure, int(tenure * 0.95))
                            
                        tabu_list[cand_h] = iteration + tenure
                        best_neighbor_found = True
                        
                        if cand_s_single > best_sharpe:
                            best_sharpe = cand_s_single
                            best_weights = cand_w_single.copy()
                            improved_this_iter = True
                            
                        break
                        
                if not best_neighbor_found:
                    # Stagnation fallback: pick absolute best even if tabu
                    best_idx_agent = indices[0]
                    current_swarm[agent_idx] = agent_w[best_idx_agent]
                    
            if improved_this_iter:
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1

            step_scale = max(0.005, step_scale * step_decay)
            
            # Log progress based on Global Best
            if iteration % 100 == 0:
                convergence_log.append((
                    iteration, best_sharpe, best_sharpe, tenure, "Swarm Phase"
                ))

        # Final metrics
        repo_best = best_by_sharpe(repository)
        if repo_best is not None and repo_best["sharpe"] > best_sharpe:
            best_sharpe = repo_best["sharpe"]
            best_weights = repo_best["weights"].copy()

        final_metrics = calc_all_metrics(
            best_weights, self.returns_data, self.cov_matrix, self.rf
        )

        final_metrics['best_sharpe'] = best_sharpe
        final_metrics['best_return'] = final_metrics['return']
        final_metrics['best_risk'] = final_metrics['risk']
        final_metrics['best_weights'] = best_weights
        final_metrics['convergence_log'] = convergence_log
        final_metrics['all_explored'] = all_explored
        final_metrics['repository'] = repository

        return final_metrics
