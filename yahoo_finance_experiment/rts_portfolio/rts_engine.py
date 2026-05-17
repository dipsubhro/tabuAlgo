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
)


# ═══════════════════════════════════════════════════════════════════════════
# MODULE A — Lévy Flight Neighborhood Generator
# ═══════════════════════════════════════════════════════════════════════════

def levy_flight(n, beta=1.5):
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
    u = np.random.normal(0, sigma_u, n)
    v = np.random.normal(0, 1, n)
    step = u / (np.abs(v) ** (1.0 / beta))
    return step


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RTS CLASS
# ═══════════════════════════════════════════════════════════════════════════

class ReactiveTabuSearch:
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
        neighbors_size=50,
        initial_tenure=10,
        weight_cap=0.10,
        oscillation_cap=0.15,
        beta=1.5,
        cycle_threshold=3,
        seed=None,
    ):
        self.returns_data = returns_data
        self.cov_matrix = cov_matrix
        self.n_assets = n_assets
        self.rf = rf
        self.max_iter = max_iter
        self.neighbors_size = neighbors_size
        self.initial_tenure = initial_tenure
        self.weight_cap = weight_cap
        self.oscillation_cap = oscillation_cap
        self.beta = beta
        self.cycle_threshold = cycle_threshold
        self.seed = seed

        # ── Reactive tenure bounds (Module B) ──
        self.min_tenure = max(3, initial_tenure // 2)
        self.max_tenure = initial_tenure * 4

        # ── Strategic oscillation schedule (Module C) ──
        # Oscillation toggles every `osc_period` iterations
        self.osc_period = max(50, max_iter // 20)

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

        When use_cap is True (oscillation phase), applies the weight cap
        via repair_weights_capped to approach constraint boundaries.
        """
        steps = levy_flight(self.n_assets, self.beta)
        candidate = current + step_scale * steps
        if use_cap:
            return repair_weights_capped(candidate, self.oscillation_cap)
        return repair_weights(candidate)

    # -------------------------------------------------------------------
    # Main search loop
    # -------------------------------------------------------------------
    def run(self):
        """
        Execute the Reactive Tabu Search.

        Returns
        -------
        dict with keys:
            best_weights, best_sharpe, best_return, best_risk,
            convergence_log, all_explored
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # ── Initialise ──────────────────────────────────────────────
        current = repair_weights(
            np.random.uniform(0, 1, self.n_assets)
        )
        current_sharpe = self._sharpe(current)

        best_weights = current.copy()
        best_sharpe = current_sharpe

        # Tabu list: deque of hashes
        tabu_list = deque(maxlen=self.max_tenure * 3)

        # Module B: visit frequency for cycle detection
        visit_count = defaultdict(int)
        tenure = self.initial_tenure

        # Step scale: starts large, decays (but Lévy still allows long jumps)
        step_scale = 0.15
        min_step_scale = 0.005
        step_decay = 0.9995

        # Tracking
        convergence_log = []          # (iter, best_sharpe, current_sharpe, tenure, phase)
        all_explored = []             # list of (risk, return) for Pareto plot
        iters_without_improvement = 0
        stagnation_threshold = max(100, self.max_iter // 10)
        diversification_count = 0
        max_diversifications = 8

        for iteration in range(self.max_iter):
            # ── Module C: Strategic Oscillation ─────────────────────
            # Alternate between normal (uncapped) and capped phases.
            # During oscillation, weights are clipped to the oscillation
            # cap before normalisation — this biases the search toward
            # more diversified portfolios and explores the constraint
            # boundary from both sides.
            cycle_pos = (iteration // self.osc_period) % 2
            oscillation_active = (cycle_pos == 1)
            if oscillation_active:
                phase = "Oscillate"
            else:
                phase = "Feasible"

            # Determine search phase label for logging
            if iters_without_improvement > stagnation_threshold // 2:
                phase = "Diversify"
            elif iters_without_improvement == 0 and iteration > 0:
                phase = "Intensify"

            # ── Generate neighbors (Module A: Lévy Flight) ─────────
            neighbors = []
            for _ in range(self.neighbors_size):
                candidate = self._generate_neighbor(
                    current, step_scale, use_cap=oscillation_active
                )
                cand_sharpe = self._sharpe(candidate)
                cand_hash = self._hash(candidate)

                # Track for Pareto plot
                cand_ret = calc_annual_return(candidate, self.returns_data)
                cand_risk = calc_annual_risk(candidate, self.cov_matrix)
                all_explored.append((cand_risk, cand_ret, cand_sharpe))

                neighbors.append((candidate, cand_sharpe, cand_hash))

            # Sort descending by Sharpe (maximisation)
            np.random.shuffle(neighbors)  # break ties randomly
            neighbors.sort(key=lambda t: t[1], reverse=True)

            # ── Module D: Multi-Objective Aspiration Criteria ──────
            accepted = None
            for cand_w, cand_s, cand_h in neighbors:
                # Aspiration: if beats global best → override tabu
                if cand_s > best_sharpe:
                    accepted = (cand_w, cand_s, cand_h)
                    break
                # Normal: accept if not tabu
                if cand_h not in tabu_list:
                    accepted = (cand_w, cand_s, cand_h)
                    break

            # Fallback: if everything is tabu, take the best anyway
            if accepted is None:
                accepted = neighbors[0]

            new_weights, new_sharpe, new_hash = accepted

            # ── Update current solution ────────────────────────────
            current = new_weights
            current_sharpe = new_sharpe

            # Add to tabu list
            tabu_list.append(new_hash)

            # ── Module B: Reactive Tenure ──────────────────────────
            visit_count[new_hash] += 1

            if new_sharpe > best_sharpe:
                # Improvement found
                best_weights = new_weights.copy()
                best_sharpe = new_sharpe
                iters_without_improvement = 0

                # Intensify: decrease tenure for fine-tuning
                tenure = max(self.min_tenure, tenure - 1)
                tabu_list = deque(tabu_list, maxlen=tenure * 3)

            else:
                iters_without_improvement += 1

                # Cycle detection: if same region visited too often
                if visit_count[new_hash] >= self.cycle_threshold:
                    # Diversify: exponential tenure increase
                    tenure = min(self.max_tenure, int(tenure * 1.5))
                    tabu_list = deque(tabu_list, maxlen=tenure * 3)

                elif iters_without_improvement > stagnation_threshold // 2:
                    # Gradual tenure increase
                    tenure = min(self.max_tenure, tenure + 1)
                    tabu_list = deque(tabu_list, maxlen=tenure * 3)

            # ── Diversification restart if stuck ───────────────────
            if iters_without_improvement >= stagnation_threshold:
                if diversification_count < max_diversifications:
                    # Restart near best with large Lévy perturbation
                    current = self._generate_neighbor(
                        best_weights, step_scale * 5.0, use_cap=False
                    )
                    current_sharpe = self._sharpe(current)

                    # Reset step scale
                    step_scale = max(0.08, step_scale * 2.0)

                    # Partially clear tabu list
                    for _ in range(len(tabu_list) // 2):
                        if tabu_list:
                            tabu_list.popleft()

                    # Partially clear visit counts
                    visit_count.clear()

                    iters_without_improvement = 0
                    diversification_count += 1

            # ── Decay step scale ───────────────────────────────────
            step_scale = max(min_step_scale, step_scale * step_decay)

            # ── Convergence log every 10 iterations ────────────────
            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                convergence_log.append(
                    (iteration, best_sharpe, current_sharpe, tenure, phase)
                )

        # ── Final: ensure best weights satisfy hard constraints ────
        best_weights = repair_weights(best_weights)
        best_sharpe = self._sharpe(best_weights)

        best_ret = calc_annual_return(best_weights, self.returns_data)
        best_risk = calc_annual_risk(best_weights, self.cov_matrix)

        return {
            'best_weights': best_weights,
            'best_sharpe': best_sharpe,
            'best_return': best_ret,
            'best_risk': best_risk,
            'convergence_log': convergence_log,
            'all_explored': all_explored,
        }
