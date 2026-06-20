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

class SingleReactiveTabuSearch:
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
        osc_period=None,
        repository_size=100,
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
        Vectorized neighbor generation using Gaussian perturbation.

        Lévy flights are reserved for diversification restarts ONLY (where
        long-range jumps genuinely help escape local optima).  Using Lévy
        for every neighbor evaluation causes the search to overshoot on the
        relatively smooth Sharpe landscape, costing more iterations than it
        saves.
        """
        noise = np.random.normal(0, step_scale, (self.neighbors_size, self.n_assets))
        candidates = current + noise
        if use_cap:
            return repair_weights_capped_2d(candidates, self.oscillation_cap)
        return repair_weights_2d(candidates)

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
            convergence_log, all_explored, repository
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
        current_ret = calc_annual_return(current, self.returns_data)
        current_risk = calc_annual_risk(current, self.cov_matrix)
        repository = []
        repository, _ = update_repository(
            repository,
            current,
            current_ret,
            current_risk,
            current_sharpe,
            max_size=self.repository_size,
        )

        # Tabu list: capacity == tenure, exactly like Normal Tabu.
        # The *3 multiplier caused the list to be 3-12× larger than Normal's
        # regardless of the reactive tenure value, throttling exploration.
        tabu_list = deque(maxlen=self.initial_tenure)

        # Module B: visit frequency for cycle detection
        visit_count = defaultdict(int)
        tenure = self.initial_tenure

        # Step scale: match Normal Tabu's fixed 0.10 baseline.
        # Starting at 0.15 with decay caused overshooting early and
        # under-exploration late; a flat 0.10 gives a stable search radius.
        step_scale = 0.10
        min_step_scale = 0.005
        step_decay = 0.9999   # Very slow decay — preserves broad exploration

        # Tracking
        convergence_log = []          # (iter, best_sharpe, current_sharpe, tenure, phase)
        all_explored = []             # list of (risk, return) for Pareto plot
        iters_without_improvement = 0
        stagnation_threshold = max(200, self.max_iter // 8)  # match Normal Tabu's threshold
        diversification_count = 0
        max_diversifications = 8

        for iteration in range(self.max_iter):
            # ── Module C: Strategic Oscillation ─────────────────────
            # Alternate between normal (uncapped) and capped phases.
            # During oscillation, weights are clipped to the oscillation
            # cap before normalisation — this biases the search toward
            # more diversified portfolios and explores the constraint
            # boundary from both sides.
            # Oscillation is active 1-in-5 phases (20 % of iterations).
            # The original 50 % split wasted half the budget in a capped
            # sub-space that penalises the high-conviction weights that
            # drive Sharpe on this dataset.
            cycle_pos = (iteration // self.osc_period) % 5
            oscillation_active = (cycle_pos == 4)
            if oscillation_active:
                phase = "Oscillate"
            else:
                phase = "Feasible"

            # Determine search phase label for logging
            if iters_without_improvement > stagnation_threshold // 2:
                phase = "Diversify"
            elif iters_without_improvement == 0 and iteration > 0:
                phase = "Intensify"

            # ── Generate neighbors (Module A: Lévy Flight - Vectorized) ──
            candidates_2d = self._generate_neighbors_2d(
                current, step_scale, use_cap=oscillation_active
            )
            
            cand_sharpes, cand_rets, cand_risks = sharpe_ratio_2d(
                candidates_2d, self.returns_data, self.cov_matrix, self.rf
            )

            neighbors = []
            for i in range(self.neighbors_size):
                cand_w = candidates_2d[i]
                cand_s = cand_sharpes[i]
                cand_h = self._hash(cand_w)

                repository, repo_added = update_repository(
                    repository,
                    cand_w,
                    cand_rets[i],
                    cand_risks[i],
                    cand_s,
                    max_size=self.repository_size,
                )

                # Track for Pareto plot
                all_explored.append((cand_risks[i], cand_rets[i], cand_s))
                neighbors.append((cand_w, cand_s, cand_h, repo_added))

            # Sort descending by Sharpe (maximisation)
            np.random.shuffle(neighbors)  # break ties randomly
            neighbors.sort(key=lambda t: t[1], reverse=True)

            # ── Module D: Multi-Objective Aspiration Criteria ──────
            accepted = None
            for cand_w, cand_s, cand_h, repo_added in neighbors:
                # Aspiration: override tabu ONLY if the move improves the
                # global best Sharpe.  Accepting moves solely because they
                # enter the Pareto repository (repo_added) drifts the
                # current solution into low-Sharpe / low-risk territory,
                # wasting moves on diversity that doesn't improve the objective.
                if cand_s > best_sharpe:
                    accepted = (cand_w, cand_s, cand_h)
                    break
                # Normal: accept if not tabu
                if cand_h not in tabu_list:
                    accepted = (cand_w, cand_s, cand_h)
                    break

            # Fallback: if everything is tabu, take the best anyway
            if accepted is None:
                accepted = neighbors[0][:3]

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
                tabu_list = deque(tabu_list, maxlen=tenure)

            else:
                iters_without_improvement += 1

                # Cycle detection: if same region visited too often
                if visit_count[new_hash] >= self.cycle_threshold:
                    # Diversify: exponential tenure increase
                    tenure = min(self.max_tenure, int(tenure * 1.5))
                    tabu_list = deque(tabu_list, maxlen=tenure)

                elif iters_without_improvement > stagnation_threshold // 2:
                    # Gradual tenure increase
                    tenure = min(self.max_tenure, tenure + 1)
                    tabu_list = deque(tabu_list, maxlen=tenure)

            # ── Diversification restart if stuck ───────────────────
            if iters_without_improvement >= stagnation_threshold:
                if diversification_count < max_diversifications:
                    # Restart near the best-known region with a moderate
                    # Gaussian jump (not Lévy ×5 which gave random points).
                    restart_anchor = sample_repository_weights(repository)
                    if restart_anchor is None:
                        restart_anchor = best_weights
                    # σ=0.10 — diverse enough to escape, close enough to stay near good region
                    noise = np.random.normal(0, 0.10, self.n_assets)
                    current = repair_weights(restart_anchor + noise)
                    current_sharpe = self._sharpe(current)

                    step_scale = 0.10  # stable post-restart scale

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
        repo_best = best_by_sharpe(repository)
        if repo_best is not None and repo_best["sharpe"] > best_sharpe:
            best_weights = repo_best["weights"].copy()
            best_sharpe = repo_best["sharpe"]

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
            'repository': repository,
        }
