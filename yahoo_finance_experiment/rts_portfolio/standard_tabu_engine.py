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
    repair_weights,
    calc_annual_return,
    calc_annual_risk,
    sharpe_ratio,
    repair_weights_2d,
    sharpe_ratio_2d,
)


class StandardTabuSearch:
    """
    Standard (Textbook) Tabu Search for portfolio weight optimisation.

    Key limitations vs Optimized RTS:
      - Gaussian noise only → gets trapped in local optima
      - Fixed step size → cannot fine-tune OR escape at the right times
      - Fixed tenure → cannot adapt to search dynamics
      - No aspiration → rejects globally-improving moves if they're tabu
      - Random restart → no intelligent diversification guidance
      - No oscillation → never explores constraint boundaries

    Parameters
    ----------
    returns_data : ndarray (T, N)
        Daily returns matrix.
    cov_matrix : ndarray (N, N)
        Daily covariance matrix.
    n_assets : int
        Number of assets.
    rf : float
        Risk-free rate (annualised).
    max_iter : int
        Maximum number of search iterations.
    neighbors_size : int
        Number of candidate neighbors generated per iteration.
    tenure : int
        Fixed tabu tenure (number of iterations a move stays tabu).
    step_scale : float
        Fixed Gaussian perturbation scale (no decay).
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
        tenure=10,
        step_scale=0.10,
        seed=None,
    ):
        self.returns_data = returns_data
        self.cov_matrix = cov_matrix
        self.n_assets = n_assets
        self.rf = rf
        self.max_iter = max_iter
        self.neighbors_size = neighbors_size
        self.tenure = tenure
        self.step_scale = step_scale
        self.seed = seed

    # -------------------------------------------------------------------
    # Evaluate a candidate
    # -------------------------------------------------------------------
    def _sharpe(self, weights):
        """Compute Sharpe Ratio for a weight vector (after repair)."""
        return sharpe_ratio(
            weights, self.returns_data, self.cov_matrix, self.rf
        )

    # -------------------------------------------------------------------
    # Hash a solution for tabu checking
    # -------------------------------------------------------------------
    @staticmethod
    def _hash(weights, precision=3):
        return hash(tuple(np.round(weights, precision)))

    # -------------------------------------------------------------------
    # Generate neighbors via Gaussian perturbation (standard approach)
    # -------------------------------------------------------------------
    def _generate_neighbors_2d(self, current):
        """
        Generate neighbors using simple Gaussian perturbation with
        a FIXED step scale. No heavy tails, no long-range jumps.

        In high dimensions, Gaussian perturbation concentrates around
        the mean distance — it cannot produce the occasional large
        jumps that Lévy flights do, so it gets trapped easily.
        """
        noise = np.random.normal(0, self.step_scale, (self.neighbors_size, self.n_assets))
        candidates = current + noise
        return repair_weights_2d(candidates)

    # -------------------------------------------------------------------
    # Main search loop
    # -------------------------------------------------------------------
    def run(self):
        """
        Execute the Standard Tabu Search.

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

        # Fixed-size tabu list (deque of hashes)
        tabu_list = deque(maxlen=self.tenure)

        # Tracking
        convergence_log = []          # (iter, best_sharpe, current_sharpe)
        all_explored = []             # list of (risk, return, sharpe)
        iters_without_improvement = 0
        stagnation_threshold = max(200, self.max_iter // 8)
        diversification_count = 0
        max_diversifications = 5

        for iteration in range(self.max_iter):
            # ── Generate neighbors (fixed Gaussian perturbation) ───
            candidates_2d = self._generate_neighbors_2d(current)

            cand_sharpes, cand_rets, cand_risks = sharpe_ratio_2d(
                candidates_2d, self.returns_data, self.cov_matrix, self.rf
            )

            neighbors = []
            for i in range(self.neighbors_size):
                cand_w = candidates_2d[i]
                cand_s = cand_sharpes[i]
                cand_h = self._hash(cand_w)

                # Track for Pareto plot
                all_explored.append((cand_risks[i], cand_rets[i], cand_s))
                neighbors.append((cand_w, cand_s, cand_h))

            # Sort descending by Sharpe (maximisation)
            np.random.shuffle(neighbors)  # break ties randomly
            neighbors.sort(key=lambda t: t[1], reverse=True)

            # ── STRICT Best-Admissible Acceptance ──────────────────
            # NO aspiration criteria — strictly reject all tabu moves.
            # This is the textbook approach: the best NON-TABU neighbor
            # is accepted, even if a tabu neighbor would be globally best.
            accepted = None
            for cand_w, cand_s, cand_h in neighbors:
                if cand_h not in tabu_list:
                    accepted = (cand_w, cand_s, cand_h)
                    break

            # Fallback: if everything is tabu, take the best anyway
            # (unavoidable — otherwise search halts entirely)
            if accepted is None:
                accepted = neighbors[0][:3]

            new_weights, new_sharpe, new_hash = accepted

            # ── Update current solution ────────────────────────────
            current = new_weights
            current_sharpe = new_sharpe

            # Add to tabu list (fixed tenure — no reactive adjustment)
            tabu_list.append(new_hash)

            if new_sharpe > best_sharpe:
                # Improvement found
                best_weights = new_weights.copy()
                best_sharpe = new_sharpe
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1

            # ── Completely random restart if stuck ─────────────────
            # Unlike Optimized RTS which uses repository-guided Lévy
            # restarts, the normal version restarts from SCRATCH —
            # completely random weights with no memory of good regions.
            if iters_without_improvement >= stagnation_threshold:
                if diversification_count < max_diversifications:
                    # Restart from completely random weights
                    current = repair_weights(
                        np.random.uniform(0, 1, self.n_assets)
                    )
                    current_sharpe = self._sharpe(current)

                    # Clear tabu list on restart
                    tabu_list.clear()

                    iters_without_improvement = 0
                    diversification_count += 1

            # ── Convergence log every 10 iterations ────────────────
            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                convergence_log.append(
                    (iteration, best_sharpe, current_sharpe)
                )

        # ── Final repair ───────────────────────────────────────────
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
