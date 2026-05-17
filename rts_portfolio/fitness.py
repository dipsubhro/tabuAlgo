"""
Fitness functions and helpers for portfolio optimization.
Replicates the exact calculations from the RMPSO reference script
(download_yahoo_finance_data(1).py) so results are directly comparable.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Weight repair
# ---------------------------------------------------------------------------

def repair_weights(weights):
    """
    Repair portfolio weights: clip negatives and normalise to sum=1.

    Matches the RMPSO reference implementation exactly:
        w_i >= 0,  sum(w) = 1  (long-only, fully invested).

    Parameters
    ----------
    weights : array-like
        Raw weight vector.
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = w.sum()
    if total <= 0.0:
        return np.full(len(w), 1.0 / len(w))
    return w / total


def repair_weights_capped(weights, cap):
    """
    Repair with per-asset weight cap for Strategic Oscillation (Module C).

    Clips each weight to [0, cap], then normalises.  When the cap is tight
    (e.g. 0.10 with 10 assets), this naturally pushes toward equal-weight.
    When the cap is relaxed (e.g. 0.15 or 1.0), it preserves the relative
    weight structure.

    The final normalisation may push some weights slightly above cap;
    this is acceptable as it represents the "boundary approach" that
    strategic oscillation is designed to explore.
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, cap)
    total = w.sum()
    if total <= 0.0:
        return np.full(len(w), 1.0 / len(w))
    return w / total


# ---------------------------------------------------------------------------
# Portfolio metrics  (identical to the RMPSO reference script)
# ---------------------------------------------------------------------------

def calc_annual_return(weights, returns_data):
    """Annual Return = mean daily return × 252"""
    w = repair_weights(weights)
    return float(np.sum(w * np.mean(returns_data, axis=0)) * 252)


def calc_annual_risk(weights, cov_matrix):
    """Annual Risk = sqrt(wT × Σ × w × 252)"""
    w = repair_weights(weights)
    return float(np.sqrt(w @ cov_matrix @ w * 252))


def sharpe_ratio(weights, returns_data, cov_matrix, rf=0.02):
    """
    Sharpe Ratio = (Annual Return - Risk Free Rate) / Annual Risk

    Parameters
    ----------
    rf : float
        Risk-free rate (annualised).  Default 2 % matches the RMPSO reference.
    """
    ret = calc_annual_return(weights, returns_data)
    risk = calc_annual_risk(weights, cov_matrix)
    if risk < 1e-10:
        return 0.0
    return (ret - rf) / risk


# Alias used internally by the engine (maximise → negate for tabu minimisation)
calc_sharpe = sharpe_ratio


def calc_all_metrics(weights, returns_data, cov_matrix, rf=0.02):
    """Calculate all portfolio metrics at once (matches RMPSO reference)."""
    w = repair_weights(weights)
    ret = calc_annual_return(w, returns_data)
    risk = calc_annual_risk(w, cov_matrix)
    s = (ret - rf) / risk if risk > 1e-10 else 0.0

    # Max Drawdown
    daily_port_returns = returns_data @ w
    cum_returns = np.cumprod(1 + daily_port_returns)
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = float(np.min(drawdowns))

    return {
        'return': ret,
        'risk': risk,
        'sharpe': s,
        'max_drawdown': max_drawdown,
    }
