import numpy as np


def normalize_long_only_weights(w):
    weights = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = weights.sum()

    if total <= 0.0:
        return np.full(weights.shape, 1.0 / len(weights), dtype=float)

    return weights / total


def compute_cvar(portfolio_returns, alpha):
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if alpha > 1:
        alpha /= 100.0
    if alpha > 1:
        raise ValueError("alpha must be in (0, 1] or (0, 100]")

    losses = -np.asarray(portfolio_returns, dtype=float)
    tail_count = max(1, int(np.ceil(alpha * losses.size)))
    worst_losses = np.sort(losses)[-tail_count:]
    return float(worst_losses.mean())
