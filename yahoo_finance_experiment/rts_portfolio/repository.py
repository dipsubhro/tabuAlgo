"""
Pareto repository helpers for Reactive Tabu Search.

The repository keeps non-dominated portfolios, where a portfolio dominates
another when it has at least as much return, no more risk, and is strictly
better on one of those two objectives.
"""

import numpy as np


def dominates(left, right):
    """Return True when ``left`` dominates ``right`` on return and risk."""
    return (
        left["return"] >= right["return"]
        and left["risk"] <= right["risk"]
        and (left["return"] > right["return"] or left["risk"] < right["risk"])
    )


def _crowding_order(repository):
    """Rank repository members by Pareto diversity, then Sharpe."""
    if len(repository) <= 2:
        return list(range(len(repository)))

    risks = np.array([item["risk"] for item in repository], dtype=float)
    returns = np.array([item["return"] for item in repository], dtype=float)
    distances = np.zeros(len(repository), dtype=float)

    for values in (risks, returns):
        order = np.argsort(values)
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = values[order[-1]] - values[order[0]]
        if span <= 1e-12:
            continue
        for pos in range(1, len(order) - 1):
            distances[order[pos]] += (
                values[order[pos + 1]] - values[order[pos - 1]]
            ) / span

    sharpes = np.array([item["sharpe"] for item in repository], dtype=float)
    return sorted(
        range(len(repository)),
        key=lambda idx: (distances[idx], sharpes[idx]),
        reverse=True,
    )


def trim_repository(repository, max_size):
    """Keep a bounded, diverse Pareto repository."""
    if len(repository) <= max_size:
        return repository
    keep = set(_crowding_order(repository)[:max_size])
    return [item for idx, item in enumerate(repository) if idx in keep]


def update_repository(repository, weights, ret, risk, sharpe, max_size=100):
    """
    Add one portfolio to the repository if it is non-dominated.

    Returns ``(repository, added)`` where ``added`` is True when the portfolio
    entered the archive after dominated members were removed.
    """
    candidate = {
        "weights": np.asarray(weights, dtype=float).copy(),
        "return": float(ret),
        "risk": float(risk),
        "sharpe": float(sharpe),
    }

    for item in repository:
        same_metrics = (
            abs(item["return"] - candidate["return"]) <= 1e-12
            and abs(item["risk"] - candidate["risk"]) <= 1e-12
        )
        if same_metrics and np.allclose(item["weights"], candidate["weights"], atol=1e-10):
            return repository, False
        if dominates(item, candidate):
            return repository, False

    repository = [item for item in repository if not dominates(candidate, item)]
    repository.append(candidate)
    repository = trim_repository(repository, max_size)
    return repository, True


def update_repository_batch(repository, weights_2d, returns, risks, sharpes, max_size=100):
    """Add a batch of evaluated portfolios to the repository."""
    added_any = False
    for weights, ret, risk, sharpe in zip(weights_2d, returns, risks, sharpes):
        repository, added = update_repository(
            repository, weights, ret, risk, sharpe, max_size=max_size
        )
        added_any = added_any or added
    return repository, added_any


def best_by_sharpe(repository):
    """Return the repository member with the highest Sharpe ratio."""
    if not repository:
        return None
    return max(repository, key=lambda item: item["sharpe"])


def sample_repository_weights(repository):
    """Sample one repository member, biased toward high-Sharpe portfolios."""
    if not repository:
        return None

    sharpes = np.array([item["sharpe"] for item in repository], dtype=float)
    shifted = sharpes - sharpes.min()
    if shifted.sum() <= 1e-12:
        probs = None
    else:
        probs = shifted / shifted.sum()

    idx = np.random.choice(len(repository), p=probs)
    return repository[idx]["weights"].copy()
