import numpy as np

from .helpers import compute_cvar, normalize_long_only_weights


def cvar_portfolio_objective(w, returns_matrix, lambda_, alpha):
    weights = normalize_long_only_weights(w)
    returns_matrix = np.asarray(returns_matrix, dtype=float)

    portfolio_returns = returns_matrix @ weights
    expected_return = float(np.mean(portfolio_returns))
    cvar = compute_cvar(portfolio_returns, alpha)

    return -(expected_return - float(lambda_) * cvar)
