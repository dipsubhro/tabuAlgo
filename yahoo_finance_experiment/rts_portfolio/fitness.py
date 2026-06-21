import numpy as np

def repair_weights(weights, cap=None):
    min_w = 0.01
    w = np.clip(np.asarray(weights, float), 0.0, cap)
    n = len(w)
    s = w.sum()
    w = w / s if s > 0 else np.full(n, 1.0/n)
    return min_w + w * max(0.0, 1.0 - n * min_w)

repair_weights_capped = repair_weights

def calc_annual_return(weights, returns_data):
    w = repair_weights(weights)
    return float(np.sum(w * np.mean(returns_data, axis=0)) * 252)

def calc_annual_risk(weights, cov_matrix):
    w = repair_weights(weights)
    return float(np.sqrt(w @ cov_matrix @ w * 252))

def sharpe_ratio(weights, returns_data, cov_matrix, rf=0.02):
    ret, risk = calc_annual_return(weights, returns_data), calc_annual_risk(weights, cov_matrix)
    return (ret - rf) / risk if risk > 1e-10 else 0.0

calc_sharpe = sharpe_ratio

def calc_all_metrics(weights, returns_data, cov_matrix, rf=0.02):
    w = repair_weights(weights)
    ret, risk = calc_annual_return(w, returns_data), calc_annual_risk(w, cov_matrix)
    c = np.cumprod(1 + returns_data @ w)
    return {'return': ret, 'risk': risk, 'sharpe': (ret-rf)/risk if risk > 1e-10 else 0.0,
            'max_drawdown': float(np.min((c - np.maximum.accumulate(c)) / np.maximum.accumulate(c)))}

def repair_weights_2d(weights_2d, cap=None):
    min_w = 0.01
    w = np.clip(weights_2d, 0.0, cap)
    tot = w.sum(axis=1, keepdims=True)
    mask = (tot <= 0).flatten()
    n = w.shape[1]
    if np.any(mask): w[mask, :], tot[mask, 0] = 1.0/n, 1.0
    w = w / tot
    return min_w + w * max(0.0, 1.0 - n * min_w)

repair_weights_capped_2d = repair_weights_2d

def sharpe_ratio_2d(weights_2d, returns_data, cov_matrix, rf=0.02):
    ret = np.sum(weights_2d * np.mean(returns_data, axis=0), axis=1) * 252
    risk = np.sqrt(np.maximum(np.sum((weights_2d @ cov_matrix) * weights_2d, axis=1) * 252, 0.0))
    s, v = np.zeros_like(ret), risk > 1e-10
    s[v] = (ret[v] - rf) / risk[v]
    return s, ret, risk
