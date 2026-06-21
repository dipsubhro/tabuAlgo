import numpy as np
import yahoo_finance_experiment.config as cfg

def repair_weights(weights, cap=None):
    min_w = getattr(cfg, 'MIN_WEIGHT', 0.0)
    max_w = getattr(cfg, 'MAX_WEIGHT', cap if cap is not None else 1.0)
    w = np.clip(np.asarray(weights, float), min_w, max_w)
    for _ in range(10):
        s = w.sum()
        if s <= 0: break
        w = w / s
        w = np.clip(w, min_w, max_w)
    return w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0/len(w))

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
    min_w = getattr(cfg, 'MIN_WEIGHT', 0.0)
    max_w = getattr(cfg, 'MAX_WEIGHT', cap if cap is not None else 1.0)
    w = np.clip(weights_2d, min_w, max_w)
    for _ in range(10):
        tot = w.sum(axis=1, keepdims=True)
        tot[tot <= 0] = 1.0
        w = w / tot
        w = np.clip(w, min_w, max_w)
    tot = w.sum(axis=1, keepdims=True)
    tot[tot <= 0] = 1.0
    return w / tot

repair_weights_capped_2d = repair_weights_2d

def sharpe_ratio_2d(weights_2d, returns_data, cov_matrix, rf=0.02):
    ret = np.sum(weights_2d * np.mean(returns_data, axis=0), axis=1) * 252
    risk = np.sqrt(np.maximum(np.sum((weights_2d @ cov_matrix) * weights_2d, axis=1) * 252, 0.0))
    s, v = np.zeros_like(ret), risk > 1e-10
    s[v] = (ret[v] - rf) / risk[v]
    return s, ret, risk
