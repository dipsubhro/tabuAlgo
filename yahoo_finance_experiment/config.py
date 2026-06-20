"""
RTS Experiment Configuration
=============================
All tunable parameters are defined here.
Edit this file to change any setting — no need to touch the runner.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM SELECTOR
# ═══════════════════════════════════════════════════════════════════════════

# Options: "SINGLE"  — Reactive Tabu Search (Lévy + Reactive Tenure + Oscillation)
#           "NORMAL"  — Standard Tabu Search (baseline, no enhancements)
ALGORITHM = "NORMAL"

# ═══════════════════════════════════════════════════════════════════════════
# DATA / UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════

# S&P 500 stocks to include in the portfolio universe
TICKERS = [
    'AAPL',   # Apple         — Technology
    'MSFT',   # Microsoft     — Technology
    'GOOGL',  # Alphabet      — Technology
    'AMZN',   # Amazon        — Consumer
    'JPM',    # JPMorgan      — Finance
    'JNJ',    # Johnson&Johnson — Healthcare
    'V',      # Visa          — Finance
    'PG',     # Procter&Gamble — Consumer
    'XOM',    # ExxonMobil    — Energy
    'NVDA',   # NVIDIA        — Technology
]

# Historical window (same as target paper)
DATA_START = '2013-01-01'
DATA_END   = '2023-01-01'

# Minimum fraction of trading days a stock must have data for (else dropped)
DATA_COVERAGE_THRESHOLD = 0.90

# ═══════════════════════════════════════════════════════════════════════════
# FINANCIAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# Annualised risk-free rate (2% = US T-bill proxy)
RF = 0.02

# ═══════════════════════════════════════════════════════════════════════════
# RTS ALGORITHM PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# Number of independent runs (different random seeds)
# NUM_RUNS = int(os.getenv("RTS_NUM_RUNS", "30"))
NUM_RUNS = 30

# Seed used to generate the per-run seed pool (reproducibility)
SEED_POOL_SEED = int(os.getenv("RTS_SEED_POOL_SEED", "786683"))

# --- Iterations ---
# Number of search steps per run
MAX_ITER = 3000   # applies to both SINGLE and NORMAL

# --- Neighbourhood ---
# Number of candidate neighbours generated each iteration
NEIGHBORS  = 100

# --- Lévy Flight (Module A) ---
# Lévy exponent β (1.5 = standard, lower = heavier tail / longer jumps)
LEVY_BETA = 1.5
# Fraction of neighbours generated via Lévy flight (rest = Gaussian)
LEVY_MIX_RATIO = 0.30

# --- Tabu Tenure ---
# Initial length of the tabu list (dynamically adjusted by reactive module)
TENURE = 10

# --- Reactive Tenure / Diversification (Module B) ---
# How many revisits to the same hash trigger a tenure increase
CYCLE_THRESHOLD = 3

# --- Portfolio Constraints ---
# Maximum weight any single asset can hold (10% cap)
WEIGHT_CAP = 0.10

# Maximum weight during strategic oscillation phase
OSC_CAP = 0.15

# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK / REFERENCE VALUES
# ═══════════════════════════════════════════════════════════════════════════

# Published results used for comparison tables and Pareto plot annotations
RMPSO_REF = {'return': None, 'risk': None, 'sharpe': 1.159}

CSO_REF = {'return': 0.168, 'risk': 0.187, 'sharpe': 0.950}

PAPER_RESULTS = {
    'BFO':       {'return': 0.142, 'risk': 0.198, 'sharpe': 0.617},
    'FWA':       {'return': 0.131, 'risk': 0.201, 'sharpe': 0.552},
    'CSO':       {'return': 0.168, 'risk': 0.187, 'sharpe': 0.950},
    'Bat':       {'return': 0.124, 'risk': 0.215, 'sharpe': 0.484},
    'mean-CVaR': {'return': 0.098, 'risk': 0.142, 'sharpe': 0.549},
}

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

# Base directory of this config file (yahoo_finance_experiment/)
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Data directory ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(_HERE, "data")

# Unified close prices — combined from individual ticker CSVs (all positive values)
# Run scripts/generate_dataset.py to rebuild from data/<TICKER>.csv files.
COMBINED_PRICES_CSV = os.path.join(DATA_DIR, "combined_prices.csv")
# Per-ticker OHLCV CSVs (from data1.zip)
TICKER_CSV = {t: os.path.join(DATA_DIR, f"{t}.csv") for t in TICKERS}

# ── Outputs directory ────────────────────────────────────────────────────────
OUTPUTS_DIR = os.path.join(_HERE, "outputs")

# Text reports — written to outputs/
OUT_CONVERGENCE_TXT   = os.path.join(OUTPUTS_DIR, "rts_convergence.txt")
OUT_COMPARISON_TXT    = os.path.join(OUTPUTS_DIR, "rts_comparison.txt")
OUT_COMPARISON_NORMAL = os.path.join(OUTPUTS_DIR, "tabu_normal_vs_optimized.txt")

# Plots — written to outputs/
OUT_PARETO_PNG        = os.path.join(OUTPUTS_DIR, "rts_pareto_front.png")
OUT_CONVERGENCE_GRAPH = os.path.join(OUTPUTS_DIR, "rts_convergence_graphs.png")
