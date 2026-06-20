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

# Options: "SINGLE" or "SWARM"
ALGORITHM = "SINGLE"

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
NUM_RUNS = int(os.getenv("RTS_NUM_RUNS", "30"))

# Seed used to generate the per-run seed pool (reproducibility)
SEED_POOL_SEED = int(os.getenv("RTS_SEED_POOL_SEED", "786683"))

# --- Iterations ---
# Number of search steps per run
MAX_ITER_SINGLE = 3000   # SingleReactiveTabuSearch
MAX_ITER_SWARM  = 2000   # SwarmReactiveTabuSearch

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

# --- Swarm (only used when ALGORITHM = "SWARM") ---
SWARM_SIZE = 10   # number of parallel solutions in the swarm

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
# OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════════════

# CSV dataset path (relative to yahoo_finance_experiment/)
# Run generate_dataset.py once to create this file.
# The runner loads from CSV if it exists; downloads & saves if not.
DATASET_CSV = "dataset.csv"

OUT_CONVERGENCE_TXT   = "rts_convergence.txt"
OUT_COMPARISON_TXT    = "rts_comparison.txt"
OUT_PARETO_PNG        = "rts_pareto_front.png"
OUT_CONVERGENCE_GRAPH = "rts_convergence_graphs.png"
