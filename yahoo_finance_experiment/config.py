import os

# Options: "RTS" (Reactive Tabu Search), "STS" (Standard Tabu Search)
ALGORITHM = "RTS"

# Data and Universe
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'XOM', 'NVDA']
DATA_START, DATA_END, DATA_COVERAGE_THRESHOLD = '2013-01-01', '2023-01-01', 0.90
RF = 0.02
NUM_RUNS = 30
SEED_POOL_SEED = int(os.getenv("RTS_SEED_POOL_SEED", "786683"))

# Shared Parameters (RTS & STS)
MAX_ITER = 3000
NEIGHBORS = 100

# STS Specific Parameters
STS_TENURE = 10
STS_STEP_SCALE = 0.10

# RTS Specific Parameters
RTS_INITIAL_TENURE = 10
RTS_LEVY_BETA = 1.5
RTS_CYCLE_THRESHOLD = 3
RTS_WEIGHT_CAP = 0.10
RTS_OSC_CAP = 0.15

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR, OUTPUTS_DIR = os.path.join(_HERE, "data"), os.path.join(_HERE, "outputs")
COMBINED_PRICES_CSV = os.path.join(DATA_DIR, "combined_prices.csv")
TICKER_CSV = {t: os.path.join(DATA_DIR, f"{t}.csv") for t in TICKERS}

OUT_CONVERGENCE_TXT   = os.path.join(OUTPUTS_DIR, "rts_convergence.txt")
OUT_COMPARISON_TXT    = os.path.join(OUTPUTS_DIR, "rts_comparison.txt")
OUT_PARETO_PNG        = os.path.join(OUTPUTS_DIR, "rts_pareto_front.png")
OUT_CONVERGENCE_GRAPH = os.path.join(OUTPUTS_DIR, "rts_convergence_graphs.png")
