import os

ALGORITHM = "SINGLE" # Options: "SINGLE", "NORMAL"
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'XOM', 'NVDA']
DATA_START, DATA_END, DATA_COVERAGE_THRESHOLD = '2013-01-01', '2023-01-01', 0.90
RF = 0.02
NUM_RUNS = 30
SEED_POOL_SEED = int(os.getenv("RTS_SEED_POOL_SEED", "786683"))

# RTS Alg Params
MAX_ITER, NEIGHBORS, TENURE = 3000, 100, 10
LEVY_BETA, LEVY_MIX_RATIO, CYCLE_THRESHOLD = 1.5, 0.30, 3
WEIGHT_CAP, OSC_CAP = 0.10, 0.15

# Benchmarks
RMPSO_REF = {'return': None, 'risk': None, 'sharpe': 1.159}
CSO_REF = {'return': 0.168, 'risk': 0.187, 'sharpe': 0.950}
PAPER_RESULTS = {
    'BFO':       {'return': 0.142, 'risk': 0.198, 'sharpe': 0.617},
    'FWA':       {'return': 0.131, 'risk': 0.201, 'sharpe': 0.552},
    'CSO':       {'return': 0.168, 'risk': 0.187, 'sharpe': 0.950},
    'Bat':       {'return': 0.124, 'risk': 0.215, 'sharpe': 0.484},
    'mean-CVaR': {'return': 0.098, 'risk': 0.142, 'sharpe': 0.549},
}

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR, OUTPUTS_DIR = os.path.join(_HERE, "data"), os.path.join(_HERE, "outputs")
COMBINED_PRICES_CSV = os.path.join(DATA_DIR, "combined_prices.csv")
TICKER_CSV = {t: os.path.join(DATA_DIR, f"{t}.csv") for t in TICKERS}

OUT_CONVERGENCE_TXT   = os.path.join(OUTPUTS_DIR, "rts_convergence.txt")
OUT_COMPARISON_TXT    = os.path.join(OUTPUTS_DIR, "rts_comparison.txt")
OUT_COMPARISON_NORMAL = os.path.join(OUTPUTS_DIR, "tabu_normal_vs_optimized.txt")
OUT_PARETO_PNG        = os.path.join(OUTPUTS_DIR, "rts_pareto_front.png")
OUT_CONVERGENCE_GRAPH = os.path.join(OUTPUTS_DIR, "rts_convergence_graphs.png")
