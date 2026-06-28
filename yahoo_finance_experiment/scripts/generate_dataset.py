"""
generate_dataset.py
====================
Builds combined_prices.csv by combining the Close column from each
individual per-ticker CSV already present in data/.

  data/AAPL.csv  ─┐
  data/MSFT.csv  ─┤  extract Close  →  combine  →  data/combined_prices.csv
  ...             ─┤
  data/NVDA.csv  ─┘

The runners (run_rts_portfolio.py, run_comparison.py) load
combined_prices.csv and compute pct_change() internally.

Usage:
    uv run python -m yahoo_finance_experiment.scripts.generate_dataset

Output:
    yahoo_finance_experiment/data/combined_prices.csv
"""

import sys
import os

# scripts/ is one level deeper than the package root, so go up two levels
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import yahoo_finance_experiment.config as cfg


def generate():
    print("\n" + "=" * 60)
    print("  DATASET GENERATOR — from individual ticker CSVs")
    print("=" * 60)
    print(f"  Tickers : {cfg.TICKERS}")
    print(f"  Source  : data/<TICKER>.csv  (Close prices)")
    print(f"  Output  : {cfg.COMBINED_PRICES_CSV}")
    print("=" * 60)

    # ── Step 1: Load Close prices from each ticker CSV ────────────
    frames = {}
    missing = []

    for ticker in cfg.TICKERS:
        csv_path = cfg.TICKER_CSV[ticker]
        if not os.path.exists(csv_path):
            missing.append(ticker)
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if "Close" not in df.columns:
            print(f"  [!] {ticker}.csv has no 'Close' column — skipping")
            continue
        frames[ticker] = df["Close"]
        print(f"  ✓ {ticker}.csv  — {len(df)} rows  "
              f"(Close: {df['Close'].min():.2f} → {df['Close'].max():.2f})")

    if missing:
        print(f"\n  [!] Missing CSV files for: {missing}")
        print(f"      Place them in: {cfg.DATA_DIR}")
        sys.exit(1)

    # ── Step 2: Combine into one Close-prices DataFrame ───────────
    combined = pd.DataFrame(frames)
    combined.index.name = "Date"
    combined = combined.sort_index().dropna()

    print(f"\n  ✓ Combined shape : {combined.shape}  (rows=trading days, cols=tickers)")
    print(f"  ✓ Date range     : {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"  ✓ All positive   : {(combined > 0).all().all()}")

    # ── Step 3: Save combined_prices.csv ─────────────────────────
    os.makedirs(os.path.dirname(cfg.COMBINED_PRICES_CSV), exist_ok=True)
    combined.to_csv(cfg.COMBINED_PRICES_CSV)

    print(f"\n  ✓ Saved: {cfg.COMBINED_PRICES_CSV}")
    print("\n  Done! Runners will load combined_prices.csv and compute")
    print("  pct_change() internally for the algorithm.\n")


if __name__ == "__main__":
    generate()
