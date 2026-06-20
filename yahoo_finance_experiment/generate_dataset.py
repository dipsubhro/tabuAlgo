"""
generate_dataset.py
====================
One-time script to download S&P 500 stock data from Yahoo Finance
and save it as a CSV file for reuse across experiments.

Usage:
    uv run python -m yahoo_finance_experiment.generate_dataset

Output:
    yahoo_finance_experiment/dataset.csv
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yahoo_finance_experiment.config as cfg


def generate():
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        print("  [!] yfinance not installed. Run: uv add yfinance")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  DATASET GENERATOR — Yahoo Finance")
    print("=" * 60)
    print(f"  Tickers : {cfg.TICKERS}")
    print(f"  Period  : {cfg.DATA_START} → {cfg.DATA_END}")
    print(f"  Output  : {cfg.DATASET_CSV}")
    print("=" * 60)

    print("\n  Downloading...")
    raw = yf.download(
        cfg.TICKERS,
        start=cfg.DATA_START,
        end=cfg.DATA_END,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw['Close']
    else:
        data = raw

    # Drop columns with too many missing values, then drop remaining NaN rows
    data = data.dropna(axis=1, thresh=int(cfg.DATA_COVERAGE_THRESHOLD * len(data)))
    data = data.dropna()

    # Compute daily returns
    returns = data.pct_change().dropna()

    print(f"\n  ✓ Downloaded  : {returns.shape[1]} stocks")
    print(f"  ✓ Trading days: {returns.shape[0]}")
    print(f"  ✓ Date range  : {returns.index[0].date()} → {returns.index[-1].date()}")

    # Save to CSV (index = date, columns = ticker symbols)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.DATASET_CSV)
    returns.to_csv(out_path)

    print(f"\n  ✓ Saved: {out_path}")
    print(f"  ✓ Shape: {returns.shape}  (rows=trading days, cols=tickers)")
    print("\n  Done! You can now run the RTS experiment.")
    print("  The runner will load from CSV automatically.\n")


if __name__ == "__main__":
    generate()
