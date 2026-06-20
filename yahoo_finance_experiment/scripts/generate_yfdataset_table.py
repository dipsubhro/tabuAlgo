"""
Generate the Yahoo Finance dataset used by the portfolio algorithms.

This script uses the same tickers, date range, adjustment setting, cleanup,
and daily-return calculation as yfdataset.py, then writes the result as a
plain text table.

Usage:
    uv run python -m yahoo_finance_experiment.scripts.generate_yfdataset_table

Output:
    yahoo_finance_experiment/data/yfdataset_table.txt
"""

import sys
import os
from pathlib import Path

# scripts/ is one level deeper than the package root, so go up two levels
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PKG_DIR     = _SCRIPTS_DIR.parent          # yahoo_finance_experiment/
_DATA_DIR    = _PKG_DIR / "data"
project_root = str(_PKG_DIR.parent)          # tabu/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

OUTPUT_FILE = _DATA_DIR / "yfdataset_table.txt"

import pandas as pd
from tabulate import tabulate

START_DATE = "2013-01-01"
END_DATE = "2023-01-01"

TICKERS = [
    "AAPL",   # Apple - Technology
    "MSFT",   # Microsoft - Technology
    "GOOGL",  # Alphabet - Technology
    "AMZN",   # Amazon - Consumer
    "JPM",    # JPMorgan - Finance
    "JNJ",    # Johnson&Johnson - Healthcare
    "V",      # Visa - Finance
    "PG",     # Procter&Gamble - Consumer
    "XOM",    # ExxonMobil - Energy
    "NVDA",   # NVIDIA - Technology
]


def download_returns_table():
    """Download adjusted close prices and convert them to daily returns."""
    try:
        import yfinance as yf
    except ImportError:
        print("  [!] yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print("  Downloading data from Yahoo Finance...")
    print(f"  Tickers : {TICKERS}")
    print(f"  Period  : {START_DATE} to {END_DATE}")

    raw = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        data = raw

    data = data.dropna(axis=1, thresh=int(0.9 * len(data)))
    data = data.dropna()

    returns = data.pct_change().dropna()
    returns.index.name = "Date"

    return returns


def write_table(returns):
    """Write the daily returns dataset as a readable text table."""
    table = returns.reset_index()
    table["Date"] = table["Date"].dt.date

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        file.write("Yahoo Finance Portfolio Dataset\n")
        file.write("================================\n")
        file.write("Source       : Yahoo Finance via yfinance\n")
        file.write(f"Date range   : {returns.index[0].date()} to {returns.index[-1].date()}\n")
        file.write(f"Assets       : {len(returns.columns)}\n")
        file.write(f"Trading days : {len(returns)}\n")
        file.write("Data type    : Daily returns from adjusted close prices\n")
        file.write(f"Tickers      : {', '.join(returns.columns)}\n\n")
        file.write(
            tabulate(
                table,
                headers="keys",
                tablefmt="grid",
                showindex=False,
                floatfmt=".8f",
            )
        )
        file.write("\n")

    print(f"  Wrote table to: {OUTPUT_FILE}")


def main():
    returns = download_returns_table()
    write_table(returns)


if __name__ == "__main__":
    main()
