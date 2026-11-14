"""Download daily prices for SPY and the 9 original sector ETFs and save them to a CSV."""
import datetime as dt
import pandas as pd
import yfinance as yf

# SPY is the benchmark (the whole S&P 500); the other 9 are the original sector ETFs.
TICKERS = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
START = "2000-01-01"
END = dt.date.today().isoformat()  # today's date

# auto_adjust=True means prices already include dividends and splits,
# so the percentage change between two days is the real return an investor got.
data = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
prices = data["Close"][TICKERS]  # keep only the closing price, one column per ticker

# Save for the backtest (data/processed is git-ignored, so no data is committed).
prices.to_csv("data/processed/prices.csv")

# Check 1: the first date each ticker has a price.
print("First available date per ticker:")
for t in TICKERS:
    print(f"  {t}: {prices[t].first_valid_index().date()}")

# Check 2: tickers with missing prices (empty cells) after their first date.
print("\nTickers with missing data (after their first date):")
found = False
for t in TICKERS:
    col = prices[t].loc[prices[t].first_valid_index():]
    if col.isna().any():
        found = True
        print(f"  {t}: {col.isna().sum()} missing days")
if not found:
    print("  none")

# Check 3: does any ETF start after the year 2000 began?
late = [t for t in TICKERS if prices[t].first_valid_index() > pd.Timestamp("2000-01-10")]
print("\nETFs starting after 2000:", late if late else "none (all have data from the first trading days of 2000)")
