"""Backtest 3 simple sector strategies against buy-and-hold SPY, without look-ahead."""
import numpy as np
import pandas as pd

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def top3_weights(past_returns):
    """Given each ETF's past return, give weight 1/3 to the 3 best ETFs, 0 to the rest.
    Rows with no past return yet (NaN) stay NaN, meaning 'no decision possible'."""
    best = past_returns.rank(axis=1, ascending=False) <= 3
    w = best.astype(float) / 3
    return w.where(past_returns.notna().any(axis=1))  # NaN where nothing was known


# Each strategy looks at MONTH-END prices and returns the weights it WANTS to hold
# after that month-end. It only sees prices up to that date (pct_change looks backwards).

def strategy_a(m):
    """A. Last year's winners: each December, pick the 3 best calendar-year performers."""
    year_ret = m[SECTORS].pct_change(12)               # return over the last 12 months
    year_ret = year_ret[year_ret.index.month == 12]    # decide only at the end of December
    w = top3_weights(year_ret)
    return w.reindex(m.index).ffill()                  # keep the same picks all year


def strategy_b(m):
    """B. 12-month momentum: each month-end, pick the 3 best trailing-12-month performers."""
    return top3_weights(m[SECTORS].pct_change(12))


def strategy_c(m):
    """C. Equal weight: every month-end, 1/9 in each sector ETF."""
    return pd.DataFrame(1 / len(SECTORS), index=m.index, columns=SECTORS)


def backtest(prices, strategy):
    """Take daily prices and a strategy function; return the strategy's monthly returns."""
    # Collapse daily prices to the last trading day of each month.
    m = prices.resample("ME").last()
    # What each ETF returned during each month (this month's price vs last month's).
    month_ret = m.pct_change()
    # Weights chosen at the END of a month...
    w = strategy(m)
    # ...can only earn money in the NEXT month. shift(1) moves them forward one month,
    # so the return of month t is always multiplied by weights decided at the end of t-1.
    w = w.shift(1)
    # Portfolio return = sum of (weight x ETF return); NaN until the strategy can trade.
    return (w * month_ret[SECTORS]).sum(axis=1, min_count=1).dropna()


def stats(r):
    """Total return, annualised return and volatility from a series of monthly returns."""
    total = (1 + r).prod() - 1
    ann = (1 + total) ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)  # monthly volatility scaled to a yearly figure
    return total, ann, vol


if __name__ == "__main__":
    prices = pd.read_csv("data/processed/prices.csv", index_col=0, parse_dates=True)

    # Benchmark: buy and hold SPY = just SPY's own monthly returns.
    spy = prices["SPY"].resample("ME").last().pct_change().dropna()
    strategies = {"A Last year's winners": strategy_a,
                  "B 12m momentum": strategy_b,
                  "C Equal weight": strategy_c}
    results = {name: backtest(prices, f) for name, f in strategies.items()}

    periods = {"2000-2026": ("2000-01-01", "2026-12-31"),
               "2000-2020": ("2000-01-01", "2020-12-31"),
               "2021-2026": ("2021-01-01", "2026-12-31")}

    rows = []
    for period, (start, end) in periods.items():
        for name, r in results.items():
            r = r.loc[start:end]
            s = spy.loc[r.index]  # SPY over exactly the same months, so the comparison is fair
            total, ann, vol = stats(r)
            _, spy_ann, _ = stats(s)
            rows.append([period, name, total, ann, vol, ann - spy_ann])
        total, ann, vol = stats(spy.loc[start:end])
        rows.append([period, "SPY buy & hold", total, ann, vol, np.nan])

    table = pd.DataFrame(rows, columns=["Period", "Strategy", "Total", "Annualised", "Volatility", "Excess vs SPY (ann.)"])
    pct = lambda x: "" if pd.isna(x) else f"{x:.1%}"
    print(table.to_string(index=False, formatters={c: pct for c in table.columns[2:]}))
