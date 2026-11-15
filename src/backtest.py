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


# Each strategy looks at MONTHLY RETURNS (one row per month, one column per asset) and
# returns the weights it WANTS to hold after that month. It only sees returns up to that
# month (rolling windows look backwards), so it works the same on real or simulated data.

def past_12m(ret):
    """Return over the last 12 months = the 12 monthly returns compounded together."""
    return (1 + ret).rolling(12).apply(np.prod, raw=True) - 1


def strategy_a(ret):
    """A. Last year's winners: each December, pick the 3 best calendar-year performers."""
    year_ret = past_12m(ret)
    year_ret = year_ret[year_ret.index.month == 12]    # decide only at the end of December
    w = top3_weights(year_ret)
    return w.reindex(ret.index).ffill()                # keep the same picks all year


def strategy_b(ret):
    """B. 12-month momentum: each month, pick the 3 best trailing-12-month performers."""
    return top3_weights(past_12m(ret))


def strategy_c(ret):
    """C. Equal weight: every month, the same weight in every asset."""
    return pd.DataFrame(1 / ret.shape[1], index=ret.index, columns=ret.columns)


def monthly_returns(prices):
    """Turn daily prices into monthly returns (last trading day of each month)."""
    return prices.resample("ME").last().pct_change()


def backtest(ret, strategy):
    """Take monthly returns of the assets and a strategy function; return the strategy's monthly returns."""
    # Weights chosen at the END of a month...
    w = strategy(ret)
    # ...can only earn money in the NEXT month. shift(1) moves them forward one month,
    # so the return of month t is always multiplied by weights decided at the end of t-1.
    w = w.shift(1)
    # Portfolio return = sum of (weight x asset return); NaN until the strategy can trade.
    return (w * ret).sum(axis=1, min_count=1).dropna()


def stats(r):
    """Total return, annualised return and volatility from a series of monthly returns."""
    total = (1 + r).prod() - 1
    ann = (1 + total) ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)  # monthly volatility scaled to a yearly figure
    return total, ann, vol


if __name__ == "__main__":
    prices = pd.read_csv("data/processed/prices.csv", index_col=0, parse_dates=True)

    # Daily prices -> monthly returns. SPY is the benchmark (buy and hold = its own returns);
    # the 9 sector ETFs are what the strategies trade.
    ret = monthly_returns(prices)
    spy = ret["SPY"].dropna()
    ret = ret[SECTORS]
    strategies = {"A Last year's winners": strategy_a,
                  "B 12m momentum": strategy_b,
                  "C Equal weight": strategy_c}
    results = {name: backtest(ret, f) for name, f in strategies.items()}

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
