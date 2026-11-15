"""Simulate many markets by bootstrapping REAL months, then test 'pick the best past strategy'.

Because we only re-draw real months (bootstrap), there is no real time order left, so no momentum
strategy can have genuine skill. Any 'winner' we find is luck by construction.
"""
import itertools
import numpy as np
import pandas as pd
from backtest import SECTORS, monthly_returns, top_k_weights, past_return, backtest, stats

N_MARKETS = 2000          # how many simulated markets
N_MONTHS = 180            # months per simulated market (15 years)
N_PAST = 120              # first 120 months = "past" (where we pick the winner)
KS = [1, 3, 5, 10, 20]    # how many strategies a person "tried"

# 3. The 20 strategies: momentum with lookback L months, hold the top-k sectors, rebalance monthly.
def momentum(L, k):
    # At each month-end: rank sectors by their return over the last L months, hold the best k.
    # (backtest() then delays the weights by one month, so there is no look-ahead.)
    return lambda ret: top_k_weights(past_return(ret, L), k)

STRATEGIES = {(L, k): momentum(L, k) for L, k in itertools.product([1, 3, 6, 12], [1, 2, 3, 4, 5])}

def excess_and_te(strat, spy):
    """Annualised excess return vs SPY, and tracking error (yearly std of the monthly gap)."""
    spy = spy.loc[strat.index]                       # compare over exactly the same months
    excess = stats(strat)[1] - stats(spy)[1]
    tracking_error = (strat - spy).std() * np.sqrt(12)
    return excess, tracking_error


def run_strategies(market):
    """Run all 20 strategies on one 180-month market (a table of monthly returns, rows 0..179).
    Returns {(L, k): ((past excess, past tracking error), (new excess, new tracking error))}."""
    spy, sectors = market["SPY"], market[SECTORS]
    results = {}
    for (L, k), strategy in STRATEGIES.items():
        r = backtest(sectors, strategy)                # monthly returns of this strategy
        results[(L, k)] = (excess_and_te(r.loc[:N_PAST - 1], spy),   # months 0-119 (past)
                           excess_and_te(r.loc[N_PAST:], spy))       # months 120-179 (new data)
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(42)  # fixed seed: same results every run

    # 1. Real monthly returns of SPY + 9 sector ETFs, 2000-2020 ONLY (nothing from 2021 on).
    prices = pd.read_csv("data/processed/prices.csv", index_col=0, parse_dates=True)
    real = monthly_returns(prices).loc[:"2020-12-31"].dropna()  # drop the first month (no prior price)

    rows = []
    for market_id in range(N_MARKETS):
        # 2. Build one market by BOOTSTRAP: draw 180 real months independently, WITH replacement
        # (a month can appear twice). Why: each draw is independent, so there is no hidden link
        # between the 'past' window and the 'new' window (without replacement, months used in the
        # past can no longer appear later, which quietly ties the two windows together).
        # We draw month NUMBERS once and use the same numbers for all 10 tickers, so each
        # month's SPY/sector returns stay together and their relationships are kept.
        order = rng.integers(0, len(real), size=N_MONTHS)
        market = real.iloc[order].reset_index(drop=True)   # months are now numbered 0..179

        # 4. Run all 20 strategies; measure them on the 'past' months and on the 'new data' months.
        results = run_strategies(market)

        # 5. Mimic someone who tried K strategies: pick K at random, keep the best PAST excess.
        keys = list(STRATEGIES)
        for K in KS:
            tried = [keys[i] for i in rng.choice(len(keys), size=K, replace=False)]
            L, k = max(tried, key=lambda key: results[key][0][0])        # the 'winner'
            (past_ex, past_te), (new_ex, _) = results[(L, k)]
            # 6. One row per (market, K).
            rows.append([market_id, K, past_ex, past_te, past_ex / past_te, L, k, new_ex])

    data = pd.DataFrame(rows, columns=["market_id", "K", "past_excess", "past_tracking_error",
                                       "past_excess_over_te", "L", "top_k", "new_excess"])
    data.to_csv("data/processed/sim_dataset.csv", index=False)

    # 7. Average past vs new-data excess return of the winners, for each K.
    print(data.groupby("K")[["past_excess", "new_excess"]].mean().map(lambda x: f"{x:.2%}"))
