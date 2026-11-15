"""Descriptive check on REAL data: repeat the 'past vs new' test in many rolling 15-year windows.

No model is trained here. All months are real and in their real order.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw charts to a file, no window needed
import matplotlib.pyplot as plt
from backtest import monthly_returns
from simulate import STRATEGIES, KS, run_strategies   # exactly the same 20 strategies and definitions

WINDOW = 180   # 120 past months + 60 new months (run_strategies splits them the same way as before)
STEP = 12      # move the window by 12 months at a time

# 1. Real monthly returns, Feb 2000 - Dec 2025 (the first month has no prior price, and the
#    unfinished Nov 2026 month is left out).
prices = pd.read_csv("data/processed/prices.csv", index_col=0, parse_dates=True)
real = monthly_returns(prices).loc[:"2025-12-31"].dropna()

# 2. Rolling windows: the last one ends in Dec 2025; step back 12 months while 180 months exist.
starts = list(range(len(real) - WINDOW, -1, -STEP))[::-1]   # oldest window first
print(f"Real months: {real.index[0]:%b %Y} to {real.index[-1]:%b %Y} ({len(real)} months)")
print(f"{len(starts)} windows of {WINDOW} months:")
for i, s in enumerate(starts, 1):
    print(f"  window {i:>2}: {real.index[s]:%b %Y} - {real.index[s + WINDOW - 1]:%b %Y} "
          f"(past to {real.index[s + 119]:%b %Y}, new from {real.index[s + 120]:%b %Y})")

# 3. For every window: run the 20 strategies, then mimic someone who tried K of them.
rng = np.random.default_rng(42)   # fixed seed
rows = []
for i, s in enumerate(starts, 1):
    market = real.iloc[s:s + WINDOW].reset_index(drop=True)      # months numbered 0..179, real order
    res = run_strategies(market)                                  # same function as the simulation
    keys = list(res)
    past = np.array([res[k][0][0] for k in keys])                 # past excess vs SPY, per strategy
    new = np.array([res[k][1][0] for k in keys])                  # new-data excess vs SPY, per strategy
    rho = pd.Series(past).corr(pd.Series(new), method="spearman") # do past ranks match new ranks?
    for K in KS:
        # 1000 random sets of K strategies; in each, the 'winner' is the best PAST excess.
        winners = [(t := rng.choice(20, size=K, replace=False))[np.argmax(past[t])] for _ in range(1000)]
        row = {"window": i, "start": f"{real.index[s]:%Y-%m}", "end": f"{real.index[s + WINDOW - 1]:%Y-%m}",
               "K": K, "avg_past_excess": past[winners].mean(), "avg_new_excess": new[winners].mean(),
               "spearman_past_vs_new": rho}
        if K == 20:   # with all 20 tried there is only ONE winner: record it in detail
            w = winners[0]
            row.update(winner_L=keys[w][0], winner_top_k=keys[w][1])
        rows.append(row)

out = pd.DataFrame(rows)
w20 = out[out["K"] == 20]

# 4. K=20 winner by window, then the summary.
print("\nK=20 winner by window (excess return vs SPY, % per year):")
show = w20[["window", "start", "end", "winner_L", "winner_top_k", "avg_past_excess", "avg_new_excess", "spearman_past_vs_new"]]
print(show.rename(columns={"avg_past_excess": "past", "avg_new_excess": "new", "spearman_past_vs_new": "spearman"})
      .round({"past": 4, "new": 4, "spearman": 2}).assign(past=lambda d: (d.past * 100).round(2), new=lambda d: (d.new * 100).round(2))
      .to_string(index=False))
lower = int((w20["avg_new_excess"] < w20["avg_past_excess"]).sum())
print(f"\nSUMMARY over {len(w20)} windows:")
print(f"  K=20 winner's new excess was LOWER than its past excess in {lower} of {len(w20)} windows")
print(f"  average past excess: {w20['avg_past_excess'].mean():.2%}   average new excess: {w20['avg_new_excess'].mean():.2%}")
print(f"  average Spearman (past vs new across the 20 strategies): {w20['spearman_past_vs_new'].mean():.2f}")
print("\nAverage over windows for every K (past excess -> new excess):")
print(out.groupby("K")[["avg_past_excess", "avg_new_excess"]].mean().map(lambda x: f"{x:.2%}").to_string())

# 5. Save the results and a chart of the K=20 winner: past vs new excess, one dot per window.
out.to_csv("results/rolling_real.csv", index=False)
plt.figure(figsize=(6, 5))
x, y = w20["avg_past_excess"] * 100, w20["avg_new_excess"] * 100
plt.scatter(x, y)
for (_, r), xi, yi in zip(w20.iterrows(), x, y):
    # Label = window end year. 2023, 2024 and 2025 sit almost on top of each other, so push them apart.
    offset = {"2023": (6, -10), "2024": (-6, 8), "2025": (-30, -12)}.get(r["end"][:4], (3, 3))
    plt.annotate(r["end"][:4], (xi, yi), fontsize=7, xytext=offset, textcoords="offset points")
lim = [min(x.min(), y.min()), max(x.max(), y.max())]
plt.plot(lim, lim, "k--", label="45-degree line (new = past)")
plt.xlabel("K=20 winner: past excess vs SPY (% per year)")
plt.ylabel("K=20 winner: new-data excess vs SPY (% per year)")
plt.title("Real rolling windows (labels = window end year)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/rolling_real.png", dpi=150)

# 6. Plain-language reminders.
print("\nNOTE: the windows overlap heavily, so they are NOT independent observations.")
print("NOTE: all months are real and in real order; no simulation and no model is used here.")
print("NOTE: this is descriptive only, not a statistical test.")
