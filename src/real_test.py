"""One illustrative test on REAL data: past = 2011-2020, new = 2021-2025, in real order.

The models are trained ONLY on simulated data (sim_dataset.csv); real data is used only to test.
Nothing is tuned after seeing these results.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw charts to a file, no window needed
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from backtest import monthly_returns
from simulate import STRATEGIES, N_PAST, KS, run_strategies   # exactly the same definitions as the simulation

FEATURES = ["K", "past_excess", "past_tracking_error", "past_excess_over_te", "L", "top_k"]
SEED = 42

# 1. Real monthly returns, Jan 2011 - Dec 2025, in their REAL order (180 months).
#    We stop at Dec 2025 so the unfinished Nov 2026 month is left out.
prices = pd.read_csv("data/processed/prices.csv", index_col=0, parse_dates=True)
real = monthly_returns(prices).loc["2011-01-01":"2025-12-31"]
past_months, new_months = real.iloc[:N_PAST], real.iloc[N_PAST:]
print(f"Months: {real.index[0]:%b %Y} to {real.index[-1]:%b %Y}, count = {len(real)}")
print(f"  past: {past_months.index[0]:%b %Y} to {past_months.index[-1]:%b %Y} ({len(past_months)} months)")
print(f"  new : {new_months.index[0]:%b %Y} to {new_months.index[-1]:%b %Y} ({len(new_months)} months)")
assert len(real) == 180 and not real.isna().any().any()

# 2. Run the same 20 momentum strategies, with the same function as the simulation
#    (months are numbered 0..179; first 120 = past, last 60 = new data).
results = run_strategies(real.reset_index(drop=True))
tab = pd.DataFrame([[L, k, p[0], p[1], p[0] / p[1], n[0]] for (L, k), (p, n) in results.items()],
                   columns=["L", "top_k", "past_excess", "past_tracking_error", "past_excess_over_te", "new_excess"])

# 3. Table of the 20 strategies, and the rank correlation between past and new excess
#    (+1 = the strategies that did best in the past also did best later; 0 = no link).
show = tab[["L", "top_k", "past_excess", "new_excess"]].copy()
show[["past_excess", "new_excess"]] = (show[["past_excess", "new_excess"]] * 100).round(2)
print("\nAll 20 strategies (excess return vs SPY, % per year):")
print(show.to_string(index=False))
rho = tab["past_excess"].corr(tab["new_excess"], method="spearman")
print(f"\nSpearman rank correlation, past vs new excess across the 20 strategies: {rho:.2f}")

# 4. Train the two models on ALL simulated rows. Real data is never used for training.
sim = pd.read_csv("data/processed/sim_dataset.csv")
lin = LinearRegression().fit(sim[FEATURES], sim["new_excess"])
hgb = HistGradientBoostingRegressor(random_state=SEED).fit(sim[FEATURES], sim["new_excess"])
sim_avg = sim["new_excess"].mean()   # 'always the average' prediction

# 5. For each K: imagine 1000 people who each tried K random strategies and kept the one with
#    the best PAST excess (the 'winner'). Predict each winner's new-data excess.
rng = np.random.default_rng(SEED)
summary = []
for K in KS:
    winners = []
    for _ in range(1000):
        tried = rng.choice(len(tab), size=K, replace=False)       # K different strategies
        winners.append(tab.loc[tab.loc[tried, "past_excess"].idxmax()])  # best past excess wins
    w = pd.DataFrame(winners).reset_index(drop=True)
    w.insert(0, "K", K)
    preds = {"trust_backtest": w["past_excess"],                  # 'it will keep doing the same'
             "always_average": pd.Series(sim_avg, index=w.index),
             "linear": lin.predict(w[FEATURES]),
             "boosting": hgb.predict(w[FEATURES])}
    row = {"K": K, "avg_past_excess": w["past_excess"].mean(), "avg_actual_new_excess": w["new_excess"].mean()}
    for name, p in preds.items():
        row[f"pred_{name}"] = np.mean(p)                          # average prediction
        row[f"mae_{name}"] = np.mean(np.abs(np.asarray(p) - w["new_excess"]))  # average miss
    summary.append(row)
    if K == 20:   # with all 20 tried there is only ONE possible winner
        print(f"\nK=20 has a single winner: L={int(w['L'][0])}, top_k={int(w['top_k'][0])}, "
              f"past {w['past_excess'][0]:.2%}, new {w['new_excess'][0]:.2%}")

summary = pd.DataFrame(summary)
print("\nPer K (averages over 1000 random subsets; MAE = average miss of the new-data excess):")
print((summary.set_index("K") * 100).round(2).T.to_string())  # all values in % per year

# 6. Save results and a chart: past vs new excess of the 20 strategies.
summary.to_csv("results/real_test.csv", index=False)
plt.figure(figsize=(6, 5))
plt.scatter(tab["past_excess"] * 100, tab["new_excess"] * 100, label="20 strategies")
lim = np.array([tab[["past_excess", "new_excess"]].min().min(), tab[["past_excess", "new_excess"]].max().max()]) * 100
plt.plot(lim, lim, "k--", label="'Trust the backtest' (new = past)")
plt.axhline(sim_avg * 100, color="red", label=f"Simulated average ({sim_avg:.2%})")
plt.xlabel("Past excess vs SPY, 2011-2020 (% per year)")
plt.ylabel("New-data excess vs SPY, 2021-2025 (% per year)")
plt.title("Real data: past vs new excess return")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/real_test.png", dpi=150)

# 7. Honest summary.
print("\n=== SUMMARY ===")
for _, r in summary.iterrows():
    maes = {m: r[f"mae_{m}"] for m in ["trust_backtest", "always_average", "linear", "boosting"]}
    best = min(maes, key=maes.get)
    beats = [m for m in ["linear", "boosting"] if maes[m] < min(maes["trust_backtest"], maes["always_average"])]
    print(f"K={int(r.K):>2}: lowest MAE = {best}; models beating BOTH baselines: {beats or 'none'}")
print("WARNING: this is ONE real 5-year window (2021-2025), so it is only illustrative, not proof.")
