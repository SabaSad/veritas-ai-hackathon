"""Can we predict a backtest winner's NEW-data excess return from what we saw on the PAST data?

Only simulated data is used here (no real 2021-2026 data).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw charts to a file, no window needed
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, brier_score_loss

SEED = 42
FEATURES = ["K", "past_excess", "past_tracking_error", "past_excess_over_te", "L", "top_k"]

# 1. Load the simulated dataset. Target 1: new-data excess (a number).
#    Target 2: 'beats SPY' = 1 if the new-data excess is above 0, else 0.
df = pd.read_csv("data/processed/sim_dataset.csv")
df["beats"] = (df["new_excess"] > 0).astype(int)

# 2. Split 80/20 BY MARKET: all rows of one market go to the same side, so the model can
#    never be tested on a market it has partly seen.
split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(split.split(df, groups=df["market_id"]))
train, test = df.iloc[train_idx], df.iloc[test_idx]
overlap = set(train["market_id"]) & set(test["market_id"])
print(f"Markets: train {train['market_id'].nunique()}, test {test['market_id'].nunique()}, "
      f"in both sets: {len(overlap)}  ->  {'OK' if not overlap else 'PROBLEM!'}")
assert not overlap

# 3. Fit the models on the TRAIN set only.
Xtr, Xte = train[FEATURES], test[FEATURES]
lin = LinearRegression().fit(Xtr, train["new_excess"])
hgb = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr, train["new_excess"])
hgb_clf = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, train["beats"])

# 4. Predictions on the TEST set. For each method: a predicted number, and a score/probability
#    for 'beats SPY' (higher = more likely to beat SPY).
majority = int(train["beats"].mean() >= 0.5)  # most common class in the training data
preds = {   # name: (predicted excess, 'beats' score, hard 0/1 'beats' prediction or None)
    "Baseline 1: trust the backtest": (test["past_excess"], test["past_excess"], (test["past_excess"] > 0).astype(int)),
    "Baseline 2: always the average": (np.full(len(test), train["new_excess"].mean()), np.full(len(test), majority), np.full(len(test), majority)),
    "Linear regression": (lin.predict(Xte), lin.predict(Xte), None),
    "Gradient boosting": (hgb.predict(Xte), hgb_clf.predict_proba(Xte)[:, 1], None),
}

# 5. Metrics on the TEST set only.
#    MAE / RMSE: average size of the error of the predicted excess (lower = better).
#    AUROC: how well the score ranks winners above losers (0.5 = coin flip, higher = better).
#    Brier: squared error of the 'beats SPY' probability (lower = better; n/a for linear regression).
def table(d, idx):
    rows = {}
    for name, (p, score, hard) in preds.items():
        p, score = np.asarray(p)[idx], np.asarray(score)[idx]
        y, ybeat = d["new_excess"], d["beats"]
        auc = roc_auc_score(ybeat, score) if len(set(score)) > 1 else 0.5   # constant score = 0.5
        brier = brier_score_loss(ybeat, np.asarray(hard)[idx] if hard is not None else score) if "Linear" not in name else np.nan
        rows[name] = [mean_absolute_error(y, p), np.sqrt(mean_squared_error(y, p)), auc, brier]
    return pd.DataFrame(rows, index=["MAE", "RMSE", "AUROC", "Brier"]).T

print("\n=== ALL TEST ROWS ===")
print(table(test, np.arange(len(test))).round(4))
# Fair reference for Brier: Baseline 2 above predicts a hard 0/1, which Brier punishes harshly.
# Predicting the training base rate as a PROBABILITY is the honest 'no-skill' Brier score.
base_rate = train["beats"].mean()
print(f"Reference Brier if we always predict P(beats SPY) = {base_rate:.2f}: "
      f"{brier_score_loss(test['beats'], np.full(len(test), base_rate)):.4f}")

# 6. The same table separately for each K (how many strategies the person tried).
for K in sorted(test["K"].unique()):
    idx = np.where(test["K"].to_numpy() == K)[0]
    print(f"\n=== K = {K} ({len(idx)} test rows) ===")
    print(table(test.iloc[idx], idx).round(4))

# 7. Calibration: sort test rows into 5 equal-sized bins by past excess (lowest to highest) and
#    compare, per bin, the average PREDICTED new excess with the average ACTUAL new excess.
bins = pd.qcut(test["past_excess"], 5, labels=[1, 2, 3, 4, 5])
calib = pd.DataFrame({"actual": test["new_excess"].groupby(bins, observed=True).mean()})
for name, (p, _, _) in preds.items():
    calib[name.split(":")[0]] = pd.Series(np.asarray(p), index=test.index).groupby(bins, observed=True).mean()
print("\n=== CALIBRATION by past-excess bin (1 = lowest past excess, 5 = highest) ===")
print((calib * 100).round(2).astype(str) + "%")

# 8. Simple chart: a perfectly calibrated method would lie on the 'actual' line.
plt.figure(figsize=(7, 4))
for col in calib.columns:
    label = {"Baseline 1": "Trust the backtest", "Baseline 2": "Always the average"}.get(col, col)
    plt.plot(calib.index.astype(int), calib[col] * 100, marker="o", lw=3 if col == "actual" else 1.5,
             color="black" if col == "actual" else None, label=label)
plt.xticks([1, 2, 3, 4, 5])   # whole-number bins only
plt.xlabel("Bin of past excess return (1 = lowest, 5 = highest)")
plt.ylabel("Average new-data excess return (%)")
plt.title("Predicted vs actual new-data excess, test set")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/calibration.png", dpi=150)
print("\nSaved figures/calibration.png")
