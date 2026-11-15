# Strategy Reality Check

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SabaSad/veritas-ai-hackathon/blob/main/notebooks/Strategy_Reality_Check.ipynb)

*This is a prototype, not financial advice.*

**Pitch:** A tool that tests an investing strategy against the S&P 500 and estimates, with a model trained on simulated markets, whether its past success is likely to be luck.

**Track:** Finance | **Building as:** individual | **Code:** https://github.com/SabaSad/veritas-ai-hackathon | **Demo video:** https://youtu.be/GKulXOXFp6c

> **Demo note:** the Gradio demo link only works while the notebook is running (about 6 minutes). Open the notebook in Colab, choose Runtime, then Run all, and open the link printed in the last cell.

## The problem
Beginner investors see rules that "beat the market" in a backtest (a test on past prices). But if you try enough rules, the best one will look great on past data by pure luck, and then fail on new data. A beginner cannot tell skill from luck. My question: if a strategy beat the S&P 500 (SPY) in the past, will it keep beating it?

## What I built
- **Real data:** daily prices from yfinance (Yahoo Finance) for SPY and the 9 original sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY), from 2000.
- **20 strategies:** momentum rules (lookback of 1, 3, 6 or 12 months; hold the top 1 to 5 ETFs; rebalance monthly). Weights decided at the end of a month only earn money in the next month, so no future data is used.
- **Simulated no-skill markets (synthetic data):** 2,000 markets, each made by drawing 180 real months from 2000-2020 at random, with replacement (the same draw for all tickers). Only the order is random, so no strategy can have real time-based skill. The first 120 months are the "past" and the last 60 are the "new data". To mimic someone who tried K strategies, I keep the one with the best past excess return (the "winner"), for K = 1, 3, 5, 10 and 20.
- **Model:** linear regression and gradient boosting predict a winner's new-data excess return from its past excess, tracking error, K and strategy settings. Trained only on simulated markets, split by whole market (1,600 train, 400 test). Baselines: (1) "trust the backtest" and (2) "always the average".
- **Real data is kept separate from simulated data:** one real window never seen by the simulation (past 2011-2020, new 2021-2025), and 11 rolling real windows (descriptive, no model).
- **Demo:** a Gradio app where you pick a window, a strategy and K, and it shows the backtest, what no-skill winners like yours did afterwards, and the model's estimate. A button reveals what really happened.

## What is my own
I chose the question myself, inspired by briefs 4.2.4 and 4.2.5. The parts I added beyond a standard backtest are: the no-skill simulated markets, the "best of K strategies" selection that creates luck on purpose, a model that estimates how much of a past edge is likely to be luck, real-data tests kept separate from the simulation, and a demo that lets you reveal what really happened.

## Results
**Simulated markets (average winner, % per year vs SPY):**

| K (strategies tried) | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Past excess | 0.63 | 2.68 | 3.23 | 4.16 | 5.06 |
| New-data excess | 0.89 | 0.87 | 0.73 | 0.74 | 0.39 |

The more strategies tried, the better the winner looks in the past, but the new-data result does not grow. This is luck from selection, because these markets contain no skill. At K = 20, the backtest said +5.06% a year, but the new-data result was +0.39%.

**Model (test markets):** The linear model tied "always the average" on error (MAE 0.0338 vs 0.0339); gradient boosting was slightly worse (0.0349). "Trust the backtest" ranked worst (AUROC 0.478). In plain words: the best forecast when nothing is predictable is the average, and the model learned to ignore the impressive past number.

**One real window the simulation never saw (2021-2025):** all 20 strategies lagged SPY in 2011-2020, so there was no impressive winner to test. The models predicted about +0.5% to +0.8%, while real winners did -0.7% to -2.3%. Only at K = 1 did gradient boosting beat both baselines; for K = 3 to 20, no model did. I did not tune anything after seeing this. "Trust the backtest" scored the lowest error for K = 3 to 20 only by accident, because every past number was negative.

**Eleven rolling real windows (descriptive):** the best-of-20 strategy's new-data excess was below its past excess in 10 of 11 windows (average +2.25% past, -2.80% new; average rank correlation between past and new across the 20 strategies: -0.10). Past excess rose with K (-0.19% at K=1 to +2.25% at K=20), but new-data excess was negative at every K. The windows overlap, so they are not independent.

**Pre-registered hypotheses (written before any results, see the plan below):**
- H1: Strategies that beat the S&P 500 in 2000 to 2020 will beat it by less, or not at all, in 2021 to 2026. Outcome: **supported** (descriptive evidence), because the best-of-20 strategy's new-data excess was below its past excess in 10 of 11 real windows (average +2.25% past, -2.80% new). The windows overlap.
- H2: The more strategies we try to find a "winner", the bigger the gap between its past and new-data performance. Outcome: **supported**, because in the simulated markets the gap grew from -0.26 points at K=1 to 4.67 points at K=20, and in the real windows from 1.41 to 5.05.
- H3: The model will estimate persistence with lower error than "trust the backtest", but only by a modest margin. Outcome: **partly supported**, because on the simulated test markets the linear model's MAE was 0.0338 against 0.0445 for "trust the backtest" (24% lower, which is not modest), but on the one unseen real window "trust the backtest" had the lowest error for K = 3 to 20.

## Where it fails
- Only one real window is fully unseen by the simulation; the others overlap the 2000-2020 months it reshuffles.
- The models barely beat "always the average", and I did not test why the linear model's AUROC (0.584) is above 0.5.
- The simulated baseline (about +0.7%) came from 2000-2020; the real 2011-2025 baseline was lower (a regime shift).
- Rolling windows overlap heavily, so "10 of 11" is not 10 independent tests.
- No trading costs; sector ETFs only; 20 momentum strategies only.
- Yahoo may revise adjusted prices, so re-runs can differ slightly.

## Credits and AI use
Data: daily prices downloaded with [yfinance](https://github.com/ranaroussi/yfinance) from [Yahoo Finance](https://finance.yahoo.com). Libraries: pandas, numpy, scikit-learn, matplotlib, Gradio. AI tools: Claude (planning and explanations) and Claude Code (helped write code). I can explain each part.

*This project is a prototype and is not financial advice.*

## How to run
**Easiest:** click the Colab badge at the top, then Runtime, then Run all.

**On your computer:**
1. Clone the repo:
   ```
   git clone https://github.com/SabaSad/veritas-ai-hackathon.git
   cd veritas-ai-hackathon
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Open the notebook:
   ```
   jupyter notebook notebooks/
   ```
5. Or run the demo directly (after the notebook or scripts have created the data): `python src/app.py`

## Pre-registered plan (written before any results)

- **Question:** if a strategy beat the S&P 500 ETF (SPY) in the past, will it keep beating it on new data, or was it luck?
- **Split rule:** real data is split by time (train up to about 2020, test 2021 to 2026). Simulated markets are split by whole market, never by single days.
- **Baselines:** (1) trust the backtest, assume a past winner keeps winning; (2) always predict the most common outcome.
- **Hypotheses:**
  - H1: Strategies that beat the S&P 500 in 2000 to 2020 will beat it by less, or not at all, in 2021 to 2026
  - H2: The more strategies we try to find a "winner", the bigger the gap between its past and new-data performance
  - H3: The model will estimate persistence with lower error than "trust the backtest", but only by a modest margin

*Note added after the results (the plan above is unchanged): the real unseen test used new years 2021-2025, the last full calendar years in the data when I ran it.*
