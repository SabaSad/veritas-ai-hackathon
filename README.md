# Project name

**Track:** Health / Finance / Climate / ...
**Team:** names

## The problem

Who has this problem, and why does it matter? (2-3 sentences)

## What we built

What goes in, what comes out, and how someone would use it.

## How it works

Data: which dataset, how many examples, how we cleaned it.

Model: what we used and why.

## Results

Our baseline scored X. Our model scored Y on held-out test data.

Where it fails, and why we think so.

## What we'd do next

One or two concrete next steps with more time.

## Sources

Datasets, pretrained models, tutorials and AI tools used, with links.

## Demo video

[Demo video link - add here]

---

*Disclaimer: this is a prototype built for a hackathon, not financial, medical, or legal advice.*

## How to run

1. Clone the repo:
   ```
   git clone <repo-url>
   cd <repo-folder>
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

## Pre-registered plan (written before any results)

- **Question:** if a strategy beat the S&P 500 ETF (SPY) in the past, will it keep beating it on new data, or was it luck?
- **Split rule:** real data is split by time (train up to about 2020, test 2021 to 2026). Simulated markets are split by whole market, never by single days.
- **Baselines:** (1) trust the backtest, assume a past winner keeps winning; (2) always predict the most common outcome.
- **Hypotheses:**
  - H1: ...
  - H2: ...
  - H3: ...
