# AI Usage Log

Track every time you use an AI tool (Claude Code, ChatGPT, GitHub Copilot, etc.) during the hackathon. This is required for the write-up's "Sources" / AI tools section.

| Date | What I asked Claude Code to do | What I changed or checked myself |
|------|-------------------------------|-----------------------------------|
| 2026-10-20 (EXAMPLE) | Set up project scaffolding: venv, requirements.txt, folder structure, README/AI_USAGE/.gitignore templates, and a check_setup.py smoke test | Reviewed the generated .gitignore to make sure no secrets or data files would be committed; ran check_setup.py myself and confirmed the printed accuracy/confusion matrix looked reasonable before trusting the environment was working |
| 2026-11-14 | Claude (chat): project idea, plan, check_prices script | Reviewed and run by me |
| 2026-11-15 | Claude Code: wrote src/train_model.py (predict winner new-data excess and beats-SPY from past data, split by market_id, 2 baselines vs linear and gradient boosting, calibration chart) | Run and checked by me: TO CONFIRM |
| 2026-11-15 | Claude Code: changed src/simulate.py to build each market by bootstrap (180 months drawn with replacement, same draws for all tickers); re-ran simulate.py and train_model.py | Run and checked by me: TO CONFIRM |
| 2026-11-15 | Claude Code: refactored src/backtest.py so strategies work on monthly returns, and generalised the helpers (top-k, lookback L) | Run and checked by me: TO CONFIRM |
| 2026-11-15 | Claude Code: wrote src/simulate.py (shuffled real months to build simulated markets, 20 momentum strategies, best-of-K winners) | Run and checked by me: TO CONFIRM |
| 2026-11-15 | Claude Code: wrote src/real_test.py (one real-data test, 2011-2020 past vs 2021-2025 new, models trained on simulated data only) and restructured src/simulate.py into importable functions (output unchanged) | Run and checked by me: TO CONFIRM |
| 2026-11-15 | Claude Code: wrote src/rolling_real.py (descriptive rolling 15-year real-data windows, K=20 winner past vs new excess, no model) | Run and checked by me: TO CONFIRM |
| 2026-11-15 | Claude Code: polished chart labels and layout only (calibration.png, real_test.png, rolling_real.png) in train_model.py, real_test.py and rolling_real.py; regenerated the charts | Run and checked by me: TO CONFIRM |
| 2026-11-15 | Claude Code: wrote src/app.py (local Gradio demo 'Strategy Reality Check') and added gradio to requirements.txt | Run and checked by me: TO CONFIRM |
| 2026-11-15 | Claude Code: improved src/app.py (dropdown of the 11 real historical windows from rolling_real.py, 'best of all 20' button that fills L, top-k and K=20, matches-found count) | Run and checked by me: TO CONFIRM |
| 2026-11-15 | Claude Code: small changes to src/app.py (note under the Historical window dropdown about months the simulation also reshuffled; reworded the verdict line) | Run and checked by me: TO CONFIRM |
