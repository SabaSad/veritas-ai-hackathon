"""Strategy Reality Check: a small local demo. Run with:  python src/app.py

Pick a historical window (15 real years: 10 'past' + 5 'new'), a momentum strategy, and how many
strategies you tried before picking it. The app shows how it did in the past, what 'winners like
yours' did afterwards in simulated no-skill markets, and (on request) what really happened next.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw charts without opening a window
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.linear_model import LinearRegression
from backtest import SECTORS, monthly_returns, backtest        # same backtest as everywhere else
from simulate import momentum, excess_and_te, run_strategies, N_PAST   # same strategy + excess definitions

FEATURES = ["K", "past_excess", "past_tracking_error", "past_excess_over_te", "L", "top_k"]
DISCLAIMER = ("*Prototype, not financial advice. Simulated markets reshuffle real 2000-2020 months; "
              "results on real and simulated data are shown separately.*")
WINDOW, STEP = 180, 12    # each window = 180 months (120 past + 60 new); windows are 12 months apart

# --- Start-up: load data, build the 11 real windows, fit the model once --------------------------
prices = pd.read_csv("data/processed/prices.csv", index_col=0, parse_dates=True)
real = monthly_returns(prices).loc[:"2025-12-31"].dropna()     # real months, Feb 2000 - Dec 2025

# Same windows as rolling_real.py: the last one ends in Dec 2025, then step back 12 months at a time.
starts = list(range(len(real) - WINDOW, -1, -STEP))[::-1]       # oldest window first
WINDOWS = {}
for s in starts:
    d = real.index[s:s + WINDOW]                                 # the 180 month-end dates of this window
    label = f"Past {d[0].year}-{d[N_PAST - 1].year}, new {d[N_PAST].year}-{d[-1].year}"
    market = real.iloc[s:s + WINDOW].reset_index(drop=True)      # months numbered 0..179, real order
    # Run all 20 strategies once per window (used by the 'best of all 20' button).
    WINDOWS[label] = dict(dates=d, market=market, results=run_strategies(market))
DEFAULT_WINDOW = next(iter(WINDOWS))                              # 'Past 2001-2010, new 2011-2015'

# Simulated no-skill winners. The linear model is fitted on ALL of them (never on real data).
sim = pd.read_csv("data/processed/sim_dataset.csv")
model = LinearRegression().fit(sim[FEATURES], sim["new_excess"])


def analyse(label, L, k, K):
    """Everything we can say about one strategy in one window. Decisions use only past months."""
    w = WINDOWS[label]
    spy = w["market"]["SPY"]
    r = backtest(w["market"][SECTORS], momentum(L, k))          # strategy monthly returns (no look-ahead)
    past_ex, past_te = excess_and_te(r.loc[:N_PAST - 1], spy)   # past excess vs SPY + tracking error
    new_ex, _ = excess_and_te(r.loc[N_PAST:], spy)              # real 'new' excess (hidden until reveal)

    # Simulated winners with the same K and a similar past excess (within 1 point; widen to 2 if < 30).
    tol, widened = 0.01, False
    similar = sim[(sim["K"] == K) & ((sim["past_excess"] - past_ex).abs() <= tol)]
    if len(similar) < 30:
        tol, widened = 0.02, True
        similar = sim[(sim["K"] == K) & ((sim["past_excess"] - past_ex).abs() <= tol)]

    # The model's estimate of this strategy's new-data excess, from past-only information.
    row = pd.DataFrame([[K, past_ex, past_te, past_ex / past_te, L, k]], columns=FEATURES)
    return dict(r=r, past_ex=past_ex, new_ex=new_ex, similar=similar, widened=widened, tol=tol,
                estimate=float(model.predict(row)[0]))


def chart(label, a, reveal):
    """Growth of $1 in the strategy vs SPY. Past only, or the whole 15 years after the reveal."""
    w = WINDOWS[label]
    r = a["r"] if reveal else a["r"].loc[:N_PAST - 1]
    x = w["dates"][r.index]                                       # turn month numbers back into dates
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, (1 + r).cumprod(), label="Your strategy")
    ax.plot(x, (1 + w["market"]["SPY"].loc[r.index]).cumprod(), label="SPY (buy and hold)")
    years = label.replace("Past ", "").split(", new ")            # e.g. ['2001-2010', '2011-2015']
    if reveal:
        ax.axvline(w["dates"][N_PAST], color="gray", ls="--")
        ax.text(w["dates"][N_PAST], ax.get_ylim()[1] * 0.97, f" {years[1]} (revealed)", va="top", fontsize=8)
    ax.set_ylabel("Growth of $1")
    ax.set_title(f"Backtest {years[0]}" + (f" + what happened {years[1]}" if reveal else ""))
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def run(label, L, k, K, reveal=False):
    """Build the chart and the text for the chosen window and strategy."""
    a = analyse(label, int(L), int(k), int(K))
    X, s = a["past_ex"], a["similar"]
    past_years, new_years = label.replace("Past ", "").split(", new ")
    lines = [f"**Past ({past_years}): {X:+.1%} a year vs SPY.**",
             f"If you trusted the backtest, you would expect: **{X:+.1%} a year vs SPY**."]
    lines.append(f"**Matches found: {len(s)}** simulated winners with the same K={K} and a past excess "
                 f"within {a['tol']*100:.0f} point{'s' if a['tol'] > 0.01 else ''} of yours.")
    if len(s):
        Y = s["new_excess"].mean()
        lo, hi = s["new_excess"].quantile([0.1, 0.9])
        note = ""
        if a["widened"]:   # say so when we had to widen the match, and when it is still thin
            note = " I widened the match from 1 to 2 points because fewer than 30 cases were within 1 point."
            if len(s) < 30:
                note += f" Only {len(s)} cases even then, so treat this with caution."
        lines.append(f"In simulated markets with no skill, winners like yours averaged **{Y:+.1%}** "
                     f"afterwards (10th-90th percentile: {lo:+.1%} to {hi:+.1%}).{note}")
        verdict = (f"In no-skill simulated markets, about **{1 - Y / X:.0%}** of an edge like this usually disappears. " if X > 0
                   else "The past excess is not positive, so there is no edge to lose. ")
        verdict += "The past number is a poor guide."
    else:
        verdict = "The past number is a poor guide."
    lines.append(f"The model's estimate for {new_years}: **{a['estimate']:+.1%}** a year vs SPY.")
    lines.append(f"**Verdict:** {verdict}")
    if reveal:
        lines.append(f"**What actually happened in {new_years} (real data): {a['new_ex']:+.1%} a year vs SPY.**")
    return chart(label, a, reveal), "\n\n".join(lines)


def best_of_all(label):
    """Look only at the PAST of this window and return the strategy with the highest past excess, with K=20."""
    results = WINDOWS[label]["results"]
    L, k = max(results, key=lambda key: results[key][0][0])      # results[key][0][0] = past excess
    return str(L), str(k), "20"


# --- The web page --------------------------------------------------------------------------------
with gr.Blocks(title="Strategy Reality Check") as demo:
    gr.Markdown("# Strategy Reality Check\nWould a strategy that beat SPY in the past keep beating it?")
    gr.Markdown(DISCLAIMER)
    window = gr.Dropdown(list(WINDOWS), value=DEFAULT_WINDOW, label="Historical window")
    gr.Markdown("<small>Windows whose new years end in 2020 or earlier use months that the simulation also "
                "reshuffled. Only the last window (Past 2011-2020, new 2021-2025) is fully unseen by the "
                "simulation.</small>")
    with gr.Row():
        L = gr.Dropdown(["1", "3", "6", "12"], value="12", label="Lookback L (months)")
        k = gr.Dropdown(["1", "2", "3", "4", "5"], value="3", label="Top-k (how many sector ETFs to hold)")
        K = gr.Dropdown(["1", "3", "5", "10", "20"], value="20", label="K: how many strategies did you try before picking this one?")
    best = gr.Button("Pick the best of all 20 strategies for this window")
    check = gr.Button("Check my strategy", variant="primary")
    plot, text = gr.Plot(label="Growth of $1"), gr.Markdown()
    reveal = gr.Button("Reveal what actually happened in the new years")
    best.click(best_of_all, [window], [L, k, K], api_name="best")                                  # fills L, top-k, K=20
    check.click(lambda w, L, k, K: run(w, L, k, K, False), [window, L, k, K], [plot, text], api_name="check")   # past only
    reveal.click(lambda w, L, k, K: run(w, L, k, K, True), [window, L, k, K], [plot, text], api_name="reveal") # adds new years

if __name__ == "__main__":
    demo.launch()   # local only (127.0.0.1), no hosting
