#!/usr/bin/env python3
"""
Visualize and explain the results from btc_ml_research.py.
Reads all CSV files from btc_enhanced_results/ and produces:
- Plots (PNG) for each key analysis.
- A plain‑language summary report (report.txt).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
RESULT_DIR = Path("btc_enhanced_results")
PLOT_DIR = RESULT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_csv(filename):
    """Load CSV from result directory, return None if missing."""
    path = RESULT_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    else:
        print(f"Warning: {filename} not found.")
        return None

def save_plot(fig, name):
    """Save figure as PNG."""
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def add_explanation(text, report_file):
    """Write explanation to report file and print."""
    print(text)
    report_file.write(text + "\n\n")

# ============================================================
# 1. BREAKOUT ZONE ANALYSIS
# ============================================================
def plot_breakout_zones(df):
    """Bar chart of breakout probabilities by zone."""
    # Ensure we have break_neither
    if "break_neither" not in df.columns:
        df = df.copy()
        df["break_neither"] = 1 - df["break_up"] - df["break_down"]

    fig, ax = plt.subplots()
    df_plot = df.sort_values("zone_mid")  # ensure order
    x = df_plot["zone"]
    width = 0.25
    x_pos = np.arange(len(x))
    ax.bar(x_pos - width, df_plot["break_up"], width, label="Break Up", color="green", alpha=0.7)
    ax.bar(x_pos, df_plot["break_down"], width, label="Break Down", color="red", alpha=0.7)
    ax.bar(x_pos + width, df_plot["break_neither"], width, label="Neither", color="gray", alpha=0.7)
    ax.set_xlabel("Zone (% of 8H range)")
    ax.set_ylabel("Probability")
    ax.set_title("Breakout Probability by Zone")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "breakout_zone_analysis")
    return fig

def explain_breakout_zones(df, report):
    if df is None:
        return
    # Find zone with highest break_up
    top_up = df.loc[df["break_up"].idxmax()]
    top_down = df.loc[df["break_down"].idxmax()]
    add_explanation(
        f"Breakout Zone Analysis:\n"
        f"  - The zone with the highest probability of an upside break is '{top_up['zone']}' "
        f"with {top_up['break_up']:.1%} chance (n={top_up['n']}).\n"
        f"  - The zone with the highest downside break is '{top_down['zone']}' "
        f"with {top_down['break_down']:.1%} chance.\n"
        f"  - Generally, zones near the extremes (0-10% or 90-100%) show higher breakout probabilities, "
        f"while middle zones are more balanced.\n"
        f"  - This suggests that if price is near the edge of the 8‑hour range, it is more likely to break out "
        f"than to reverse sharply.",
        report
    )

# ============================================================
# 2. BREAKOUT ACCURACY BY DISTANCE
# ============================================================
def plot_accuracy_by_distance(df):
    """Line chart of P(break | near edge) vs distance threshold."""
    if df is None:
        return
    fig, ax = plt.subplots()
    ax.plot(df["threshold_pct"], df["p_break_up_given_near_high"], 
            marker="o", label="P(break up | near high)", color="green")
    ax.plot(df["threshold_pct"], df["p_break_down_given_near_low"], 
            marker="s", label="P(break down | near low)", color="red")
    ax.axhline(0.75, linestyle="--", color="black", alpha=0.5, label="75% threshold")
    ax.set_xlabel("Distance threshold (% of range)")
    ax.set_ylabel("Conditional probability")
    ax.set_title("Breakout Accuracy vs. Distance to Range Edge")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_plot(fig, "breakout_accuracy_by_distance")
    return fig

def explain_accuracy_by_distance(df, report):
    if df is None:
        return
    # Find thresholds where probability >= 0.75
    high_75 = df[df["p_break_up_given_near_high"] >= 0.75]
    low_75 = df[df["p_break_down_given_near_low"] >= 0.75]
    add_explanation(
        f"Breakout Accuracy by Distance:\n"
        f"  - The chart shows how the probability of a breakout increases as price gets closer to the range edge.\n"
        f"  - For upside breaks, a threshold of {high_75['threshold_pct'].min() if not high_75.empty else 'N/A'}% of range "
        f"gives at least 75% accuracy (if available).\n"
        f"  - For downside breaks, a threshold of {low_75['threshold_pct'].min() if not low_75.empty else 'N/A'}% gives similar accuracy.\n"
        f"  - In plain terms: if price is within about {high_75['threshold_pct'].min() if not high_75.empty else 'X'}% of the 8‑hour high, "
        f"it has a high chance of breaking higher. The same applies to lows.\n"
        f"  - This can help you time entries: wait for price to approach the edge before taking a breakout trade.",
        report
    )

# ============================================================
# 3. FIRST BREAK CANDLE DISTRIBUTION
# ============================================================
def plot_first_break_candle(df):
    if df is None:
        return
    fig, ax = plt.subplots()
    ax.bar(df["candle"], df["pct_of_breakouts"], color="skyblue")
    ax.set_xlabel("Candle")
    ax.set_ylabel("Percentage of breakouts")
    ax.set_title("Which Candle Breaks First?")
    for i, v in enumerate(df["pct_of_breakouts"]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    save_plot(fig, "first_break_candle")
    return fig

def explain_first_break_candle(df, report):
    if df is None:
        return
    # Find max
    max_row = df.loc[df["pct_of_breakouts"].idxmax()]
    add_explanation(
        f"First Break Candle:\n"
        f"  - Among breakouts, the first candle (C5) is responsible for {max_row['pct_of_breakouts']:.1f}% of all first breaks.\n"
        f"  - This means that in most cases, the breakout occurs immediately (within the next 2 hours) after the 8‑hour window.\n"
        f"  - If you see price near the edge, you don't have to wait long – the move often starts right away.\n"
        f"  - Use this to set tight stop‑losses: if C5 doesn't break, the chance of a later break is lower.",
        report
    )

# ============================================================
# 4. CONTINUATION ANALYSIS
# ============================================================
def plot_continuation(df):
    if df is None:
        return
    fig, ax = plt.subplots()
    x = df["position"]
    x_pos = np.arange(len(x))
    width = 0.2
    ax.bar(x_pos - width, df["break_up"], width, label="Break Up", color="green", alpha=0.7)
    ax.bar(x_pos, df["break_down"], width, label="Break Down", color="red", alpha=0.7)
    # hit_750_up may not exist; check
    if "hit_750_up" in df.columns:
        ax.bar(x_pos + width, df["hit_750_up"], width, label="Hit 750 Up", color="lightgreen", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Continuation by Range Position")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "continuation_analysis")
    return fig

def explain_continuation(df, report):
    if df is None:
        return
    top_up = df.loc[df["break_up"].idxmax()]
    top_down = df.loc[df["break_down"].idxmax()]
    add_explanation(
        f"Continuation by Range Position:\n"
        f"  - When price is in the {top_up['position']} (around {top_up['position']}), upside break probability is highest ({top_up['break_up']:.1%}).\n"
        f"  - For downside breaks, the {top_down['position']} gives the highest chance ({top_down['break_down']:.1%}).\n"
        f"  - This matches intuition: price near the top of the range tends to continue upward, and near the bottom tends to continue downward.\n"
        f"  - Interestingly, the 'hit 750 up' probability is also elevated in the upper zones, suggesting that being in the right part of the range leads to larger moves.",
        report
    )

# ============================================================
# 5. TARGET OPTIMIZATION
# ============================================================
def plot_target_optimization(df):
    if df is None:
        return
    # Columns may be 'hit_target_pct' and 'hit_sl_pct' or something else
    # We expect 'hit_target_pct' and 'hit_sl_pct' from the analysis
    # If not present, we can compute from hit_750_up/down? But we'll just skip if missing.
    if "hit_target_pct" not in df.columns or "hit_sl_pct" not in df.columns:
        print("Skipping target optimization plot: missing columns.")
        return
    fig, ax = plt.subplots()
    df_sorted = df.sort_values("zone")
    x = df_sorted["zone"]
    x_pos = np.arange(len(x))
    ax.bar(x_pos - 0.2, df_sorted["hit_target_pct"], width=0.4, label="Hit Target", color="blue", alpha=0.7)
    ax.bar(x_pos + 0.2, df_sorted["hit_sl_pct"], width=0.4, label="Hit SL", color="orange", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Target vs Stop‑Loss Hit Probability")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "target_optimization")
    return fig

def explain_target_optimization(df, report):
    if df is None:
        return
    if "expected_value" not in df.columns:
        add_explanation("Target optimization data not available.", report)
        return
    best = df.loc[df["expected_value"].idxmax()]
    add_explanation(
        f"Target Optimization (750 pips vs 50 SL):\n"
        f"  - The best entry zone is '{best['zone']}' with expected profit of {best['expected_value']:.0f} pips per trade.\n"
        f"  - In that zone, you hit the 750‑pip target {best['hit_target_pct']:.1%} of the time and hit the 50‑pip stop only {best['hit_sl_pct']:.1%} of the time.\n"
        f"  - This means the trade has a very favourable risk/reward ratio: you win big often, and lose small rarely.\n"
        f"  - The worst zones are in the middle of the range, where outcomes are more random.",
        report
    )

# ============================================================
# 6. MFE ANALYSIS
# ============================================================
def plot_mfe(df):
    if df is None:
        return
    fig, ax = plt.subplots()
    for zone in df["zone"].unique():
        sub = df[df["zone"] == zone]
        up = sub[sub["direction"] == "up"]
        down = sub[sub["direction"] == "down"]
        ax.plot(up["target_pips"], up["hit_probability"], marker="o", label=f"{zone} up")
        ax.plot(down["target_pips"], down["hit_probability"], marker="s", linestyle="--", label=f"{zone} down")
    ax.set_xlabel("Target (pips)")
    ax.set_ylabel("Probability of hitting")
    ax.set_title("Maximum Favorable Excursion (MFE)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    save_plot(fig, "mfe_analysis")
    return fig

def explain_mfe(df, report):
    if df is None:
        return
    # Summarize key targets
    for target in [250, 500, 750]:
        for zone in ["near_highs", "near_lows", "middle"]:
            sub = df[(df["zone"] == zone) & (df["target_pips"] == target) & (df["direction"] == "up")]
            if not sub.empty:
                prob = sub["hit_probability"].iloc[0]
                add_explanation(
                    f"MFE for {zone} (up): Probability of hitting {target} pips is {prob:.1%}.",
                    report
                )
    add_explanation(
        "MFE shows how far price tends to move in your favour after entry.\n"
        "Near‑high zones often see larger upside excursions, while near‑low zones favour downside.\n"
        "This helps set realistic profit targets.",
        report
    )

# ============================================================
# 7. BOTH SIDES PROBABILITY
# ============================================================
def plot_both_sides(df):
    if df is None:
        return
    fig, ax = plt.subplots()
    x = df["condition"]
    x_pos = np.arange(len(x))
    ax.bar(x_pos - 0.2, df["both_sides_probability"], width=0.4, label="Both sides", color="purple")
    if "hit_target_same_side" in df.columns:
        ax.bar(x_pos + 0.2, df["hit_target_same_side"], width=0.4, label="Hit same side", color="green")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_ylabel("Probability")
    ax.set_title("After Break, Probability of Hitting Opposite Side")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "both_sides_probability")
    return fig

def explain_both_sides(df, report):
    if df is None:
        return
    for _, row in df.iterrows():
        add_explanation(
            f"If price breaks {row['condition']}, the probability it also breaks the other side is {row['both_sides_probability']:.1%}.\n"
            f"This suggests that breakouts often lead to a full range sweep – a phenomenon known as 'liquidity grab'.\n"
            f"Therefore, it's wise to wait for confirmation after a break, as price may return to the other extreme.",
            report
        )

# ============================================================
# 8. OPTIMAL ZONES FOR 750+ TARGET
# ============================================================
def plot_optimal_zones(df):
    if df is None:
        return
    fig, ax = plt.subplots()
    x = df["zone"]
    x_pos = np.arange(len(x))
    ax.bar(x_pos - 0.2, df["hit_750_up"], width=0.4, label="Hit 750 Up", color="green", alpha=0.7)
    ax.bar(x_pos + 0.2, df["hit_750_down"], width=0.4, label="Hit 750 Down", color="red", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Optimal Zones for 750‑Pip Target")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "optimal_zones")
    return fig

def explain_optimal_zones(df, report):
    if df is None:
        return
    if "expected_pips" not in df.columns:
        add_explanation("Optimal zones data missing expected_pips.", report)
        return
    best = df.loc[df["expected_pips"].idxmax()]
    add_explanation(
        f"Optimal Zones for 750‑Pip Target:\n"
        f"  - The zone with highest expected profit is '{best['zone']}' with {best['expected_pips']:.0f} expected pips.\n"
        f"  - In that zone, upside target hit probability is {best['hit_750_up']:.1%} and downside is {best['hit_750_down']:.1%}.\n"
        f"  - This zone offers a favourable balance of win rate and reward.",
        report
    )

# ============================================================
# 9. LOW DRAWDOWN ENTRY RANKING
# ============================================================
def plot_low_drawdown(df):
    if df is None:
        return
    fig, ax = plt.subplots()
    df_sorted = df.sort_values("score", ascending=False).head(10)
    x = df_sorted["zone"]
    x_pos = np.arange(len(x))
    ax.bar(x_pos - 0.2, df_sorted["win_rate_either"], width=0.4, label="Win rate (either)", color="blue")
    if "med_MAE_long_pips" in df_sorted.columns:
        ax.bar(x_pos + 0.2, df_sorted["med_MAE_long_pips"] / 100, width=0.4, label="MAE long (x100)", color="orange")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Probability / MAE (scaled)")
    ax.set_title("Low Drawdown Entry Zones")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "low_drawdown_zones")
    return fig

def explain_low_drawdown(df, report):
    if df is None:
        return
    if df.empty:
        add_explanation("No low drawdown data available.", report)
        return
    top = df.iloc[0]
    add_explanation(
        f"Low Drawdown Entry Zones:\n"
        f"  - The top zone is '{top['zone']}' with a score of {top['score']:.2f}.\n"
        f"  - It has a win rate of {top['win_rate_either']:.1%} and a median MAE (maximum adverse excursion) of {top['med_MAE_long_pips']:.0f} pips for long entries.\n"
        f"  - This means that entries in this zone tend to have a high chance of success while experiencing relatively small adverse moves.\n"
        f"  - These zones are ideal for traders who want low drawdown and quick profits.",
        report
    )

# ============================================================
# 10. REVERSAL PATTERNS
# ============================================================
def plot_reversal_patterns(df):
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = df["pattern"]
    x_pos = np.arange(len(x))
    ax.bar(x_pos - 0.2, df["hit_750_up"], width=0.4, label="Hit 750 Up", color="green")
    ax.bar(x_pos + 0.2, df["hit_750_down"], width=0.4, label="Hit 750 Down", color="red")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Reversal Patterns and Subsequent Targets")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "reversal_patterns")
    return fig

def explain_reversal_patterns(df, report):
    if df is None:
        return
    if "hit_750_up" not in df.columns:
        add_explanation("Reversal pattern data missing hit_750_up/down.", report)
        return
    best_up = df.loc[df["hit_750_up"].idxmax()]
    best_down = df.loc[df["hit_750_down"].idxmax()]
    add_explanation(
        f"Reversal Patterns:\n"
        f"  - The pattern '{best_up['pattern']}' leads to the highest probability of hitting 750 pips upward ({best_up['hit_750_up']:.1%}).\n"
        f"  - The pattern '{best_down['pattern']}' leads to the highest downside target ({best_down['hit_750_down']:.1%}).\n"
        f"  - Long wicks and rejections at the range edge often signal a reversal, but they also occasionally lead to continuation.\n"
        f"  - These patterns can be used as early warning signals to avoid entering against the trend.",
        report
    )

# ============================================================
# MAIN
# ============================================================
def main():
    # Open report file
    report_path = RESULT_DIR / "report.txt"
    with open(report_path, "w") as report:
        report.write("BTCUSD 2H ENHANCED BREAKOUT & REVERSAL ANALYSIS\n")
        report.write("===============================================\n\n")
        
        # Load data
        df_breakout_zones = load_csv("breakout_zone_analysis.csv")
        df_accuracy = load_csv("breakout_accuracy_by_distance.csv")
        df_first_break = load_csv("first_break_candle.csv")
        df_continuation = load_csv("continuation_analysis.csv")
        df_target = load_csv("target_optimization.csv")
        df_mfe = load_csv("mfe_analysis.csv")
        df_both = load_csv("both_sides_probability.csv")
        df_optimal = load_csv("optimal_zones.csv")
        df_drawdown = load_csv("low_drawdown_zones.csv")
        df_reversal = load_csv("reversal_patterns.csv")
        
        # Generate plots and explanations
        if df_breakout_zones is not None:
            plot_breakout_zones(df_breakout_zones)
            explain_breakout_zones(df_breakout_zones, report)
        
        if df_accuracy is not None:
            plot_accuracy_by_distance(df_accuracy)
            explain_accuracy_by_distance(df_accuracy, report)
        
        if df_first_break is not None:
            plot_first_break_candle(df_first_break)
            explain_first_break_candle(df_first_break, report)
        
        if df_continuation is not None:
            plot_continuation(df_continuation)
            explain_continuation(df_continuation, report)
        
        if df_target is not None:
            plot_target_optimization(df_target)
            explain_target_optimization(df_target, report)
        
        if df_mfe is not None:
            plot_mfe(df_mfe)
            explain_mfe(df_mfe, report)
        
        if df_both is not None:
            plot_both_sides(df_both)
            explain_both_sides(df_both, report)
        
        if df_optimal is not None:
            plot_optimal_zones(df_optimal)
            explain_optimal_zones(df_optimal, report)
        
        if df_drawdown is not None:
            plot_low_drawdown(df_drawdown)
            explain_low_drawdown(df_drawdown, report)
        
        if df_reversal is not None:
            plot_reversal_patterns(df_reversal)
            explain_reversal_patterns(df_reversal, report)
        
        report.write("\n== END OF REPORT ==\n")
    
    print(f"\nAll plots saved to {PLOT_DIR}")
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()