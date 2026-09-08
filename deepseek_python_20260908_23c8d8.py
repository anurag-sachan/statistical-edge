#!/usr/bin/env python3

"""
BTCUSD 2H -> Enhanced Breakout & Reversal Analysis (Extended)

New features:
- Per-candle break indicators (C5, C6, C7) for the 8H range.
- Breakout direction accuracy vs. distance to high/low.
- First break candle distribution.
- Pullback magnitude after break.
- Deeper reversal & continuation analysis.
- Zone ranking by expected profit and MAE.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "binance/BTCUSD-2h-2025-12-04_to_2026-09-07.csv"
OUTPUT_DIR = Path("btc_enhanced_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_PATTERN_N = 30
PIP_THRESHOLD = 750  # Target in pips (0.01 = 1 pip for BTC)
SL_PIPS = 50
RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    df = pd.read_csv(INPUT_FILE)
    required = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.dropna().copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    
    print(f"DATA: {len(df)} rows, {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    return df

# ============================================================
# ENHANCED FEATURES (extended)
# ============================================================

def build_enhanced_dataset(df):
    """Build dataset with precise breakout levels, timing, and per-candle breaks."""
    
    rows = []
    
    for i in range(4, len(df) - 3):
        candles = [df.iloc[i-4], df.iloc[i-3], df.iloc[i-2], df.iloc[i-1]]
        future = [df.iloc[i], df.iloc[i+1], df.iloc[i+2]]
        
        C1, C2, C3, C4 = candles
        C5, C6, C7 = future
        
        row = {}
        
        # ============================================================
        # 8-HOUR WINDOW ANALYSIS
        # ============================================================
        
        window_high = max(c["high"] for c in candles)
        window_low = min(c["low"] for c in candles)
        window_range = window_high - window_low
        window_mid = (window_high + window_low) / 2
        
        # C4 position in window (0-100%)
        C4_close_position = (C4["close"] - window_low) / max(window_range, 1e-12)
        row["C4_close_position_pct"] = C4_close_position * 100
        
        # Distance to break levels
        row["C4_dist_to_high_pct"] = (window_high - C4["close"]) / max(window_range, 1e-12) * 100
        row["C4_dist_to_low_pct"] = (C4["close"] - window_low) / max(window_range, 1e-12) * 100
        row["C4_dist_to_high_pips"] = (window_high - C4["close"]) * 10000
        row["C4_dist_to_low_pips"] = (C4["close"] - window_low) * 10000
        
        # ============================================================
        # BREAKOUT LEVEL ANALYSIS - ZONES
        # ============================================================
        
        for pct in [25, 50, 75, 90]:
            upper_level = window_low + (pct / 100) * window_range
            lower_level = window_high - (pct / 100) * window_range
            
            row[f"break_above_{pct}pct"] = int(C4["close"] > upper_level)
            row[f"break_below_{pct}pct"] = int(C4["close"] < lower_level)
            row[f"dist_to_upper_{pct}pct_pct"] = (upper_level - C4["close"]) / max(window_range, 1e-12) * 100
            row[f"dist_to_lower_{pct}pct_pct"] = (C4["close"] - lower_level) / max(window_range, 1e-12) * 100
        
        # ============================================================
        # C3 & C4 HIGH/LOW RELATIONSHIP
        # ============================================================
        
        row["C4_vs_C3_high_pct"] = (C4["high"] - C3["high"]) / max(window_range, 1e-12) * 100
        row["C4_vs_C3_low_pct"] = (C4["low"] - C3["low"]) / max(window_range, 1e-12) * 100
        C3_range = C3["high"] - C3["low"]
        row["C4_close_in_C3_pct"] = (C4["close"] - C3["low"]) / max(C3_range, 1e-12) * 100
        
        # ============================================================
        # WICK ANALYSIS FOR REVERSAL DETECTION
        # ============================================================
        
        def get_wick_info(candle):
            body = abs(candle["close"] - candle["open"])
            upper_wick = candle["high"] - max(candle["open"], candle["close"])
            lower_wick = min(candle["open"], candle["close"]) - candle["low"]
            total_wick = upper_wick + lower_wick
            return {
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "upper_wick_ratio": upper_wick / max(body + total_wick, 1e-12),
                "lower_wick_ratio": lower_wick / max(body + total_wick, 1e-12),
                "wick_body_ratio": total_wick / max(body, 1e-12)
            }
        
        for n, c in enumerate([C5, C6, C7], start=5):
            wick = get_wick_info(c)
            row[f"C{n}_upper_wick_pips"] = wick["upper_wick"] * 10000
            row[f"C{n}_lower_wick_pips"] = wick["lower_wick"] * 10000
            row[f"C{n}_upper_wick_ratio"] = wick["upper_wick_ratio"]
            row[f"C{n}_lower_wick_ratio"] = wick["lower_wick_ratio"]
            row[f"C{n}_wick_body_ratio"] = wick["wick_body_ratio"]
        
        # ============================================================
        # REVERSAL PATTERNS
        # ============================================================
        
        for n, c in enumerate([C5, C6, C7], start=5):
            wick = get_wick_info(c)
            row[f"C{n}_long_upper_wick"] = int(wick["upper_wick_ratio"] > 0.6)
            row[f"C{n}_long_lower_wick"] = int(wick["lower_wick_ratio"] > 0.6)
            row[f"C{n}_rejected_high"] = int(
                c["high"] > window_high and 
                wick["upper_wick_ratio"] > 0.5 and
                c["close"] < window_high
            )
            row[f"C{n}_rejected_low"] = int(
                c["low"] < window_low and 
                wick["lower_wick_ratio"] > 0.5 and
                c["close"] > window_low
            )
        
        # ============================================================
        # PER-CANDLE BREAK INDICATORS (NEW)
        # ============================================================
        
        # Break of the 8H high/low by each future candle
        row["C5_break_high"] = int(C5["high"] > window_high)
        row["C5_break_low"] = int(C5["low"] < window_low)
        row["C6_break_high"] = int(C6["high"] > window_high)
        row["C6_break_low"] = int(C6["low"] < window_low)
        row["C7_break_high"] = int(C7["high"] > window_high)
        row["C7_break_low"] = int(C7["low"] < window_low)
        
        # Close break (for stronger signal)
        row["C5_close_above_high"] = int(C5["close"] > window_high)
        row["C5_close_below_low"] = int(C5["close"] < window_low)
        row["C6_close_above_high"] = int(C6["close"] > window_high)
        row["C6_close_below_low"] = int(C6["close"] < window_low)
        row["C7_close_above_high"] = int(C7["close"] > window_high)
        row["C7_close_below_low"] = int(C7["close"] < window_low)
        
        # First break candle (1=C5, 2=C6, 3=C7, 0=none within C5-C7)
        first_break = 0
        if row["C5_break_high"] or row["C5_break_low"]:
            first_break = 1
        elif row["C6_break_high"] or row["C6_break_low"]:
            first_break = 2
        elif row["C7_break_high"] or row["C7_break_low"]:
            first_break = 3
        row["first_break_candle"] = first_break
        
        # ============================================================
        # C5 CONTINUATION FROM BETWEEN C3/C4 HIGHS/LOWS
        # ============================================================
        
        high_min = min(C3["high"], C4["high"])
        high_max = max(C3["high"], C4["high"])
        low_min = min(C3["low"], C4["low"])
        low_max = max(C3["low"], C4["low"])
        
        row["C5_above_both_highs"] = int(C5["high"] > C3["high"] and C5["high"] > C4["high"])
        row["C5_above_higher_high"] = int(C5["high"] > high_max)
        row["C5_below_both_lows"] = int(C5["low"] < C3["low"] and C5["low"] < C4["low"])
        row["C5_below_lower_low"] = int(C5["low"] < low_min)
        row["C5_high_between_highs"] = int(high_min <= C5["high"] <= high_max)
        row["C5_low_between_lows"] = int(low_min <= C5["low"] <= low_max)
        
        # Break from between the highs/lows (if C5 high > higher high)
        row["C5_breaks_above_from_between"] = int(
            (C3["high"] > C4["high"] and C5["high"] > C3["high"]) or
            (C4["high"] > C3["high"] and C5["high"] > C4["high"])
        )
        
        # ============================================================
        # C3+C4 COMBINED BODY VS WICK (NEW)
        # ============================================================
        
        # Total body of C3 and C4 (absolute)
        body_C3 = abs(C3["close"] - C3["open"])
        body_C4 = abs(C4["close"] - C4["open"])
        total_body = body_C3 + body_C4
        # Total range of C3+C4 (high to low of the two candles)
        combined_high = max(C3["high"], C4["high"])
        combined_low = min(C3["low"], C4["low"])
        combined_range = combined_high - combined_low
        # Wicks: total wick = combined_range - total_body (ignoring gaps)
        total_wick = combined_range - total_body
        row["C3_C4_total_body_pips"] = total_body * 10000
        row["C3_C4_total_wick_pips"] = total_wick * 10000
        row["C3_C4_body_ratio"] = total_body / max(combined_range, 1e-12)
        row["C3_C4_wick_ratio"] = total_wick / max(combined_range, 1e-12)
        
        # ============================================================
        # FUTURE TARGETS - MFE/MAE
        # ============================================================
        
        base_price = C4["close"]
        
        for n, c in enumerate([C5, C6, C7], start=5):
            max_up = (c["high"] - base_price) / base_price * 10000
            max_down = (base_price - c["low"]) / base_price * 10000
            row[f"C{n}_max_up_pips"] = max_up
            row[f"C{n}_max_down_pips"] = max_down
            row[f"C{n}_net_change_pips"] = (c["close"] - base_price) / base_price * 10000
            row[f"C{n}_hit_750_up"] = int(max_up >= 750)
            row[f"C{n}_hit_750_down"] = int(max_down >= 750)
            row[f"C{n}_hit_50_SL"] = int(max_down >= 50 or max_up >= 50)
            row[f"C{n}_time_to_target"] = n - 4
        
        # ============================================================
        # COMBINED TARGET ANALYSIS
        # ============================================================
        
        max_up_all = max(c["high"] for c in [C5, C6, C7]) - base_price
        max_down_all = base_price - min(c["low"] for c in [C5, C6, C7])
        row["max_up_all_pips"] = max_up_all / base_price * 10000
        row["max_down_all_pips"] = max_down_all / base_price * 10000
        row["hit_750_up"] = int(max_up_all >= 750)
        row["hit_750_down"] = int(max_down_all >= 750)
        
        # ============================================================
        # PULLBACK MAGNITUDE (NEW)
        # ============================================================
        
        # After a break up (C5 high > window_high), what is the minimum low within C5-C7 relative to the break?
        # We'll compute the maximum retracement (in pips and % of range) back into the range.
        if row["C5_break_high"] or row["C6_break_high"] or row["C7_break_high"]:
            # Find the first break up and then the subsequent lowest low after that break
            # For simplicity, we consider the overall minimum low of C5-C7 after the first break.
            # But we need to know when the break occurred.
            # We'll compute the minimum low of all future candles after the break (including the break candle)
            # and the maximum high after break (to see if it continues).
            # This is a simplified approach; we can refine later.
            min_low_after_break = min(c["low"] for c in [C5, C6, C7])
            # Pullback from the highest high reached (which is at least window_high)
            max_high_after_break = max(c["high"] for c in [C5, C6, C7])
            pullback_pips = (max_high_after_break - min_low_after_break) * 10000
            pullback_pct_range = pullback_pips / (window_range * 10000) * 100
            row["pullback_pips"] = pullback_pips
            row["pullback_pct_of_range"] = pullback_pct_range
        else:
            row["pullback_pips"] = 0
            row["pullback_pct_of_range"] = 0
        
        # ============================================================
        # OTHER TARGETS
        # ============================================================
        
        row["C5_direction"] = int(C5["close"] > C5["open"])
        row["C6_direction"] = int(C6["close"] > C6["open"])
        row["C7_direction"] = int(C7["close"] > C7["open"])
        
        row["future_break_up"] = int(max(c["high"] for c in [C5, C6, C7]) > window_high)
        row["future_break_down"] = int(min(c["low"] for c in [C5, C6, C7]) < window_low)
        
        row["timestamp"] = C4["timestamp"]
        rows.append(row)
    
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "enhanced_dataset.csv", index=False)
    print(f"Enhanced dataset: {len(result)} rows, {len(result.columns)} features")
    return result

# ============================================================
# NEW: BREAKOUT DIRECTION ACCURACY BY DISTANCE
# ============================================================

def breakout_direction_accuracy(df):
    """Find distance thresholds that give 75% accuracy in predicting breakout direction."""
    print("\n" + "="*70)
    print("BREAKOUT DIRECTION ACCURACY VS. DISTANCE TO RANGE EXTREMES")
    print("="*70)
    
    # Define thresholds (in % of range) for "near" high/low
    thresholds = np.arange(2, 31, 2)  # 2% to 30% in steps of 2%
    results = []
    
    for thresh in thresholds:
        # Near high: distance to high < thresh% of range
        near_high = df["C4_dist_to_high_pct"] < thresh
        # Near low: distance to low < thresh% of range
        near_low = df["C4_dist_to_low_pct"] < thresh
        # Middle: not near either (optional)
        
        if near_high.sum() < MIN_PATTERN_N or near_low.sum() < MIN_PATTERN_N:
            continue
        
        # For near_high, we predict up if actual break up occurs
        subset_high = df[near_high]
        p_break_up = subset_high["future_break_up"].mean()
        p_break_down = subset_high["future_break_down"].mean()
        # Accuracy if we predict up whenever near high: TP = break_up, FP = break_down, FN = break_up? Actually accuracy = P(predict up | actual up) * P(actual up) + P(predict down | actual down) * P(actual down) but we only predict up for near_high, so accuracy is simply P(actual up) because we always predict up. But that's not a good metric; we want P(actual up | near_high). That's the conditional probability.
        # So we report that.
        
        # For near_low, we predict down
        subset_low = df[near_low]
        p_break_down_low = subset_low["future_break_down"].mean()
        
        # Also compute overall accuracy of a rule: predict up if near_high, down if near_low, else random (or neutral)
        # But simpler: we report the conditional probabilities.
        
        results.append({
            "threshold_pct": thresh,
            "n_near_high": len(subset_high),
            "p_break_up_given_near_high": p_break_up,
            "p_break_down_given_near_high": p_break_down,
            "n_near_low": len(subset_low),
            "p_break_down_given_near_low": p_break_down_low,
            "p_break_up_given_near_low": subset_low["future_break_up"].mean()
        })
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "breakout_accuracy_by_distance.csv", index=False)
    
    # Find thresholds where P(break_up|near_high) >= 0.75 or P(break_down|near_low) >= 0.75
    high_75 = result_df[result_df["p_break_up_given_near_high"] >= 0.75]
    low_75 = result_df[result_df["p_break_down_given_near_low"] >= 0.75]
    
    print("\nThresholds where probability of breaking up when near high >= 75%:")
    if not high_75.empty:
        print(high_75[["threshold_pct", "n_near_high", "p_break_up_given_near_high"]].to_string(index=False))
    else:
        print("None found (try lower/higher thresholds or more data).")
    
    print("\nThresholds where probability of breaking down when near low >= 75%:")
    if not low_75.empty:
        print(low_75[["threshold_pct", "n_near_low", "p_break_down_given_near_low"]].to_string(index=False))
    else:
        print("None found.")
    
    return result_df

# ============================================================
# NEW: FIRST BREAK CANDLE DISTRIBUTION
# ============================================================

def first_break_candle_analysis(df):
    """Analyze which candle (C5, C6, C7) is most likely to be the first to break."""
    print("\n" + "="*70)
    print("FIRST BREAK CANDLE DISTRIBUTION")
    print("="*70)
    
    # Only consider cases where a breakout occurs (future_break_up or future_break_down)
    breakout_cases = df[(df["future_break_up"] == 1) | (df["future_break_down"] == 1)]
    if len(breakout_cases) < 10:
        print("Not enough breakout cases.")
        return pd.DataFrame()
    
    # Distribution of first_break_candle (1,2,3)
    dist = breakout_cases["first_break_candle"].value_counts().sort_index()
    total = dist.sum()
    results = []
    for candle in [1,2,3]:
        n = dist.get(candle, 0)
        results.append({
            "candle": f"C{candle+4}",  # C5, C6, C7
            "count": n,
            "pct_of_breakouts": n / total * 100 if total > 0 else 0
        })
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "first_break_candle.csv", index=False)
    print(result_df.to_string(index=False))
    return result_df

# ============================================================
# NEW: PULLBACK ANALYSIS AFTER BREAK
# ============================================================

def pullback_after_break(df):
    """Analyze pullback magnitude after a breakout."""
    print("\n" + "="*70)
    print("PULLBACK MAGNITUDE AFTER BREAKOUT")
    print("="*70)
    
    # Only cases with breakout up or down (we'll treat separately)
    for direction in ["up", "down"]:
        if direction == "up":
            mask = df["future_break_up"] == 1
        else:
            mask = df["future_break_down"] == 1
        subset = df[mask]
        if len(subset) < MIN_PATTERN_N:
            continue
        
        # Pullback statistics
        pullback_pips = subset["pullback_pips"]
        pullback_pct = subset["pullback_pct_of_range"]
        
        print(f"\nDirection: {direction.upper()} (n={len(subset)})")
        print(f"  Median pullback (pips): {pullback_pips.median():.0f}")
        print(f"  Mean pullback (pips): {pullback_pips.mean():.0f}")
        print(f"  Median pullback (% of range): {pullback_pct.median():.1f}%")
        print(f"  Mean pullback (% of range): {pullback_pct.mean():.1f}%")
        print(f"  Percent of cases with pullback > 50 pips: {(pullback_pips > 50).mean():.1%}")
        print(f"  Percent of cases with pullback > 100 pips: {(pullback_pips > 100).mean():.1%}")
    
    # Save detailed stats
    result = df[df["pullback_pips"] > 0][["future_break_up", "future_break_down", "pullback_pips", "pullback_pct_of_range"]].copy()
    result.to_csv(OUTPUT_DIR / "pullback_stats.csv", index=False)

# ============================================================
# EXISTING FUNCTIONS (slightly extended)
# ============================================================

def breakout_zone_analysis(df):
    """Analyze breakout probability from different zones within range (existing)."""
    print("\n" + "="*70)
    print("BREAKOUT PROBABILITY BY ZONE")
    print("="*70)
    
    results = []
    for zone_start in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        for zone_end in [zone_start + 10, zone_start + 20, zone_start + 30]:
            if zone_end > 100:
                continue
            mask = (df["C4_close_position_pct"] >= zone_start) & (df["C4_close_position_pct"] < zone_end)
            subset = df[mask]
            if len(subset) < MIN_PATTERN_N:
                continue
            break_up = subset["future_break_up"].mean()
            break_down = subset["future_break_down"].mean()
            hit_750_up = subset["hit_750_up"].mean()
            hit_750_down = subset["hit_750_down"].mean()
            results.append({
                "zone": f"{zone_start}-{zone_end}%",
                "n": len(subset),
                "break_up": break_up,
                "break_down": break_down,
                "hit_750_up": hit_750_up,
                "hit_750_down": hit_750_down,
                "zone_mid": (zone_start + zone_end) / 2
            })
    
    result = pd.DataFrame(results)
    result = result.sort_values("break_up", ascending=False)
    result.to_csv(OUTPUT_DIR / "breakout_zone_analysis.csv", index=False)
    print("\nTop zones for upside breakout:")
    print(result.head(10).to_string(index=False))
    return result

def reversal_pattern_analysis(df):
    """Analyze reversal patterns with wick analysis (existing)."""
    print("\n" + "="*70)
    print("REVERSAL PATTERN ANALYSIS")
    print("="*70)
    
    patterns = {
        "C5_long_upper_wick": df["C5_long_upper_wick"] == 1,
        "C5_long_lower_wick": df["C5_long_lower_wick"] == 1,
        "C5_rejected_high": df["C5_rejected_high"] == 1,
        "C5_rejected_low": df["C5_rejected_low"] == 1,
        "C6_rejected_high": df["C6_rejected_high"] == 1,
        "C6_rejected_low": df["C6_rejected_low"] == 1,
    }
    results = []
    for name, condition in patterns.items():
        subset = df[condition]
        if len(subset) < MIN_PATTERN_N:
            continue
        next_move_up = df.loc[subset.index, "C6_direction"].mean() if "C6_direction" in df else 0
        hit_750_up = df.loc[subset.index, "hit_750_up"].mean()
        hit_750_down = df.loc[subset.index, "hit_750_down"].mean()
        results.append({
            "pattern": name,
            "n": len(subset),
            "next_candle_up": next_move_up,
            "hit_750_up": hit_750_up,
            "hit_750_down": hit_750_down,
            "reversal_quality": 1 - abs(next_move_up - 0.5) * 2
        })
    result = pd.DataFrame(results)
    result = result.sort_values("reversal_quality", ascending=False)
    result.to_csv(OUTPUT_DIR / "reversal_patterns.csv", index=False)
    print(result.to_string(index=False))
    return result

def continuation_analysis(df):
    """Analyze continuation from 25%, 50%, 75% positions (existing)."""
    print("\n" + "="*70)
    print("CONTINUATION BY RANGE POSITION")
    print("="*70)
    positions = [(0, 25, "bottom_quarter"), (25, 50, "lower_mid"), (50, 75, "upper_mid"), (75, 100, "top_quarter")]
    results = []
    for pos_start, pos_end, pos_name in positions:
        mask = (df["C4_close_position_pct"] >= pos_start) & (df["C4_close_position_pct"] < pos_end)
        subset = df[mask]
        if len(subset) < MIN_PATTERN_N:
            continue
        c5_up = subset["C5_direction"].mean()
        break_up = subset["future_break_up"].mean()
        break_down = subset["future_break_down"].mean()
        hit_750_up = subset["hit_750_up"].mean()
        hit_750_down = subset["hit_750_down"].mean()
        results.append({
            "position": pos_name,
            "n": len(subset),
            "C5_up": c5_up,
            "break_up": break_up,
            "break_down": break_down,
            "hit_750_up": hit_750_up,
            "hit_750_down": hit_750_down,
            "edge": hit_750_up - hit_750_down
        })
    result = pd.DataFrame(results)
    result.to_csv(OUTPUT_DIR / "continuation_analysis.csv", index=False)
    print(result.to_string(index=False))
    return result

def target_optimization(df):
    """Find optimal entry zones for 750+ pip target with 50 pip SL (existing)."""
    print("\n" + "="*70)
    print("TARGET OPTIMIZATION")
    print("="*70)
    results = []
    for zone_pct in [5, 10, 15, 20]:
        upper_mask = df["C4_close_position_pct"] >= (100 - zone_pct)
        lower_mask = df["C4_close_position_pct"] <= zone_pct
        mid_mask = (df["C4_close_position_pct"] >= 40) & (df["C4_close_position_pct"] <= 60)
        for mask, zone_name in [(upper_mask, f"upper_{zone_pct}%"), (lower_mask, f"lower_{zone_pct}%"), (mid_mask, "middle_50%")]:
            subset = df[mask]
            if len(subset) < MIN_PATTERN_N:
                continue
            hit_target = subset["hit_750_up"].mean() if "upper" in zone_name else subset["hit_750_down"].mean()
            hit_sl = 1 - hit_target
            ev = hit_target * PIP_THRESHOLD - (1 - hit_target) * SL_PIPS
            results.append({
                "zone": zone_name,
                "n": len(subset),
                "hit_target_pct": hit_target,
                "hit_sl_pct": hit_sl,
                "expected_value": ev,
                "risk_reward_ratio": hit_target / max(1 - hit_target, 0.01)
            })
    result = pd.DataFrame(results)
    result = result.sort_values("expected_value", ascending=False)
    result.to_csv(OUTPUT_DIR / "target_optimization.csv", index=False)
    print(result.to_string(index=False))
    return result

def mfe_analysis(df):
    """Analyze Maximum Favorable Excursion for different entries (existing)."""
    print("\n" + "="*70)
    print("MAXIMUM FAVORABLE EXCURSION (MFE) ANALYSIS")
    print("="*70)
    zones = [
        ("near_highs", df["C4_close_position_pct"] >= 75),
        ("near_lows", df["C4_close_position_pct"] <= 25),
        ("middle", (df["C4_close_position_pct"] >= 35) & (df["C4_close_position_pct"] <= 65)),
        ("all", pd.Series(True, index=df.index))
    ]
    results = []
    for zone_name, mask in zones:
        subset = df[mask]
        if len(subset) < MIN_PATTERN_N:
            continue
        mfe_up = subset["max_up_all_pips"].median()
        mfe_down = subset["max_down_all_pips"].median()
        for target in [250, 500, 750, 1000, 1500]:
            hit_up = (subset["max_up_all_pips"] >= target).mean()
            hit_down = (subset["max_down_all_pips"] >= target).mean()
            results.append({"zone": zone_name, "direction": "up", "target_pips": target, "hit_probability": hit_up, "n": len(subset)})
            results.append({"zone": zone_name, "direction": "down", "target_pips": target, "hit_probability": hit_down, "n": len(subset)})
    result = pd.DataFrame(results)
    result.to_csv(OUTPUT_DIR / "mfe_analysis.csv", index=False)
    print("\nMFE Summary by Zone:")
    for zone_name, mask in zones:
        subset = df[mask]
        if len(subset) < MIN_PATTERN_N:
            continue
        print(f"\n{zone_name.upper()} (n={len(subset)}):")
        print(f"  Median MFE Up: {subset['max_up_all_pips'].median():.0f} pips")
        print(f"  Median MFE Down: {subset['max_down_all_pips'].median():.0f} pips")
        print(f"  Hit 750 Up: {subset['hit_750_up'].mean():.2%}")
        print(f"  Hit 750 Down: {subset['hit_750_down'].mean():.2%}")
    return result

def both_sides_probability(df):
    """Probability of hitting both sides after breaking one side (existing)."""
    print("\n" + "="*70)
    print("PROBABILITY OF HITTING BOTH SIDES")
    print("="*70)
    break_conditions = {"broke_up": df["future_break_up"] == 1, "broke_down": df["future_break_down"] == 1}
    results = []
    for name, condition in break_conditions.items():
        subset = df[condition]
        if len(subset) < MIN_PATTERN_N:
            continue
        both = (subset["future_break_up"] & subset["future_break_down"]).mean()
        hit_target = subset["hit_750_up"].mean() if "up" in name else subset["hit_750_down"].mean()
        results.append({
            "condition": name,
            "n": len(subset),
            "both_sides_probability": both,
            "hit_target_same_side": hit_target,
            "hit_target_opposite": subset["hit_750_down"].mean() if "up" in name else subset["hit_750_up"].mean()
        })
    result = pd.DataFrame(results)
    result.to_csv(OUTPUT_DIR / "both_sides_probability.csv", index=False)
    print(result.to_string(index=False))
    return result

def find_optimal_zones(df):
    """Find zones with highest probability of 750+ pip moves (existing)."""
    print("\n" + "="*70)
    print("OPTIMAL ZONES FOR 750+ PIP TARGET")
    print("="*70)
    df["position_bin"] = pd.cut(df["C4_close_position_pct"], bins=10, labels=False) * 10 + 5
    zone_results = []
    for bin_val in sorted(df["position_bin"].unique()):
        mask = df["position_bin"] == bin_val
        subset = df[mask]
        if len(subset) < MIN_PATTERN_N:
            continue
        hit_750_up = subset["hit_750_up"].mean()
        hit_750_down = subset["hit_750_down"].mean()
        hit_750_either = hit_750_up + hit_750_down
        hit_sl = 1 - (hit_750_up + hit_750_down)  # simplified
        zone_results.append({
            "zone": f"{bin_val-5}-{bin_val+5}%",
            "n": len(subset),
            "hit_750_up": hit_750_up,
            "hit_750_down": hit_750_down,
            "hit_750_either": hit_750_either,
            "risk_reward": hit_750_either / max(1 - hit_750_either, 0.01),
            "expected_pips": hit_750_either * 750 - (1 - hit_750_either) * 50
        })
    result = pd.DataFrame(zone_results)
    result = result.sort_values("expected_pips", ascending=False)
    result.to_csv(OUTPUT_DIR / "optimal_zones.csv", index=False)
    print(result.to_string(index=False))
    return result

# ============================================================
# NEW: LOW DRAWDOWN ENTRY RANKING
# ============================================================

def low_drawdown_entry_ranking(df):
    """Rank entry conditions by low maximum adverse excursion (MAE) and high win rate."""
    print("\n" + "="*70)
    print("LOW DRAWDOWN ENTRY ZONES (MAE ANALYSIS)")
    print("="*70)
    
    # Define entry zones based on position in range
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df["pos_bin"] = pd.cut(df["C4_close_position_pct"], bins=bins, labels=[f"{bins[i]}-{bins[i+1]}%" for i in range(len(bins)-1)])
    
    results = []
    for zone, group in df.groupby("pos_bin"):
        if len(group) < MIN_PATTERN_N:
            continue
        # Compute win rate for hitting 750 up (or down) – we'll consider both
        win_rate_up = group["hit_750_up"].mean()
        win_rate_down = group["hit_750_down"].mean()
        win_rate_either = (group["hit_750_up"] | group["hit_750_down"]).mean()
        # MAE: maximum adverse excursion (max drawdown) - we can use max_down_all_pips for up entries and max_up_all_pips for down entries
        # For a long entry (buy), the adverse move is max_down_all_pips; for short, max_up_all_pips.
        # We'll compute median MAE for both directions separately.
        mae_up = group["max_down_all_pips"].median()  # worst adverse move for long
        mae_down = group["max_up_all_pips"].median()  # worst adverse move for short
        # Expected profit considering SL=50, target=750, and win rate
        expected_pips = win_rate_either * 750 - (1 - win_rate_either) * 50
        results.append({
            "zone": zone,
            "n": len(group),
            "win_rate_up": win_rate_up,
            "win_rate_down": win_rate_down,
            "win_rate_either": win_rate_either,
            "med_MAE_long_pips": mae_up,
            "med_MAE_short_pips": mae_down,
            "expected_pips": expected_pips,
            "score": win_rate_either / (mae_up + 1)  # simple score: high win, low MAE
        })
    
    result = pd.DataFrame(results)
    result = result.sort_values("score", ascending=False)
    result.to_csv(OUTPUT_DIR / "low_drawdown_zones.csv", index=False)
    print("\nTop zones by low drawdown score:")
    print(result.head(10).to_string(index=False))
    return result

# ============================================================
# MAIN
# ============================================================

def main():
    df = load_data()
    dataset = build_enhanced_dataset(df)
    
    # New analyses
    breakout_direction_accuracy(dataset)
    first_break_candle_analysis(dataset)
    pullback_after_break(dataset)
    low_drawdown_entry_ranking(dataset)
    
    # Existing analyses (extended)
    breakout_zone_analysis(dataset)
    reversal_pattern_analysis(dataset)
    continuation_analysis(dataset)
    target_optimization(dataset)
    mfe_analysis(dataset)
    both_sides_probability(dataset)
    find_optimal_zones(dataset)
    
    print("\n" + "="*70)
    print(f"All results saved to: {OUTPUT_DIR.resolve()}")
    print("="*70)

if __name__ == "__main__":
    main()