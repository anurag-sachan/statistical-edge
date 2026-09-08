#!/usr/bin/env python3

"""
BTCUSD 2H -> 2H/4H statistical edge + ML research.

Input:
    BTCUSD-2h-2025-12-04_to_2026-09-07.csv

Expected columns:
    timestamp, open, high, low, close, volume, trades

The script:

- Builds rolling 4 x 2H input windows.
- Aggregates C1+C2 and C3+C4 into 4H candles.
- Generates candle-structure features.
- Generates continuation/reversal/engulfing/consolidation features.
- Predicts C5/C6/C7.
- Tests conditional relationships.
- Runs walk-forward ML.
- Saves reports/results.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    mannwhitneyu,
)

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "binance/BTCUSD-2h-2025-12-04_to_2026-09-07.csv"

OUTPUT_DIR = Path("btc_ml_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_PATTERN_N = 50

TRAIN_FRACTION = 0.70
N_WALK_FORWARD_SPLITS = 5

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(INPUT_FILE)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    for c in required:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna().copy()

    # Binance timestamps are microseconds.
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="us",
        utc=True
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    print()
    print("DATA")
    print("=" * 70)
    print("Rows:", len(df))
    print("Start:", df["datetime"].iloc[0])
    print("End:", df["datetime"].iloc[-1])

    return df


# ============================================================
# CANDLE FEATURES
# ============================================================

def candle_features(prefix, o, h, l, c, v, trades):

    rng = max(h - l, 1e-12)

    body = abs(c - o)

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    close_location = (c - l) / rng
    open_location = (o - l) / rng

    return {
        f"{prefix}_open": o,
        f"{prefix}_high": h,
        f"{prefix}_low": l,
        f"{prefix}_close": c,

        f"{prefix}_range": rng,
        f"{prefix}_body": body,
        f"{prefix}_body_ratio": body / rng,

        f"{prefix}_upper_wick": upper_wick,
        f"{prefix}_lower_wick": lower_wick,

        f"{prefix}_upper_wick_ratio": upper_wick / rng,
        f"{prefix}_lower_wick_ratio": lower_wick / rng,

        f"{prefix}_close_location": close_location,
        f"{prefix}_open_location": open_location,

        f"{prefix}_return": (c / o) - 1,

        f"{prefix}_volume": v,
        f"{prefix}_trades": trades,
    }


# ============================================================
# AGGREGATE 2H -> 4H
# ============================================================

def aggregate_4h(c1, c2):

    o = c1["open"]
    h = max(c1["high"], c2["high"])
    l = min(c1["low"], c2["low"])
    c = c2["close"]

    v = c1["volume"] + c2["volume"]
    trades = c1["trades"] + c2["trades"]

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "trades": trades,
    }


# ============================================================
# MAIN FEATURE CONSTRUCTION
# ============================================================

def build_dataset(df):

    rows = []

    for i in range(4, len(df) - 3):

        candles = [
            df.iloc[i - 4],
            df.iloc[i - 3],
            df.iloc[i - 2],
            df.iloc[i - 1],
        ]

        future = [
            df.iloc[i],
            df.iloc[i + 1],
            df.iloc[i + 2],
        ]

        C1, C2, C3, C4 = candles
        C5, C6, C7 = future

        row = {}

        # ----------------------------------------------------
        # Individual 2H candles
        # ----------------------------------------------------

        for n, c in enumerate(candles, start=1):

            row.update(
                candle_features(
                    f"C{n}",
                    c["open"],
                    c["high"],
                    c["low"],
                    c["close"],
                    c["volume"],
                    c["trades"],
                )
            )

        # ----------------------------------------------------
        # 4H candles
        # ----------------------------------------------------

        H1 = aggregate_4h(C1, C2)
        H2 = aggregate_4h(C3, C4)

        row.update(
            candle_features(
                "H4_1",
                H1["open"],
                H1["high"],
                H1["low"],
                H1["close"],
                H1["volume"],
                H1["trades"],
            )
        )

        row.update(
            candle_features(
                "H4_2",
                H2["open"],
                H2["high"],
                H2["low"],
                H2["close"],
                H2["volume"],
                H2["trades"],
            )
        )

        # ----------------------------------------------------
        # 4H relationship
        # ----------------------------------------------------

        h4_range_1 = H1["high"] - H1["low"]
        h4_range_2 = H2["high"] - H2["low"]

        row["H4_range_ratio"] = (
            h4_range_2 /
            max(h4_range_1, 1e-12)
        )

        row["H4_return_change"] = (
            H2["close"] / H1["close"] - 1
        )

        row["H4_same_direction"] = int(
            np.sign(H1["close"] - H1["open"])
            ==
            np.sign(H2["close"] - H2["open"])
        )

        row["H4_opposite_direction"] = int(
            np.sign(H1["close"] - H1["open"])
            !=
            np.sign(H2["close"] - H2["open"])
        )

        # H4 engulfing

        row["H4_bullish_engulf"] = int(
            H2["close"] > H2["open"]
            and H1["close"] < H1["open"]
            and H2["open"] <= H1["close"]
            and H2["close"] >= H1["open"]
        )

        row["H4_bearish_engulf"] = int(
            H2["close"] < H2["open"]
            and H1["close"] > H1["open"]
            and H2["open"] >= H1["close"]
            and H2["close"] <= H1["open"]
        )

        # Inside bar

        row["H4_inside"] = int(
            H2["high"] <= H1["high"]
            and H2["low"] >= H1["low"]
        )

        # ----------------------------------------------------
        # First four candle range
        # ----------------------------------------------------

        window_high = max(c["high"] for c in candles)
        window_low = min(c["low"] for c in candles)

        window_range = (
            window_high -
            window_low
        )

        window_mid = (
            window_high +
            window_low
        ) / 2

        row["window_high"] = window_high
        row["window_low"] = window_low
        row["window_range"] = window_range

        row["C4_close_position"] = (
            C4["close"] - window_low
        ) / max(window_range, 1e-12)

        # ----------------------------------------------------
        # Range compression / expansion
        # ----------------------------------------------------

        ranges = [
            C1["high"] - C1["low"],
            C2["high"] - C2["low"],
            C3["high"] - C3["low"],
            C4["high"] - C4["low"],
        ]

        row["C2_C1_range_ratio"] = (
            ranges[1] / max(ranges[0], 1e-12)
        )

        row["C3_C2_range_ratio"] = (
            ranges[2] / max(ranges[1], 1e-12)
        )

        row["C4_C3_range_ratio"] = (
            ranges[3] / max(ranges[2], 1e-12)
        )

        row["C4_vs_avg_range"] = (
            ranges[3] /
            max(np.mean(ranges[:3]), 1e-12)
        )

        row["range_trend"] = np.polyfit(
            np.arange(4),
            ranges,
            1
        )[0]

        # ----------------------------------------------------
        # C3/C4 relationships
        # ----------------------------------------------------

        row["C4_break_C3_high"] = int(
            C4["high"] > C3["high"]
        )

        row["C4_break_C3_low"] = int(
            C4["low"] < C3["low"]
        )

        row["C4_close_above_C3_high"] = int(
            C4["close"] > C3["high"]
        )

        row["C4_close_below_C3_low"] = int(
            C4["close"] < C3["low"]
        )

        row["C4_inside_C3"] = int(
            C4["high"] <= C3["high"]
            and
            C4["low"] >= C3["low"]
        )

        row["C3_C4_return"] = (
            C4["close"] /
            C3["close"] - 1
        )

        row["C4_body_vs_C3_body"] = (
            abs(C4["close"] - C4["open"])
            /
            max(
                abs(C3["close"] - C3["open"]),
                1e-12
            )
        )

        row["C4_volume_ratio"] = (
            C4["volume"] /
            max(C3["volume"], 1e-12)
        )

        row["C4_trades_ratio"] = (
            C4["trades"] /
            max(C3["trades"], 1e-12)
        )

        # ----------------------------------------------------
        # Continuation
        # ----------------------------------------------------

        directions = []

        for c in candles:

            directions.append(
                np.sign(
                    c["close"] -
                    c["open"]
                )
            )

        row["all_bullish"] = int(
            all(x > 0 for x in directions)
        )

        row["all_bearish"] = int(
            all(x < 0 for x in directions)
        )

        row["last2_bullish"] = int(
            directions[-1] > 0
            and directions[-2] > 0
        )

        row["last2_bearish"] = int(
            directions[-1] < 0
            and directions[-2] < 0
        )

        row["direction_sum"] = sum(directions)

        # ----------------------------------------------------
        # Future targets
        # ----------------------------------------------------

        base_close = C4["close"]

        for n, c in enumerate(
            future,
            start=5
        ):

            row[f"C{n}_direction"] = int(
                c["close"] > c["open"]
            )

            row[f"C{n}_return"] = (
                c["close"] /
                base_close - 1
            )

            row[f"C{n}_high_excursion"] = (
                c["high"] -
                base_close
            ) / max(window_range, 1e-12)

            row[f"C{n}_low_excursion"] = (
                c["low"] -
                base_close
            ) / max(window_range, 1e-12)

            row[f"C{n}_range"] = (
                c["high"] -
                c["low"]
            )

            row[f"C{n}_break_high"] = int(
                c["high"] > window_high
            )

            row[f"C{n}_break_low"] = int(
                c["low"] < window_low
            )

        # ----------------------------------------------------
        # First-touch target
        # ----------------------------------------------------

        future_high = max(
            c["high"]
            for c in future
        )

        future_low = min(
            c["low"]
            for c in future
        )

        up_hit = future_high > window_high
        down_hit = future_low < window_low

        row["future_break_up"] = int(up_hit)
        row["future_break_down"] = int(down_hit)

        if up_hit and down_hit:

            # Determine which happened first.
            outcome = None

            for c in future:

                if c["high"] > window_high:
                    outcome = 1
                    break

                if c["low"] < window_low:
                    outcome = -1
                    break

            row["first_touch"] = outcome

        elif up_hit:

            row["first_touch"] = 1

        elif down_hit:

            row["first_touch"] = -1

        else:

            row["first_touch"] = 0

        row["timestamp"] = C4["timestamp"]

        rows.append(row)

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT_DIR / "feature_dataset.csv",
        index=False
    )

    print()
    print("FEATURE DATASET")
    print("=" * 70)
    print("Rows:", len(result))
    print("Features:", len(result.columns))

    return result


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def correlation_analysis(df):

    print()
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    targets = [
        "C5_high_excursion",
        "C5_low_excursion",
        "C6_high_excursion",
        "C6_low_excursion",
        "C7_high_excursion",
        "C7_low_excursion",
        "C5_return",
        "C6_return",
        "C7_return",
    ]

    ignore = set(targets)

    numeric = df.select_dtypes(
        include=np.number
    )

    rows = []

    for feature in numeric.columns:

        if feature in ignore:
            continue

        x = numeric[feature]

        for target in targets:

            if target not in numeric:
                continue

            y = numeric[target]

            mask = (
                x.notna()
                &
                y.notna()
            )

            if mask.sum() < MIN_PATTERN_N:
                continue

            try:

                pearson = pearsonr(
                    x[mask],
                    y[mask]
                )

                spearman = spearmanr(
                    x[mask],
                    y[mask]
                )

                rows.append({
                    "feature": feature,
                    "target": target,
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_r": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                    "n": mask.sum(),
                })

            except Exception:
                pass

    result = pd.DataFrame(rows)

    result["abs_spearman"] = (
        result["spearman_r"].abs()
    )

    result = result.sort_values(
        "abs_spearman",
        ascending=False
    )

    result.to_csv(
        OUTPUT_DIR / "correlations.csv",
        index=False
    )

    print(
        result.head(30).to_string(
            index=False
        )
    )

    return result


# ============================================================
# CONDITIONAL PATTERN ANALYSIS
# ============================================================

def conditional_patterns(df):

    print()
    print("CONDITIONAL PATTERN ANALYSIS")
    print("=" * 70)

    patterns = {

        "H4_bullish_engulf":
            df["H4_bullish_engulf"] == 1,

        "H4_bearish_engulf":
            df["H4_bearish_engulf"] == 1,

        "H4_inside":
            df["H4_inside"] == 1,

        "C4_break_C3_high":
            df["C4_break_C3_high"] == 1,

        "C4_break_C3_low":
            df["C4_break_C3_low"] == 1,

        "C4_inside_C3":
            df["C4_inside_C3"] == 1,

        "all_bullish":
            df["all_bullish"] == 1,

        "all_bearish":
            df["all_bearish"] == 1,

        "last2_bullish":
            df["last2_bullish"] == 1,

        "last2_bearish":
            df["last2_bearish"] == 1,

        "high_contraction":
            df["C4_vs_avg_range"] < 0.6,

        "strong_expansion":
            df["C4_vs_avg_range"] > 1.5,

        "close_near_high":
            df["C4_close_position"] > 0.8,

        "close_near_low":
            df["C4_close_position"] < 0.2,

        "positive_directional_imbalance":
            df["direction_sum"] >= 3,

        "negative_directional_imbalance":
            df["direction_sum"] <= -3,
    }

    targets = [
        "C5_direction",
        "C6_direction",
        "C7_direction",
        "future_break_up",
        "future_break_down",
    ]

    rows = []

    for name, condition in patterns.items():

        subset = df[condition]

        if len(subset) < MIN_PATTERN_N:
            continue

        for target in targets:

            base = df[target].mean()

            conditional = subset[target].mean()

            edge = conditional - base

            rows.append({
                "pattern": name,
                "target": target,
                "n": len(subset),
                "baseline": base,
                "conditional": conditional,
                "edge": edge,
            })

    result = pd.DataFrame(rows)

    result["abs_edge"] = (
        result["edge"].abs()
    )

    result = result.sort_values(
        "abs_edge",
        ascending=False
    )

    result.to_csv(
        OUTPUT_DIR /
        "conditional_patterns.csv",
        index=False
    )

    print(
        result.head(50).to_string(
            index=False
        )
    )

    return result


# ============================================================
# EXTENSION ANALYSIS
# ============================================================

def extension_analysis(df):

    print()
    print("EXTENSION ANALYSIS")
    print("=" * 70)

    # Bin C4 range relative to previous 3 candles.

    df = df.copy()

    df["range_regime"] = pd.qcut(
        df["C4_vs_avg_range"],
        q=5,
        labels=[
            "very_low",
            "low",
            "medium",
            "high",
            "very_high",
        ],
        duplicates="drop"
    )

    targets = [
        "C5_high_excursion",
        "C5_low_excursion",
        "C6_high_excursion",
        "C6_low_excursion",
        "C7_high_excursion",
        "C7_low_excursion",
    ]

    result = (
        df.groupby(
            "range_regime",
            observed=True
        )[targets]
        .agg([
            "mean",
            "median",
            "count"
        ])
    )

    result.to_csv(
        OUTPUT_DIR /
        "extension_by_range_regime.csv"
    )

    print(result)

    return result


# ============================================================
# MODEL DATA
# ============================================================

def prepare_ml_data(df):

    # Features must not include targets.

    target_columns = [
        c for c in df.columns
        if (
            c.startswith("C5_")
            or c.startswith("C6_")
            or c.startswith("C7_")
            or c.startswith("future_")
            or c == "first_touch"
        )
    ]

    exclude = set(target_columns)

    exclude.update([
        "timestamp",
    ])

    features = [
        c
        for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]

    X = (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return X, features


# ============================================================
# WALK-FORWARD CLASSIFICATION
# ============================================================

def walk_forward_classifier(
    X,
    y,
    name
):

    print()
    print(
        f"MODEL: {name}"
    )
    print("=" * 70)

    n = len(X)

    # Expanding-window walk-forward.

    test_size = n // (
        N_WALK_FORWARD_SPLITS + 1
    )

    results = []

    predictions = []

    for split in range(
        1,
        N_WALK_FORWARD_SPLITS + 1
    ):

        train_end = (
            test_size * split
        )

        test_end = min(
            train_end + test_size,
            n
        )

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        if len(X_test) == 0:
            continue

        model = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        proba = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            pred
        )

        balanced = balanced_accuracy_score(
            y_test,
            pred
        )

        try:
            auc = roc_auc_score(
                y_test,
                proba
            )
        except:
            auc = np.nan

        results.append({
            "split": split,
            "train_n": len(X_train),
            "test_n": len(X_test),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "roc_auc": auc,
        })

        p = pd.DataFrame({
            "actual": y_test.values,
            "prediction": pred,
            "probability": proba,
            "split": split,
        })

        predictions.append(p)

        print(
            f"Split {split}: "
            f"accuracy={accuracy:.4f} "
            f"balanced={balanced:.4f} "
            f"AUC={auc:.4f}"
        )

    results_df = pd.DataFrame(results)

    predictions_df = pd.concat(
        predictions,
        ignore_index=True
    )

    results_df.to_csv(
        OUTPUT_DIR /
        f"model_{name}_metrics.csv",
        index=False
    )

    predictions_df.to_csv(
        OUTPUT_DIR /
        f"model_{name}_predictions.csv",
        index=False
    )

    print()
    print(
        "Average:"
    )

    print(
        results_df[
            [
                "accuracy",
                "balanced_accuracy",
                "roc_auc",
            ]
        ].mean()
    )

    return results_df


# ============================================================
# REGRESSION
# ============================================================

def walk_forward_regression(
    X,
    y,
    name
):

    print()
    print(
        f"REGRESSION: {name}"
    )
    print("=" * 70)

    n = len(X)

    test_size = n // (
        N_WALK_FORWARD_SPLITS + 1
    )

    results = []

    for split in range(
        1,
        N_WALK_FORWARD_SPLITS + 1
    ):

        train_end = (
            test_size * split
        )

        test_end = min(
            train_end + test_size,
            n
        )

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        if len(X_test) == 0:
            continue

        model = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=20,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )

        correlation = np.corrcoef(
            y_test,
            pred
        )[0, 1]

        results.append({
            "split": split,
            "train_n": len(X_train),
            "test_n": len(X_test),
            "mae": mae,
            "rmse": rmse,
            "prediction_correlation":
                correlation,
        })

        print(
            f"Split {split}: "
            f"MAE={mae:.5f} "
            f"RMSE={rmse:.5f} "
            f"corr={correlation:.4f}"
        )

    result = pd.DataFrame(results)

    result.to_csv(
        OUTPUT_DIR /
        f"regression_{name}_metrics.csv",
        index=False
    )

    print()
    print(
        result[
            [
                "mae",
                "rmse",
                "prediction_correlation",
            ]
        ].mean()
    )

    return result


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def feature_importance(
    X,
    y,
    target_name
):

    model = ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X,
        y
    )

    result = pd.DataFrame({
        "feature": X.columns,
        "importance":
            model.feature_importances_,
    })

    result = result.sort_values(
        "importance",
        ascending=False
    )

    result.to_csv(
        OUTPUT_DIR /
        f"feature_importance_{target_name}.csv",
        index=False
    )

    print()
    print(
        f"FEATURE IMPORTANCE: {target_name}"
    )
    print("=" * 70)

    print(
        result.head(30).to_string(
            index=False
        )
    )

    return result


# ============================================================
# BASELINE
# ============================================================

def baseline_report(df):

    print()
    print("BASELINES")
    print("=" * 70)

    for target in [
        "C5_direction",
        "C6_direction",
        "C7_direction",
        "future_break_up",
        "future_break_down",
    ]:

        p = df[target].mean()

        naive_accuracy = max(
            p,
            1 - p
        )

        print(
            f"{target:25s} "
            f"P= {p:.4f} "
            f"naive accuracy= "
            f"{naive_accuracy:.4f}"
        )



# ============================================================
# INSTRUMENT BEHAVIOUR / C5 -> C6 RANGE MICROSTRUCTURE RESEARCH
# ============================================================

# IMPORTANT:
# C4 close is only the context point inside the prior 8H range.
# The predictive/reference anchor for this section is C5 HIGH / C5 LOW.
# Outcomes are measured on C6 only, while max projections use C5+C6.
# This avoids treating C4 close as an entry or target anchor.

ADV_FUTURE_BARS = 2                  # C5 + C6 only
REFERENCE_LEVELS_PIPS = [50,100,250,500,750,800,1000,1500,2000]
ADV_PIP_SIZE = 0.01                  # BTC: 0.01 price = 1 pip


def _pip(x):
    return x / ADV_PIP_SIZE


def _zone_pct(x):
    if x < 0.10: return "0-10%"
    if x < 0.25: return "10-25%"
    if x < 0.40: return "25-40%"
    if x < 0.50: return "40-50%"
    if x < 0.60: return "50-60%"
    if x < 0.75: return "60-75%"
    if x < 0.90: return "75-90%"
    return "90-100%"


def _c5_side(c5, hi, lo):
    """Classify C5 by which prior-8H edge it breaks; 0 if neither/both."""
    up = c5["high"] > hi
    dn = c5["low"] < lo
    if up and not dn:
        return 1
    if dn and not up:
        return -1
    return 0


def _c5c6_stats(c4, c5, c6):
    """
    Two complementary views of the same C5->C6 sequence.

    1) C4-close anchored: maximum MFE/MAE reached anywhere across C5+C6.
       This answers how far price travelled from the already-known C4 close.

    2) C5 HIGH/LOW anchored: C6 extension and adverse excursion relative to
       the actual C5 HIGH/LOW. This answers what happened after C5 established
       its extremes.
    """
    c4_close = float(c4["close"])
    c5_hi = float(c5["high"])
    c5_lo = float(c5["low"])
    c5_close = float(c5["close"])
    c6_hi = float(c6["high"])
    c6_lo = float(c6["low"])
    c6_close = float(c6["close"])

    # ------------------------------------------------------------
    # C4-close anchor: full C5+C6 path (maximum distance).
    # ------------------------------------------------------------
    c4_long_mfe = max(0.0, max(c5_hi, c6_hi) - c4_close)
    c4_long_mae = max(0.0, c4_close - min(c5_lo, c6_lo))
    c4_short_mfe = max(0.0, c4_close - min(c5_lo, c6_lo))
    c4_short_mae = max(0.0, max(c5_hi, c6_hi) - c4_close)

    # ------------------------------------------------------------
    # C5 HIGH/LOW anchor: C6 extension/adverse excursion only.
    # ------------------------------------------------------------
    long_mfe = max(0.0, c6_hi - c5_hi)
    long_mae = max(0.0, c5_hi - c6_lo)
    short_mfe = max(0.0, c5_lo - c6_lo)
    short_mae = max(0.0, c6_hi - c5_lo)

    return {
        "C5_range_pips": _pip(c5_hi - c5_lo),
        "C6_range_pips": _pip(c6_hi - c6_lo),

        # C5 H/L distances from the C4 close: the bridge between the two views.
        "C4_to_C5_high_pips": _pip(c5_hi - c4_close),
        "C4_to_C5_low_pips": _pip(c4_close - c5_lo),

        # Full C5+C6 maximum excursion from C4 close.
        "MFE_long_C4_close_C5C6_pips": _pip(c4_long_mfe),
        "MAE_long_C4_close_C5C6_pips": _pip(c4_long_mae),
        "MFE_short_C4_close_C5C6_pips": _pip(c4_short_mfe),
        "MAE_short_C4_close_C5C6_pips": _pip(c4_short_mae),

        # C6 extension beyond C5 extremes.
        "C6_extend_above_C5_high_pips": _pip(long_mfe),
        "C6_extend_below_C5_low_pips": _pip(short_mfe),
        "C6_close_from_C5_high_pips": _pip(c6_close - c5_hi),
        "C6_close_from_C5_low_pips": _pip(c6_close - c5_lo),
        "MFE_long_C5_high_pips": _pip(long_mfe),
        "MAE_long_C5_high_pips": _pip(long_mae),
        "MFE_short_C5_low_pips": _pip(short_mfe),
        "MAE_short_C5_low_pips": _pip(short_mae),

        # C5+C6 maximum projection, explicitly anchored to C5 H/L.
        "max_projection_up_C5C6_from_C5_high_pips": _pip(max(0.0, max(c5_hi, c6_hi) - c5_hi)),
        "max_projection_down_C5C6_from_C5_low_pips": _pip(max(0.0, c5_lo - min(c5_lo, c6_lo))),
        "C6_breaks_C5_high": int(c6_hi > c5_hi),
        "C6_breaks_C5_low": int(c6_lo < c5_lo),
        "C6_close_above_C5_high": int(c6_close > c5_hi),
        "C6_close_below_C5_low": int(c6_close < c5_lo),
        "C6_inside_C5_range": int(c5_lo <= c6_close <= c5_hi),
        "C6_high_sweep_C5_high_reject": int(c6_hi > c5_hi and c6_close < c5_hi),
        "C6_low_sweep_C5_low_reject": int(c6_lo < c5_lo and c6_close > c5_lo),
    }


def _first_level_time_from_c5(future, c5_hi, c5_lo, levels):
    """First C6/C5+C6 bar that reaches a favourable level from C5 HIGH/LOW."""
    out = {}
    for n in levels:
        up_level = c5_hi + n * ADV_PIP_SIZE
        down_level = c5_lo - n * ADV_PIP_SIZE
        up_bar = np.nan
        down_bar = np.nan
        for j, c in enumerate(future, start=1):
            if pd.isna(up_bar) and c["high"] >= up_level:
                up_bar = j
            if pd.isna(down_bar) and c["low"] <= down_level:
                down_bar = j
            if not pd.isna(up_bar) and not pd.isna(down_bar):
                break
        out[f"up_from_C5_high_touch_{n}_pips_bar"] = up_bar
        out[f"down_from_C5_low_touch_{n}_pips_bar"] = down_bar
    return out


def instrument_behavior_analysis(raw_df):
    """Study what C6 does relative to C5 HIGH/LOW; C5+C6 give max projection."""
    print("\n" + "="*70)
    print("C5 HIGH/LOW -> C6 INSTRUMENT BEHAVIOUR RESEARCH")
    print("="*70)

    rows = []
    # Need C1..C4 + C5 + C6.
    for i in range(4, len(raw_df) - 1):
        c1, c2, c3, c4 = [raw_df.iloc[i-k] for k in range(4, 0, -1)]
        c5 = raw_df.iloc[i]
        c6 = raw_df.iloc[i + 1]
        future = [c5, c6]

        prior_hi = max(c["high"] for c in (c1, c2, c3, c4))
        prior_lo = min(c["low"] for c in (c1, c2, c3, c4))
        prior_rr = max(prior_hi - prior_lo, 1e-12)
        c5_pos_close = np.clip((c5["close"] - prior_lo) / prior_rr, 0, 1)
        c5_high_pos = np.clip((c5["high"] - prior_lo) / prior_rr, 0, 1)
        c5_low_pos = np.clip((c5["low"] - prior_lo) / prior_rr, 0, 1)

        side = _c5_side(c5, prior_hi, prior_lo)
        stats = _c5c6_stats(c4, c5, c6)

        row = {
            "timestamp": c4["timestamp"],
            "C4_close": c4["close"],
            "prior_8h_high": prior_hi,
            "prior_8h_low": prior_lo,
            "prior_8h_range_pips": _pip(prior_rr),
            # C5 is now the reference candle; these are context only.
            "C5_close": c5["close"],
            "C5_high": c5["high"],
            "C5_low": c5["low"],
            "C5_close_position_pct_8h": c5_pos_close * 100,
            "C5_high_position_pct_8h": c5_high_pos * 100,
            "C5_low_position_pct_8h": c5_low_pos * 100,
            "C5_high_zone_8h": _zone_pct(c5_high_pos),
            "C5_low_zone_8h": _zone_pct(c5_low_pos),
            "C5_breaks_prior_8h_high": int(c5["high"] > prior_hi),
            "C5_breaks_prior_8h_low": int(c5["low"] < prior_lo),
            "C5_high_break_only": int(c5["high"] > prior_hi and c5["low"] >= prior_lo),
            "C5_low_break_only": int(c5["low"] < prior_lo and c5["high"] <= prior_hi),
            "C5_break_side": side,
            "C5_high_sweep_reject_prior_8h": int(c5["high"] > prior_hi and c5["close"] < prior_hi),
            "C5_low_sweep_reject_prior_8h": int(c5["low"] < prior_lo and c5["close"] > prior_lo),
            "C5_high_distance_from_prior_8h_high_pips": _pip(max(0.0, c5["high"] - prior_hi)),
            "C5_low_distance_from_prior_8h_low_pips": _pip(max(0.0, prior_lo - c5["low"])),
            **stats,
        }
        row.update(_first_level_time_from_c5(future, c5["high"], c5["low"], REFERENCE_LEVELS_PIPS))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "instrument_behavior_rows.csv", index=False)

    # --------------------------------------------------------
    # 12/18: C5-side outcomes. These are the primary reports.
    # --------------------------------------------------------
    zone = out.groupby("C5_high_zone_8h", observed=False).agg(
        n=("timestamp", "size"),
        median_C5_range_pips=("C5_range_pips", "median"),
        median_MFE_long=("MFE_long_C5_high_pips", "median"),
        p75_MFE_long=("MFE_long_C5_high_pips", lambda x: x.quantile(.75)),
        median_MAE_long=("MAE_long_C5_high_pips", "median"),
        median_MFE_short=("MFE_short_C5_low_pips", "median"),
        p75_MFE_short=("MFE_short_C5_low_pips", lambda x: x.quantile(.75)),
        median_MAE_short=("MAE_short_C5_low_pips", "median"),
        C6_break_C5_high=("C6_breaks_C5_high", "mean"),
        C6_break_C5_low=("C6_breaks_C5_low", "mean"),
        C6_close_above_C5_high=("C6_close_above_C5_high", "mean"),
        C6_close_below_C5_low=("C6_close_below_C5_low", "mean"),
        C6_inside_C5_range=("C6_inside_C5_range", "mean"),
    ).reset_index().rename(columns={"C5_high_zone_8h": "C5_zone"})
    zone.to_csv(OUTPUT_DIR / "instrument_behavior_by_range_zone.csv", index=False)

    # Side-specific C5 HIGH / C5 LOW conditioning.
    # Upside outcomes use C5 HIGH location; downside outcomes use C5 LOW location.
    side_zone_rows = []
    for side, zone_col, mfe_col, mae_col, break_col, close_col in [
        ("up_from_C5_high", "C5_high_zone_8h", "MFE_long_C5_high_pips", "MAE_long_C5_high_pips", "C6_breaks_C5_high", "C6_close_above_C5_high"),
        ("down_from_C5_low", "C5_low_zone_8h", "MFE_short_C5_low_pips", "MAE_short_C5_low_pips", "C6_breaks_C5_low", "C6_close_below_C5_low"),
    ]:
        for zone_name, g in out.groupby(zone_col, observed=False):
            side_zone_rows.append({
                "reference_side": side,
                "C5_zone": zone_name,
                "n": len(g),
                "median_MFE_pips": g[mfe_col].median(),
                "p75_MFE_pips": g[mfe_col].quantile(.75),
                "p90_MFE_pips": g[mfe_col].quantile(.90),
                "median_MAE_pips": g[mae_col].median(),
                "p75_MAE_pips": g[mae_col].quantile(.75),
                # Same side measured from C4 close across the full C5+C6 path.
                "median_MFE_from_C4_pips": g[
                    "MFE_long_C4_close_C5C6_pips" if side == "up_from_C5_high"
                    else "MFE_short_C4_close_C5C6_pips"
                ].median(),
                "p75_MFE_from_C4_pips": g[
                    "MFE_long_C4_close_C5C6_pips" if side == "up_from_C5_high"
                    else "MFE_short_C4_close_C5C6_pips"
                ].quantile(.75),
                "median_MAE_from_C4_pips": g[
                    "MAE_long_C4_close_C5C6_pips" if side == "up_from_C5_high"
                    else "MAE_short_C4_close_C5C6_pips"
                ].median(),
                "p75_MAE_from_C4_pips": g[
                    "MAE_long_C4_close_C5C6_pips" if side == "up_from_C5_high"
                    else "MAE_short_C4_close_C5C6_pips"
                ].quantile(.75),
                "median_C4_to_C5_reference_pips": g[
                    "C4_to_C5_high_pips" if side == "up_from_C5_high"
                    else "C4_to_C5_low_pips"
                ].median(),
                "C6_break_reference": g[break_col].mean(),
                "C6_close_beyond_reference": g[close_col].mean(),
            })
    side_zone = pd.DataFrame(side_zone_rows)
    side_zone.to_csv(OUTPUT_DIR / "c5c6_behavior_by_side_zone.csv", index=False)

    # --------------------------------------------------------
    # 13: C5+C6 max projection as % of the PRIOR 8H range.
    # --------------------------------------------------------
    out["MFE_long_pct_prior_8h"] = out["MFE_long_C5_high_pips"] / out["prior_8h_range_pips"] * 100
    out["MFE_short_pct_prior_8h"] = out["MFE_short_C5_low_pips"] / out["prior_8h_range_pips"] * 100
    penetration_rows = []
    for side, zone_col, mfe_col in [
        ("up_from_C5_high", "C5_high_zone_8h", "MFE_long_pct_prior_8h"),
        ("down_from_C5_low", "C5_low_zone_8h", "MFE_short_pct_prior_8h"),
    ]:
        for zone_name, g in out.groupby(zone_col, observed=False):
            penetration_rows.append({
                "reference_side": side,
                "C5_zone": zone_name,
                "n": len(g),
                "median_MFE_pct_prior_8h": g[mfe_col].median(),
                "p75_MFE_pct_prior_8h": g[mfe_col].quantile(.75),
                "p90_MFE_pct_prior_8h": g[mfe_col].quantile(.90),
            })
    penetration = pd.DataFrame(penetration_rows)
    penetration.to_csv(OUTPUT_DIR / "range_penetration_by_zone.csv", index=False)

    # --------------------------------------------------------
    # 14: probability that C6 extends X pips beyond C5 HIGH/LOW.
    # --------------------------------------------------------
    target_rows = []
    for direction, zone_col, col in [
        ("up_from_C5_high", "C5_high_zone_8h", "MFE_long_C5_high_pips"),
        ("down_from_C5_low", "C5_low_zone_8h", "MFE_short_C5_low_pips"),
    ]:
        for zone_name, g in out.groupby(zone_col, observed=False):
            for level in REFERENCE_LEVELS_PIPS:
                target_rows.append({
                    "zone": zone_name,
                    "direction": direction,
                    "level_pips": level,
                    "n": len(g),
                    "hit_probability": (g[col] >= level).mean(),
                    "median_mfe_pips": g[col].median(),
                    "p75_mfe_pips": g[col].quantile(.75),
                })
    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(OUTPUT_DIR / "reference_excursion_levels.csv", index=False)

    # --------------------------------------------------------
    # 16: C5 structural conditions -> C6 continuation/reversion.
    # --------------------------------------------------------
    structures = {
        "C5_high_break_only": out["C5_high_break_only"] == 1,
        "C5_low_break_only": out["C5_low_break_only"] == 1,
        "C5_high_sweep_rejection": out["C5_high_sweep_reject_prior_8h"] == 1,
        "C5_low_sweep_rejection": out["C5_low_sweep_reject_prior_8h"] == 1,
        "C5_breaks_both_8h_edges": (out["C5_breaks_prior_8h_high"] == 1) & (out["C5_breaks_prior_8h_low"] == 1),
        "C5_inside_prior_8h": (out["C5_breaks_prior_8h_high"] == 0) & (out["C5_breaks_prior_8h_low"] == 0),
    }
    sr = []
    for name, mask in structures.items():
        g = out[mask]
        if len(g) < MIN_PATTERN_N:
            continue
        sr.append({
            "pattern": name,
            "n": len(g),
            "C6_break_C5_high": g.C6_breaks_C5_high.mean(),
            "C6_break_C5_low": g.C6_breaks_C5_low.mean(),
            "C6_close_above_C5_high": g.C6_close_above_C5_high.mean(),
            "C6_close_below_C5_low": g.C6_close_below_C5_low.mean(),
            "C6_inside_C5_range": g.C6_inside_C5_range.mean(),
            "median_MFE_long": g.MFE_long_C5_high_pips.median(),
            "median_MFE_short": g.MFE_short_C5_low_pips.median(),
            "median_MAE_long": g.MAE_long_C5_high_pips.median(),
            "median_MAE_short": g.MAE_short_C5_low_pips.median(),
        })
    pd.DataFrame(sr).to_csv(OUTPUT_DIR / "C3C4_C5_structure_behavior.csv", index=False)

    print("\nC5 HIGH/LOW -> C6 BEHAVIOUR BY C5 HIGH ZONE")
    print(zone.to_string(index=False))
    print("\nC6 EXTENSION PROBABILITIES FROM C5 HIGH/LOW")
    print(target_df[target_df.level_pips.isin([500, 750, 800, 1000])].to_string(index=False))
    print("\nC5 STRUCTURE -> C6")
    print(pd.DataFrame(sr).to_string(index=False))
    print("\nSaved C5/C6 instrument behaviour files to:", OUTPUT_DIR.resolve())
    return out


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    dataset = build_dataset(
        df
    )

    baseline_report(
        dataset
    )

    correlation_analysis(
        dataset
    )

    conditional_patterns(
        dataset
    )

    extension_analysis(
        dataset
    )

    # Descriptive instrument-behaviour study. C4 is context; the
    # C5 HIGH / C5 LOW are the reference levels and C6 is the outcome.
    instrument_behavior_analysis(
        df
    )

    X, features = prepare_ml_data(
        dataset
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    for target in [
        "C5_direction",
        "C6_direction",
        "C7_direction",
        "future_break_up",
        "future_break_down",
    ]:

        y = dataset[target].astype(int)

        walk_forward_classifier(
            X,
            y,
            target
        )

        feature_importance(
            X,
            y,
            target
        )

    # --------------------------------------------------------
    # FIRST TOUCH
    # --------------------------------------------------------

    first_touch = dataset[
        "first_touch"
    ].replace(
        -1,
        0
    )

    # Binary:
    # 1 = upside first
    # 0 = downside/neither first

    walk_forward_classifier(
        X,
        first_touch.astype(int),
        "first_touch_up"
    )

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    for target in [
        "C5_high_excursion",
        "C5_low_excursion",
        "C6_high_excursion",
        "C6_low_excursion",
        "C7_high_excursion",
        "C7_low_excursion",
    ]:

        y = dataset[target]

        walk_forward_regression(
            X,
            y,
            target
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()
