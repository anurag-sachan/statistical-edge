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
# INSTRUMENT BEHAVIOUR / C4 -> C5/C6 -> C7 RESEARCH
# ============================================================

# This section is deliberately anchored to C4 CLOSE because C4 is already
# inside the completed 8H range.  The study asks two practical questions:
#
#   1) From C4 close, how far can C5/C6 extend against a hypothetical
#      long/short before price reverts or resumes continuation?
#      -> MAE / max pullback.
#
#   2) After C5/C6 establish their combined high/low, how far can C7 project
#      beyond that extreme?
#      -> C7 MFE / projection, which is more useful for discovering natural
#         target distances.
#
# No fixed TP/SL is assumed. Reference levels are descriptive only.

ADV_PIP_SIZE = 0.01                  # BTC: 0.01 price = 1 pip
C7_REFERENCE_LEVELS_PIPS = [50,100,250,500,750,800,1000,1500,2000]


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


def _c4_sequence_stats(c4, c5, c6, c7):
    """Calculate exact C4 -> C5/C6 MAE and C7 projection statistics."""
    entry = float(c4["close"])

    # C5/C6 combined extremes. These are the maximum adverse extensions
    # against either hypothetical direction before C7.
    c56_high = max(float(c5["high"]), float(c6["high"]))
    c56_low = min(float(c5["low"]), float(c6["low"]))

    long_mae_price = max(0.0, entry - c56_low)
    short_mae_price = max(0.0, c56_high - entry)

    # Also retain the favourable C5/C6 extension from C4. This tells us how
    # far the move can travel before C7 even starts its projection phase.
    c56_up_from_c4 = max(0.0, c56_high - entry)
    c56_down_from_c4 = max(0.0, entry - c56_low)

    # C7 projection is measured ONLY beyond the C5/C6 combined extreme.
    # This avoids double-counting the C5/C6 move as a C7 target.
    c7_up_projection_price = max(0.0, float(c7["high"]) - c56_high)
    c7_down_projection_price = max(0.0, c56_low - float(c7["low"]))

    # Raw C7 excursion from C4 close is retained for comparison.
    c7_up_from_c4 = max(0.0, float(c7["high"]) - entry)
    c7_down_from_c4 = max(0.0, entry - float(c7["low"]))

    # Direction of C7 close relative to C4 close.
    if float(c7["close"]) > entry:
        c7_direction = 1
    elif float(c7["close"]) < entry:
        c7_direction = -1
    else:
        c7_direction = 0

    return {
        "C5_high": float(c5["high"]),
        "C5_low": float(c5["low"]),
        "C6_high": float(c6["high"]),
        "C6_low": float(c6["low"]),
        "C7_high": float(c7["high"]),
        "C7_low": float(c7["low"]),
        "C7_close": float(c7["close"]),
        "C5C6_high": c56_high,
        "C5C6_low": c56_low,
        "C5C6_up_from_C4_pips": _pip(c56_up_from_c4),
        "C5C6_down_from_C4_pips": _pip(c56_down_from_c4),
        "MAE_long_C5C6_pips": _pip(long_mae_price),
        "MAE_short_C5C6_pips": _pip(short_mae_price),
        "C7_MFE_up_from_C4_pips": _pip(c7_up_from_c4),
        "C7_MFE_down_from_C4_pips": _pip(c7_down_from_c4),
        "C7_projection_up_pips": _pip(c7_up_projection_price),
        "C7_projection_down_pips": _pip(c7_down_projection_price),
        "C7_direction": c7_direction,
        "C7_continues_above_C5C6_high": int(float(c7["high"]) > c56_high),
        "C7_continues_below_C5C6_low": int(float(c7["low"]) < c56_low),
        "C7_closes_above_C5C6_high": int(float(c7["close"]) > c56_high),
        "C7_closes_below_C5C6_low": int(float(c7["close"]) < c56_low),
    }


def _first_level_from_c7_reference(c7_projection_pips, levels, prefix):
    return {
        f"{prefix}_touch_{level}_pips": int(c7_projection_pips >= level)
        for level in levels
    }


def c4_c5_c6_c7_behavior_analysis(raw_df):
    """Primary descriptive study: C4 close -> C5/C6 MAE -> C7 projection."""
    print("\n" + "="*70)
    print("C4 CLOSE -> C5/C6 MAE -> C7 PROJECTION RESEARCH")
    print("="*70)

    rows = []
    # Need C1..C4 to define the completed 8H range, then C5/C6/C7.
    for i in range(4, len(raw_df) - 3):
        c1, c2, c3, c4 = [raw_df.iloc[i-k] for k in range(4, 0, -1)]
        c5, c6, c7 = raw_df.iloc[i], raw_df.iloc[i+1], raw_df.iloc[i+2]

        range_high = max(float(c["high"]) for c in (c1, c2, c3, c4))
        range_low = min(float(c["low"]) for c in (c1, c2, c3, c4))
        range_size = max(range_high - range_low, 1e-12)
        entry = float(c4["close"])
        pos = np.clip((entry - range_low) / range_size, 0, 1)

        stats = _c4_sequence_stats(c4, c5, c6, c7)
        row = {
            "timestamp": c4["timestamp"],
            "C4_close": entry,
            "range_high": range_high,
            "range_low": range_low,
            "range_pips": _pip(range_size),
            "C4_position_pct": pos * 100,
            "C4_zone": _zone_pct(pos),
            "dist_high_pips": _pip(range_high - entry),
            "dist_low_pips": _pip(entry - range_low),
            "dist_high_pct_range": (range_high - entry) / range_size * 100,
            "dist_low_pct_range": (entry - range_low) / range_size * 100,
            **stats,
        }

        row.update(_first_level_from_c7_reference(
            stats["C7_projection_up_pips"], C7_REFERENCE_LEVELS_PIPS, "C7_up"
        ))
        row.update(_first_level_from_c7_reference(
            stats["C7_projection_down_pips"], C7_REFERENCE_LEVELS_PIPS, "C7_down"
        ))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "c4_c5_c6_c7_behavior_rows.csv", index=False)

    # --------------------------------------------------------
    # MAE from C4 close during C5+C6.
    # --------------------------------------------------------
    zone = out.groupby("C4_zone", observed=False).agg(
        n=("timestamp", "size"),
        median_C4_to_C5C6_MAE_long=("MAE_long_C5C6_pips", "median"),
        p75_C4_to_C5C6_MAE_long=("MAE_long_C5C6_pips", lambda x: x.quantile(.75)),
        p90_C4_to_C5C6_MAE_long=("MAE_long_C5C6_pips", lambda x: x.quantile(.90)),
        median_C4_to_C5C6_MAE_short=("MAE_short_C5C6_pips", "median"),
        p75_C4_to_C5C6_MAE_short=("MAE_short_C5C6_pips", lambda x: x.quantile(.75)),
        p90_C4_to_C5C6_MAE_short=("MAE_short_C5C6_pips", lambda x: x.quantile(.90)),
        median_C5C6_up_from_C4=("C5C6_up_from_C4_pips", "median"),
        median_C5C6_down_from_C4=("C5C6_down_from_C4_pips", "median"),
        median_C7_projection_up=("C7_projection_up_pips", "median"),
        p75_C7_projection_up=("C7_projection_up_pips", lambda x: x.quantile(.75)),
        p90_C7_projection_up=("C7_projection_up_pips", lambda x: x.quantile(.90)),
        median_C7_projection_down=("C7_projection_down_pips", "median"),
        p75_C7_projection_down=("C7_projection_down_pips", lambda x: x.quantile(.75)),
        p90_C7_projection_down=("C7_projection_down_pips", lambda x: x.quantile(.90)),
        c7_breaks_C5C6_high=("C7_continues_above_C5C6_high", "mean"),
        c7_breaks_C5C6_low=("C7_continues_below_C5C6_low", "mean"),
        c7_close_above_C5C6_high=("C7_closes_above_C5C6_high", "mean"),
        c7_close_below_C5C6_low=("C7_closes_below_C5C6_low", "mean"),
    ).reset_index()
    zone.to_csv(OUTPUT_DIR / "c4_c5_c6_c7_by_zone.csv", index=False)

    # --------------------------------------------------------
    # C7 projection target distribution. Unlike the old report,
    # these levels are measured from the C5/C6 extreme, not C4.
    # --------------------------------------------------------
    target_rows = []
    for zone_name, g in out.groupby("C4_zone", observed=False):
        for direction, col in [("up", "C7_projection_up_pips"), ("down", "C7_projection_down_pips")]:
            for level in C7_REFERENCE_LEVELS_PIPS:
                target_rows.append({
                    "zone": zone_name,
                    "direction": direction,
                    "reference": "C5C6_high" if direction == "up" else "C5C6_low",
                    "level_pips": level,
                    "n": len(g),
                    "hit_probability": (g[col] >= level).mean(),
                    "median_projection_pips": g[col].median(),
                    "p75_projection_pips": g[col].quantile(.75),
                    "p90_projection_pips": g[col].quantile(.90),
                })
    targets = pd.DataFrame(target_rows)
    targets.to_csv(OUTPUT_DIR / "c7_projection_levels.csv", index=False)

    # --------------------------------------------------------
    # A compact C5/C6/C7 sequence report for the visualizer.
    # --------------------------------------------------------
    sequence = pd.DataFrame({
        "metric": [
            "C4 -> C5/C6 max adverse extension for long",
            "C4 -> C5/C6 max adverse extension for short",
            "C5/C6 high -> C7 upside projection",
            "C5/C6 low -> C7 downside projection",
            "C7 breaks above C5/C6 high",
            "C7 breaks below C5/C6 low",
            "C7 closes above C5/C6 high",
            "C7 closes below C5/C6 low",
        ],
        "median_all": [
            out["MAE_long_C5C6_pips"].median(),
            out["MAE_short_C5C6_pips"].median(),
            out["C7_projection_up_pips"].median(),
            out["C7_projection_down_pips"].median(),
            out["C7_continues_above_C5C6_high"].mean() * 100,
            out["C7_continues_below_C5C6_low"].mean() * 100,
            out["C7_closes_above_C5C6_high"].mean() * 100,
            out["C7_closes_below_C5C6_low"].mean() * 100,
        ],
        "p75": [
            out["MAE_long_C5C6_pips"].quantile(.75),
            out["MAE_short_C5C6_pips"].quantile(.75),
            out["C7_projection_up_pips"].quantile(.75),
            out["C7_projection_down_pips"].quantile(.75),
            np.nan, np.nan, np.nan, np.nan,
        ],
        "p90": [
            out["MAE_long_C5C6_pips"].quantile(.90),
            out["MAE_short_C5C6_pips"].quantile(.90),
            out["C7_projection_up_pips"].quantile(.90),
            out["C7_projection_down_pips"].quantile(.90),
            np.nan, np.nan, np.nan, np.nan,
        ],
        "unit": ["pips", "pips", "pips", "pips", "%", "%", "%", "%"],
    })
    sequence.to_csv(OUTPUT_DIR / "c4_c5_c6_c7_summary.csv", index=False)

    print("\nC4 -> C5/C6 MAE + C7 PROJECTION BY RANGE ZONE")
    print(zone.to_string(index=False))
    print("\nC7 PROJECTION LEVELS (measured beyond C5/C6 extreme)")
    print(targets[targets.level_pips.isin([500, 750, 800, 1000])].to_string(index=False))
    print("\nSaved C4/C5/C6/C7 behaviour files to:", OUTPUT_DIR.resolve())
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

    # Primary descriptive instrument-behaviour study: C4 close ->
    # C5/C6 max adverse extension -> C7 projection beyond C5/C6 extreme.
    c4_c5_c6_c7_behavior_analysis(
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
