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
# INSTRUMENT BEHAVIOUR / RANGE MICROSTRUCTURE RESEARCH
# ============================================================

# IMPORTANT: this section is deliberately NOT optimized around the
# user's 750/800 target. 750/800 is included as one reference level,
# while the main outputs describe how BTC actually travels: MFE, MAE,
# excursion distributions, range penetration, continuation/reversion,
# and time-to-level. This preserves the instrument's behaviour.

ADV_FUTURE_BARS = 12                 # 24H after C4 on 2H data
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


def _path_stats(future, entry):
    """Pure descriptive excursion statistics; no TP/SL assumption."""
    highs = np.array([c["high"] for c in future], dtype=float)
    lows = np.array([c["low"] for c in future], dtype=float)
    closes = np.array([c["close"] for c in future], dtype=float)

    up = highs - entry
    down = entry - lows
    return {
        "mfe_up_pips": _pip(up.max()),
        "mfe_down_pips": _pip(down.max()),
        "mae_long_pips": _pip(max(0.0, down.max())),
        "mae_short_pips": _pip(max(0.0, up.max())),
        "close_move_pips": _pip(closes[-1] - entry),
    }


def _first_level_time(future, entry, direction, levels):
    """Return first 2H bar on which each favourable level is touched."""
    out = {f"{direction}_touch_{n}_pips_bar": np.nan for n in levels}
    for n in levels:
        level = entry + direction * n * ADV_PIP_SIZE
        for j, c in enumerate(future, start=1):
            hit = c["high"] >= level if direction == 1 else c["low"] <= level
            if hit:
                out[f"{direction}_touch_{n}_pips_bar"] = j
                break
    return out


def _first_break_sequence(future, hi, lo):
    """No invented intrabar ordering when both range edges are hit."""
    first = 0
    first_bar = np.nan
    opposite = 0
    ambiguous = 0
    for j, c in enumerate(future, start=1):
        up = c["high"] > hi
        dn = c["low"] < lo
        if first == 0:
            if up and dn:
                ambiguous = 1
                break
            if up:
                first, first_bar = 1, j
            elif dn:
                first, first_bar = -1, j
        elif first == 1 and dn:
            opposite = 1
            break
        elif first == -1 and up:
            opposite = 1
            break
    return first, first_bar, opposite, ambiguous


def instrument_behavior_analysis(raw_df):
    """Deep descriptive study of BTC's 8H -> future behaviour."""
    print("\\n" + "="*70)
    print("INSTRUMENT BEHAVIOUR / ENTRY-ZONE RESEARCH")
    print("="*70)

    rows = []
    max_future = min(ADV_FUTURE_BARS, len(raw_df) - 4)
    for i in range(4, len(raw_df) - max_future + 1):
        c1,c2,c3,c4 = [raw_df.iloc[i-k] for k in range(4,0,-1)]
        future = [raw_df.iloc[i+j] for j in range(max_future)]

        hi=max(c["high"] for c in (c1,c2,c3,c4))
        lo=min(c["low"] for c in (c1,c2,c3,c4))
        rr=max(hi-lo,1e-12)
        entry=c4["close"]
        pos=np.clip((entry-lo)/rr,0,1)

        c34_hi=max(c3["high"],c4["high"])
        c34_lo=min(c3["low"],c4["low"])
        body_hi=max(c3["open"],c3["close"],c4["open"],c4["close"])
        body_lo=min(c3["open"],c3["close"],c4["open"],c4["close"])
        body_rr=max(body_hi-body_lo,1e-12)

        first, first_bar, opposite, ambiguous=_first_break_sequence(future,hi,lo)
        ps=_path_stats(future,entry)

        row={
            "timestamp":c4["timestamp"],
            "C4_close":entry,
            "range_high":hi,
            "range_low":lo,
            "range_pips":_pip(rr),
            "C4_position_pct":pos*100,
            "C4_zone":_zone_pct(pos),
            "dist_high_pips":_pip(hi-entry),
            "dist_low_pips":_pip(entry-lo),
            "dist_high_pct_range":(hi-entry)/rr*100,
            "dist_low_pct_range":(entry-lo)/rr*100,
            "C3_high":c3["high"], "C4_high":c4["high"],
            "C3_low":c3["low"], "C4_low":c4["low"],
            "C3C4_high":c34_hi, "C3C4_low":c34_lo,
            "C3C4_body_high":body_hi, "C3C4_body_low":body_lo,
            "C3C4_body_range_pips":_pip(body_rr),
            "C3C4_body_pct_8h":body_rr/rr*100,
            "C3C4_wick_range_pct_8h":(c34_hi-c34_lo)/rr*100,
            "C5_close_between_C3_C4_highs":int(min(c3["high"],c4["high"])<=future[0]["close"]<=max(c3["high"],c4["high"])),
            "C5_close_between_C3_C4_lows":int(min(c3["low"],c4["low"])<=future[0]["close"]<=max(c3["low"],c4["low"])),
            "C5_above_both_C3_C4_highs":int(future[0]["close"]>max(c3["high"],c4["high"])),
            "C5_below_both_C3_C4_lows":int(future[0]["close"]<min(c3["low"],c4["low"])),
            "C5_breaks_8h_high":int(future[0]["high"]>hi),
            "C5_breaks_8h_low":int(future[0]["low"]<lo),
            "C5_high_sweep_reject":int(future[0]["high"]>hi and future[0]["close"]<hi),
            "C5_low_sweep_reject":int(future[0]["low"]<lo and future[0]["close"]>lo),
            "first_break_side":first,
            "first_break_bar":first_bar,
            "opposite_side_after_first_break":opposite,
            "ambiguous_first_break":ambiguous,
            **ps,
        }
        row.update(_first_level_time(future,entry,1,REFERENCE_LEVELS_PIPS))
        row.update(_first_level_time(future,entry,-1,REFERENCE_LEVELS_PIPS))
        rows.append(row)

    out=pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR/"instrument_behavior_rows.csv",index=False)

    # --------------------------------------------------------
    # Descriptive range-zone report: this is the primary report.
    # --------------------------------------------------------
    zone=out.groupby("C4_zone",observed=False).agg(
        n=("timestamp","size"),
        median_range_pips=("range_pips","median"),
        mean_range_pips=("range_pips","mean"),
        median_MFE_up=("mfe_up_pips","median"),
        median_MFE_down=("mfe_down_pips","median"),
        p75_MFE_up=("mfe_up_pips",lambda x:x.quantile(.75)),
        p75_MFE_down=("mfe_down_pips",lambda x:x.quantile(.75)),
        p90_MFE_up=("mfe_up_pips",lambda x:x.quantile(.90)),
        p90_MFE_down=("mfe_down_pips",lambda x:x.quantile(.90)),
        median_MAE_long=("mae_long_pips","median"),
        median_MAE_short=("mae_short_pips","median"),
        up_first=("first_break_side",lambda x:(x==1).mean()),
        down_first=("first_break_side",lambda x:(x==-1).mean()),
        ambiguous_first=("ambiguous_first_break","mean"),
        opposite_after_first=("opposite_side_after_first_break","mean"),
    ).reset_index()
    zone.to_csv(OUTPUT_DIR/"instrument_behavior_by_range_zone.csv",index=False)

    # --------------------------------------------------------
    # Range penetration: how much of the existing 8H range is
    # normally consumed before extension? This avoids forcing a
    # fixed target.
    # --------------------------------------------------------
    out["MFE_up_pct_8h"] = out["mfe_up_pips"] / out["range_pips"] * 100
    out["MFE_down_pct_8h"] = out["mfe_down_pips"] / out["range_pips"] * 100
    penetration=out.groupby("C4_zone",observed=False).agg(
        n=("timestamp","size"),
        median_MFE_up_pct_8h=("MFE_up_pct_8h","median"),
        p75_MFE_up_pct_8h=("MFE_up_pct_8h",lambda x:x.quantile(.75)),
        p90_MFE_up_pct_8h=("MFE_up_pct_8h",lambda x:x.quantile(.90)),
        median_MFE_down_pct_8h=("MFE_down_pct_8h","median"),
        p75_MFE_down_pct_8h=("MFE_down_pct_8h",lambda x:x.quantile(.75)),
        p90_MFE_down_pct_8h=("MFE_down_pct_8h",lambda x:x.quantile(.90)),
    ).reset_index()
    penetration.to_csv(OUTPUT_DIR/"range_penetration_by_zone.csv",index=False)

    # --------------------------------------------------------
    # Reference target distribution. These are descriptive hit
    # probabilities from C4, NOT TP-before-SL trading signals.
    # --------------------------------------------------------
    target_rows=[]
    for zone_name,g in out.groupby("C4_zone",observed=False):
        for d,col in [("up","mfe_up_pips"),("down","mfe_down_pips")]:
            for level in REFERENCE_LEVELS_PIPS:
                target_rows.append({
                    "zone":zone_name,"direction":d,
                    "level_pips":level,"n":len(g),
                    "hit_probability":(g[col]>=level).mean(),
                    "median_mfe_pips":g[col].median(),
                    "p75_mfe_pips":g[col].quantile(.75),
                })
    target_df=pd.DataFrame(target_rows)
    target_df.to_csv(OUTPUT_DIR/"reference_excursion_levels.csv",index=False)

    # --------------------------------------------------------
    # C3/C4 -> C5 structural relations.
    # --------------------------------------------------------
    structures={
        "C5_between_C3_C4_highs":out["C5_close_between_C3_C4_highs"]==1,
        "C5_between_C3_C4_lows":out["C5_close_between_C3_C4_lows"]==1,
        "C5_above_both_C3_C4_highs":out["C5_above_both_C3_C4_highs"]==1,
        "C5_below_both_C3_C4_lows":out["C5_below_both_C3_C4_lows"]==1,
        "C5_high_sweep_rejection":out["C5_high_sweep_reject"]==1,
        "C5_low_sweep_rejection":out["C5_low_sweep_reject"]==1,
        "C5_high_break":out["C5_breaks_8h_high"]==1,
        "C5_low_break":out["C5_breaks_8h_low"]==1,
    }
    sr=[]
    for name,mask in structures.items():
        g=out[mask]
        if len(g)<MIN_PATTERN_N: continue
        sr.append({
            "pattern":name,"n":len(g),
            "up_first":(g.first_break_side==1).mean(),
            "down_first":(g.first_break_side==-1).mean(),
            "opposite_after_first":g.opposite_side_after_first_break.mean(),
            "median_MFE_up":g.mfe_up_pips.median(),
            "median_MFE_down":g.mfe_down_pips.median(),
            "p75_MFE_up":g.mfe_up_pips.quantile(.75),
            "p75_MFE_down":g.mfe_down_pips.quantile(.75),
            "median_MAE_long":g.mae_long_pips.median(),
            "median_MAE_short":g.mae_short_pips.median(),
        })
    pd.DataFrame(sr).to_csv(OUTPUT_DIR/"C3C4_C5_structure_behavior.csv",index=False)

    # --------------------------------------------------------
    # Print only the facts most useful for understanding BTC.
    # --------------------------------------------------------
    print("\\nRANGE-ZONE BEHAVIOUR")
    print(zone.to_string(index=False))
    print("\\nREFERENCE EXCURSION LEVELS (not TP/SL optimized)")
    print(target_df[target_df.level_pips.isin([500,750,800,1000])].to_string(index=False))
    print("\\nC3/C4 -> C5 STRUCTURE")
    print(pd.DataFrame(sr).to_string(index=False))
    print("\\nSaved instrument behaviour files to:", OUTPUT_DIR.resolve())
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

    # Descriptive instrument-behaviour study. This is deliberately
    # independent of the ML classifier and does not optimize for a
    # fixed 750/800-pip objective.
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
