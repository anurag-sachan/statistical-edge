#!/usr/bin/env python3
"""
BTC ML Results Visualizer
=========================

Reads the CSV files produced by the BTCUSD 2H -> 4H ML research script
and creates a visual report under:

    btc_ml_results/visualizations/

Expected input files include:
    correlations.csv
    conditional_patterns.csv
    extension_by_range_regime.csv
    model_*_metrics.csv
    model_*_predictions.csv
    regression_*_metrics.csv
    feature_importance_*.csv
    instrument_behavior_rows.csv
    instrument_behavior_by_range_zone.csv
    range_penetration_by_zone.csv
    reference_excursion_levels.csv
    C3C4_C5_structure_behavior.csv

Usage:
    python visualize_btc_ml_results.py

Optional:
    python visualize_btc_ml_results.py --results-dir btc_ml_results
    python visualize_btc_ml_results.py --top-n 20
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_RESULTS_DIR = Path("btc_ml_results")
DEFAULT_TOP_N = 20


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[WARN] Empty file: {path}")
            return None
        return df
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def save_fig(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output}")


def clean_name(name: str) -> str:
    return (
        str(name)
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def find_files(results_dir: Path, pattern: str) -> list[Path]:
    return sorted(results_dir.glob(pattern))


def no_data(output_dir: Path, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(
        0.5,
        0.5,
        f"No data available\n{title}",
        ha="center",
        va="center",
        fontsize=16,
    )
    ax.set_axis_off()
    save_fig(fig, output_dir / filename)


# ---------------------------------------------------------------------
# 1. Walk-forward model performance
# ---------------------------------------------------------------------

def plot_model_metrics(results_dir: Path, output_dir: Path) -> None:
    files = find_files(results_dir, "model_*_metrics.csv")

    rows = []

    for path in files:
        df = safe_read_csv(path)
        if df is None:
            continue

        required = {"split", "accuracy", "balanced_accuracy", "roc_auc"}
        if not required.issubset(df.columns):
            continue

        target = path.stem.replace("model_", "").replace("_metrics", "")

        temp = df.copy()
        temp["target"] = target
        rows.append(temp)

    if not rows:
        no_data(output_dir, "Walk-forward model metrics", "01_model_performance.png")
        return

    all_metrics = pd.concat(rows, ignore_index=True)

    # -------------------------------------------------------------
    # Average metrics by target
    # -------------------------------------------------------------
    summary = (
        all_metrics.groupby("target")[
            ["accuracy", "balanced_accuracy", "roc_auc"]
        ]
        .mean()
        .sort_values("balanced_accuracy", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(summary))
    width = 0.25

    ax.bar(x - width, summary["accuracy"], width, label="Accuracy")
    ax.bar(x, summary["balanced_accuracy"], width, label="Balanced accuracy")
    ax.bar(x + width, summary["roc_auc"], width, label="ROC-AUC")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [clean_name(v) for v in summary.index],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Walk-forward ML performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_fig(fig, output_dir / "01_model_performance.png")

    # -------------------------------------------------------------
    # Split-by-split performance
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))

    for target, group in all_metrics.groupby("target"):
        group = group.sort_values("split")
        ax.plot(
            group["split"],
            group["balanced_accuracy"],
            marker="o",
            label=clean_name(target),
        )

    ax.axhline(0.5, linestyle="--", linewidth=1, label="50% reference")
    ax.set_xlabel("Walk-forward split")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Balanced accuracy across walk-forward splits")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    save_fig(fig, output_dir / "02_walk_forward_stability.png")


# ---------------------------------------------------------------------
# 2. Classification prediction quality
# ---------------------------------------------------------------------

def plot_prediction_probabilities(
    results_dir: Path,
    output_dir: Path,
) -> None:
    files = find_files(results_dir, "model_*_predictions.csv")

    for path in files:
        df = safe_read_csv(path)
        if df is None:
            continue

        required = {"actual", "prediction", "probability", "split"}
        if not required.issubset(df.columns):
            continue

        target = path.stem.replace("model_", "").replace("_predictions", "")

        # Probability distribution
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            df.loc[df["actual"] == 0, "probability"],
            bins=30,
            alpha=0.65,
            label="Actual 0",
        )
        ax.hist(
            df.loc[df["actual"] == 1, "probability"],
            bins=30,
            alpha=0.65,
            label="Actual 1",
        )

        ax.axvline(0.5, linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted probability of class 1")
        ax.set_ylabel("Observations")
        ax.set_title(f"Prediction probability distribution — {clean_name(target)}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        save_fig(
            fig,
            output_dir / f"03_probability_{target}.png",
        )

        # Actual vs predicted confusion-style counts
        confusion = (
            pd.crosstab(
                df["actual"],
                df["prediction"],
                rownames=["Actual"],
                colnames=["Prediction"],
            )
            .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(confusion.values)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["Actual 0", "Actual 1"])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        ax.set_title(f"Prediction matrix — {clean_name(target)}")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    int(confusion.iloc[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                )

        fig.colorbar(im, ax=ax, label="Count")
        save_fig(
            fig,
            output_dir / f"04_confusion_{target}.png",
        )


# ---------------------------------------------------------------------
# 3. Feature importance
# ---------------------------------------------------------------------

def plot_feature_importance(
    results_dir: Path,
    output_dir: Path,
    top_n: int,
) -> None:
    files = find_files(results_dir, "feature_importance_*.csv")

    for path in files:
        df = safe_read_csv(path)
        if df is None:
            continue

        required = {"feature", "importance"}
        if not required.issubset(df.columns):
            continue

        target = path.stem.replace("feature_importance_", "")

        df = (
            df[["feature", "importance"]]
            .dropna()
            .sort_values("importance", ascending=False)
            .head(top_n)
            .sort_values("importance")
        )

        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))

        ax.barh(df["feature"], df["importance"])
        ax.set_xlabel("Feature importance")
        ax.set_ylabel("Feature")
        ax.set_title(
            f"Top {len(df)} features — {clean_name(target)}"
        )
        ax.grid(axis="x", alpha=0.25)

        save_fig(
            fig,
            output_dir / f"05_feature_importance_{target}.png",
        )


# ---------------------------------------------------------------------
# 4. Correlations
# ---------------------------------------------------------------------

def plot_correlations(
    results_dir: Path,
    output_dir: Path,
    top_n: int,
) -> None:
    path = results_dir / "correlations.csv"
    df = safe_read_csv(path)

    if df is None:
        no_data(output_dir, "Correlation analysis", "06_correlations.png")
        return

    required = {"feature", "target", "spearman_r", "abs_spearman"}
    if not required.issubset(df.columns):
        no_data(output_dir, "Correlation analysis", "06_correlations.png")
        return

    df = (
        df.sort_values("abs_spearman", ascending=False)
        .head(top_n)
        .sort_values("spearman_r")
    )

    labels = [
        f"{row.feature} → {row.target}"
        for row in df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(13, max(7, len(df) * 0.35)))

    ax.barh(labels, df["spearman_r"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("Feature → target")
    ax.set_title(f"Top {len(df)} feature/target correlations")
    ax.grid(axis="x", alpha=0.25)

    save_fig(fig, output_dir / "06_correlations.png")


# ---------------------------------------------------------------------
# 5. Conditional pattern edge
# ---------------------------------------------------------------------

def plot_conditional_patterns(
    results_dir: Path,
    output_dir: Path,
    top_n: int,
) -> None:
    path = results_dir / "conditional_patterns.csv"
    df = safe_read_csv(path)

    if df is None:
        no_data(output_dir, "Conditional pattern analysis", "07_conditional_edges.png")
        return

    required = {"pattern", "target", "baseline", "conditional", "edge"}
    if not required.issubset(df.columns):
        no_data(output_dir, "Conditional pattern analysis", "07_conditional_edges.png")
        return

    df = (
        df.assign(abs_edge=df["edge"].abs())
        .sort_values("abs_edge", ascending=False)
        .head(top_n)
        .sort_values("edge")
    )

    labels = [
        f"{row.pattern} → {row.target}"
        for row in df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(13, max(7, len(df) * 0.38)))

    ax.barh(labels, df["edge"] * 100)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Conditional edge vs baseline (percentage points)")
    ax.set_ylabel("Pattern → target")
    ax.set_title("Strongest conditional pattern edges")
    ax.grid(axis="x", alpha=0.25)

    save_fig(fig, output_dir / "07_conditional_edges.png")

    # Probability comparison: baseline vs conditional
    df2 = df.sort_values("abs_edge", ascending=False).head(min(12, len(df)))
    x = np.arange(len(df2))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(
        x - width / 2,
        df2["baseline"] * 100,
        width,
        label="Baseline",
    )
    ax.bar(
        x + width / 2,
        df2["conditional"] * 100,
        width,
        label="Conditional",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{r.pattern}\n{r.target}"
            for r in df2.itertuples()
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Probability (%)")
    ax.set_title("Baseline vs conditional probability")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_fig(fig, output_dir / "08_baseline_vs_conditional.png")


# ---------------------------------------------------------------------
# 6. Extension / range regime
# ---------------------------------------------------------------------

def plot_extension_regimes(
    results_dir: Path,
    output_dir: Path,
) -> None:
    path = results_dir / "extension_by_range_regime.csv"
    df = safe_read_csv(path)

    if df is None:
        no_data(output_dir, "Range regime analysis", "09_range_regimes.png")
        return

    # This CSV is written by pandas groupby().agg(), so columns may be
    # represented as a MultiIndex when loaded depending on pandas version.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in col if str(x) != "nan").strip("_")
            for col in df.columns
        ]

    # The first column is usually the range_regime index after CSV export.
    if "range_regime" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "range_regime"})

    mean_cols = [
        c for c in df.columns
        if c.endswith("_mean")
    ]

    if not mean_cols:
        no_data(output_dir, "Range regime analysis", "09_range_regimes.png")
        return

    # Plot mean C5/C6/C7 high and low excursion separately.
    preferred = [
        "C5_high_excursion_mean",
        "C5_low_excursion_mean",
        "C6_high_excursion_mean",
        "C6_low_excursion_mean",
        "C7_high_excursion_mean",
        "C7_low_excursion_mean",
    ]

    cols = [c for c in preferred if c in df.columns]

    if not cols:
        cols = mean_cols[:6]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(df))
    width = 0.12
    offsets = (
        np.arange(len(cols)) - (len(cols) - 1) / 2
    ) * width

    for offset, col in zip(offsets, cols):
        ax.bar(
            x + offset,
            df[col],
            width,
            label=clean_name(col.replace("_mean", "")),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["range_regime"].astype(str))
    ax.set_xlabel("C4 range regime")
    ax.set_ylabel("Mean normalized excursion")
    ax.set_title("Future excursion by C4 range regime")
    ax.axhline(0, linewidth=1)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    save_fig(fig, output_dir / "09_range_regimes.png")


# ---------------------------------------------------------------------
# 7. Regression metrics
# ---------------------------------------------------------------------

def plot_regression_metrics(
    results_dir: Path,
    output_dir: Path,
) -> None:
    files = find_files(results_dir, "regression_*_metrics.csv")

    rows = []

    for path in files:
        df = safe_read_csv(path)
        if df is None:
            continue

        required = {
            "split",
            "mae",
            "rmse",
            "prediction_correlation",
        }

        if not required.issubset(df.columns):
            continue

        target = path.stem.replace("regression_", "").replace("_metrics", "")

        temp = df.copy()
        temp["target"] = target
        rows.append(temp)

    if not rows:
        no_data(output_dir, "Regression metrics", "10_regression_metrics.png")
        return

    all_metrics = pd.concat(rows, ignore_index=True)

    summary = (
        all_metrics.groupby("target")[
            ["mae", "rmse", "prediction_correlation"]
        ]
        .mean()
    )

    # MAE / RMSE
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(summary))
    width = 0.36

    ax.bar(
        x - width / 2,
        summary["mae"],
        width,
        label="MAE",
    )
    ax.bar(
        x + width / 2,
        summary["rmse"],
        width,
        label="RMSE",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [clean_name(v) for v in summary.index],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Error")
    ax.set_title("Walk-forward regression error")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_fig(fig, output_dir / "10_regression_errors.png")

    # Prediction correlation
    fig, ax = plt.subplots(figsize=(14, 7))

    corr = summary["prediction_correlation"].sort_values()

    ax.bar(
        [clean_name(v) for v in corr.index],
        corr.values,
    )
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Prediction correlation")
    ax.set_title("Regression prediction correlation")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)

    save_fig(fig, output_dir / "11_regression_correlation.png")



# ---------------------------------------------------------------------
# 8. C5 HIGH/LOW -> C6 instrument behaviour / projection research
# ---------------------------------------------------------------------

ZONE_ORDER = [
    "0-10%", "10-25%", "25-40%", "40-50%",
    "50-60%", "60-75%", "75-90%", "90-100%",
]


def _ordered_zone(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.Categorical(df[column], categories=ZONE_ORDER, ordered=True)
    return df.sort_values(column)


def plot_instrument_behavior_by_zone(results_dir: Path, output_dir: Path) -> None:
    """
    Compare the two anchors: C4 close across the full C5+C6 path versus
    C5 HIGH/LOW into the extended C6 candle.
    """
    path = results_dir / "c5c6_behavior_by_side_zone.csv"
    df = safe_read_csv(path)
    if df is None:
        no_data(output_dir, "C4 close vs C5 HIGH/LOW MFE/MAE", "12_instrument_behavior_by_zone.png")
        return

    required = {
        "reference_side", "C5_zone", "n",
        "median_MFE_pips", "median_MAE_pips",
        "median_MFE_from_C4_pips", "median_MAE_from_C4_pips",
        "median_C4_to_C5_reference_pips",
    }
    if not required.issubset(df.columns):
        no_data(output_dir, "C4 close vs C5 HIGH/LOW MFE/MAE", "12_instrument_behavior_by_zone.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    for ax, side in zip(axes, ["up_from_C5_high", "down_from_C5_low"]):
        sub = _ordered_zone(df[df["reference_side"] == side], "C5_zone")
        x = np.arange(len(sub))
        width = 0.18

        # Full C5+C6 travel from C4 close.
        ax.bar(x - 1.5 * width, sub["median_MFE_from_C4_pips"], width, label="Median MFE from C4")
        ax.bar(x - 0.5 * width, sub["median_MAE_from_C4_pips"], width, label="Median MAE from C4")

        # C6 extension/adverse movement after C5 establishes its H/L.
        ax.bar(x + 0.5 * width, sub["median_MFE_pips"], width, label="Median MFE from C5 H/L")
        ax.bar(x + 1.5 * width, sub["median_MAE_pips"], width, label="Median MAE from C5 H/L")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["C5_zone"].astype(str))
        ax.set_xlabel("C5 reference location inside prior 8H range")
        ax.set_title(clean_name(side))
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Maximum excursion (pips)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "C4 close → C5 HIGH/LOW → C6: MFE and MAE comparison\n"
        "C4 = full C5+C6 travel; C5 H/L = C6 extension beyond the C5 extreme"
    )
    save_fig(fig, output_dir / "12_instrument_behavior_by_zone.png")


def plot_range_penetration(results_dir: Path, output_dir: Path) -> None:
    """C5+C6 maximum projection as a percentage of the prior 8H range."""
    path = results_dir / "range_penetration_by_zone.csv"
    df = safe_read_csv(path)
    if df is None:
        no_data(output_dir, "C5/C6 max projection", "13_range_penetration.png")
        return

    required = {"reference_side", "C5_zone", "median_MFE_pct_prior_8h", "p75_MFE_pct_prior_8h"}
    if not required.issubset(df.columns):
        no_data(output_dir, "C5/C6 max projection", "13_range_penetration.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, side in zip(axes, ["up_from_C5_high", "down_from_C5_low"]):
        sub = _ordered_zone(df[df["reference_side"] == side], "C5_zone")
        x = np.arange(len(sub))
        width = 0.28
        ax.bar(x - width / 2, sub["median_MFE_pct_prior_8h"], width, label="Median projection")
        ax.bar(x + width / 2, sub["p75_MFE_pct_prior_8h"], width, label="75th percentile projection")
        ax.axhline(100, linestyle="--", linewidth=1, label="100% of prior 8H range")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["C5_zone"].astype(str))
        ax.set_xlabel("C5 reference location inside prior 8H range")
        ax.set_title(clean_name(side))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Maximum C5+C6 projection (% of prior 8H range)")
    axes[0].legend(fontsize=8)
    fig.suptitle("C5 HIGH / LOW anchored max projection across C5+C6")
    save_fig(fig, output_dir / "13_range_penetration.png")


def plot_reference_excursion_levels(results_dir: Path, output_dir: Path) -> None:
    """Probability that C6 extends X pips beyond C5 HIGH or C5 LOW."""
    path = results_dir / "reference_excursion_levels.csv"
    df = safe_read_csv(path)
    if df is None:
        no_data(output_dir, "C5 HIGH/LOW excursion levels", "14_reference_excursion_levels.png")
        return

    required = {"zone", "direction", "level_pips", "hit_probability"}
    if not required.issubset(df.columns):
        no_data(output_dir, "C5 HIGH/LOW excursion levels", "14_reference_excursion_levels.png")
        return

    summary = (
        df.groupby(["direction", "level_pips"], as_index=False)["hit_probability"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    for direction, group in summary.groupby("direction"):
        group = group.sort_values("level_pips")
        ax.plot(
            group["level_pips"], group["hit_probability"] * 100,
            marker="o", label=clean_name(direction),
        )

    ax.set_xlabel("C6 extension beyond C5 HIGH / LOW (pips)")
    ax.set_ylabel("Probability of reaching level (%)")
    ax.set_title(
        "C6 continuation probability from C5 HIGH / LOW\n"
        "C5 is the reference; not C4 close and not TP-before-SL"
    )
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.25)
    save_fig(fig, output_dir / "14_reference_excursion_levels.png")


def plot_reference_excursion_by_zone(results_dir: Path, output_dir: Path) -> None:
    """C5 HIGH/LOW excursion probabilities split by C5 location."""
    path = results_dir / "reference_excursion_levels.csv"
    df = safe_read_csv(path)
    if df is None:
        no_data(output_dir, "C5 HIGH/LOW excursion by zone", "15_excursion_levels_by_zone.png")
        return

    required = {"zone", "direction", "level_pips", "hit_probability"}
    if not required.issubset(df.columns):
        no_data(output_dir, "C5 HIGH/LOW excursion by zone", "15_excursion_levels_by_zone.png")
        return

    level_order = sorted(df["level_pips"].dropna().unique())
    levels = [v for v in [500, 750, 800, 1000] if v in level_order]
    if not levels:
        levels = level_order

    zone_names = [z for z in ZONE_ORDER if z in df["zone"].astype(str).unique()]
    if not zone_names:
        zone_names = sorted(df["zone"].astype(str).unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, direction in zip(axes, ["up_from_C5_high", "down_from_C5_low"]):
        sub = df[(df["direction"] == direction) & (df["level_pips"].isin(levels))].copy()
        matrix = (
            sub.pivot_table(index="zone", columns="level_pips", values="hit_probability", aggfunc="mean")
            .reindex(index=zone_names, columns=levels)
        )
        im = ax.imshow(matrix.values * 100, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels([str(int(v)) for v in matrix.columns])
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index.astype(str))
        ax.set_xlabel("C6 extension beyond C5 reference (pips)")
        ax.set_title(clean_name(direction))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value * 100:.1f}%", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label="Probability (%)")

    axes[0].set_ylabel("C5 high position inside prior 8H range")
    fig.suptitle(
        "C6 extension probability from C5 HIGH / LOW by C5 location\n"
        "C4 close is context only"
    )
    save_fig(fig, output_dir / "15_excursion_levels_by_zone.png")


def plot_c3c4_c5_structure(results_dir: Path, output_dir: Path) -> None:
    """C5 structure -> C6 continuation/reversion."""
    path = results_dir / "C3C4_C5_structure_behavior.csv"
    df = safe_read_csv(path)
    if df is None:
        no_data(output_dir, "C5 structure -> C6", "16_C3C4_C5_structure.png")
        return

    required = {
        "pattern", "n", "C6_break_C5_high", "C6_break_C5_low",
        "C6_close_above_C5_high", "C6_close_below_C5_low",
    }
    if not required.issubset(df.columns):
        no_data(output_dir, "C5 structure -> C6", "16_C3C4_C5_structure.png")
        return

    df = df.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(14, max(7, len(df) * 0.45)))
    y = np.arange(len(df))
    width = 0.19
    ax.barh(y - 1.5 * width, df["C6_break_C5_high"] * 100, width, label="C6 breaks C5 high")
    ax.barh(y - 0.5 * width, df["C6_break_C5_low"] * 100, width, label="C6 breaks C5 low")
    ax.barh(y + 0.5 * width, df["C6_close_above_C5_high"] * 100, width, label="C6 closes above C5 high")
    ax.barh(y + 1.5 * width, df["C6_close_below_C5_low"] * 100, width, label="C6 closes below C5 low")
    ax.set_yticks(y)
    ax.set_yticklabels([clean_name(v) for v in df["pattern"]])
    ax.set_xlabel("Probability (%)")
    ax.set_title("C5 structure as the signal; C6 High/Low as the outcome")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, output_dir / "16_C3C4_C5_structure.png")


def plot_mfe_mae_distribution(results_dir: Path, output_dir: Path) -> None:
    """
    Distribution comparison:
      - C4-close anchored MFE/MAE over the complete C5+C6 path
      - C5 HIGH/LOW anchored MFE/MAE during the extended C6 candle
    """
    path = results_dir / "instrument_behavior_rows.csv"
    df = safe_read_csv(path)
    if df is None:
        no_data(output_dir, "C4 vs C5 anchored MFE/MAE", "17_mfe_mae_distribution.png")
        return

    required = {
        "MFE_long_C4_close_C5C6_pips", "MAE_long_C4_close_C5C6_pips",
        "MFE_short_C4_close_C5C6_pips", "MAE_short_C4_close_C5C6_pips",
        "MFE_long_C5_high_pips", "MAE_long_C5_high_pips",
        "MFE_short_C5_low_pips", "MAE_short_C5_low_pips",
    }
    if not required.issubset(df.columns):
        no_data(output_dir, "C4 vs C5 anchored MFE/MAE", "17_mfe_mae_distribution.png")
        return

    all_values = pd.concat([
        df[c] for c in required
    ]).dropna()
    if all_values.empty:
        no_data(output_dir, "C4 vs C5 anchored MFE/MAE", "17_mfe_mae_distribution.png")
        return

    xmax = all_values.quantile(0.99)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)

    panels = [
        (axes[0], "Long side",
         "MFE_long_C4_close_C5C6_pips", "MAE_long_C4_close_C5C6_pips",
         "MFE_long_C5_high_pips", "MAE_long_C5_high_pips",
         "C4 close → full C5+C6", "C5 HIGH → C6"),
        (axes[1], "Short side",
         "MFE_short_C4_close_C5C6_pips", "MAE_short_C4_close_C5C6_pips",
         "MFE_short_C5_low_pips", "MAE_short_C5_low_pips",
         "C4 close → full C5+C6", "C5 LOW → C6"),
    ]

    for ax, title, c4_mfe, c4_mae, c5_mfe, c5_mae, c4_label, c5_label in panels:
        ax.hist(df[c4_mfe].clip(upper=xmax), bins=50, alpha=0.42, label=f"{c4_label} MFE")
        ax.hist(df[c4_mae].clip(upper=xmax), bins=50, alpha=0.28, label=f"{c4_label} MAE")
        ax.hist(df[c5_mfe].clip(upper=xmax), bins=50, alpha=0.42, label=f"{c5_label} MFE")
        ax.hist(df[c5_mae].clip(upper=xmax), bins=50, alpha=0.28, label=f"{c5_label} MAE")
        ax.set_title(title)
        ax.set_xlabel("Maximum excursion (pips)")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Observations")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "MFE / MAE: C4-close context versus C5 HIGH/LOW + extended C6\n"
        "C4 measures total travel through C5+C6; C5 measures the post-C5 extension"
    )
    save_fig(fig, output_dir / "17_mfe_mae_distribution.png")


def plot_first_break_behaviour(results_dir: Path, output_dir: Path) -> None:
    """Show C6 continuation/rejection after C5 establishes HIGH/LOW."""
    path = results_dir / "c5c6_behavior_by_side_zone.csv"
    df = safe_read_csv(path)
    if df is None:
        no_data(output_dir, "C5 HIGH/LOW -> C6", "18_first_break_behaviour.png")
        return

    required = {"reference_side", "C5_zone", "C6_break_reference", "C6_close_beyond_reference"}
    if not required.issubset(df.columns):
        no_data(output_dir, "C5 HIGH/LOW -> C6", "18_first_break_behaviour.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, side in zip(axes, ["up_from_C5_high", "down_from_C5_low"]):
        sub = _ordered_zone(df[df["reference_side"] == side], "C5_zone")
        x = np.arange(len(sub))
        width = 0.32
        ax.bar(x - width / 2, sub["C6_break_reference"] * 100, width, label="C6 breaks C5 reference")
        ax.bar(x + width / 2, sub["C6_close_beyond_reference"] * 100, width, label="C6 closes beyond C5 reference")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["C5_zone"].astype(str))
        ax.set_xlabel("C5 reference location inside prior 8H range")
        ax.set_title(clean_name(side))
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Probability (%)")
    axes[0].legend()
    fig.suptitle("C5 HIGH / LOW -> C6 continuation and close beyond the reference")
    save_fig(fig, output_dir / "18_first_break_behaviour.png")


# 9. Consolidated summary CSV
# ---------------------------------------------------------------------

def build_summary(results_dir: Path, output_dir: Path) -> None:
    rows = []

    for path in find_files(results_dir, "model_*_metrics.csv"):
        df = safe_read_csv(path)
        if df is None:
            continue

        cols = [
            c for c in
            ["accuracy", "balanced_accuracy", "roc_auc"]
            if c in df.columns
        ]

        if not cols:
            continue

        target = path.stem.replace("model_", "").replace("_metrics", "")

        row = {"model": target}

        for col in cols:
            row[f"mean_{col}"] = df[col].mean()
            row[f"std_{col}"] = df[col].std()

        rows.append(row)

    if rows:
        summary = pd.DataFrame(rows).sort_values(
            "mean_balanced_accuracy",
            ascending=False,
        )
        summary.to_csv(
            output_dir / "model_summary.csv",
            index=False,
        )

        print(f"[OK] {output_dir / 'model_summary.csv'}")


# ---------------------------------------------------------------------
# 10. HTML report
# ---------------------------------------------------------------------

def build_html_report(
    results_dir: Path,
    output_dir: Path,
) -> None:
    images = sorted(output_dir.glob("*.png"))

    model_summary = output_dir / "model_summary.csv"

    summary_html = ""

    if model_summary.exists():
        df = pd.read_csv(model_summary)

        # Format numerical values for readability.
        formatted = df.copy()

        for col in formatted.columns:
            if col != "model":
                formatted[col] = formatted[col].map(
                    lambda x: f"{x:.4f}"
                    if pd.notna(x)
                    else ""
                )

        summary_html = (
            "<h2>Model summary</h2>"
            + formatted.to_html(
                index=False,
                classes="summary",
                border=0,
            )
        )

    image_html = []

    for image in images:
        image_html.append(
            f"""
            <section>
                <h2>{clean_name(image.stem)}</h2>
                <img src="{image.name}" alt="{image.stem}">
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BTC ML Results Dashboard</title>
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 1500px;
    margin: 0 auto;
    padding: 30px;
    background: #f5f5f5;
    color: #222;
}}

h1 {{
    margin-bottom: 5px;
}}

.subtitle {{
    color: #666;
    margin-bottom: 30px;
}}

section {{
    background: white;
    margin: 25px 0;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 0 auto;
}}

.summary {{
    width: 100%;
    border-collapse: collapse;
}}

.summary th,
.summary td {{
    padding: 9px;
    border-bottom: 1px solid #ddd;
    text-align: right;
}}

.summary th:first-child,
.summary td:first-child {{
    text-align: left;
}}
</style>
</head>

<body>
<h1>BTC ML Results Dashboard</h1>
<div class="subtitle">
Generated from <code>{results_dir}</code>
</div>

{summary_html}

{''.join(image_html)}

</body>
</html>
"""

    report_path = output_dir / "btc_ml_dashboard.html"
    report_path.write_text(html, encoding="utf-8")

    print(f"[OK] {report_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize BTC ML research results."
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing btc_ml_results CSV files.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top features/patterns/correlations to display.",
    )

    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = results_dir / "visualizations"

    if not results_dir.exists():
        raise SystemExit(
            f"Results directory does not exist: {results_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("BTC ML RESULTS VISUALIZER")
    print("=" * 70)
    print(f"Input : {results_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print()

    plot_model_metrics(
        results_dir,
        output_dir,
    )

    plot_prediction_probabilities(
        results_dir,
        output_dir,
    )

    plot_feature_importance(
        results_dir,
        output_dir,
        args.top_n,
    )

    plot_correlations(
        results_dir,
        output_dir,
        args.top_n,
    )

    plot_conditional_patterns(
        results_dir,
        output_dir,
        args.top_n,
    )

    plot_extension_regimes(
        results_dir,
        output_dir,
    )

    plot_regression_metrics(
        results_dir,
        output_dir,
    )

    # New descriptive instrument-behaviour visualizations.
    plot_instrument_behavior_by_zone(
        results_dir,
        output_dir,
    )

    plot_range_penetration(
        results_dir,
        output_dir,
    )

    plot_reference_excursion_levels(
        results_dir,
        output_dir,
    )

    plot_reference_excursion_by_zone(
        results_dir,
        output_dir,
    )

    plot_c3c4_c5_structure(
        results_dir,
        output_dir,
    )

    plot_mfe_mae_distribution(
        results_dir,
        output_dir,
    )

    plot_first_break_behaviour(
        results_dir,
        output_dir,
    )

    build_summary(
        results_dir,
        output_dir,
    )

    build_html_report(
        results_dir,
        output_dir,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("Open:")
    print(output_dir / "btc_ml_dashboard.html")


if __name__ == "__main__":
    main()
