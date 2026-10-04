"""Run the complete V1 data-to-feature validation pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.validate_market_data import (
    assert_valid,
    load_market_data,
    validate_dataframe,
    write_validation_outputs,
)
from src.evaluation.split import assign_splits, summarize_splits
from src.features.build_features import FEATURE_COLUMNS, build_features
from src.features.build_labels import LABEL_COLUMNS, build_labels, make_model_ready

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "market_data_v1.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def build_feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in FEATURE_COLUMNS:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        rows.append(
            {
                "feature_name": column,
                "data_type": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "min": numeric.min(),
                "max": numeric.max(),
                "mean": numeric.mean(),
                "std": numeric.std(),
                "is_model_input": True,
            }
        )
    return pd.DataFrame(rows)


def create_feature_overview(market_data: pd.DataFrame, labeled_features: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for symbol, group in market_data.groupby("symbol"):
        normalized = group["adj_close"] / group["adj_close"].iloc[0]
        axes[0, 0].plot(group["date"], normalized, label=symbol)
    axes[0, 0].set_title("Normalized adjusted close")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    for symbol, group in labeled_features.groupby("symbol"):
        axes[0, 1].plot(group["date"], group["return_1d"], label=symbol, alpha=0.7)
    axes[0, 1].set_title("Daily log return")
    axes[0, 1].grid(alpha=0.25)

    for symbol, group in labeled_features.groupby("symbol"):
        axes[1, 0].plot(group["date"], group["rolling_volatility_20d"], label=symbol)
    axes[1, 0].set_title("20-day rolling volatility")
    axes[1, 0].grid(alpha=0.25)

    for symbol, group in labeled_features.groupby("symbol"):
        axes[1, 1].plot(group["date"], group["rsi_14"], label=symbol)
    axes[1, 1].axhline(70, color="grey", linestyle="--", linewidth=0.8)
    axes[1, 1].axhline(30, color="grey", linestyle="--", linewidth=0.8)
    axes[1, 1].set_title("14-day RSI")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(alpha=0.25)

    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=30)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_pipeline() -> dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    market_data = load_market_data(DATA_PATH)
    report = validate_dataframe(market_data)
    write_validation_outputs(
        report,
        RESULTS_DIR / "data_validation_report.json",
        RESULTS_DIR / "data_quality_summary.csv",
    )
    assert_valid(report)

    features = build_features(market_data)
    labeled_features = build_labels(features)
    model_ready = make_model_ready(labeled_features)
    split_frame = assign_splits(model_ready)

    labeled_features.to_csv(PROCESSED_DIR / "features_v1.csv", index=False)
    model_ready.to_csv(PROCESSED_DIR / "model_ready_v1.csv", index=False)
    split_frame.to_csv(PROCESSED_DIR / "splits_v1.csv", index=False)

    feature_summary = build_feature_summary(labeled_features)
    feature_summary.to_csv(RESULTS_DIR / "feature_summary_v1.csv", index=False)
    split_summary = summarize_splits(split_frame)
    split_summary.to_csv(RESULTS_DIR / "split_summary_v1.csv", index=False)
    create_feature_overview(market_data, labeled_features, FIGURES_DIR / "feature_overview.png")

    run_summary = {
        "status": "passed",
        "market_rows": int(len(market_data)),
        "feature_rows": int(len(labeled_features)),
        "model_ready_rows": int(len(model_ready)),
        "split_rows": int(len(split_frame)),
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "outputs": [
            "data/processed/features_v1.csv",
            "data/processed/model_ready_v1.csv",
            "data/processed/splits_v1.csv",
            "results/data_validation_report.json",
            "results/data_quality_summary.csv",
            "results/feature_summary_v1.csv",
            "results/split_summary_v1.csv",
            "results/figures/feature_overview.png",
        ],
    }
    (RESULTS_DIR / "v1_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_summary


if __name__ == "__main__":
    summary = run_pipeline()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
