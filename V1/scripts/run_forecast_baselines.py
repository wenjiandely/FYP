"""Run the aligned five-session ARIMA/XGBoost baseline experiment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "V1"))

from src.models.forecast_baselines import (
    MODEL_FEATURES,
    PATH_TARGETS,
    PRICE_TARGETS,
    predict_arima,
    predict_naive,
    predict_xgboost,
)
from src.targets.forecast_targets import build_forecast_targets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.duplicated(["model", "split", "symbol", "date"]).any():
        raise ValueError("Duplicate model/split/symbol/date forecast origins")
    model_names = ["naive_zero", "naive_last", "arima", "xgboost"]
    rows = []
    for model in model_names:
        subset = predictions.loc[predictions["model"].eq(model)] if "model" in predictions else pd.DataFrame()
        if subset.empty:
            continue
        for (split, symbol), group in subset.groupby(["split", "symbol"]):
            required_values = [
                "adj_close", *PATH_TARGETS, *PRICE_TARGETS,
                "target_mean_price_return_5d", "target_cumulative_log_return_5d",
                *[f"{model}_pred_return_{step}d" for step in range(1, 6)],
                *[f"{model}_pred_adj_close_{step}d" for step in range(1, 6)],
                f"{model}_pred_mean_price_return_5d",
            ]
            if not np.isfinite(group[required_values].to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite scoring values: {model}/{split}/{symbol}")
            for step, target in enumerate(PATH_TARGETS, start=1):
                error = group[f"{model}_pred_return_{step}d"] - group[target]
                rows.append({"model": model, "split": split, "symbol": symbol, "metric": f"return_{step}d_mae", "value": error.abs().mean(), "n": len(error)})
                rows.append({"model": model, "split": split, "symbol": symbol, "metric": f"return_{step}d_rmse", "value": np.sqrt(np.mean(error ** 2)), "n": len(error)})
                normalized_price_error = (
                    group[f"{model}_pred_adj_close_{step}d"] - group[f"target_adj_close_{step}d"]
                ) / group["adj_close"]
                rows.append({"model": model, "split": split, "symbol": symbol, "metric": f"normalized_price_{step}d_mae", "value": normalized_price_error.abs().mean(), "n": len(normalized_price_error)})
                rows.append({"model": model, "split": split, "symbol": symbol, "metric": f"normalized_price_{step}d_rmse", "value": np.sqrt(np.mean(normalized_price_error ** 2)), "n": len(normalized_price_error)})
                for metric, value in _direction_metrics(group[target], group[f"{model}_pred_return_{step}d"]).items():
                    rows.append({"model": model, "split": split, "symbol": symbol, "metric": f"return_{step}d_{metric}", "value": value, "n": len(group)})
            for target, pred, metric_names in (
                ("target_mean_price_return_5d", f"{model}_pred_mean_price_return_5d", ("mean_return_mae", "mean_return_rmse")),
                ("target_cumulative_log_return_5d", f"{model}_pred_return_5d_cumulative", ("endpoint_return_mae", "endpoint_return_rmse")),
            ):
                if target.startswith("target_cumulative"):
                    group = group.copy()
                    group[pred] = group[[f"{model}_pred_return_{i}d" for i in range(1, 6)]].sum(axis=1)
                error = group[pred] - group[target]
                rows.append({"model": model, "split": split, "symbol": symbol, "metric": metric_names[0], "value": error.abs().mean(), "n": len(error)})
                rows.append({"model": model, "split": split, "symbol": symbol, "metric": metric_names[1], "value": np.sqrt(np.mean(error ** 2)), "n": len(error)})
            for prefix, actual, predicted in (
                ("mean_trend", group["target_mean_price_return_5d"], group[f"{model}_pred_mean_price_return_5d"]),
                ("endpoint", group["target_cumulative_log_return_5d"], group[[f"{model}_pred_return_{step}d" for step in range(1, 6)]].sum(axis=1)),
            ):
                for metric, value in _direction_metrics(actual, predicted).items():
                    rows.append({"model": model, "split": split, "symbol": symbol, "metric": f"{prefix}_{metric}", "value": value, "n": len(group)})
    result = pd.DataFrame(rows)
    macro = (
        result.groupby(["model", "split", "metric"], as_index=False)
        .agg(value=("value", "mean"), n=("symbol", "nunique"))
    )
    macro["symbol"] = "MACRO"
    return pd.concat([result, macro[result.columns]], ignore_index=True)


def _direction_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Score up versus non-up; majority is a retrospective class-balance reference."""
    actual_up = actual.gt(0)
    predicted_up = predicted.gt(0)
    up_count = int(actual_up.sum())
    non_up_count = len(actual_up) - up_count
    up_recall = predicted_up.loc[actual_up].mean() if up_count else np.nan
    non_up_recall = (~predicted_up.loc[~actual_up]).mean() if non_up_count else np.nan
    up_rate = float(actual_up.mean())
    return {
        "direction_accuracy": float(predicted_up.eq(actual_up).mean()),
        "balanced_accuracy": float((up_recall + non_up_recall) / 2),
        "actual_up_rate": up_rate,
        "predicted_up_rate": float(predicted_up.mean()),
        "majority_baseline_accuracy": max(up_rate, 1 - up_rate),
        "random_expected_accuracy": 0.5,
    }


def _plot_mean_price_return(predictions: pd.DataFrame, output_path: Path) -> None:
    sample = predictions.loc[predictions["split"].eq("test")]
    symbols = sorted(sample["symbol"].unique())
    figure, axes = plt.subplots(len(symbols), 1, figsize=(13, 3 * len(symbols)), sharex=True, squeeze=False)
    colors = {"arima": "#2676b8", "xgboost": "#d66b27"}
    for axis, symbol in zip(axes[:, 0], symbols):
        part = sample.loc[sample["symbol"].eq(symbol)]
        actual = part[["date", "target_mean_price_return_5d"]].drop_duplicates("date").sort_values("date")
        axis.plot(actual["date"], actual["target_mean_price_return_5d"], label="Realized target (observed)", color="#222222", linewidth=1.1)
        for model in ("arima", "xgboost"):
            model_rows = part.loc[part["model"].eq(model)].sort_values("date")
            axis.plot(model_rows["date"], model_rows[f"{model}_pred_mean_price_return_5d"], label=f"{model.upper()} forecast", color=colors[model], alpha=0.9, linewidth=1)
        axis.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
        axis.set(title=symbol, ylabel="5-session mean-price return")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("Forecast origin")
    figure.suptitle("Test set: five-session mean-price-return summary (derived from forecast path)", y=1.002)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_daily_direction_accuracy(metrics: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {"naive_zero": "#777777", "naive_last": "#a6a6a6", "arima": "#2676b8", "xgboost": "#d66b27"}
    for axis, suffix, title in zip(axes, ("direction_accuracy", "balanced_accuracy"), ("Direction accuracy", "Balanced direction accuracy")):
        for model in ("naive_zero", "naive_last", "arima", "xgboost"):
            rows = metrics.loc[
                metrics["split"].eq("test") & metrics["symbol"].eq("MACRO")
                & metrics["model"].eq(model)
                & metrics["metric"].str.fullmatch(rf"return_[1-5]d_{suffix}")
            ].copy()
            rows["horizon"] = rows["metric"].str.extract(r"return_(\d)d").astype(int)
            rows = rows.sort_values("horizon")
            axis.plot(rows["horizon"], rows["value"], marker="o", label=model.replace("_", " ").upper(), color=colors[model])
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Fair-coin expectation (50%)")
        axis.set(xticks=range(1, 6), xlabel="Future session k (one-session return)", ylabel=title, ylim=(0.4, 0.65), title=title)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.25)
    up_rows = metrics.loc[
        metrics["split"].eq("test") & metrics["symbol"].eq("MACRO") & metrics["model"].eq("naive_zero")
        & metrics["metric"].str.fullmatch(r"return_[1-5]d_actual_up_rate")
    ].sort_values("metric")
    axes[0].plot(range(1, 6), up_rows["value"], color="#519647", marker="s", linestyle="--", label="Always UP (actual up-rate)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("2025 test: each symbol has equal weight; vertical axis shown from 40% to 65%")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_forecast_path_example(predictions: pd.DataFrame, run_dir: Path) -> None:
    """Show the first test origin, fixed before inspecting model performance."""
    test = predictions.loc[predictions["split"].eq("test")]
    origin = test["date"].min()
    sample = test.loc[test["date"].eq(origin)]
    symbols = sorted(sample["symbol"].unique())
    figure, axes = plt.subplots(len(symbols), 2, figsize=(14, 3.3 * len(symbols)), squeeze=False)
    rows = []
    colors = {"actual": "#222222", "arima": "#2676b8", "xgboost": "#d66b27"}
    for axis_pair, symbol in zip(axes, symbols):
        part = sample.loc[sample["symbol"].eq(symbol)]
        reference = part.iloc[0]
        for name in colors:
            source = reference if name == "actual" else part.loc[part["model"].eq(name)].iloc[0]
            return_prefix = "target_return" if name == "actual" else f"{name}_pred_return"
            price_prefix = "target_adj_close" if name == "actual" else f"{name}_pred_adj_close"
            returns = np.array([source[f"{return_prefix}_{step}d"] for step in range(1, 6)])
            prices = np.array([source[f"{price_prefix}_{step}d"] for step in range(1, 6)])
            indexed_prices = 100 * prices / reference["adj_close"]
            label = "Actual (realized later)" if name == "actual" else name.upper()
            axis_pair[0].plot(range(1, 6), returns, marker="o", color=colors[name], label=label)
            axis_pair[1].plot(range(6), np.r_[100, indexed_prices], marker="o", color=colors[name], label=label)
            axis_pair[1].hlines(indexed_prices.mean(), 1, 5, color=colors[name], linestyle=":", linewidth=1.3)
            for step in range(1, 6):
                rows.append({"symbol": symbol, "origin": origin, "series": name, "session": step, "daily_log_return": returns[step - 1], "adj_close": prices[step - 1], "mean_adj_close_5d": prices.mean()})
        axis_pair[0].set(title=f"{symbol}: five future daily returns", xticks=range(1, 6), xlabel="Future session k", ylabel="One-session log return")
        axis_pair[0].yaxis.set_major_formatter(PercentFormatter(1))
        axis_pair[0].axhline(0, color="#777777", linestyle="--", linewidth=0.7)
        axis_pair[1].set(title=f"{symbol}: price paths; dotted = five-price means", xticks=range(6), xlabel="Session (0 = forecast origin)", ylabel="Adjusted close indexed to 100 at origin")
        for axis in axis_pair:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, frameon=False)
    figure.suptitle(f"One forecast issued after {origin:%Y-%m-%d}: no future observation updates within this five-step path", y=1.002)
    figure.tight_layout()
    figure.savefig(run_dir / "five_session_path_example.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(rows).to_csv(run_dir / "five_session_path_example.csv", index=False)


def write_forecast_evaluation(predictions: pd.DataFrame, run_dir: Path) -> None:
    """Write matching per-symbol/macro metrics, readable daily summary and figures."""
    metrics = _score(predictions)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    daily = metrics.loc[metrics["metric"].str.fullmatch(r"return_[1-5]d_(direction_accuracy|balanced_accuracy|actual_up_rate|predicted_up_rate|majority_baseline_accuracy|random_expected_accuracy|mae|rmse)")].copy()
    daily["horizon"] = daily["metric"].str.extract(r"return_(\d)d").astype(int)
    daily["metric"] = daily["metric"].str.replace(r"return_[1-5]d_", "", regex=True)
    summary = daily.pivot(index=["model", "split", "symbol", "horizon"], columns="metric", values="value").reset_index()
    summary.to_csv(run_dir / "daily_forecast_summary.csv", index=False)
    _plot_mean_price_return(predictions, run_dir / "mean_price_return_all_models.png")
    _plot_daily_direction_accuracy(metrics, run_dir / "daily_direction_accuracy.png")
    _plot_forecast_path_example(predictions, run_dir)


def main() -> None:
    data_path = ROOT / "V1/data/processed/market_data_v1.csv"
    feature_path = ROOT / "V1/data/processed/features_v1.csv"
    market = pd.read_csv(data_path, parse_dates=["date"])
    features = pd.read_csv(feature_path, parse_dates=["date"])
    targets = build_forecast_targets(market)
    # The legacy V1 file carries a one-day label with a name reused by the new path contract.
    features = features.drop(columns=["target_return_1d", "target_direction_1d"], errors="ignore")
    frame = features.merge(targets, on=["symbol", "date"], validate="one_to_one")
    frame["feature_complete"] = frame[MODEL_FEATURES].notna().all(axis=1)
    frame["eligible"] = frame["eligible"] & frame["feature_complete"]

    output_root = ROOT / "V1/results/aligned_baselines_v2"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    naive = predict_naive(frame)
    # Keep all models in one long table with an explicit shared schema.
    naive_parts = []
    for model in ("naive_zero", "naive_last"):
        common_columns = {
            "symbol", "date", "split", "adj_close", *PATH_TARGETS, *PRICE_TARGETS,
            "target_mean_price_5d", "target_mean_price_return_5d", "target_cumulative_log_return_5d",
            "target_direction_5d", "target_end_date", "purged",
        }
        part = naive.loc[naive["model"].eq(model)].copy()
        part = part[[column for column in part if column.startswith(model + "_") or column in common_columns | {"model"}]]
        naive_parts.append(part)
    arima = predict_arima(frame, model_dir=run_dir / "models" / "arima")
    arima["model"] = "arima"
    device = os.environ.get("XGB_DEVICE", "cuda:0")
    xgboost, retraining_log = predict_xgboost(
        frame, device=device, model_dir=run_dir / "models" / "xgboost"
    )
    xgboost["model"] = "xgboost"
    combined = pd.concat([*naive_parts, arima, xgboost], ignore_index=True, sort=False)
    origin_sets = {
        model: set(map(tuple, group[["symbol", "date", "split"]].drop_duplicates().to_numpy()))
        for model, group in combined.groupby("model")
    }
    if len(origin_sets) != 4 or len({frozenset(origins) for origins in origin_sets.values()}) != 1:
        counts = {name: len(origins) for name, origins in origin_sets.items()}
        raise RuntimeError(f"models do not share identical forecast origins: {counts}")
    contract_rows = [
        {
            "model": model,
            "forecast_origins": len(origins),
            "symbols": combined.loc[combined["model"].eq(model), "symbol"].nunique(),
            "splits": ";".join(sorted(combined.loc[combined["model"].eq(model), "split"].unique())),
        }
        for model, origins in sorted(origin_sets.items())
    ]
    pd.DataFrame(contract_rows).to_csv(run_dir / "prediction_contract.csv", index=False)
    combined.to_csv(run_dir / "predictions.csv", index=False)
    write_forecast_evaluation(combined, run_dir)
    retraining_log.to_csv(run_dir / "xgboost_retraining_log.csv", index=False)

    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        revision = "unknown"
    summary = {
        "run_id": run_id,
        "code_revision": revision,
        "data_sha256": {"market_data_v1.csv": _sha256(data_path), "features_v1.csv": _sha256(feature_path)},
        "symbols": sorted(market["symbol"].unique().tolist()),
        "target": "next five one-session adjusted-close log returns; reconstructed path and arithmetic mean price",
        "split": {"train": "2019-2023", "validation": "2024", "test": "2025", "purge": "all five target sessions must remain within origin split"},
        "models": {"arima": "per-symbol ARIMA(1,0,1), refit every 21 scored origins; refit checkpoints saved", "xgboost": "pooled five direct regressors with one-hot symbol columns, refit every 63 scored market dates; all horizon checkpoints saved"},
        "xgboost_device": device,
        "seed": 42,
        "fit_artifacts": "New aligned model checkpoints are saved under models/; legacy checkpoints are not reused.",
        "rows": {"target_rows": len(targets), "purged_origins": int(targets["purged"].sum()), "eligible_complete_feature_rows": int(frame["eligible"].sum())},
    }
    (run_dir / "run_config.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Run complete: {run_dir}")
    print(f"XGBoost device: {device}; predictions: {len(combined)}")


if __name__ == "__main__":
    main()
