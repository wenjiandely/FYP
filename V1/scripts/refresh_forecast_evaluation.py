"""Recompute metrics and figures from an existing forecast run; does not train."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from run_forecast_baselines import _sha256, write_forecast_evaluation
from src.models.forecast_baselines import _add_forecast_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Existing aligned_baselines_v2 run directory")
    run_dir = parser.parse_args().run_dir.resolve()
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.is_file():
        parser.error(f"predictions.csv not found in {run_dir}")

    original_sha256 = _sha256(predictions_path)
    audit_path = run_dir / "evaluation_refresh.json"
    previous_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    predictions = pd.read_csv(predictions_path, parse_dates=["date"], float_precision="round_trip")
    for model, group in predictions.groupby("model"):
        columns = [f"{model}_pred_return_{step}d" for step in range(1, 6)]
        reconstructed = _add_forecast_outputs(group, group[columns].to_numpy(dtype=float), model)
        for column in reconstructed:
            if column.startswith(f"{model}_pred_"):
                predictions.loc[group.index, column] = reconstructed[column].to_numpy()
    write_forecast_evaluation(predictions, run_dir)
    legacy_figure = run_dir / "mean_trend_all_symbols.png"
    if legacy_figure.exists():
        shutil.copyfile(run_dir / "mean_price_return_all_models.png", legacy_figure)
    predictions.to_csv(predictions_path, index=False)
    audit_path.write_text(json.dumps({
        "original_predictions_sha256": previous_audit.get("original_predictions_sha256", original_sha256),
        "latest_refresh_input_sha256": original_sha256,
        "refreshed_predictions_sha256": _sha256(predictions_path),
        "method": "Preserve all five daily return forecasts; reconstruct prices and mean-price summaries with expm1 for stable zero-return scoring.",
        "retrained": False,
        "direction": "return > 0 versus return <= 0",
        "majority_baseline": "Retrospective per-symbol/per-split class-balance diagnostic, not a test-fitted predictor.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Refreshed evaluation from existing predictions: {run_dir}")
    print("No models were trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
