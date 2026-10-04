"""Build next-day labels and model-ready records for V1."""

from __future__ import annotations

import pandas as pd

from src.features.build_features import FEATURE_COLUMNS

LABEL_COLUMNS = ["target_return_1d", "target_direction_1d"]


def build_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Append next-trading-day return and direction labels."""

    required = {"symbol", "date", "return_1d", *FEATURE_COLUMNS}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"feature data is missing columns: {missing}")

    result = features.sort_values(["symbol", "date"]).copy()
    result["target_return_1d"] = result.groupby("symbol")["return_1d"].shift(-1)
    direction = pd.Series(pd.NA, index=result.index, dtype="Int64")
    known_target = result["target_return_1d"].notna()
    direction.loc[known_target] = result.loc[known_target, "target_return_1d"].gt(0).astype("int64")
    result["target_direction_1d"] = direction
    return result[["symbol", "date"] + FEATURE_COLUMNS + LABEL_COLUMNS].reset_index(drop=True)


def make_model_ready(features_with_labels: pd.DataFrame) -> pd.DataFrame:
    """Remove warm-up and unlabeled rows without imputing values."""

    required = FEATURE_COLUMNS + LABEL_COLUMNS
    missing = sorted(set(required) - set(features_with_labels.columns))
    if missing:
        raise ValueError(f"labeled feature data is missing columns: {missing}")
    return features_with_labels.dropna(subset=required).reset_index(drop=True)
