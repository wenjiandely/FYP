"""Deterministic time-based split utilities for V1."""

from __future__ import annotations

import pandas as pd

SPLIT_BOUNDS = {
    "train": (pd.Timestamp("2019-01-02"), pd.Timestamp("2023-12-31")),
    "validation": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
    "test": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
}


def assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign train, validation, or test using inclusive calendar bounds."""

    if "date" not in frame.columns:
        raise ValueError("frame must contain a date column")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["split"] = pd.Series(pd.NA, index=result.index, dtype="string")
    for split_name, (start, end) in SPLIT_BOUNDS.items():
        mask = result["date"].between(start, end, inclusive="both")
        result.loc[mask, "split"] = split_name
    if result["split"].isna().any():
        dates = result.loc[result["split"].isna(), "date"]
        raise ValueError(f"records fall outside configured split bounds: {dates.min()} to {dates.max()}")
    return result


def summarize_splits(split_frame: pd.DataFrame) -> pd.DataFrame:
    """Return split counts and date ranges by split and symbol."""

    required = {"symbol", "date", "split"}
    missing = sorted(required - set(split_frame.columns))
    if missing:
        raise ValueError(f"split data is missing columns: {missing}")
    summary = (
        split_frame.groupby(["split", "symbol"], observed=False)
        .agg(
            row_count=("date", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .reset_index()
        .sort_values(["split", "symbol"])
    )
    summary["start_date"] = summary["start_date"].dt.date.astype(str)
    summary["end_date"] = summary["end_date"].dt.date.astype(str)
    return summary
