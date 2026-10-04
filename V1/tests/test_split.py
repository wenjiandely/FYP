from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.split import assign_splits, summarize_splits


def split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2023-12-29", "2024-01-02", "2025-01-02"]),
            "feature": [1.0, 2.0, 3.0],
        }
    )


def test_assign_splits_has_non_overlapping_fixed_periods() -> None:
    result = assign_splits(split_frame())
    assert result["split"].tolist() == ["train", "validation", "test"]
    assert result.groupby("split")["date"].agg(["min", "max"]).shape[0] == 3


def test_split_summary_contains_counts_and_ranges() -> None:
    result = assign_splits(split_frame())
    summary = summarize_splits(result)
    assert summary["row_count"].sum() == 3
    assert set(summary["split"]) == {"train", "validation", "test"}


def test_dates_outside_configured_bounds_fail() -> None:
    frame = split_frame()
    frame.loc[0, "date"] = pd.Timestamp("2018-12-31")
    with pytest.raises(ValueError, match="outside configured split bounds"):
        assign_splits(frame)
