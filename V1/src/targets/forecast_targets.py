"""Construct aligned five-session path targets and purge split boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.split import SPLIT_BOUNDS


def build_forecast_targets(market_data: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "adj_close"}
    missing = sorted(required - set(market_data.columns))
    if missing:
        raise ValueError(f"market data is missing columns: {missing}")

    frame = market_data[["symbol", "date", "adj_close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    for _, group in frame.groupby("symbol", sort=False):
        group = group.copy()
        prices = group["adj_close"].astype(float)
        for step in range(1, 6):
            future_price = prices.shift(-step)
            prior_price = prices.shift(-(step - 1))
            group[f"target_return_{step}d"] = np.log(future_price / prior_price)
            group[f"target_date_{step}d"] = group["date"].shift(-step)
            group[f"target_adj_close_{step}d"] = future_price
        future_price_columns = [f"target_adj_close_{step}d" for step in range(1, 6)]
        complete_path = group[future_price_columns].notna().all(axis=1)
        group["target_mean_price_5d"] = group[future_price_columns].mean(axis=1).where(complete_path)
        group["target_mean_price_return_5d"] = group["target_mean_price_5d"] / prices - 1
        group["target_cumulative_log_return_5d"] = np.log(prices.shift(-5) / prices)
        group["target_direction_5d"] = (
            group["target_cumulative_log_return_5d"].gt(0).astype("float").where(
                group["target_cumulative_log_return_5d"].notna()
            )
        )
        group["target_end_date"] = group["target_date_5d"]
        rows.append(group)

    result = pd.concat(rows, ignore_index=True)
    result["split"] = pd.NA
    for split_name, (start, end) in SPLIT_BOUNDS.items():
        result.loc[result["date"].between(start, end), "split"] = split_name
    result["target_end_split"] = pd.NA
    for split_name, (start, end) in SPLIT_BOUNDS.items():
        result.loc[result["target_end_date"].between(start, end), "target_end_split"] = split_name

    # A scored origin is retained only when all five future sessions stay in its split.
    complete_path = result[[f"target_return_{step}d" for step in range(1, 6)]].notna().all(axis=1)
    result["purged"] = result["split"].notna() & (
        result["target_end_split"].isna()
        | result["split"].fillna("").ne(result["target_end_split"].fillna(""))
    )
    result["eligible"] = complete_path & result["split"].notna() & ~result["purged"]
    return result
