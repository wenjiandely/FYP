"""Build deterministic price and technical features for V1."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "lag_return_1d",
    "lag_return_5d",
    "rolling_volatility_5d",
    "rolling_volatility_20d",
    "high_low_range",
    "volume_change_1d",
    "sma_5",
    "sma_20",
    "sma_60",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
]
RELATIVE_FEATURE_COLUMNS = [
    "spy_relative_return",
    "spy_relative_volatility",
    "excess_return_5d",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + RELATIVE_FEATURE_COLUMNS


def _wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & average_gain.gt(0), 100)
    rsi = rsi.mask((average_loss == 0) & average_gain.eq(0), 50)
    return rsi


def _build_single_symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    result = group.sort_values("date").copy()
    adjusted_close = result["adj_close"]
    log_price = np.log(adjusted_close)
    result["return_1d"] = log_price.diff()
    result["return_5d"] = log_price.diff(5)
    result["return_20d"] = log_price.diff(20)
    result["lag_return_1d"] = result["return_1d"].shift(1)
    result["lag_return_5d"] = result["return_5d"].shift(1)
    result["rolling_volatility_5d"] = result["return_1d"].rolling(5).std()
    result["rolling_volatility_20d"] = result["return_1d"].rolling(20).std()
    result["high_low_range"] = (result["high"] - result["low"]) / result["close"]
    result["volume_change_1d"] = result["volume"].pct_change()
    result["sma_5"] = adjusted_close.rolling(5).mean()
    result["sma_20"] = adjusted_close.rolling(20).mean()
    result["sma_60"] = adjusted_close.rolling(60).mean()
    result["ema_12"] = adjusted_close.ewm(span=12, adjust=False).mean()
    result["ema_26"] = adjusted_close.ewm(span=26, adjust=False).mean()
    result["rsi_14"] = _wilder_rsi(adjusted_close)
    result["macd"] = result["ema_12"] - result["ema_26"]
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    return result


def build_features(market_data: pd.DataFrame) -> pd.DataFrame:
    """Build the V1 feature table from canonical market data."""

    required = {"symbol", "date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = sorted(required - set(market_data.columns))
    if missing:
        raise ValueError(f"market data is missing columns: {missing}")

    ordered = (
        market_data.copy()
        .assign(date=lambda frame: pd.to_datetime(frame["date"]))
        .sort_values(["symbol", "date"])
    )
    enriched = pd.concat(
        [_build_single_symbol_features(group) for _, group in ordered.groupby("symbol", sort=False)],
        ignore_index=True,
    )

    spy = enriched.loc[
        enriched["symbol"] == "SPY", ["date", "return_1d", "return_5d", "rolling_volatility_20d"]
    ].rename(
        columns={
            "return_1d": "spy_return_1d",
            "return_5d": "spy_return_5d",
            "rolling_volatility_20d": "spy_volatility_20d",
        }
    )
    enriched = enriched.merge(spy, on="date", how="left", validate="many_to_one")
    enriched["spy_relative_return"] = enriched["return_1d"] - enriched["spy_return_1d"]
    enriched["spy_relative_volatility"] = enriched["rolling_volatility_20d"] - enriched["spy_volatility_20d"]
    enriched["excess_return_5d"] = enriched["return_5d"] - enriched["spy_return_5d"]

    output_columns = ["symbol", "date"] + FEATURE_COLUMNS
    return enriched[output_columns].sort_values(["symbol", "date"]).reset_index(drop=True)
