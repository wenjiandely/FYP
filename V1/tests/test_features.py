from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import build_features
from src.features.build_labels import build_labels, make_model_ready


def market_frame(periods: int = 65) -> pd.DataFrame:
    dates = pd.date_range("2019-01-02", periods=periods, freq="D")
    rows = []
    for symbol, multiplier in [("AAPL", 1.0), ("NVDA", 2.0), ("SPY", 1.5)]:
        for index, date in enumerate(dates):
            close = multiplier * (100 + index)
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": close - 1,
                    "high": close + 1,
                    "low": close - 2,
                    "close": close,
                    "adj_close": close,
                    "volume": 1000 + index,
                }
            )
    return pd.DataFrame(rows)


def test_return_and_lag_features_use_past_values() -> None:
    frame = market_frame()
    features = build_features(frame)
    aapl = features.loc[features["symbol"] == "AAPL"].sort_values("date").reset_index(drop=True)
    expected_return = np.log(101 / 100)
    assert np.isnan(aapl.loc[0, "return_1d"])
    assert np.isclose(aapl.loc[1, "return_1d"], expected_return)
    assert np.isnan(aapl.loc[1, "lag_return_1d"])
    assert np.isclose(aapl.loc[2, "lag_return_1d"], expected_return)


def test_warmup_windows_remain_missing() -> None:
    features = build_features(market_frame())
    aapl = features.loc[features["symbol"] == "AAPL"].sort_values("date").reset_index(drop=True)
    assert aapl["rolling_volatility_20d"].iloc[:20].isna().all()
    assert aapl["sma_60"].iloc[:59].isna().all()
    assert aapl["sma_60"].iloc[60:].notna().all()


def test_labels_use_next_day_and_last_label_is_missing() -> None:
    labeled = build_labels(build_features(market_frame()))
    aapl = labeled.loc[labeled["symbol"] == "AAPL"].sort_values("date").reset_index(drop=True)
    expected_next = np.log(101 / 100)
    assert np.isclose(aapl.loc[0, "target_return_1d"], expected_next)
    assert aapl["target_return_1d"].iloc[-1] != aapl["target_return_1d"].iloc[-1]
    ready = make_model_ready(labeled)
    assert ready.iloc[-1]["target_return_1d"] == labeled.loc[labeled["target_return_1d"].notna(), "target_return_1d"].iloc[-1]


def test_relative_spy_features_are_date_aligned() -> None:
    labeled = build_labels(build_features(market_frame()))
    spy = labeled.loc[labeled["symbol"] == "SPY"]
    assert (spy["spy_relative_return"].dropna() == 0).all()
    assert (spy["spy_relative_volatility"].dropna() == 0).all()
    assert (spy["excess_return_5d"].dropna() == 0).all()
