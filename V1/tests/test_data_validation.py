from __future__ import annotations

import pandas as pd

from src.data.validate_market_data import validate_dataframe


def valid_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "date": ["2019-01-02", "2019-01-03"],
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "adj_close": [11.0, 12.0],
            "volume": [100, 110],
        }
    )


def test_validation_reports_missing_required_columns() -> None:
    report = validate_dataframe(valid_rows().drop(columns=["volume"]))
    assert not report["passed"]
    assert "missing required columns" in report["errors"][0]


def test_validation_reports_duplicate_symbol_date() -> None:
    frame = pd.concat([valid_rows(), valid_rows().iloc[[0]]], ignore_index=True)
    report = validate_dataframe(frame)
    assert not report["passed"]
    assert any("duplicate symbol/date" in error for error in report["errors"])


def test_validation_reports_ohlc_violation() -> None:
    frame = valid_rows()
    frame.loc[0, "high"] = 8.0
    report = validate_dataframe(frame)
    assert not report["passed"]
    assert any("OHLC relationship" in error for error in report["errors"])
