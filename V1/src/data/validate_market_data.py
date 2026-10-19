"""Validation utilities for the V1 market data file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]
EXPECTED_SYMBOLS = ["AAPL", "NVDA", "ORCL", "QCOM", "SPY"]
EXPECTED_RECORDS_PER_SYMBOL = 1760
EXPECTED_TOTAL_RECORDS = 8800
EXPECTED_MIN_DATE = pd.Timestamp("2019-01-02")
EXPECTED_MAX_DATE = pd.Timestamp("2025-12-31")
NUMERIC_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class DataValidationError(ValueError):
    """Raised when market data does not satisfy the V1 data contract."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Market data validation failed:\n- " + "\n- ".join(errors))


def load_market_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load the canonical V1 market data file."""

    frame = pd.read_csv(path, parse_dates=["date"])
    return frame


def _check(name: str, passed: bool, expected: Any, actual: Any, details: str = "") -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
        "details": details,
    }


def validate_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serializable validation report for a market data frame."""

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    checks.append(
        _check(
            "required_columns",
            not missing_columns,
            REQUIRED_COLUMNS,
            list(frame.columns),
            f"missing={missing_columns}",
        )
    )
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")
        return {"passed": False, "errors": errors, "checks": checks, "rows": int(len(frame))}

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    symbols = sorted(work["symbol"].dropna().unique().tolist())
    unknown_symbols = sorted(set(symbols) - set(EXPECTED_SYMBOLS))
    missing_symbols = sorted(set(EXPECTED_SYMBOLS) - set(symbols))
    symbols_passed = not unknown_symbols and not missing_symbols
    checks.append(
        _check(
            "symbols",
            symbols_passed,
            EXPECTED_SYMBOLS,
            symbols,
            f"unknown={unknown_symbols}; missing={missing_symbols}",
        )
    )
    if unknown_symbols or missing_symbols:
        errors.append(f"unexpected or missing symbols: unknown={unknown_symbols}, missing={missing_symbols}")

    duplicate_mask = work.duplicated(["symbol", "date"], keep=False)
    checks.append(_check("unique_symbol_date", not duplicate_mask.any(), 0, int(duplicate_mask.sum())))
    if duplicate_mask.any():
        errors.append(f"duplicate symbol/date rows: {int(duplicate_mask.sum())}")

    invalid_dates = int(work["date"].isna().sum())
    checks.append(_check("valid_dates", invalid_dates == 0, 0, invalid_dates))
    if invalid_dates:
        errors.append(f"invalid dates: {invalid_dates}")

    sorted_by_symbol = True
    unsorted_symbols: list[str] = []
    for symbol, group in work.groupby("symbol", sort=False):
        if not group["date"].is_monotonic_increasing:
            sorted_by_symbol = False
            unsorted_symbols.append(str(symbol))
    checks.append(_check("date_order", sorted_by_symbol, "ascending per symbol", unsorted_symbols))
    if not sorted_by_symbol:
        errors.append(f"dates are not ascending for symbols: {unsorted_symbols}")

    min_date = work["date"].min()
    max_date = work["date"].max()
    date_range_passed = min_date == EXPECTED_MIN_DATE and max_date == EXPECTED_MAX_DATE
    checks.append(
        _check(
            "date_range",
            date_range_passed,
            {"min": EXPECTED_MIN_DATE.date().isoformat(), "max": EXPECTED_MAX_DATE.date().isoformat()},
            {
                "min": None if pd.isna(min_date) else min_date.date().isoformat(),
                "max": None if pd.isna(max_date) else max_date.date().isoformat(),
            },
        )
    )
    if not date_range_passed:
        errors.append(f"unexpected date range: {min_date} to {max_date}")

    missing_by_column = {column: int(work[column].isna().sum()) for column in REQUIRED_COLUMNS}
    missing_passed = not any(missing_by_column.values())
    checks.append(_check("missing_values", missing_passed, 0, missing_by_column))
    if not missing_passed:
        errors.append(f"missing values by column: {missing_by_column}")

    non_finite_by_column = {
        column: int((~np.isfinite(work[column].to_numpy(dtype="float64"))).sum())
        for column in NUMERIC_COLUMNS
    }
    numeric_finite = all(count == 0 for count in non_finite_by_column.values())
    checks.append(_check("numeric_values", numeric_finite, 0, non_finite_by_column))
    if not numeric_finite:
        errors.append(f"non-numeric or non-finite numeric values: {non_finite_by_column}")

    positive_price_mask = work[["open", "high", "low", "close", "adj_close"]].le(0).any(axis=1)
    negative_volume_mask = work["volume"] < 0
    checks.append(_check("positive_prices", not positive_price_mask.any(), 0, int(positive_price_mask.sum())))
    checks.append(_check("non_negative_volume", not negative_volume_mask.any(), 0, int(negative_volume_mask.sum())))
    if positive_price_mask.any():
        errors.append(f"non-positive price rows: {int(positive_price_mask.sum())}")
    if negative_volume_mask.any():
        errors.append(f"negative volume rows: {int(negative_volume_mask.sum())}")

    ohlc_violations = {
        "high_below_open": int((work["high"] < work["open"]).sum()),
        "high_below_close": int((work["high"] < work["close"]).sum()),
        "low_above_open": int((work["low"] > work["open"]).sum()),
        "low_above_close": int((work["low"] > work["close"]).sum()),
    }
    ohlc_passed = not any(ohlc_violations.values())
    checks.append(_check("ohlc_relationships", ohlc_passed, 0, ohlc_violations))
    if not ohlc_passed:
        errors.append(f"OHLC relationship violations: {ohlc_violations}")

    counts = work["symbol"].value_counts().sort_index().to_dict()
    count_passed = len(work) == EXPECTED_TOTAL_RECORDS and all(
        counts.get(symbol) == EXPECTED_RECORDS_PER_SYMBOL for symbol in EXPECTED_SYMBOLS
    )
    checks.append(
        _check(
            "record_counts",
            count_passed,
            {"total": EXPECTED_TOTAL_RECORDS, "per_symbol": EXPECTED_RECORDS_PER_SYMBOL},
            {"total": int(len(work)), "per_symbol": {key: int(value) for key, value in counts.items()}},
        )
    )
    if not count_passed:
        errors.append(f"unexpected record counts: total={len(work)}, per_symbol={counts}")

    quality_rows = []
    for column in REQUIRED_COLUMNS:
        quality_rows.append(
            {
                "scope": "column",
                "name": column,
                "rows": int(len(work)),
                "missing_count": int(work[column].isna().sum()),
                "unique_count": int(work[column].nunique(dropna=True)),
            }
        )
    for symbol, count in counts.items():
        quality_rows.append(
            {
                "scope": "symbol",
                "name": symbol,
                "rows": int(count),
                "missing_count": int(work.loc[work["symbol"] == symbol].isna().sum().sum()),
                "unique_count": int(work.loc[work["symbol"] == symbol, "date"].nunique()),
            }
        )

    return {
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "rows": int(len(work)),
        "symbols": symbols,
        "min_date": None if pd.isna(min_date) else min_date.date().isoformat(),
        "max_date": None if pd.isna(max_date) else max_date.date().isoformat(),
        "quality_rows": quality_rows,
    }


def assert_valid(report: dict[str, Any]) -> None:
    """Raise a useful exception if a validation report failed."""

    if not report["passed"]:
        raise DataValidationError(report["errors"])


def write_validation_outputs(
    report: dict[str, Any], report_path: Union[str, Path], quality_path: Union[str, Path]
) -> None:
    """Write the JSON report and tabular quality summary."""

    report_path = Path(report_path)
    quality_path = Path(quality_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(report.get("quality_rows", [])).to_csv(quality_path, index=False)
