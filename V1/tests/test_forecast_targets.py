import numpy as np
import pandas as pd

from src.targets.forecast_targets import build_forecast_targets
from src.models.forecast_baselines import (
    MODEL_FEATURES,
    PATH_TARGETS,
    PRICE_TARGETS,
    eligible_training_rows,
    predict_naive,
)


def test_five_step_daily_returns_and_mean_price_are_hand_calculated():
    prices = np.array([100, 101, 99, 103, 102, 105, 107, 106], dtype=float)
    dates = pd.bdate_range("2023-12-18", periods=len(prices))
    source = pd.DataFrame({"symbol": "TEST", "date": dates, "adj_close": prices})

    targets = build_forecast_targets(source)
    first = targets.iloc[0]

    expected_returns = np.log(prices[1:6] / prices[:5])
    np.testing.assert_allclose([first[f"target_return_{step}d"] for step in range(1, 6)], expected_returns)
    np.testing.assert_allclose(first["target_mean_price_5d"], prices[1:6].mean())
    np.testing.assert_allclose(first["target_mean_price_return_5d"], prices[1:6].mean() / prices[0] - 1)
    np.testing.assert_allclose(first["target_cumulative_log_return_5d"], np.log(prices[5] / prices[0]))
    assert first["target_end_date"] == dates[5]


def test_split_boundaries_purge_origins_whose_path_crosses_split():
    dates = pd.bdate_range("2023-12-25", "2024-01-12")
    prices = np.arange(100, 100 + len(dates), dtype=float)
    source = pd.DataFrame({"symbol": "TEST", "date": dates, "adj_close": prices})

    targets = build_forecast_targets(source)
    crossing = targets.loc[targets["date"].eq(pd.Timestamp("2023-12-29"))].iloc[0]
    assert crossing["split"] == "train"
    assert crossing["target_end_split"] == "validation"
    assert crossing["purged"]
    assert not crossing["eligible"]


def test_last_five_origins_have_incomplete_targets():
    dates = pd.bdate_range("2024-01-02", periods=10)
    source = pd.DataFrame({"symbol": "TEST", "date": dates, "adj_close": np.arange(10) + 100.0})

    targets = build_forecast_targets(source)
    assert targets.tail(5)[[f"target_return_{step}d" for step in range(1, 6)]].isna().any(axis=None)
    assert targets.iloc[-1]["target_direction_5d"] != targets.iloc[-1]["target_direction_5d"]


def test_naive_predictions_share_path_contract_and_realized_prices():
    dates = pd.bdate_range("2024-01-02", periods=12)
    prices = np.linspace(100, 112, len(dates))
    frame = build_forecast_targets(
        pd.DataFrame({"symbol": "TEST", "date": dates, "adj_close": prices})
    )
    frame["return_1d"] = np.log(frame["adj_close"] / frame["adj_close"].shift(1))
    frame["return_1d"] = frame["return_1d"].fillna(0)

    predictions = predict_naive(frame)

    assert set(PRICE_TARGETS).issubset(predictions.columns)
    assert set(predictions["naive_zero_pred_return_1d"].dropna().unique()) == {0.0}
    assert predictions.groupby("model").size().to_dict() == {"naive_last": 7, "naive_zero": 7}


def test_training_rows_exclude_purged_origins_and_unobserved_endpoints():
    dates = pd.bdate_range("2023-12-18", periods=20)
    source = pd.DataFrame({"symbol": "TEST", "date": dates, "adj_close": np.arange(100, 120, dtype=float)})
    targets = build_forecast_targets(source)
    for column in MODEL_FEATURES:
        targets[column] = 1.0

    train = eligible_training_rows(targets, pd.Timestamp("2024-01-08"))

    assert train["target_end_date"].lt(pd.Timestamp("2024-01-08")).all()
    assert train["eligible"].all()
    assert train["target_end_split"].eq(train["split"]).all()
    assert set(PATH_TARGETS).issubset(train.columns)
