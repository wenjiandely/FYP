import numpy as np
import pandas as pd
import pytest

from scripts.run_forecast_baselines import _direction_metrics, _score
from src.models.forecast_baselines import predict_naive
from src.targets.forecast_targets import build_forecast_targets


@pytest.fixture
def naive_predictions():
    dates = pd.bdate_range("2024-01-02", periods=12)
    prices = np.array([183.40399169921875, 185, 181, 186, 180, 187, 182, 186, 183, 188, 184, 189])
    frame = build_forecast_targets(pd.DataFrame({"symbol": "TEST", "date": dates, "adj_close": prices}))
    frame["return_1d"] = np.log(frame["adj_close"] / frame["adj_close"].shift(1)).fillna(0)
    return predict_naive(frame)


def test_direction_accuracy_and_balance_have_different_references():
    actual = pd.Series([0.01, 0.02, 0.03, -0.01])
    scores = _direction_metrics(actual, pd.Series([0.01] * 4))
    assert scores["direction_accuracy"] == 0.75
    assert scores["balanced_accuracy"] == 0.5
    assert scores["actual_up_rate"] == 0.75
    assert scores["majority_baseline_accuracy"] == 0.75
    assert scores["random_expected_accuracy"] == 0.5


def test_zero_is_non_up_and_missing_class_has_no_balanced_accuracy():
    scores = _direction_metrics(pd.Series([0, 0.1, -0.1, -0.1]), pd.Series([-0.01, 0.01, 0, 0.01]))
    assert scores["direction_accuracy"] == 0.75
    assert scores["balanced_accuracy"] == pytest.approx((1 + 2 / 3) / 2)
    absent_class = _direction_metrics(pd.Series([0, -0.1]), pd.Series([0, 0]))
    assert np.isnan(absent_class["balanced_accuracy"])


def test_zero_return_forecasts_stay_exactly_zero_after_price_reconstruction(naive_predictions):
    zero = naive_predictions.loc[naive_predictions["model"].eq("naive_zero")]
    assert zero["naive_zero_pred_mean_price_return_5d"].eq(0).all()
    metrics = _score(naive_predictions).set_index(["model", "split", "symbol", "metric"])
    actual_non_up_rate = zero["target_mean_price_return_5d"].le(0).mean()
    assert metrics.loc[("naive_zero", "validation", "TEST", "mean_trend_direction_accuracy"), "value"] == actual_non_up_rate


def test_scoring_rejects_missing_forecasts_and_duplicate_origins(naive_predictions):
    duplicate = pd.concat([naive_predictions, naive_predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        _score(duplicate)
    missing = naive_predictions.copy()
    missing.loc[0, "naive_zero_pred_return_1d"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        _score(missing)


def test_scoring_uses_individual_daily_returns_and_symbol_macro(naive_predictions):
    metrics = _score(naive_predictions).set_index(["model", "split", "symbol", "metric"])
    for step in range(1, 6):
        actual = naive_predictions.loc[naive_predictions["model"].eq("naive_last")]
        expected = (actual[f"target_return_{step}d"].gt(0) == actual[f"naive_last_pred_return_{step}d"].gt(0)).mean()
        assert metrics.loc[("naive_last", "validation", "TEST", f"return_{step}d_direction_accuracy"), "value"] == expected
        assert metrics.loc[("naive_last", "validation", "MACRO", f"return_{step}d_direction_accuracy"), "value"] == expected
