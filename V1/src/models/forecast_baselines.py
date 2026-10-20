"""Aligned ARIMA and GPU XGBoost walk-forward baseline forecasts."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PATH_TARGETS = [f"target_return_{step}d" for step in range(1, 6)]
PRICE_TARGETS = [f"target_adj_close_{step}d" for step in range(1, 6)]
MODEL_FEATURES = [
    "return_1d", "return_5d", "return_20d", "lag_return_1d", "lag_return_5d",
    "rolling_volatility_5d", "rolling_volatility_20d", "high_low_range", "volume_change_1d",
    "sma_5", "sma_20", "sma_60", "ema_12", "ema_26", "rsi_14", "macd", "macd_signal",
    "spy_relative_return", "spy_relative_volatility", "excess_return_5d",
]


def eligible_training_rows(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return complete eligible rows whose five-session labels are observable."""
    return frame.loc[
        frame["eligible"]
        & frame["date"].lt(cutoff)
        & frame["target_end_date"].lt(cutoff)
        & frame[[*MODEL_FEATURES, *PATH_TARGETS]].notna().all(axis=1)
    ].copy()


def _add_forecast_outputs(frame: pd.DataFrame, predicted_returns: np.ndarray, model: str) -> pd.DataFrame:
    result = frame[["symbol", "date", "split", "adj_close"] + PATH_TARGETS + PRICE_TARGETS + [
        "target_mean_price_5d", "target_mean_price_return_5d", "target_cumulative_log_return_5d",
        "target_direction_5d", "target_end_date", "purged",
    ]].copy()
    cumulative_returns = np.cumsum(predicted_returns, axis=1)
    predicted_prices = result["adj_close"].to_numpy(dtype=float)[:, None] * np.exp(cumulative_returns)
    for step in range(1, 6):
        result[f"{model}_pred_return_{step}d"] = predicted_returns[:, step - 1]
        result[f"{model}_pred_adj_close_{step}d"] = predicted_prices[:, step - 1]
    mean_price_return = np.expm1(cumulative_returns).mean(axis=1)
    result[f"{model}_pred_mean_price_5d"] = result["adj_close"].to_numpy(dtype=float) * (1 + mean_price_return)
    result[f"{model}_pred_mean_price_return_5d"] = mean_price_return
    return result


def predict_naive(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[frame["eligible"] & frame["split"].isin(["validation", "test"])].copy()
    output = []
    for model, mode in (("naive_zero", "zero"), ("naive_last", "last")):
        if mode == "zero":
            returns = np.zeros((len(eligible), 5))
        else:
            returns = np.repeat(eligible["return_1d"].to_numpy(dtype=float)[:, None], 5, axis=1)
        prediction = _add_forecast_outputs(eligible, returns, model)
        prediction["model"] = model
        output.append(prediction)
    return pd.concat(output, ignore_index=True)


def predict_arima(
    frame: pd.DataFrame,
    refit_every: int = 21,
    model_dir: Path | None = None,
) -> pd.DataFrame:
    """Fit per-symbol ARIMA(1,0,1); refit every 21 origins, forecast five increments."""
    from statsmodels.tsa.arima.model import ARIMA

    scored = frame.loc[frame["eligible"] & frame["split"].isin(["validation", "test"])].copy()
    predictions: list[dict] = []
    for symbol, all_rows in frame.groupby("symbol", sort=False):
        all_rows = all_rows.sort_values("date").reset_index(drop=True)
        score_dates = set(scored.loc[scored["symbol"].eq(symbol), "date"])
        scored_positions = [i for i, value in enumerate(all_rows["date"]) if value in score_dates]
        result = None
        last_fit = -refit_every
        previous_position = None
        for position in scored_positions:
            history = all_rows.loc[:position, "return_1d"].dropna().astype(float)
            history = pd.Series(history.to_numpy(), index=pd.RangeIndex(len(history)), dtype=float)
            if len(history) < 40:
                continue
            if result is None or position - last_fit >= refit_every:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = ARIMA(history, order=(1, 0, 1), trend="n").fit()
                    fitted_order = "1,0,1"
                except Exception:
                    result = ARIMA(history, order=(1, 0, 0), trend="n").fit()
                    fitted_order = "1,0,0"
                if model_dir is not None:
                    checkpoint_dir = model_dir / symbol
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_date = all_rows.iloc[position]["date"]
                    (checkpoint_dir / f"{checkpoint_date:%Y%m%d}.json").write_text(
                        json.dumps(
                            {
                                "order": fitted_order,
                                "origin": checkpoint_date.strftime("%Y-%m-%d"),
                                "params": {name: float(value) for name, value in zip(result.param_names, result.params)},
                                "aic": float(result.aic),
                                "training_observations": int(result.nobs),
                            },
                            indent=2,
                        ) + "\n",
                        encoding="utf-8",
                    )
                last_fit = position
            elif previous_position is not None:
                observed = all_rows.loc[position, "return_1d"]
                if pd.notna(observed):
                    result = result.append([float(observed)], refit=False)
            row = all_rows.iloc[position]
            predicted = np.asarray(result.forecast(steps=5), dtype=float)
            predictions.append({"symbol": symbol, "date": row["date"], "returns": predicted})
            previous_position = position
    indexed = scored.reset_index(names="source_index")
    pred_map = {(item["symbol"], item["date"]): item["returns"] for item in predictions}
    valid = indexed.loc[indexed.apply(lambda r: (r["symbol"], r["date"]) in pred_map, axis=1)].copy()
    returns = np.vstack([pred_map[(row.symbol, row.date)] for row in valid.itertuples()]) if len(valid) else np.empty((0, 5))
    return _add_forecast_outputs(valid, returns, "arima")


def predict_xgboost(
    frame: pd.DataFrame,
    refit_every: int = 63,
    device: str = "cuda:0",
    random_state: int = 42,
    model_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled direct regressors; every training label must be observable by cutoff."""
    from xgboost import XGBRegressor

    data = frame.copy()
    symbol_features = pd.get_dummies(data["symbol"], prefix="symbol", dtype=float)
    data = pd.concat([data, symbol_features], axis=1)
    feature_columns = MODEL_FEATURES + list(symbol_features.columns)
    scored = data.loc[data["eligible"] & data["split"].isin(["validation", "test"])].copy()
    # Refit on a shared trading-date cadence, using only origins whose full labels matured.
    dates = sorted(scored["date"].drop_duplicates())
    refit_dates = set(dates[::refit_every])
    predictions = []
    audit = []
    active_models = None
    active_cutoff = None
    for date in dates:
        if active_models is None or date in refit_dates:
            train = eligible_training_rows(data, date)
            train = train.loc[train[feature_columns].notna().all(axis=1)]
            if len(train) < 100:
                continue
            active_models = []
            for target in PATH_TARGETS:
                model = XGBRegressor(
                    objective="reg:squarederror", n_estimators=350, max_depth=4,
                    learning_rate=0.03, subsample=0.85, colsample_bytree=0.85,
                    reg_lambda=1.0, tree_method="hist", device=device,
                    random_state=random_state, n_jobs=1,
                )
                model.fit(train[feature_columns], train[target])
                if model_dir is not None:
                    checkpoint_dir = model_dir / date.strftime("%Y%m%d")
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.get_booster().save_model(checkpoint_dir / f"{target}.json")
                active_models.append(model)
            active_cutoff = date
            audit.append({"cutoff": date, "training_rows": len(train), "max_target_end_date": train["target_end_date"].max()})
        current = scored.loc[scored["date"].eq(date)].copy()
        if active_models is None or current.empty:
            continue
        matrix = current[feature_columns]
        values = np.column_stack([model.predict(matrix) for model in active_models])
        current["training_cutoff"] = active_cutoff
        predictions.append((current, values))
    output = []
    for current, values in predictions:
        pred = _add_forecast_outputs(current, values, "xgboost")
        pred["training_cutoff"] = current["training_cutoff"].to_numpy()
        output.append(pred)
    return (pd.concat(output, ignore_index=True) if output else pd.DataFrame(), pd.DataFrame(audit))
