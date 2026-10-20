# Baseline Analysis V2

> **Legacy exploratory results, not final evidence.** The old ARIMA implementation compares its step-`h` one-day return forecast with XGBoost's cumulative origin-to-`h`-day return target for horizons 2–5. These are different quantities. The reported multi-horizon aggregate metrics and conclusions must not be used as a fair ARIMA-versus-XGBoost comparison. The final target and rerun requirements are defined in [`../../docs/project_scope_v2.md`](../../docs/project_scope_v2.md).

## Current Result

The current experiment compares per-symbol expanding-window ARIMA with pooled XGBoost regressors. Both models predict adjusted-close log returns at horizons one through five trading days.

The averages below are across five symbols and five horizons.

| Split | Model | MAE | RMSE | Directional accuracy |
|---|---|---:|---:|---:|
| Validation | ARIMA | 0.024600 | 0.032283 | 0.572857 |
| Validation | XGBoost | 0.025365 | 0.032818 | 0.493333 |
| Test | ARIMA | 0.028743 | 0.040433 | 0.549726 |
| Test | XGBoost | 0.029385 | 0.040529 | 0.500467 |

The overall RMSE gap is small. XGBoost is not uniformly worse: at the five-day test horizon it has lower RMSE than ARIMA for AAPL, NVDA, and QCOM, but higher RMSE for ORCL and SPY. Its directional accuracy is notably weaker for NVDA and SPY in this run.

## Why XGBoost Is Currently Weaker

### 1. Very small panel for a tree model

The dataset contains approximately 6,000 training rows across only five instruments. XGBoost can partition this space effectively, but the available sample is small for learning stable nonlinear interactions across many technical features.

### 2. The symbol is not an input feature

The current XGBoost feature matrix contains technical and SPY-relative features, but not an encoded `symbol`. Therefore, the model is pooled across AAPL, NVDA, ORCL, QCOM, and SPY and must use the same mapping for instruments with different volatility and return dynamics. ARIMA is fitted separately for each symbol, so it automatically has a symbol-specific model.

### 3. ARIMA receives expanding-window updates

The ARIMA implementation appends each newly observed return before producing the next forecast. XGBoost is trained on the 2019–2023 training set and then evaluated on 2024 and 2025 without retraining. This is a deliberate first baseline, but it gives ARIMA a stronger ability to adapt to regime changes.

### 4. The target has a weak predictable signal

Daily and five-day stock returns are noisy. Technical indicators may explain conditional volatility or regime state without reliably predicting the sign of the next return. XGBoost minimizes squared error, not directional accuracy, so a numerically small prediction can reduce RMSE while still producing a weak sign forecast.

### 5. The tree configuration is intentionally conservative

The first run uses fixed parameters rather than validation-driven tuning. It does not yet use early stopping, symbol-specific models, target transformations, or rolling retraining. This makes the comparison reproducible but does not represent a tuned XGBoost system.

## What To Show In The Meeting

1. `rmse_by_horizon_validation_test.png`: both models degrade as the horizon increases; the gap is small.
2. `test_comparison_by_symbol_5d.png`: XGBoost performance varies substantially by symbol.
3. `actual_vs_predicted_spy_nvda_5d.png`: shows how predictions track or miss regime changes.
4. `test_residual_distribution_5d.png`: compares error spread rather than only the mean score.
5. `xgboost_feature_importance_top15.png`: shows that EMA, volatility, moving-average, and RSI features dominate the first tree model.
6. `test_rmse_heatmap_5d.png`: makes the symbol-level asymmetry immediately visible.

## Highest-Value Follow-Up Experiments

1. Add one-hot `symbol` features or train one XGBoost model per symbol.
2. Add monthly or quarterly walk-forward XGBoost retraining using only information available at each date.
3. Tune `max_depth`, `learning_rate`, `n_estimators`, `subsample`, and `min_child_weight` on the 2024 validation set only.
4. Add naive zero-return and last-return baselines before claiming either model is useful.
5. Compare separate models for return magnitude and direction rather than using only squared-error regression.

The current result should therefore be reported as: ARIMA is the stronger first reference under the current protocol, while XGBoost is close in RMSE but less reliable for directional prediction and has not yet received symbol encoding or walk-forward retraining.
