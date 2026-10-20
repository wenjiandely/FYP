# Aligned Numerical Baselines V2

## Experiment Contract

This implementation follows [`../../docs/project_scope_v2.md`](../../docs/project_scope_v2.md). A forecast is made after the close at origin `t`, using the five subsequent instrument trading sessions. The five target values are consecutive one-session log returns, not five cumulative origin-to-horizon returns. Forecast prices are reconstructed by compounding predicted daily returns from the origin adjusted close. The primary smoothed statistic is the arithmetic mean of the five reconstructed/realized adjusted closes relative to the origin close. Supervisor confirmation of that arithmetic-mean interpretation is still pending.

## Models and Information Timing

| Model | Fit design | Output |
| --- | --- | --- |
| Zero-return reference | No fitting | Five zero daily returns |
| Last-return reference | No fitting | Repeats the origin's last daily return |
| ARIMA | Independent per-symbol ARIMA(1,0,1), refit every 21 scored origins; observed returns update the fitted state between refits | Five sequential daily-return forecasts |
| XGBoost | Pooled direct regressor per path step; one-hot symbol columns; CUDA device configured by `XGB_DEVICE`; retrain every 63 scored market dates | Five daily-return forecasts |

XGBoost uses the explicitly allowlisted numerical feature columns in `src/models/forecast_baselines.py`. This prevents old one-day target columns from entering the model. At each XGBoost fitting date, only rows with `target_end_date < cutoff` and complete features/targets enter training. The strict inequality is conservative: no training label extends to the current forecast origin.

Validation forecasts use all fully matured historical rows available at each training cutoff. Test forecasts may use validation-period observations only after their five-session labels have matured. Test-period outcomes are never used to fit a prediction made before those outcomes are observable. The fixed initial XGBoost parameters are a baseline, not validation-optimized hyperparameters.

## Split Purging and Metrics

Splits are assigned by forecast origin using 2019–2023 train, 2024 validation and 2025 test bounds. An origin is eligible only when all five future trading dates exist and its fifth target date belongs to the same calendar split. The trailing incomplete origins are excluded. Features with warm-up nulls are excluded from model fitting and scoring; the target audit table still records them.

`metrics.csv` reports per-model, per-split and per-symbol daily return MAE/RMSE for each of five steps, origin-normalized reconstructed price MAE/RMSE, five-day mean-price-return MAE/RMSE and trend direction accuracy, plus cumulative endpoint-return MAE/RMSE. Evaluation refreshes also report per-step direction accuracy, balanced accuracy, actual up-rate, majority-class accuracy, and a 50% random-direction reference. Direction means positive (`> 0`) versus non-positive (`<= 0`) return. Majority accuracy is the accuracy of always predicting the more frequent class within that symbol and split. Macro rows equally average the five symbol-level values; they are not pooled by price scale.

Direction accuracy is a percentage only after multiplying the stored fraction by 100. An independent fair-coin guess has expected accuracy 50% even when the observed classes are imbalanced. This expectation is not a lower bound on a fitted model or a finite random sample. An always-up or majority-class predictor can exceed 50% without learning temporal information. Balanced accuracy averages up-class and non-up-class recall and has a 50% constant-predictor reference when both classes are represented. Majority accuracy is a retrospective class-balance diagnostic, not a predictor whose class should be selected using test outcomes. The overlapping five-session targets at adjacent origins are statistically dependent, so a result above 50% alone does not establish significance or trading value.

The figure `mean_price_return_all_models.png` plots the realized five-session arithmetic mean-price-return target and the ARIMA/XGBoost forecasts of that same derived summary. “Realized target (observed)” is ground truth computed from future closes, not a model. Its horizontal axis enumerates forecast origins; it is not one five-step forecast connected over time. `daily_direction_accuracy.png` instead shows classification accuracy separately for each one-session return in the predicted five-step path, alongside always-up and fair-coin references. `five_session_path_example.png` shows all five symbols at the first test origin, selected without reference to performance: left panels show individual daily returns, right panels show reconstructed prices with the origin indexed to 100 and dotted five-price means. The actual future observations are evaluation targets, not model inputs.

`daily_forecast_summary.csv` contains readable daily-return metrics by split, model, symbol and horizon. The 50% reference is an expectation, not an additional randomly fitted model. Refresh these outputs without retraining with `V1/.venv/bin/python V1/scripts/refresh_forecast_evaluation.py V1/results/aligned_baselines_v2/<run-id>`. Refreshing preserves the original five daily-return forecasts exactly and reconstructs their derived prices and mean-price summaries. The refreshed mean return uses `mean(expm1(cumsum(predicted_returns)))`, algebraically equivalent to mean reconstructed price divided by origin price minus one, to keep an exactly zero-return forecast exactly zero. `evaluation_refresh.json` records before/after prediction-file hashes and states that no retraining occurred.

## Corrected Run: 20 October 2026

Run `20261020T065637Z` scores 2,460 common validation/test origins per model: 247 origins per symbol in 2024 and 245 per symbol in 2025. Every model covers AAPL, NVDA, ORCL, QCOM and SPY. Test metrics below are equally weighted macro-averages across the five symbols; errors are fractions, so `0.026` means 2.6%.

| Model | Five-day mean-price-return MAE | Five-day mean-price-return RMSE | Mean-trend direction accuracy |
| --- | ---: | ---: | ---: |
| Zero-return | 0.02603 | 0.03588 | 43.51% |
| ARIMA | 0.02604 | 0.03582 | 49.96% |
| XGBoost | 0.02610 | 0.03555 | 55.18% |

Direction is scored as positive versus non-positive; a zero-return forecast is therefore classified non-positive. The originally reported zero-return accuracy of 44.08% was affected by floating-point cancellation in the derived mean price; stable reconstruction corrects it to 43.51%. Learned-model direction scores are unchanged. Always predicting a positive mean trend scores 56.49%, exceeding both ARIMA and XGBoost. XGBoost has slightly lower RMSE, but its MAE is marginally worse than both references. The differences are small and do not establish reliable predictive or trading value. The arithmetic five-close mean remains a provisional interpretation pending supervisor confirmation. Inspect the per-symbol result rows because the instruments have different volatility scales.

### Daily Direction Results After Evaluation Refresh

The following 2025 test results equally weight AAPL, NVDA, ORCL, QCOM and SPY. Each symbol contributes 245 forecast origins to each horizon (1,225 origin-symbol pairs per model/horizon). Session `k` evaluates `log(P[t+k] / P[t+k-1])` predicted using information only through origin `t`. In particular session 5 is not the cumulative return from `t` to `t+5`.

| Future session | ARIMA accuracy | XGBoost accuracy | Always-up accuracy | ARIMA balanced accuracy | XGBoost balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51.84% | 55.43% | 54.94% | 52.17% | 53.51% |
| 2 | 51.84% | 53.31% | 54.86% | 51.59% | 52.15% |
| 3 | 51.67% | 51.76% | 54.61% | 51.95% | 50.90% |
| 4 | 51.35% | 51.92% | 54.69% | 51.07% | 50.78% |
| 5 | 51.35% | 50.94% | 54.45% | 51.60% | 49.63% |

Only XGBoost session 1 exceeds the always-up accuracy, by approximately 0.49 percentage points. No significance analysis has been performed. XGBoost session 5 has balanced accuracy below 50%. These results do not demonstrate a stable directional advantage across the five-session path. Regression quality is a separate question: direction accuracy discards the size of each error, so it must be presented alongside the daily MAE/RMSE and reconstructed-price errors already in `metrics.csv`.

For the five-day mean trend, ARIMA balanced accuracy is 50.35% and XGBoost balanced accuracy is 52.48%. XGBoost predicts a positive mean trend at 67.84% of origins while the actual up-rate is 56.49%. Its raw 55.18% mean-trend hit-rate therefore cannot be read as evidence of outperforming an untrained positive-trend reference.

## Outputs and Reproduction

Run from repository root with `XGB_DEVICE=cuda:0 python V1/scripts/run_forecast_baselines.py`. Outputs are written to a new timestamped subdirectory of `results/aligned_baselines_v2/`; existing legacy results are not overwritten. `run_config.json` records code revision, source data SHA-256 hashes, split/target definitions, seed and device. `xgboost_retraining_log.csv` records each cutoff and the latest training target endpoint. `prediction_contract.csv` records scored origins per model; the runner fails if variants do not share identical origins. XGBoost Booster JSON files are saved for all five horizons at every refit cutoff. ARIMA parameter JSON files are saved for each symbol and refit cutoff. The fresh aligned run does not load or reuse legacy model files.

Main code:

- `src/targets/forecast_targets.py`: five daily increments, future price path, mean-price target and split purge flags.
- `src/models/forecast_baselines.py`: common forecast reconstruction, naive baselines, per-symbol ARIMA and pooled direct XGBoost.
- `scripts/run_forecast_baselines.py`: reproducible run, artifact writing and metric calculation.
- `tests/test_forecast_targets.py`: hand-computed target and boundary-purge tests.

Prior V1/V2/V3 results remain historical exploratory artifacts, but their mismatched multi-step label semantics prevent using them as final ARIMA/XGBoost comparisons.
