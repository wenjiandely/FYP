# V1 Data and Forecasting Experiments

Status: **The aligned five-session ARIMA/XGBoost training pipeline is implemented. Historical V1/V2/V3 comparisons remain exploratory and must not be treated as final evidence.**

This directory contains the first reproducible experiment assets for SP73, “Handling and Processing Time-Series Data using Agentic AI”. The current version includes:

All supervisor-facing documentation and experiment notes in this repository are maintained in English.

- A canonical five-symbol daily market-data file for AAPL, NVDA, ORCL, QCOM, and SPY;
- A normalized Google News RSS records file for the same five symbols;
- An import script that verifies and re-imports the shared ZIP;
- A data dictionary, file hashes, feature schema, and result schema;
- Data validation, feature, label, time-split, and one-command pipeline scripts;
- Unit tests and an experiment Notebook skeleton;
- Exploratory model scripts and results, clearly superseded for final evaluation by the current protocol.
- Aligned five-session target construction and purged walk-forward ARIMA/XGBoost baselines.

See [`../docs/project_scope_v2.md`](../docs/project_scope_v2.md) for the canonical target, fair-comparison protocol, Agentic AI architecture, and implementation plan. See [`docs/technical_design_v1.md`](docs/technical_design_v1.md) for the original data-pipeline record.

## Run

Run the validation pipeline in the prepared environment:

```bash
V1/.venv/bin/python V1/scripts/run_v1_validation.py
```

Run the tests:

```bash
V1/.venv/bin/pytest V1/tests
```

Dependencies are defined in `requirements.txt` and installed in `V1/.venv`. The complete pipeline run has passed.

Validation result: `10 passed`; 8,800 feature rows, 8,500 model-ready rows, and 8,500 time-split rows were generated.

## Current Implementation

- `src/data/validate_market_data.py`: validates fields, dates, numeric values, OHLC relationships, duplicate keys, and five-symbol record counts;
- `scripts/import_shared_dataset.py`: imports the shared ZIP and checks split files against the archive's combined files;
- `src/features/build_features.py`: builds returns, lagged values, volatility, moving averages, RSI, MACD, and SPY-relative features;
- `src/features/build_labels.py`: builds next-day return and direction labels and removes unusable rows;
- `src/evaluation/split.py`: applies fixed time splits for 2019–2023 training, 2024 validation, and 2025 testing;
- `scripts/run_v1_validation.py`: generates features, labels, splits, quality reports, summary tables, and an overview figure.

## Imported shared dataset

The shared archive is imported into the canonical files below:

- `data/processed/market_data_v1.csv`: 8,800 price rows, 1,760 per symbol;
- `data/processed/news_v1.csv`: 30,902 Google News RSS records;
- `data/metadata/shared_dataset_import.json`: archive hash, source inventory, and import audit;
- `data/metadata/news_quality_v1.json`: news completeness and timestamp-quality audit;
- `data/raw/SP73_Data_2019_2025_source_links.txt`: source links supplied with the archive.

The news file contains article titles and RSS summaries only. It is not the full FNSPID dataset, and no relevance filter has been applied yet. News should be timestamp-aligned and relevance-filtered before model training.

To re-import the archive:

```bash
V1/.venv/bin/python V1/scripts/import_shared_dataset.py /path/to/SP73_Data_2019_2025_share\(1\).zip
```

## Aligned Baseline Experiment

The current experiment contract is defined in [`../docs/project_scope_v2.md`](../docs/project_scope_v2.md). Each forecast origin predicts the next five **daily** adjusted-close log returns. The price path is reconstructed from those increments, then summarized by the arithmetic mean of the five predicted prices. The mean-price interpretation remains provisional pending supervisor confirmation.

Run on the designated CUDA server from the repository root:

```bash
XGB_DEVICE=cuda:0 python V1/scripts/run_forecast_baselines.py
```

The runner scores zero-return and last-return references, per-symbol ARIMA(1,0,1), and pooled direct-horizon XGBoost with one-hot symbol indicators. XGBoost retrains every 63 scored market sessions and ARIMA refits every 21 origins. At each cutoff, an XGBoost training row is admitted only when it is non-purged and its fifth target session is strictly earlier than the cutoff. Forecast origins are purged unless the complete five-session path remains inside their calendar split. Metrics are emitted per split, per symbol and as an equally weighted macro average; final test results cover 2025 only. New ARIMA parameter JSONs and XGBoost Booster JSON weights are saved at each refit; no legacy weights are reused.

Each run writes to a new timestamped directory under `results/aligned_baselines_v2/`, preserving prior experiment outputs. Artifacts include predictions, metric rows, XGBoost retraining cutoffs, shared-origin contract, run configuration and an all-symbol five-day mean-trend plot. The feature list is explicitly allowlisted; legacy `target_return_1d` and `target_direction_1d` columns in the V1 feature table are never used as inputs. The corrected 20 October 2026 run is summarized in [`docs/aligned_baselines_v2.md`](docs/aligned_baselines_v2.md).

Refresh evaluation from saved forecasts without retraining:

```bash
V1/.venv/bin/python V1/scripts/refresh_forecast_evaluation.py V1/results/aligned_baselines_v2/20261020T065637Z
```

This writes daily direction and balanced accuracy for all five horizons, a readable `daily_forecast_summary.csv`, mean-trend comparisons including ARIMA, and a fixed-origin five-session path example. The figures distinguish realized observations from forecasts and include class-balance references. The zero-return mean trend is reconstructed with stable arithmetic to avoid floating-point sign errors. Original daily-return predictions are preserved. See the experiment notes for the corrected zero-return score and the limitations of accuracy near 50%.
