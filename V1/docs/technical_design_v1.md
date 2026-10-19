# V1 Technical Design and Implementation Record

- Project: SP73 Handling and Processing Time-Series Data using Agentic AI
- Version: V1
- Document status: V1 validation-stage record
- Created: 2026-10-02 (Asia/Shanghai)
- Current progress: The V1 data-to-feature validation code has been extended to five symbols and fully executed; all 10 unit tests pass; the shared news archive has been imported and audited; model training and Agentic AI have not started.

## 1. V1 Scope and Boundaries

V1 prepares traceable data assets and a reproducible feature pipeline. It does not start predictive modeling yet. The initial experiment uses five liquid US instruments with different roles:

| Symbol | Type | Role |
| --- | --- | --- |
| AAPL | Equity | Large technology-company sample |
| NVDA | Equity | Higher-volatility technology-company sample |
| ORCL | Equity | Additional technology-company comparison sample |
| QCOM | Equity | Additional semiconductor and wireless comparison sample |
| SPY | ETF | Market benchmark and relative-feature reference |

The requested period is 2019-01-01 to 2025-12-31. The first available trading record is 2019-01-02.

The current implementation does not include:

- ARIMA, Random Forest, XGBoost, MLP, or LSTM;
- Model training or hyperparameter tuning;
- LLM calls or an Agentic AI workflow;
- Unverified predictive metrics.

## 2. Data Sources

### 2.1 Primary source

- Source: Yahoo Finance Chart API
- Endpoint: `https://query2.finance.yahoo.com/v8/finance/chart/{SYMBOL}`
- Parameters: `period1=1546300800`, `period2=1767225600`, `interval=1d`, `events=history`, `includeAdjustedClose=true`
- Download date: 2026-10-02
- Frequency: daily
- Currency: USD
- Raw files: `data/raw/yahoo_chart_<SYMBOL>_20190101_20251231.json`
- Unified file: `data/processed/market_data_v1.csv`

`period2` is the exclusive Unix-time boundary. The returned data includes records through 2025-12-31. Because the Yahoo Finance endpoint may apply rate limits, the raw responses are retained locally and should be reused for subsequent experiments.

### 2.2 Shared archive import

The shared archive is imported with `scripts/import_shared_dataset.py`. The import checks ZIP CRC integrity, verifies the five per-symbol price files against the combined price file, verifies the five news files against the combined news file, normalizes field names, and writes the canonical files:

- `data/processed/market_data_v1.csv`;
- `data/processed/news_v1.csv`;
- `data/metadata/shared_dataset_import.json`;
- `data/metadata/news_quality_v1.json`.

The archive hash and source links are retained in the metadata and `data/raw/SP73_Data_2019_2025_source_links.txt`. The archive itself is not duplicated inside the repository.

### 2.3 Backup sources

- Stooq was considered as a backup source, but the current environment returned a browser-verification page; that response was not saved as data.
- The Yahoo Finance CSV download endpoint returned HTTP 401 during preparation and was not used.

Any future source change must document field definitions, adjustment rules, download date, and differences from the current dataset. Sources must not be mixed without a documented reconciliation step.

## 3. Current Data Assets

### 3.1 Raw files

| File | Content | Records |
| --- | --- | ---: |
| `data/raw/yahoo_chart_AAPL_20190101_20251231.json` | AAPL API response | 1,760 |
| `data/raw/yahoo_chart_NVDA_20190101_20251231.json` | NVDA API response | 1,760 |
| `data/raw/yahoo_chart_SPY_20190101_20251231.json` | SPY API response | 1,760 |
| `data/raw/SP73_Data_2019_2025_source_links.txt` | Shared archive source links | N/A |

### 3.2 Unified market-data file

File: `data/processed/market_data_v1.csv`

- Total records: 8,800;
- Records per symbol: 1,760;
- Symbols: AAPL, NVDA, ORCL, QCOM, SPY;
- Date range: 2019-01-02 to 2025-12-31;
- Missing values in the current CSV: 0;
- Row grain: `symbol + date`;
- Ordering: grouped by symbol and ascending date;
- Price values preserve the precision returned by Yahoo Finance;
- `adj_close` is retained for return and label construction. Its use must remain consistent across all experiments.

The imported news table contains 30,902 Google News RSS search records. News provenance and limitations are documented in `docs/news_data_v1.md`. It is not the full FNSPID dataset.

## 4. Data Dictionary

| Field | Type | Meaning | Planned use |
| --- | --- | --- | --- |
| `symbol` | string | Equity or ETF ticker | Panel grouping and index |
| `date` | ISO date | Trading date | Time index and splitting |
| `open` | float | Daily opening price | OHLCV features |
| `high` | float | Daily high price | Range and volatility features |
| `low` | float | Daily low price | Range and volatility features |
| `close` | float | Daily closing price | Price-series input and reference |
| `adj_close` | float | Adjusted closing price | Returns and label candidates |
| `volume` | integer | Daily trading volume | Volume-change and liquidity features |

## 5. Feature and Label Design

The schema is defined in `data/processed/features_v1_template.csv`. The implemented pipeline generates one row per `symbol + date`.

### 5.1 Price and lagged features

- `return_1d`, `return_5d`, `return_20d`;
- `lag_return_1d`, `lag_return_5d`;
- `rolling_volatility_5d`, `rolling_volatility_20d`;
- `high_low_range`;
- `volume_change_1d`.

Returns and price indicators use `adj_close`. Warm-up values remain missing and are not replaced with zero.

### 5.2 Technical indicators

- `sma_5`, `sma_20`, `sma_60`;
- `ema_12`, `ema_26`;
- `rsi_14` using a Wilder-style exponential average;
- `macd = ema_12 - ema_26`;
- `macd_signal`, the 9-day exponential average of MACD.

Any future indicator must document its formula, window, and price field.

### 5.3 Market-relative features

SPY is used as the market reference:

- `spy_relative_return = asset_return_1d - spy_return_1d`;
- `spy_relative_volatility = asset_rolling_volatility_20d - spy_rolling_volatility_20d`;
- `excess_return_5d = asset_return_5d - spy_return_5d`.

The SPY reference is joined by date before relative features are calculated. No future date is used. SPY's own relative values are zero when both sides are available.

### 5.4 Prediction labels

- Regression: `target_return_1d`, the next-trading-day log return;
- Classification: `target_direction_1d`, equal to 1 when the next-day return is positive and 0 otherwise.

The final row of each symbol has no next-day label and is removed from `model_ready_v1.csv`.

## 6. Implemented Code Structure

```text
src/
  data/
    download_market_data.py      # Future network download and raw-file cache
    validate_market_data.py      # Field, date, missing-value, and duplicate checks
  features/
    build_features.py            # Return, technical, and market-relative features
    build_labels.py              # Regression and classification labels
  models/
    baselines.py                 # Future naive baseline and ARIMA
    tree_models.py               # Future RF/XGBoost
    neural_models.py             # Future MLP/LSTM
  evaluation/
    split.py                     # Fixed time split and future walk-forward support
    metrics.py                   # Future prediction and trading metrics
    report.py                    # Future result tables and figures
```

Implemented modules are `validate_market_data.py`, `build_features.py`, `build_labels.py`, `split.py`, `import_shared_dataset.py`, and `run_v1_validation.py`. Future models must reuse the current time-split and output conventions.

Run the complete pipeline with:

```bash
V1/.venv/bin/python V1/scripts/run_v1_validation.py
```

The command validates the raw data, builds features and labels, removes unusable rows, assigns fixed time splits, and writes reports, summaries, and a feature overview figure.

## 7. Future Training Design

### 7.1 Baseline order

1. Zero-return or historical-mean naive baseline;
2. ARIMA;
3. Random Forest or XGBoost;
4. MLP;
5. LSTM.

If the workload must be reduced, the main comparison table should retain `naive + ARIMA + XGBoost + LSTM`; Random Forest and MLP can be extension experiments.

### 7.2 Time splits

The current pipeline uses chronological splits without random shuffling:

- Training: 2019–2023;
- Validation: 2024;
- Final test: 2025.

The final test set must not be used for model selection before hyperparameters are frozen. Future walk-forward experiments must record window length, retraining frequency, and forecast horizon.

### 7.3 Evaluation metrics

Regression: `MAE`, `RMSE`, and directional accuracy.

Classification: balanced accuracy, precision, recall, F1, and ROC-AUC.

If a trading backtest is added: cost-adjusted cumulative return, annualized Sharpe, maximum drawdown, and turnover.

## 8. Future Agentic AI Interface

Agentic AI is outside the current validation implementation. The proposed future flow is:

```text
SEC Retriever -> Filing Parser -> Evidence Summarizer
              -> Structured Feature Extractor -> Model Dataset
```

Each text feature should retain `filing_id`, `event_date`, evidence location, model version, prompt version, and generation timestamp. The public availability time of each filing must precede the corresponding prediction time.

Planned ablation groups are:

- A: price, volume, and lagged features;
- B: A + technical and market-relative features;
- C: B + dictionary/FinBERT text features;
- D: B + Agentic AI structured features.

## 9. Progress Record

| Date | Item | Status | Notes |
| --- | --- | --- | --- |
| 2026-09-20 | Initial literature review, research questions, and baseline protocol | Completed | See `docs/literature/` |
| 2026-10-02 | Created the V1 directory and technical document | Completed | This document |
| 2026-10-02 | Downloaded AAPL, NVDA, and SPY market data | Completed | Yahoo Chart API |
| 2026-10-19 | Imported the shared five-symbol price and news archive | Completed | 8,800 price rows; 30,902 news records |
| 2026-10-19 | Generated the unified market-data CSV | Completed | 8,800 rows; no missing fields |
| 2026-10-02 | Created feature and result schemas | Completed | Schemas match the implemented output |
| 2026-10-02 | Implemented validation, feature, label, and time-split code | Completed | `src/` and `scripts/` |
| 2026-10-02 | Added first-version unit tests | Completed | `tests/` |
| 2026-10-02 | Created the V1 virtual environment | Completed | `V1/.venv/` is not version-controlled |
| 2026-10-19 | Installed dependencies and ran the complete pipeline | Completed | 8,800 feature rows; 8,500 model-ready rows |
| 2026-10-02 | Ran the V1 unit tests | Completed | 10 passed |
| Pending | Train the first baseline model | Not started | Outside the current scope |
| Pending | Add SEC/Agentic AI features | Not started | After numerical baselines |

## 10. Reproducibility and Validation

SHA-256 hashes for the raw and unified data files are stored in `data/metadata/sha256sums.txt`. Any future data update must regenerate and record the hashes.

Completed static checks:

- Each of five symbols contains 1,760 trading records;
- The CSV contains 8,800 rows;
- Required fields are present;
- The raw CSV contains no missing or `null` values;
- The date range matches the V1 specification.

Completed code checks and runtime validation:

- Data-validation unit tests: passed;
- Return, label, and SPY-relative feature tests: passed;
- Time-boundary and non-overlap tests: passed;
- Feature-summary table and overview figure generation: passed;
- Full command: `V1/.venv/bin/python V1/scripts/run_v1_validation.py`;
- Test command: `V1/.venv/bin/pytest V1/tests -q`, result `10 passed`.

Checks still requiring future confirmation:

- Corporate-action and adjustment consistency beyond the current Yahoo `adj_close` convention;
- Time-zone and exchange-calendar policy;
- Model training and predictive metrics.

Actual V1 run results:

- `features_v1.csv`: 8,800 rows and 24 columns, retaining warm-up and final-label missing values;
- `model_ready_v1.csv`: 8,500 rows and 24 columns, with 1,700 rows per symbol;
- `splits_v1.csv`: 8,500 rows and 25 columns;
- Training set: 1,199 rows per symbol, from 2019-03-28 to 2023-12-29;
- Validation set: 252 rows per symbol, from 2024-01-02 to 2024-12-31;
- Test set: 249 rows per symbol, from 2025-01-02 to 2025-12-30;
- `feature_overview.png` generated successfully for all five symbols.
- `news_v1.csv` contains 30,902 rows; all required fields and article IDs passed the import checks.

## 11. V1 Output Contract

A successful run generates:

- `data/processed/features_v1.csv`: features, labels, and warm-up missing values;
- `data/processed/model_ready_v1.csv`: rows with complete features and labels;
- `data/processed/splits_v1.csv`: the added `train`, `validation`, or `test` assignment;
- `results/data_validation_report.json`: per-check validation results;
- `results/data_quality_summary.csv`: field- and symbol-level quality summary;
- `results/feature_summary_v1.csv`: feature counts, missing values, ranges, means, and standard deviations;
- `results/split_summary_v1.csv`: split and symbol record summary;
- `results/figures/feature_overview.png`: adjusted-close, return, volatility, and RSI overview;
- `results/v1_run_summary.json`: row counts, fields, and output list for the run.
