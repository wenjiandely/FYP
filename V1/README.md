# V1 Initial Validation Package

Status: **V1 validation code is implemented and has completed a full run; model experiments have not started.**

This directory contains the first reproducible experiment assets for SP73, “Handling and Processing Time-Series Data using Agentic AI”. The current version includes:

All supervisor-facing documentation and experiment notes in this repository are maintained in English.

- A canonical five-symbol daily market-data file for AAPL, NVDA, ORCL, QCOM, and SPY;
- A normalized Google News RSS records file for the same five symbols;
- An import script that verifies and re-imports the shared ZIP;
- A data dictionary, file hashes, feature schema, and result schema;
- Data validation, feature, label, time-split, and one-command pipeline scripts;
- Unit tests and an experiment Notebook skeleton;
- No predictive-model training code and no fabricated model results.

See [`docs/technical_design_v1.md`](docs/technical_design_v1.md) for the detailed technical record.

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
