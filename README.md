# Handling and Processing Time-Series Data using Agentic AI
This is a final year project for NUSRI.

## Current Experiment Definition

The current project-level target and engineering plan are maintained in [`docs/project_scope_v2.md`](docs/project_scope_v2.md). The working target is a five-trading-session daily adjusted-close path forecast plus the mean of those five future adjusted closes; the exact meaning of “average” remains pending supervisor confirmation. The aligned ARIMA/XGBoost pipeline has been implemented and run; its corrected metrics and limitations are recorded in [`V1/docs/aligned_baselines_v2.md`](V1/docs/aligned_baselines_v2.md). Earlier one-day labels and mismatched model comparisons remain historical exploratory results.

## Project Aim

The project broadly aims to:

- Build a clean US stock and ETF dataset from open or freemium sources.
- Implement two or three baseline prediction methods for comparison.
- Engineer price-based, technical, and text-derived features.
- Design an Agentic AI workflow to extract and summarize public market signals.
- Compare baseline models with the Agentic AI-enhanced approach.
- Evaluate results using forecasting and classification metrics.
- Produce a reproducible prototype that can later be extended into a trading decision-support tool.

An alternative structure is one shared baseline strategy plus one additional strategy from each student, giving three strategies for comparison. The Agentic AI workflow is an important learning and implementation component. A competition-oriented extension is optional.

## Proposed Work Packages

The proposed strategy can be refined after the first validation results.

### Common Work

**CW0: Data collection and preprocessing**

This must be fixed and documented first. Both students should use the same raw data, cleaning rules, labels, time splits, and evaluation protocol.

**CW1: Agentic AI pipeline setup**

Both students should focus on the intelligent information-gathering side. The work should include an Agent that collects Securities and Exchange Commission (SEC) filings, extracts relevant text, summarizes market events, and converts the output into structured features. These features should later be integrated into the prediction pipeline and compared with pure numerical ML models.

**CW2: Feature engineering**

After the common data pipeline is stable, each student may develop an individual feature-engineering component. The feature definitions, availability time, and leakage controls must be documented because they can affect the final comparison.

### Student 1: Machine Learning and Agentic AI Pipeline

Student 1 focuses on quantitative forecasting and contributes CW0–CW2, together with baseline models such as ARIMA and Random Forest. Responsibilities also include chronological train/validation/test splitting, backtesting, and model evaluation.

### Student 2: Machine Learning and Agentic AI Pipeline

Student 2 focuses on quantitative forecasting and contributes CW0–CW2, together with baseline models such as XGBoost and LSTM. Responsibilities also include chronological train/validation/test splitting, backtesting, and model evaluation.

**CW3:** Include MLP processing as part of the Agentic AI pipeline and compare its performance with the other models.

During Semester 2, if the project is progressing well, an original strategy may be proposed and tested from December 2026 onward.

## Suggested Methodology

### 1. Data Collection

The initial examples below use US stocks because the project brief does not assume prior familiarity with Chinese stocks. The asset universe can later be changed to Chinese stocks or ETFs.

Use open or freemium sources such as Yahoo Finance for daily OHLCV data, Stooq as a backup source, and SEC EDGAR for filings such as 10-K, 10-Q, and 8-K. The study may focus on liquid US or Chinese stocks and ETFs such as AAPL, MSFT, NVDA, SPY, QQQ, and sector ETFs. The dataset should cover at least three to five years of daily records to support stable training and evaluation. Start with a small asset set and expand only after the pipeline is reliable.

### 2. Data Preprocessing

Clean missing values, align dates across assets, remove duplicate records, and define a consistent price-adjustment policy. Create labels such as next-day direction, five-day direction, or next-day return. Split the dataset chronologically to avoid look-ahead bias and data leakage.

### 3. Feature Engineering

Generate technical indicators such as moving averages, RSI, MACD, returns, volatility, and volume changes. Add market-relative features such as comparisons with SPY or sector ETFs. If filings are used, derive features such as filing frequency, sentiment, keyword counts, risk indicators, and structured event summaries.

### 4. Baseline Modeling

Implement two or three standard methods first. A suitable initial set is ARIMA, XGBoost, and LSTM. ARIMA provides a classical statistical benchmark, XGBoost provides a strong tabular ML baseline, and LSTM models sequential dependencies. These models form the comparison foundation before the Agentic AI layer is introduced.

### 5. Agentic AI Integration

Build a simple multi-agent workflow:

1. A retrieval agent obtains relevant SEC filings or public market text.
2. A parsing or evidence agent extracts relevant passages and metadata.
3. A summarization agent produces a concise market-event summary.
4. A coordinator agent converts the summary into structured daily features.

Append these features to the ML dataset and test whether contextual information improves prediction. CW3 should be included in the broader Agentic AI pipeline comparison.

### 6. Evaluation

For regression outputs, use MAE, RMSE, and directional accuracy. For classification outputs, use accuracy, precision, recall, F1-score, balanced accuracy, and ROC-AUC. If time permits, include a small paper-trading simulation or long-only backtest with transaction-cost assumptions to assess practical value.

### 7. Comparison and Analysis

Compare results across stocks and ETFs, and compare baseline models with Agentic AI-enhanced models. Discuss where Agentic AI features help, where they fail, and whether the added complexity is justified. Include focused error analysis for high-volatility periods and filing or earnings-release windows.

## Recommended Data Sources

- **Yahoo Finance / yfinance:** US stocks and ETF daily historical data.
- **Stooq:** backup historical price data.
- **SEC EDGAR API:** official filings and structured filing metadata.
- **Kaggle:** optional early-stage datasets for prototyping.
- **Nasdaq Data Link:** optional free macroeconomic or sector datasets.

## Expected Deliverables

- A literature review on stock forecasting and Agentic AI in finance for Chapter 2.
- A cleaned selected-stock and ETF dataset.
- A reproducible feature-generation pipeline.
- Two or three baseline model implementations.
- An Agentic AI module that extracts and summarizes public market signals.
- A final comparison and analysis report.
- An independently runnable prototype or experiment Notebook from each student.

All project files intended for supervisor review should be maintained in English.
