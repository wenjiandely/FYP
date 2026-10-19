# V1 Price Data Quality Check Record

Check date: 2026-10-19

| Check | Result |
| --- | --- |
| Number of symbols | 5 |
| Symbols | AAPL, NVDA, ORCL, QCOM, SPY |
| Records per symbol | 1,760 |
| Total records | 8,800 |
| Earliest date | 2019-01-02 |
| Latest date | 2025-12-31 |
| Required fields present | Yes |
| Missing values / `null` | 0 |
| Duplicate key check | Passed; no duplicate `symbol + date` rows |
| OHLC business rules | Passed |
| Adjustment consistency | Yahoo Finance `adj_close` is used; the rule must remain fixed during model experiments |
| Time zone / trading calendar | Dates are parsed from Yahoo daily records; no separate trading-calendar package is used |

This record covers structural and basic business-rule checks for the current price CSV. News quality is recorded separately in `news_quality_v1.json`. Neither is a complete financial-data audit.
