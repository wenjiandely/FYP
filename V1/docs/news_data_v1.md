# V1 News Data Record

## Source and scope

The imported file `data/processed/news_v1.csv` comes from Google News RSS search records supplied in the shared archive. It is not the full FNSPID dataset. The archive records the official FNSPID links separately, but the large original dataset was not copied into this repository.

The file contains titles and RSS summaries. It does not contain full third-party article text. The normalized fields are:

| Field | Meaning |
| --- | --- |
| `article_id` | Stable ID generated for the source record |
| `symbol` | Project ticker |
| `published_at_utc` | Parsed publication timestamp in UTC |
| `published_date_utc` | UTC calendar date used for date-level joins |
| `title` | Article title |
| `summary` | RSS description or summary |
| `source` | Publisher name |
| `source_url` | Publisher domain |
| `article_url` | Google News article link |
| `search_query` | Query used to retrieve the record |
| `source_dataset` | Provenance label |
| `content_scope` | `title_and_rss_summary_only` |

## Inventory

| Symbol | Records |
| --- | ---: |
| AAPL | 6,973 |
| NVDA | 8,010 |
| ORCL | 3,741 |
| QCOM | 3,839 |
| SPY | 8,339 |
| Total | 30,902 |

Coverage is 2019-01-01 through 2025-12-31. All required fields are populated and all `article_id` values are unique in the imported combined file.

## Limitations before modeling

- No company or topic relevance filter has been applied.
- ORCL searches contain some ordinary-language uses of “oracle”; QCOM and SPY also contain broad technology or market news.
- 30,220 of 30,902 timestamps are exactly midnight UTC or 23:00 UTC. These records should be treated as date-level news rather than exact intraday timestamps.
- The safe first join is to the next available trading date after publication, with a documented policy for records published during or after market hours.
- Google News RSS summaries should be used as source evidence, not assumed to be full article text.

The first text experiment should add a relevance flag, a publication-availability date, and an audit trail before sentiment or Agent features are generated.
