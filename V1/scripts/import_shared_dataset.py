"""Import and normalize the shared SP73 price and news archive.

The source ZIP is intentionally kept outside the repository. This script makes
the import reproducible from that ZIP and writes only the canonical price table,
one normalized news table, and audit metadata into V1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT.parent / "SP73_Data_2019_2025_share(1).zip"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
EXPECTED_SYMBOLS = ["AAPL", "NVDA", "ORCL", "QCOM", "SPY"]
EXPECTED_START = pd.Timestamp("2019-01-01")
EXPECTED_END = pd.Timestamp("2025-12-31")
PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
NEWS_COLUMNS = [
    "article_id",
    "symbol",
    "published_at_utc",
    "published_date_utc",
    "title",
    "summary",
    "source",
    "source_url",
    "article_url",
    "search_query",
    "source_dataset",
    "content_scope",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    return parser.parse_args()


def read_csv_from_zip(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with archive.open(name) as handle:
        return pd.read_csv(handle, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prices(archive: zipfile.ZipFile, root: str) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    source_name = f"{root}/prices/prices_all_2019_2025.csv"
    prices = read_csv_from_zip(archive, source_name)
    prices = prices.rename(columns={"ticker": "symbol", "adjusted_close": "adj_close"})
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    prices = prices[PRICE_COLUMNS].sort_values(["symbol", "date"]).reset_index(drop=True)
    quality = {
        "rows": int(len(prices)),
        "symbols": sorted(prices["symbol"].unique().tolist()),
        "rows_by_symbol": {key: int(value) for key, value in prices["symbol"].value_counts().sort_index().items()},
        "min_date": prices["date"].min().date().isoformat(),
        "max_date": prices["date"].max().date().isoformat(),
        "duplicate_symbol_date": int(prices.duplicated(["symbol", "date"]).sum()),
        "missing_by_column": {key: int(value) for key, value in prices.isna().sum().items()},
    }
    return prices, quality


def validate_price_files(archive: zipfile.ZipFile, root: str, combined: pd.DataFrame) -> dict[str, int]:
    results: dict[str, int] = {}
    for symbol in EXPECTED_SYMBOLS:
        split_name = f"{root}/prices/{symbol}_2019_2025.csv"
        split = read_csv_from_zip(archive, split_name)
        split = split.rename(columns={"ticker": "symbol", "adjusted_close": "adj_close"})
        split["date"] = pd.to_datetime(split["date"], errors="raise")
        split = split[PRICE_COLUMNS].sort_values(["symbol", "date"]).reset_index(drop=True)
        merged = combined.loc[combined["symbol"] == symbol].reset_index(drop=True)
        if not split.equals(merged):
            raise ValueError(f"split and merged price files differ for {symbol}")
        results[symbol] = int(len(split))
    return results


def normalize_news(archive: zipfile.ZipFile, root: str) -> tuple[pd.DataFrame, dict[str, object]]:
    source_name = f"{root}/news/news_google_rss_2019_2025.csv"
    news = read_csv_from_zip(archive, source_name)
    news = news.rename(columns={"ticker": "symbol", "published_at": "published_at_utc"})
    news["published_at_utc"] = pd.to_datetime(news["published_at_utc"], utc=True, errors="raise")
    news["published_date_utc"] = news["published_at_utc"].dt.strftime("%Y-%m-%d")
    news["source_dataset"] = "Google News RSS search records"
    news["content_scope"] = "title_and_rss_summary_only"
    news = news[NEWS_COLUMNS].sort_values(["symbol", "published_at_utc", "article_id"]).reset_index(drop=True)
    timestamps = news["published_at_utc"]
    duplicate_ids = int(news["article_id"].duplicated().sum())
    quality = {
        "rows": int(len(news)),
        "symbols": sorted(news["symbol"].unique().tolist()),
        "rows_by_symbol": {key: int(value) for key, value in news["symbol"].value_counts().sort_index().items()},
        "min_date": timestamps.min().date().isoformat(),
        "max_date": timestamps.max().date().isoformat(),
        "duplicate_article_id": duplicate_ids,
        "missing_by_column": {key: int(value) for key, value in news.isna().sum().items()},
        "timestamp_exact_midnight_or_23": int(
            timestamps.dt.strftime("%H:%M:%S%z").isin(["00:00:00+0000", "23:00:00+0000"]).sum()
        ),
        "relevance_filter_applied": False,
    }
    return news, quality


def validate_news_files(archive: zipfile.ZipFile, root: str, combined: pd.DataFrame) -> dict[str, int]:
    results: dict[str, int] = {}
    for symbol in EXPECTED_SYMBOLS:
        split_name = f"{root}/news/{symbol}_news_2019_2025.csv"
        split = read_csv_from_zip(archive, split_name)
        split = split.rename(columns={"ticker": "symbol", "published_at": "published_at_utc"})
        split["published_at_utc"] = pd.to_datetime(split["published_at_utc"], utc=True, errors="raise")
        split = split.sort_values(["symbol", "published_at_utc", "article_id"]).reset_index(drop=True)
        merged = combined.loc[combined["symbol"] == symbol, [
            "article_id", "symbol", "published_at_utc", "title", "summary", "source", "source_url", "article_url", "search_query"
        ]].reset_index(drop=True)
        source_columns = [
            "article_id", "symbol", "published_at_utc", "title", "summary", "source", "source_url", "article_url", "search_query"
        ]
        if not split[source_columns].equals(merged):
            raise ValueError(f"split and merged news files differ for {symbol}")
        results[symbol] = int(len(split))
    return results


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    output.to_csv(path, index=False, encoding="utf-8")


def write_manifest(manifest: dict[str, object]) -> None:
    rows = [
        {
            "file": "data/processed/market_data_v1.csv",
            "source": manifest["price_source"],
            "period_start": manifest["prices"]["min_date"],
            "period_end": manifest["prices"]["max_date"],
            "records": manifest["prices"]["rows"],
            "notes": "Canonical five-symbol long-format price table",
        },
        {
            "file": "data/processed/news_v1.csv",
            "source": manifest["news_source"],
            "period_start": manifest["news"]["min_date"],
            "period_end": manifest["news"]["max_date"],
            "records": manifest["news"]["rows"],
            "notes": "Title and RSS summary only; no full article text",
        },
        {
            "file": "data/raw/SP73_Data_2019_2025_source_links.txt",
            "source": "Shared archive metadata",
            "period_start": "",
            "period_end": "",
            "records": "",
            "notes": "Original source links and FNSPID reference",
        },
    ]
    pd.DataFrame(rows).to_csv(METADATA_DIR / "data_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    if not args.archive.exists():
        raise FileNotFoundError(f"shared archive not found: {args.archive}")
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        roots = sorted({name.split("/", 1)[0] for name in archive.namelist() if "/" in name})
        if len(roots) != 1:
            raise ValueError(f"expected one archive root, found: {roots}")
        root = roots[0]
        prices, price_quality = normalize_prices(archive, root)
        news, news_quality = normalize_news(archive, root)
        price_split_counts = validate_price_files(archive, root, prices)
        news_split_counts = validate_news_files(archive, root, news)

    if price_quality["symbols"] != EXPECTED_SYMBOLS or price_quality["rows"] != 8800:
        raise ValueError(f"unexpected price inventory: {price_quality}")
    if news_quality["symbols"] != EXPECTED_SYMBOLS:
        raise ValueError(f"unexpected news inventory: {news_quality}")
    required_news = ["article_id", "symbol", "published_at_utc", "title", "summary", "source", "article_url"]
    missing_news = {column: int(news[column].isna().sum()) for column in required_news if news[column].isna().any()}
    if missing_news:
        raise ValueError(f"required news fields contain missing values: {missing_news}")
    if news["article_id"].duplicated().any():
        raise ValueError("duplicate article_id values in merged news records")
    if not news["published_at_utc"].between(
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
    ).all():
        raise ValueError("news publication timestamp outside expected period")

    write_csv(prices, PROCESSED_DIR / "market_data_v1.csv")
    write_csv(news, PROCESSED_DIR / "news_v1.csv")

    source_archive = RAW_DIR / "SP73_Data_2019_2025_source_links.txt"
    source_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        source_archive.write_text(
            archive.read(f"{root}/source_links.txt").decode("utf-8", errors="replace"), encoding="utf-8"
        )

    manifest = {
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive_name": args.archive.name,
        "source_archive_sha256": sha256_file(args.archive),
        "source_root": root,
        "price_source": "Yahoo Finance Chart API export included in shared archive",
        "news_source": "Google News RSS search records included in shared archive",
        "fnspid_status": "source link recorded only; FNSPID full dataset is not included",
        "prices": price_quality,
        "price_split_file_counts": price_split_counts,
        "news": news_quality,
        "news_split_file_counts": news_split_counts,
        "outputs": {
            "prices": "data/processed/market_data_v1.csv",
            "news": "data/processed/news_v1.csv",
            "source_links": "data/raw/SP73_Data_2019_2025_source_links.txt",
        },
    }
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (METADATA_DIR / "shared_dataset_import.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (METADATA_DIR / "news_quality_v1.json").write_text(
        json.dumps(news_quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
