"""News ingestion: yfinance + Finnhub -> deduped, recent NewsItem list.

Both sources are best-effort: a failed source logs and contributes nothing
rather than failing the fetch. Items are deduped by normalized-headline hash
and capped to the most recent max_items.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import requests
import yfinance as yf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class NewsItem(BaseModel):
    ticker: str
    source: str
    headline: str
    summary: str = ""
    url: str = ""
    published_at: datetime
    content_hash: str


def _hash_headline(headline: str) -> str:
    normalized = re.sub(r"\W+", " ", headline.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def _yfinance_news(ticker: str) -> list[NewsItem]:
    items: list[NewsItem] = []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", ticker, exc)
        return items

    for entry in raw:
        # yfinance >= 0.2.50 nests fields under 'content'; older is flat.
        content = entry.get("content", entry)
        headline = content.get("title")
        if not headline:
            continue
        pub = content.get("pubDate") or content.get("providerPublishTime")
        if isinstance(pub, (int, float)):
            published = datetime.fromtimestamp(pub, tz=timezone.utc)
        elif isinstance(pub, str):
            try:
                published = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except ValueError:
                continue
        else:
            continue
        url = ""
        canonical = content.get("canonicalUrl")
        if isinstance(canonical, dict):
            url = canonical.get("url", "")
        elif isinstance(content.get("link"), str):
            url = content["link"]
        provider = content.get("provider")
        source = provider.get("displayName", "yahoo") if isinstance(provider, dict) else "yahoo"
        items.append(NewsItem(
            ticker=ticker,
            source=source,
            headline=headline,
            summary=(content.get("summary") or "")[:500],
            url=url,
            published_at=published,
            content_hash=_hash_headline(headline),
        ))
    return items


def _finnhub_news(ticker: str, api_key: str, lookback_days: int) -> list[NewsItem]:
    items: list[NewsItem] = []
    to_date = datetime.now(tz=timezone.utc).date()
    from_date = to_date - timedelta(days=lookback_days)
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "token": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        logger.warning("finnhub news failed for %s: %s", ticker, exc)
        return items

    for entry in raw if isinstance(raw, list) else []:
        headline = entry.get("headline")
        epoch = entry.get("datetime")
        if not headline or not epoch:
            continue
        items.append(NewsItem(
            ticker=ticker,
            source=entry.get("source", "finnhub"),
            headline=headline,
            summary=(entry.get("summary") or "")[:500],
            url=entry.get("url", ""),
            published_at=datetime.fromtimestamp(epoch, tz=timezone.utc),
            content_hash=_hash_headline(headline),
        ))
    return items


def fetch_news(
    ticker: str,
    lookback_days: int = 7,
    max_items: int = 25,
    finnhub_api_key: str | None = None,
) -> list[NewsItem]:
    """Fetch, merge, dedupe, and recency-filter news for one ticker."""
    items = _yfinance_news(ticker)
    if finnhub_api_key:
        items += _finnhub_news(ticker, finnhub_api_key, lookback_days)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
    seen: set[str] = set()
    deduped: list[NewsItem] = []
    for item in sorted(items, key=lambda i: i.published_at, reverse=True):
        if item.published_at < cutoff or item.content_hash in seen:
            continue
        seen.add(item.content_hash)
        deduped.append(item)
    return deduped[:max_items]


def persist_news(conn: sqlite3.Connection, items: list[NewsItem]) -> None:
    """Insert items into news_items, skipping already-seen hashes."""
    now = datetime.now(tz=timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT OR IGNORE INTO news_items
            (ticker, source, headline, url, published_at, content_hash, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (i.ticker, i.source, i.headline, i.url, i.published_at.isoformat(),
             i.content_hash, now)
            for i in items
        ],
    )
    conn.commit()
