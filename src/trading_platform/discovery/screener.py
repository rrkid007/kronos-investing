"""Discovery Agent — suggests watchlist candidates. Agents propose, the human
approves: nothing here mutates the watchlist; suggestions go to a table and
the report, and `discover_stocks.py --add` is the human's pen.

Funnel (cheap -> expensive):
1. Candidate selection: universe minus watchlist, gap sectors first, then
   least-recently-screened (the universe rotates through over weekly runs).
2. Quant screen per candidate: OHLCV refresh through the standard quality
   gate, then the SAME TechnicalAgent rubric and fundamentals scoring the
   daily pipeline uses — suggestions are judged exactly as members will be.
3. Ranking: combined score = technical and fundamental scores at their live
   relative weights (0.25 : 0.35, renormalized) + a sector-gap bonus
   (+8 sector absent from watchlist, +4 underrepresented). A fundamentals
   quality floor blocks momentum junk regardless of technical score.

Replacement pressure: past max_watchlist_size, the report flags the weakest
incumbent (lowest average final score over recent runs, not currently held)
so the list doesn't grow without bound.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import date, datetime, timezone

from pydantic import BaseModel

from trading_platform.agents.fundamentals import score_snapshot
from trading_platform.agents.technical import TechnicalAgent
from trading_platform.core.config import AppConfig
from trading_platform.data.fundamentals import fetch_fundamentals
from trading_platform.data.market_data import MarketDataService
from trading_platform.data.universe import load_universe, mark_screened

logger = logging.getLogger(__name__)

# Live relative weights of the two PIT-screenable signals (0.25 : 0.35).
TECH_WEIGHT = 0.25 / 0.60
FUND_WEIGHT = 0.35 / 0.60
GAP_BONUS_ABSENT = 8.0
GAP_BONUS_UNDERREPRESENTED = 4.0


class Suggestion(BaseModel):
    ticker: str
    name: str = ""
    sector: str = ""
    technical_score: float
    fundamental_score: float
    sector_gap_bonus: float
    combined_score: float
    rationale: str


def sector_counts(config: AppConfig) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in config.watchlist.tickers:
        counts[entry.sector] = counts.get(entry.sector, 0) + 1
    return counts


def sector_bonus(sector: str, counts: dict[str, int]) -> float:
    if sector not in counts:
        return GAP_BONUS_ABSENT
    average = sum(counts.values()) / len(counts)
    if counts[sector] < average:
        return GAP_BONUS_UNDERREPRESENTED
    return 0.0


def select_candidates(
    conn: sqlite3.Connection, config: AppConfig, max_candidates: int
) -> list[sqlite3.Row]:
    """Gap sectors first, then least-recently-screened — the screen rotates
    through the whole universe across weekly runs."""
    watchlist = set(config.watchlist.symbols)
    counts = sector_counts(config)
    rows = [r for r in load_universe(conn) if r["ticker"] not in watchlist]
    rows.sort(key=lambda r: (
        0 if r["sector"] not in counts else 1,
        r["last_screened_at"] or "",       # never-screened ('' sorts first)
        r["ticker"],
    ))
    return rows[:max_candidates]


def screen(
    conn: sqlite3.Connection,
    config: AppConfig,
    market_data: MarketDataService,
    as_of: date | None = None,
    fundamentals_fetcher=fetch_fundamentals,
) -> list[Suggestion]:
    """Run the funnel; persists the suggestion batch and returns it."""
    settings = config.settings.discovery
    candidates = select_candidates(conn, config, settings.max_candidates)
    counts = sector_counts(config)
    tech_agent = TechnicalAgent()
    suggestions: list[Suggestion] = []
    screened: list[str] = []

    for row in candidates:
        ticker = row["ticker"]
        screened.append(ticker)
        data = market_data.refresh_and_validate(ticker, as_of=as_of)
        if not data.ok:
            logger.info("discovery skip %s: %s", ticker, data.detail())
            continue
        price = float(data.df["close"].iloc[-1])
        if price < settings.min_price:
            continue

        tech = tech_agent.analyze(ticker, "discovery", data.df)
        if tech.confidence <= 0:
            continue

        try:
            scored = score_snapshot(fundamentals_fetcher(ticker))
        except Exception as exc:
            logger.info("discovery skip %s: fundamentals failed: %s", ticker, exc)
            continue
        if scored["coverage"] < 0.5:
            continue  # not enough data to trust the quality floor
        fund_score = scored["final"]
        if fund_score < settings.fundamentals_floor:
            continue  # quality floor: no momentum junk

        bonus = sector_bonus(row["sector"], counts)
        combined = round(TECH_WEIGHT * tech.score + FUND_WEIGHT * fund_score + bonus, 2)
        suggestions.append(Suggestion(
            ticker=ticker,
            name=row["name"] or "",
            sector=row["sector"],
            technical_score=tech.score,
            fundamental_score=fund_score,
            sector_gap_bonus=bonus,
            combined_score=combined,
            rationale=_rationale(row["sector"], bonus, tech, scored),
        ))

    mark_screened(conn, screened)
    suggestions.sort(key=lambda s: s.combined_score, reverse=True)
    top = suggestions[: settings.top_n]
    _persist_batch(conn, top)
    return top


def _rationale(sector: str, bonus: float, tech, scored: dict) -> str:
    parts = []
    if bonus == GAP_BONUS_ABSENT:
        parts.append(f"fills sector gap ({sector} absent from watchlist)")
    elif bonus == GAP_BONUS_UNDERREPRESENTED:
        parts.append(f"adds underrepresented sector ({sector})")
    parts.append(f"technical {tech.score:.0f} ({tech.details.get('trend', 'n/a')})")
    subs = scored["sub_scores"]
    parts.append(
        f"fundamentals {scored['final']:.0f} "
        f"(growth {subs['growth']:.0f}, profitability {subs['profitability']:.0f}, "
        f"valuation {subs['valuation']:.0f})"
    )
    return "; ".join(parts)


def _persist_batch(conn: sqlite3.Connection, suggestions: list[Suggestion]) -> str:
    batch_id = uuid.uuid4().hex[:8]
    now = datetime.now(tz=timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO watchlist_suggestions
            (batch_id, created_at, ticker, sector, combined_score, technical_score,
             fundamental_score, sector_gap_bonus, rationale)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (batch_id, now, s.ticker, s.sector, s.combined_score, s.technical_score,
             s.fundamental_score, s.sector_gap_bonus, s.rationale)
            for s in suggestions
        ],
    )
    conn.commit()
    return batch_id


def weakest_incumbent(
    conn: sqlite3.Connection, config: AppConfig, lookback_runs: int = 10
) -> dict | None:
    """The watchlist member with the lowest average final score over recent
    runs that is NOT currently held — the natural candidate to drop."""
    held = {
        r["ticker"] for r in conn.execute("SELECT ticker FROM positions").fetchall()
    }
    rows = conn.execute(
        """
        SELECT d.ticker, AVG(d.final_score) AS avg_score, COUNT(*) AS n
        FROM decisions d
        WHERE d.run_id IN (
            SELECT run_id FROM runs ORDER BY started_at DESC LIMIT ?
        )
        GROUP BY d.ticker
        """,
        (lookback_runs,),
    ).fetchall()
    candidates = [
        r for r in rows
        if r["ticker"] in config.watchlist.symbols and r["ticker"] not in held
    ]
    if not candidates:
        return None
    weakest = min(candidates, key=lambda r: r["avg_score"])
    return {
        "ticker": weakest["ticker"],
        "avg_final_score": round(weakest["avg_score"], 2),
        "n_runs": weakest["n"],
    }
