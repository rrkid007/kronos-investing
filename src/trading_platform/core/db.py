"""SQLite engine: WAL mode, single-writer discipline, versioned schema.

The orchestrator is the only writer; the dashboard reads. Migrations are
applied in order by PRAGMA user_version — to evolve the schema, add an entry
to MIGRATIONS and bump SCHEMA_VERSION.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 5

# Version 1: full base schema. Version 2: broker routing columns (phase 13).
# Version 3: discovery — candidate universe + watchlist suggestions (phase 14).
# Version 4: macro indicator series cache (phase 15).
# Version 5: pre-approval research memos (phase 16).
MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE runs (
        run_id      TEXT PRIMARY KEY,
        run_date    TEXT NOT NULL,
        started_at  TEXT NOT NULL,
        finished_at TEXT,
        status      TEXT NOT NULL CHECK (status IN ('running','completed','failed','partial')),
        notes       TEXT
    );

    CREATE TABLE run_stages (
        run_id     TEXT NOT NULL REFERENCES runs(run_id),
        stage      TEXT NOT NULL,
        ticker     TEXT NOT NULL DEFAULT '',
        status     TEXT NOT NULL CHECK (status IN ('pending','running','completed','failed','skipped')),
        detail     TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (run_id, stage, ticker)
    );

    CREATE TABLE agent_scores (
        run_id     TEXT NOT NULL REFERENCES runs(run_id),
        agent      TEXT NOT NULL,
        ticker     TEXT NOT NULL,
        score      REAL NOT NULL CHECK (score >= 0 AND score <= 100),
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        direction  TEXT,
        details    TEXT,
        data_as_of TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (run_id, agent, ticker)
    );

    CREATE TABLE decisions (
        run_id           TEXT NOT NULL REFERENCES runs(run_id),
        ticker           TEXT NOT NULL,
        action           TEXT NOT NULL CHECK (action IN ('buy','sell','hold','watchlist')),
        final_score      REAL NOT NULL,
        signal_breakdown TEXT,
        sizing_hint      REAL,
        reason           TEXT,
        created_at       TEXT NOT NULL,
        PRIMARY KEY (run_id, ticker)
    );

    CREATE TABLE risk_events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id     TEXT NOT NULL REFERENCES runs(run_id),
        ticker     TEXT,
        approved   INTEGER NOT NULL,
        violations TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE orders (
        order_id   TEXT PRIMARY KEY,
        run_id     TEXT NOT NULL REFERENCES runs(run_id),
        ticker     TEXT NOT NULL,
        side       TEXT NOT NULL CHECK (side IN ('buy','sell')),
        qty        REAL NOT NULL CHECK (qty > 0),
        status     TEXT NOT NULL CHECK (status IN
                     ('awaiting_approval','approved','rejected','expired','filled','cancelled')),
        created_at TEXT NOT NULL,
        decided_at TEXT,
        expires_at TEXT,
        notes      TEXT,
        UNIQUE (run_id, ticker, side)
    );

    CREATE TABLE fills (
        fill_id    TEXT PRIMARY KEY,
        order_id   TEXT NOT NULL REFERENCES orders(order_id),
        ticker     TEXT NOT NULL,
        side       TEXT NOT NULL,
        qty        REAL NOT NULL,
        price      REAL NOT NULL,
        slippage   REAL NOT NULL DEFAULT 0,
        commission REAL NOT NULL DEFAULT 0,
        filled_at  TEXT NOT NULL
    );

    CREATE TABLE positions (
        ticker     TEXT PRIMARY KEY,
        qty        REAL NOT NULL,
        avg_cost   REAL NOT NULL,
        opened_at  TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE account (
        id           INTEGER PRIMARY KEY CHECK (id = 1),
        cash         REAL NOT NULL,
        realized_pnl REAL NOT NULL DEFAULT 0,
        updated_at   TEXT NOT NULL
    );

    CREATE TABLE account_snapshots (
        snapshot_date  TEXT PRIMARY KEY,
        cash           REAL NOT NULL,
        equity         REAL NOT NULL,
        unrealized_pnl REAL NOT NULL,
        realized_pnl   REAL NOT NULL,
        created_at     TEXT NOT NULL
    );

    CREATE TABLE price_cache (
        ticker    TEXT NOT NULL,
        date      TEXT NOT NULL,
        open      REAL,
        high      REAL,
        low       REAL,
        close     REAL,
        adj_close REAL,
        volume    INTEGER,
        PRIMARY KEY (ticker, date)
    );

    CREATE TABLE news_items (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT NOT NULL,
        source       TEXT,
        headline     TEXT NOT NULL,
        url          TEXT,
        published_at TEXT,
        content_hash TEXT UNIQUE,
        fetched_at   TEXT NOT NULL
    );

    CREATE TABLE filings (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT NOT NULL,
        form_type    TEXT NOT NULL,
        filing_date  TEXT NOT NULL,
        accession_no TEXT UNIQUE,
        analyzed_at  TEXT,
        summary      TEXT
    );

    CREATE INDEX idx_agent_scores_ticker ON agent_scores(ticker);
    CREATE INDEX idx_orders_status ON orders(status);
    CREATE INDEX idx_fills_ticker ON fills(ticker);
    CREATE INDEX idx_news_ticker ON news_items(ticker);
    """,
    2: """
    ALTER TABLE orders ADD COLUMN broker TEXT NOT NULL DEFAULT 'local';
    ALTER TABLE orders ADD COLUMN broker_order_id TEXT;
    """,
    3: """
    CREATE TABLE universe (
        ticker           TEXT PRIMARY KEY,
        name             TEXT,
        sector           TEXT,
        refreshed_at     TEXT NOT NULL,
        last_screened_at TEXT
    );

    CREATE TABLE watchlist_suggestions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id          TEXT NOT NULL,
        created_at        TEXT NOT NULL,
        ticker            TEXT NOT NULL,
        sector            TEXT,
        combined_score    REAL NOT NULL,
        technical_score   REAL,
        fundamental_score REAL,
        sector_gap_bonus  REAL NOT NULL DEFAULT 0,
        rationale         TEXT,
        status            TEXT NOT NULL DEFAULT 'suggested'
                          CHECK (status IN ('suggested','added','dismissed'))
    );
    CREATE INDEX idx_suggestions_batch ON watchlist_suggestions(batch_id);
    """,
    4: """
    CREATE TABLE macro_indicators (
        series     TEXT NOT NULL,
        date       TEXT NOT NULL,
        value      REAL NOT NULL,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (series, date)
    );
    """,
    5: """
    CREATE TABLE research_memos (
        order_id       TEXT PRIMARY KEY REFERENCES orders(order_id),
        ticker         TEXT NOT NULL,
        run_id         TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        model          TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        memo_md        TEXT NOT NULL
    );
    """,
}


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a connection with WAL mode and foreign keys enabled."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations. Idempotent."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {current} is newer than this code "
            f"supports ({SCHEMA_VERSION}) — update the application"
        )
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.executescript(MIGRATIONS[version])
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
