from trading_platform.core.db import SCHEMA_VERSION, connect, init_db

EXPECTED_TABLES = {
    "runs",
    "run_stages",
    "agent_scores",
    "decisions",
    "risk_events",
    "orders",
    "fills",
    "positions",
    "account",
    "account_snapshots",
    "price_cache",
    "news_items",
    "filings",
}


def _tables(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r["name"] for r in rows}


def test_init_creates_all_tables(tmp_path):
    conn = connect(tmp_path / "test.sqlite")
    init_db(conn)
    assert EXPECTED_TABLES <= _tables(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def test_init_is_idempotent(tmp_path):
    path = tmp_path / "test.sqlite"
    conn = connect(path)
    init_db(conn)
    init_db(conn)  # second call must be a no-op, not an error
    conn.close()
    conn = connect(path)
    init_db(conn)  # reopen and re-init too
    assert EXPECTED_TABLES <= _tables(conn)
    conn.close()


def test_wal_mode_enabled(tmp_path):
    conn = connect(tmp_path / "test.sqlite")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()
