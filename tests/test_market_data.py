"""MarketDataService tests — yfinance is patched out; no network in tests."""

from datetime import date

import pandas as pd
import pytest

import trading_platform.data.market_data as md
from tests.fixtures import make_ohlcv
from trading_platform.core.db import connect, init_db

AS_OF = date(2026, 6, 11)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.sqlite")
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def service(conn, monkeypatch):
    frame = make_ohlcv()
    monkeypatch.setattr(md, "_download", lambda ticker, start, end: frame.copy())
    return md.MarketDataService(conn), frame


def test_refresh_upserts_all_rows(service):
    svc, frame = service
    result = svc.refresh("AAPL", as_of=AS_OF)
    assert result.error is None
    assert result.rows_upserted == len(frame)
    assert result.last_date == frame.index[-1].date()


def test_refresh_is_idempotent(service, conn):
    svc, frame = service
    svc.refresh("AAPL", as_of=AS_OF)
    svc.refresh("AAPL", as_of=AS_OF)  # second refresh upserts, never duplicates
    count = conn.execute("SELECT COUNT(*) FROM price_cache WHERE ticker='AAPL'").fetchone()[0]
    assert count == len(frame)


def test_load_roundtrip_preserves_values(service):
    svc, frame = service
    svc.refresh("AAPL", as_of=AS_OF)
    loaded = svc.load("AAPL")
    assert len(loaded) == len(frame)
    assert loaded.index[0] == frame.index[0]
    assert loaded["close"].iloc[-1] == pytest.approx(frame["close"].iloc[-1])
    assert loaded["adj_close"].iloc[100] == pytest.approx(frame["adj_close"].iloc[100])


def test_refresh_updates_changed_adjusted_prices(service, conn, monkeypatch):
    """A retroactive split adjustment must overwrite cached adj_close."""
    svc, frame = service
    svc.refresh("AAPL", as_of=AS_OF)

    adjusted = frame.copy()
    adjusted["adj_close"] = adjusted["adj_close"] * 0.5
    monkeypatch.setattr(md, "_download", lambda ticker, start, end: adjusted.copy())
    svc.refresh("AAPL", as_of=AS_OF)

    loaded = svc.load("AAPL")
    assert loaded["adj_close"].iloc[0] == pytest.approx(frame["adj_close"].iloc[0] * 0.5)


def test_download_failure_returns_error_not_exception(conn, monkeypatch):
    def boom(ticker, start, end):
        raise ConnectionError("DNS failure")

    monkeypatch.setattr(md, "_download", boom)
    svc = md.MarketDataService(conn)
    result = svc.refresh("AAPL", as_of=AS_OF)
    assert result.error is not None
    assert "DNS failure" in result.error


def test_empty_download_is_an_error(conn, monkeypatch):
    monkeypatch.setattr(md, "_download", lambda t, s, e: pd.DataFrame())
    svc = md.MarketDataService(conn)
    result = svc.refresh("UNKNOWN", as_of=AS_OF)
    assert result.error == "no data returned"


def _corrupt_last_bar(frame):
    """Mimic Yahoo's intraday partial bar: open above high (prior-day open)."""
    bad = frame.copy()
    bad.iloc[-1, bad.columns.get_loc("open")] = bad.iloc[-1]["high"] * 1.02
    return bad


def test_partial_last_bar_dropped_on_intraday_refresh(conn, monkeypatch):
    frame = make_ohlcv()  # ends at AS_OF
    monkeypatch.setattr(md, "_download", lambda t, s, e: _corrupt_last_bar(frame))
    svc = md.MarketDataService(conn)
    result = svc.refresh("AAPL", as_of=AS_OF)

    assert result.error is None
    assert result.rows_upserted == len(frame) - 1
    assert result.last_date < AS_OF  # partial bar excluded
    loaded = svc.load("AAPL")
    assert loaded.index[-1].date() < AS_OF


def test_inconsistent_historical_bar_is_kept_for_gate_to_catch(conn, monkeypatch):
    """Only the as_of-dated bar is treated as partial; historical corruption
    must reach the quality gate, not be silently repaired."""
    frame = make_ohlcv()
    bad = frame.copy()
    bad.iloc[100, bad.columns.get_loc("open")] = bad.iloc[100]["high"] * 1.05
    monkeypatch.setattr(md, "_download", lambda t, s, e: bad)
    svc = md.MarketDataService(conn)
    data = svc.refresh_and_validate("AAPL", as_of=AS_OF)
    assert not data.ok
    assert "high_low_consistency" in data.quality.summary()


def test_stale_partial_bar_purged_from_cache(conn, monkeypatch):
    """A partial bar cached by an intraday run disappears after a clean refresh."""
    frame = make_ohlcv()
    monkeypatch.setattr(md, "_download", lambda t, s, e: _corrupt_last_bar(frame))
    svc = md.MarketDataService(conn)
    # Force-cache the bad bar by writing the corrupted frame pre-fix style:
    conn.execute(
        "INSERT OR REPLACE INTO price_cache (ticker, date, open, high, low, close, adj_close, volume) "
        "VALUES ('AAPL', ?, 999, 100, 99, 99.5, 99.5, 1000)",
        (AS_OF.isoformat(),),
    )
    conn.commit()
    svc.refresh("AAPL", as_of=AS_OF)
    loaded = svc.load("AAPL")
    assert loaded.index[-1].date() < AS_OF  # purged, not lingering


def test_refresh_and_validate_runs_quality_gate(service):
    svc, frame = service
    data = svc.refresh_and_validate("AAPL", as_of=frame.index[-1].date())
    assert data.ok
    assert data.quality is not None and data.quality.passed
    assert data.df is not None and len(data.df) == len(frame)
