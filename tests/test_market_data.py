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


def test_refresh_and_validate_runs_quality_gate(service):
    svc, frame = service
    data = svc.refresh_and_validate("AAPL", as_of=frame.index[-1].date())
    assert data.ok
    assert data.quality is not None and data.quality.passed
    assert data.df is not None and len(data.df) == len(frame)
