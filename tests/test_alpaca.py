"""Alpaca paper broker — submission, sync, reconciliation, migration. No network."""

from datetime import date

import pytest
import requests

import trading_platform.execution.alpaca as alpaca_mod
from trading_platform.core.config import load_config
from trading_platform.core.db import MIGRATIONS, SCHEMA_VERSION, connect, init_db
from trading_platform.execution.alpaca import AlpacaError, AlpacaPaperBroker, alpaca_symbol
from trading_platform.execution.alpaca_sync import sync_alpaca
from trading_platform.execution.approval import approve_and_submit
from trading_platform.execution.orders import submit_order
from trading_platform.execution.paper_broker import ensure_account, get_account
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeAlpacaAPI:
    """Routes requests.request calls; records payloads; serves canned state."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.orders: dict[str, dict] = {}     # broker_order_id -> order info
        self.positions: list[dict] = []
        self.account = {"cash": "100000", "equity": "100000"}
        self.fail_submit = False
        self._counter = 0

    def __call__(self, method, url, headers=None, json=None, timeout=None):
        path = url.replace("https://paper-api.alpaca.markets", "")
        self.calls.append((method, path, json))

        class Resp:
            status_code = 200
            text = ""
            def __init__(self, payload):
                self._payload = payload
            def json(self):
                return self._payload

        if method == "POST" and path == "/v2/orders":
            if self.fail_submit:
                r = Resp({"message": "insufficient buying power"})
                r.status_code = 403
                r.text = "insufficient buying power"
                return r
            self._counter += 1
            broker_id = f"alp-{self._counter:04d}"
            self.orders[broker_id] = {
                "id": broker_id, "status": "accepted",
                "client_order_id": json["client_order_id"],
                "symbol": json["symbol"], "side": json["side"], "qty": json["qty"],
            }
            return Resp(self.orders[broker_id])
        if method == "GET" and path.startswith("/v2/orders/"):
            broker_id = path.rsplit("/", 1)[1]
            return Resp(self.orders[broker_id])
        if method == "GET" and path == "/v2/positions":
            return Resp(self.positions)
        if method == "GET" and path == "/v2/account":
            return Resp(self.account)
        raise AssertionError(f"unexpected call {method} {path}")


@pytest.fixture
def api(monkeypatch):
    fake = FakeAlpacaAPI()
    monkeypatch.setattr(requests, "request", fake)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test-secret")
    return fake


@pytest.fixture
def alpaca_config():
    config = load_config(REPO_ROOT / "config").model_copy(deep=True)
    config.settings.execution.broker = "alpaca_paper"
    return config


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.sqlite")
    init_db(c)
    ensure_account(c, 100_000.0)
    for i, d in enumerate(["2026-06-09", "2026-06-10", "2026-06-11"]):
        c.execute(
            "INSERT INTO runs (run_id, run_date, started_at, status) "
            "VALUES (?, ?, ?, 'completed')", (f"run-{i}", d, f"{d}T22:00:00"),
        )
    c.commit()
    yield c
    c.close()


# --- migration ----------------------------------------------------------------

def test_v1_database_migrates_to_v2(tmp_path):
    db = tmp_path / "old.sqlite"
    old = connect(db)
    old.executescript(MIGRATIONS[1])
    old.execute("PRAGMA user_version = 1")
    old.commit()
    old.close()

    upgraded = connect(db)
    init_db(upgraded)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    cols = {r["name"] for r in upgraded.execute("PRAGMA table_info(orders)").fetchall()}
    assert {"broker", "broker_order_id"} <= cols
    upgraded.close()


# --- broker client --------------------------------------------------------------

def test_missing_keys_refused(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    with pytest.raises(AlpacaError, match="keys are not set"):
        AlpacaPaperBroker("", "")


def test_symbol_mapping():
    assert alpaca_symbol("BRK-B") == "BRK.B"
    assert alpaca_symbol("AAPL") == "AAPL"


def test_submit_payload_market_on_open(api):
    broker = AlpacaPaperBroker("k", "s")
    broker.submit_market_on_open("BRK-B", "buy", 7, client_order_id="ord123")
    method, path, payload = api.calls[0]
    assert (method, path) == ("POST", "/v2/orders")
    assert payload == {
        "symbol": "BRK.B", "qty": "7", "side": "buy", "type": "market",
        "time_in_force": "opg", "client_order_id": "ord123",
    }


# --- approval routing -------------------------------------------------------------

def test_local_broker_approval_makes_no_http_calls(api, conn):
    config = load_config(REPO_ROOT / "config")  # broker: local
    oid = submit_order(conn, "run-2", "AAPL", "buy", 10, "2026-06-11", auto_approve=False)
    message = approve_and_submit(conn, config, oid)
    assert "local" in message
    assert api.calls == []


def test_alpaca_approval_submits_and_stores_broker_id(api, conn, alpaca_config):
    oid = submit_order(conn, "run-2", "AAPL", "buy", 10, "2026-06-11", auto_approve=False)
    message = approve_and_submit(conn, alpaca_config, oid)
    assert "alpaca paper" in message

    row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (oid,)).fetchone()
    assert row["status"] == "approved"
    assert row["broker"] == "alpaca_paper"
    assert row["broker_order_id"] == "alp-0001"
    assert api.orders["alp-0001"]["client_order_id"] == oid  # idempotency key


def test_failed_submission_keeps_order_approved_for_retry(api, conn, alpaca_config):
    api.fail_submit = True
    oid = submit_order(conn, "run-2", "AAPL", "buy", 10, "2026-06-11", auto_approve=False)
    message = approve_and_submit(conn, alpaca_config, oid)
    assert "submission failed" in message

    row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (oid,)).fetchone()
    assert row["status"] == "approved"
    assert row["broker_order_id"] is None
    assert "will retry" in row["notes"]


# --- sync -------------------------------------------------------------------------

def approved_alpaca_order(conn, api, config, run="run-1", ticker="AAPL", side="buy",
                          qty=10, run_date="2026-06-10"):
    oid = submit_order(conn, run, ticker, side, qty, run_date, auto_approve=False)
    approve_and_submit(conn, config, oid)
    return conn.execute("SELECT * FROM orders WHERE order_id = ?", (oid,)).fetchone()


def test_sync_records_real_fill_through_shared_accounting(api, conn, alpaca_config):
    order = approved_alpaca_order(conn, api, alpaca_config)
    api.orders[order["broker_order_id"]].update(
        status="filled", filled_avg_price="201.37", filled_at="2026-06-11T13:30:01Z",
    )
    api.positions = [{"symbol": "AAPL", "qty": "10", "avg_entry_price": "201.37"}]
    api.account = {"cash": str(100_000 - 10 * 201.37), "equity": "100000"}

    result = sync_alpaca(conn, alpaca_config, date(2026, 6, 11))
    assert result["filled"] == 1
    assert result["drift"] == []  # ledgers agree

    pos = conn.execute("SELECT * FROM positions WHERE ticker='AAPL'").fetchone()
    assert pos["qty"] == 10
    assert pos["avg_cost"] == pytest.approx(201.37)  # Alpaca's REAL price, no slippage model
    assert get_account(conn)["cash"] == pytest.approx(100_000 - 10 * 201.37)
    assert conn.execute(
        "SELECT status FROM orders WHERE order_id=?", (order["order_id"],)
    ).fetchone()[0] == "filled"
    fill = conn.execute("SELECT * FROM fills").fetchone()
    assert fill["price"] == pytest.approx(201.37)
    assert fill["filled_at"] == "2026-06-11"


def test_sync_cancels_terminal_broker_states(api, conn, alpaca_config):
    order = approved_alpaca_order(conn, api, alpaca_config)
    api.orders[order["broker_order_id"]]["status"] = "canceled"

    result = sync_alpaca(conn, alpaca_config, date(2026, 6, 11))
    assert result["cancelled"] == 1
    row = conn.execute("SELECT * FROM orders WHERE order_id=?", (order["order_id"],)).fetchone()
    assert row["status"] == "cancelled"
    assert "canceled" in row["notes"]


def test_sync_retries_unsubmitted_approved_orders(api, conn, alpaca_config):
    # Approval failed to reach Alpaca earlier; order has no broker_order_id.
    api.fail_submit = True
    order = approved_alpaca_order(conn, api, alpaca_config)
    assert order["broker_order_id"] is None

    api.fail_submit = False
    result = sync_alpaca(conn, alpaca_config, date(2026, 6, 11))
    assert result["resubmitted"] == 1
    row = conn.execute("SELECT * FROM orders WHERE order_id=?", (order["order_id"],)).fetchone()
    assert row["broker_order_id"] is not None


def test_reconciliation_reports_drift(api, conn, alpaca_config):
    # Local thinks we hold MSFT; Alpaca disagrees and cash differs.
    conn.execute(
        "INSERT INTO positions (ticker, qty, avg_cost, opened_at, updated_at) "
        "VALUES ('MSFT', 5, 400.0, '2026-06-01', '2026-06-01')"
    )
    conn.commit()
    api.positions = [{"symbol": "NVDA", "qty": "3", "avg_entry_price": "100"}]
    api.account = {"cash": "95000", "equity": "98000"}

    result = sync_alpaca(conn, alpaca_config, date(2026, 6, 11))
    drift = "\n".join(result["drift"])
    assert "MSFT: local qty 5 vs alpaca 0" in drift
    assert "NVDA: local qty 0 vs alpaca 3" in drift
    assert "cash: local 100,000.00 vs alpaca 95,000.00" in drift


def test_sync_without_keys_degrades_gracefully(conn, alpaca_config, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    result = sync_alpaca(conn, alpaca_config, date(2026, 6, 11))
    assert result["filled"] == 0
    assert "sync skipped" in result["drift"][0]


def test_paper_base_url_is_hardcoded():
    """Live trading must be structurally impossible."""
    assert alpaca_mod.PAPER_BASE_URL == "https://paper-api.alpaca.markets"
    src = Path(alpaca_mod.__file__).read_text(encoding="utf-8")
    assert "api.alpaca.markets" not in src.replace("paper-api.alpaca.markets", "")
