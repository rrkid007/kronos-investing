"""Read-model queries for the dashboard. Read-only except nothing —
order approval mutations live in execution.orders, not here."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from trading_platform.pipeline import AGENT_STAGES


def latest_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()


def account_overview(conn: sqlite3.Connection) -> dict:
    account = conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
    snapshot = conn.execute(
        "SELECT * FROM account_snapshots ORDER BY snapshot_date DESC LIMIT 1"
    ).fetchone()
    positions = []
    for p in conn.execute("SELECT * FROM positions ORDER BY ticker").fetchall():
        last = conn.execute(
            "SELECT close FROM price_cache WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (p["ticker"],),
        ).fetchone()
        price = last["close"] if last else p["avg_cost"]
        positions.append({
            "ticker": p["ticker"],
            "qty": p["qty"],
            "avg_cost": p["avg_cost"],
            "last": price,
            "value": p["qty"] * price,
            "unrealized": (price - p["avg_cost"]) * p["qty"],
            "unrealized_pct": (price / p["avg_cost"] - 1) * 100 if p["avg_cost"] else 0,
            "opened_at": p["opened_at"][:10],
        })
    return {
        "cash": account["cash"] if account else None,
        "realized_pnl": account["realized_pnl"] if account else 0.0,
        "snapshot": dict(snapshot) if snapshot else None,
        "positions": positions,
        "positions_value": sum(p["value"] for p in positions),
    }


def equity_points(conn: sqlite3.Connection) -> list[tuple[str, float]]:
    rows = conn.execute(
        "SELECT snapshot_date, equity FROM account_snapshots ORDER BY snapshot_date"
    ).fetchall()
    return [(r["snapshot_date"], r["equity"]) for r in rows]


def pending_orders(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT o.*, r.run_date FROM orders o JOIN runs r ON o.run_id = r.run_id
        WHERE o.status = 'awaiting_approval' ORDER BY r.run_date, o.ticker
        """
    ).fetchall()
    out = []
    for o in rows:
        notes = json.loads(o["notes"] or "{}")
        memo = conn.execute(
            "SELECT recommendation, memo_md FROM research_memos WHERE order_id = ?",
            (o["order_id"],),
        ).fetchone()
        out.append({
            "order_id": o["order_id"],
            "run_date": o["run_date"],
            "ticker": o["ticker"],
            "side": o["side"],
            "qty": o["qty"],
            "est_value": notes.get("est_value"),
            "final_score": notes.get("final_score"),
            "reason": notes.get("reason", ""),
            "expires_at": o["expires_at"],
            "memo_recommendation": memo["recommendation"] if memo else None,
            "memo_md": memo["memo_md"] if memo else None,
        })
    return out


def recent_order_activity(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        """
        SELECT o.ticker, o.side, o.qty, o.status, r.run_date,
               f.price AS fill_price, f.filled_at
        FROM orders o
        JOIN runs r ON o.run_id = r.run_id
        LEFT JOIN fills f ON f.order_id = o.order_id
        WHERE o.status != 'awaiting_approval'
        ORDER BY o.created_at DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def decisions_for_run(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM decisions WHERE run_id = ? ORDER BY final_score DESC",
        (run_id,),
    ).fetchall()
    out = []
    for d in rows:
        payload = json.loads(d["signal_breakdown"] or "{}")
        risk = payload.get("risk")
        out.append({
            "ticker": d["ticker"],
            "action": d["action"],
            "final_score": d["final_score"],
            "sizing_hint": d["sizing_hint"],
            "reason": d["reason"],
            "coverage": payload.get("coverage"),
            "risk_approved": risk.get("approved") if risk else None,
            "risk_errors": [
                c["detail"] for c in (risk or {}).get("checks", []) if not c["passed"]
            ],
        })
    return out


def leaderboard(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """Per ticker: per-agent scores + final score + action, sorted by final."""
    scores: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT ticker, agent, score, confidence FROM agent_scores WHERE run_id = ?",
        (run_id,),
    ).fetchall():
        scores.setdefault(r["ticker"], {})[r["agent"]] = {
            "score": r["score"], "confidence": r["confidence"],
        }
    decisions = {
        d["ticker"]: d for d in conn.execute(
            "SELECT ticker, action, final_score FROM decisions WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    }
    rows = []
    for ticker, agent_scores in scores.items():
        decision = decisions.get(ticker)
        rows.append({
            "ticker": ticker,
            "agents": {a: agent_scores.get(a) for a in AGENT_STAGES},
            "final_score": decision["final_score"] if decision else None,
            "action": decision["action"] if decision else None,
        })
    rows.sort(key=lambda r: r["final_score"] or 0, reverse=True)
    return rows


def run_health(conn: sqlite3.Connection, run_id: str) -> dict:
    failures = conn.execute(
        "SELECT stage, ticker, detail FROM run_stages "
        "WHERE run_id = ? AND status = 'failed' ORDER BY stage, ticker",
        (run_id,),
    ).fetchall()
    n_stages = conn.execute(
        "SELECT COUNT(*) FROM run_stages WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    return {
        "failures": [dict(f) for f in failures],
        "n_stages": n_stages,
    }


def latest_suggestions(conn: sqlite3.Connection) -> list[dict]:
    batch = conn.execute(
        "SELECT batch_id FROM watchlist_suggestions ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if batch is None:
        return []
    rows = conn.execute(
        "SELECT * FROM watchlist_suggestions WHERE batch_id = ? "
        "ORDER BY combined_score DESC",
        (batch["batch_id"],),
    ).fetchall()
    return [dict(r) for r in rows]


def list_report_dates(reports_dir: Path) -> list[str]:
    daily = reports_dir / "daily"
    if not daily.exists():
        return []
    return sorted((p.stem for p in daily.glob("*.md")), reverse=True)
