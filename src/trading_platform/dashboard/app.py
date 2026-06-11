"""Dashboard — FastAPI + Jinja2 + HTMX, server-rendered, single user.

Read-only against SQLite except the two approval actions (approve/reject),
which delegate to execution.orders. Binds 127.0.0.1 by default; there is no
auth layer, so don't expose it beyond localhost.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from trading_platform.analytics.performance import performance_summary
from trading_platform.core.config import AppConfig, load_config
from trading_platform.core.db import connect, init_db
from trading_platform.dashboard import queries
from trading_platform.execution.approval import approve_and_submit
from trading_platform.execution.orders import reject_order
from trading_platform.pipeline import AGENT_STAGES

TEMPLATES_DIR = Path(__file__).parent / "templates"


def equity_svg(points: list[tuple[str, float]], width: int = 900, height: int = 180) -> str:
    """Server-rendered equity sparkline; no JS chart library."""
    if len(points) < 2:
        return ""
    values = [v for _, v in points]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pad = 8
    step = (width - 2 * pad) / (len(values) - 1)

    def xy(i: int, v: float) -> tuple[float, float]:
        return (pad + i * step,
                pad + (height - 2 * pad) * (1 - (v - lo) / span))

    coords = [xy(i, v) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = (f"{coords[0][0]:.1f},{height - pad} " + line +
            f" {coords[-1][0]:.1f},{height - pad}")
    up = values[-1] >= values[0]
    color = "var(--up)" if up else "var(--down)"
    return f"""<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img"
     aria-label="equity curve {points[0][0]} to {points[-1][0]}">
  <polygon points="{area}" fill="{color}" opacity="0.08"/>
  <polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.5"/>
  <circle cx="{coords[-1][0]:.1f}" cy="{coords[-1][1]:.1f}" r="3" fill="{color}"/>
</svg>"""


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_config()
    app = FastAPI(title="Investment Research Platform", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals["agent_stages"] = AGENT_STAGES

    def db() -> sqlite3.Connection:
        conn = connect(config.db_path)
        init_db(conn)
        return conn

    def approvals_context(conn) -> dict:
        return {"pending": queries.pending_orders(conn),
                "recent": queries.recent_order_activity(conn)}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        conn = db()
        try:
            run = queries.latest_run(conn)
            run_id = run["run_id"] if run else None
            points = queries.equity_points(conn)
            context = {
                "request": request,
                "run": dict(run) if run else None,
                "account": queries.account_overview(conn),
                "equity_svg": equity_svg(points),
                "equity_points": points,
                "performance": performance_summary(conn, config.settings.benchmarks),
                "decisions": queries.decisions_for_run(conn, run_id) if run_id else [],
                "leaderboard": queries.leaderboard(conn, run_id) if run_id else [],
                "health": queries.run_health(conn, run_id) if run_id else None,
                **approvals_context(conn),
            }
            return templates.TemplateResponse(request, "index.html", context)
        finally:
            conn.close()

    @app.get("/partials/approvals", response_class=HTMLResponse)
    def approvals_partial(request: Request):
        conn = db()
        try:
            return templates.TemplateResponse(
                request, "partials/approvals.html", approvals_context(conn)
            )
        finally:
            conn.close()

    def _decide(request: Request, order_id: str, action) -> HTMLResponse:
        conn = db()
        try:
            flash = None
            try:
                flash = action(conn, order_id)  # approval returns a status message
            except ValueError as exc:
                flash = str(exc)
            return templates.TemplateResponse(
                request, "partials/approvals.html",
                {"flash": flash, **approvals_context(conn)},
            )
        finally:
            conn.close()

    @app.post("/orders/{order_id}/approve", response_class=HTMLResponse)
    def approve(request: Request, order_id: str):
        return _decide(request, order_id,
                       lambda conn, oid: approve_and_submit(conn, config, oid))

    @app.post("/orders/{order_id}/reject", response_class=HTMLResponse)
    def reject(request: Request, order_id: str):
        return _decide(request, order_id, reject_order)

    @app.get("/reports", response_class=HTMLResponse)
    def reports(request: Request):
        return templates.TemplateResponse(request, "reports.html", {
            "dates": queries.list_report_dates(config.reports_dir),
        })

    @app.get("/reports/{report_date}", response_class=PlainTextResponse)
    def report_view(report_date: str):
        # stem-only lookup prevents path traversal
        safe = "".join(c for c in report_date if c.isdigit() or c == "-")
        path = config.reports_dir / "daily" / f"{safe}.md"
        if not path.exists():
            return PlainTextResponse("report not found", status_code=404)
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    return app
