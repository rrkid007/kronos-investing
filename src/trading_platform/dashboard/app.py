"""Dashboard — FastAPI + Jinja2 + HTMX, server-rendered, single user.

Read-only against SQLite except for: the two approval actions
(approve/reject), the Settings page (which writes validated edits back to the
config YAML via core.config_writer), watchlist edits, API-key/credentials
storage (core.secrets, written to a gitignored .env — never the YAML), and the
in-app scheduler controls. Binds 127.0.0.1 by default; there is no auth layer,
so don't expose it beyond localhost.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from trading_platform.analytics.performance import performance_summary
from trading_platform.core.config import AppConfig, load_config
from trading_platform.core import config_writer, secrets
from trading_platform.core.db import connect, init_db
from trading_platform.dashboard import queries
from trading_platform.dashboard.scheduler import RunScheduler
from trading_platform.dashboard.settings_schema import build_schema, coerce, coercion_map
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


def secrets_descriptor(config: AppConfig) -> list[dict]:
    """The API keys the platform reads, derived from the configured env-var
    names. Values are never included — only whether each is currently set."""
    s = config.settings
    items = [
        (s.news.finnhub_api_key_env, "Finnhub API key",
         "News agent — Finnhub free tier"),
        (s.memo.api_key_env, "External LLM API key",
         "Research memos — OpenAI-compatible endpoint (optional)"),
        (s.execution.alpaca_key_env, "Alpaca API key ID",
         "Paper trading — only used when broker = alpaca_paper"),
        (s.execution.alpaca_secret_env, "Alpaca API secret",
         "Paper trading — only used when broker = alpaca_paper"),
    ]
    seen: set[str] = set()
    out: list[dict] = []
    for env, label, help_text in items:
        if not env or env in seen:
            continue
        seen.add(env)
        out.append({"env": env, "label": label, "help": help_text,
                    "set": secrets.is_set(env)})
    return out


def create_app(
    config: AppConfig | None = None,
    *,
    scheduler: RunScheduler | None = None,
    start_scheduler: bool = True,
) -> FastAPI:
    config = config or load_config()
    # Mutable holder so Settings edits can hot-swap the live config without a
    # restart. Every route reads cfg() rather than closing over `config`.
    config_dir = config.root / "config"
    secrets_path = config.root / ".env"
    secrets.load_env_file(secrets_path)  # make stored keys live for this process
    state = {"config": config, "config_dir": config_dir, "secrets_path": secrets_path}

    def cfg() -> AppConfig:
        return state["config"]

    app = FastAPI(title="Investment Research Platform", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals["agent_stages"] = AGENT_STAGES

    scheduler = scheduler or RunScheduler(get_config=cfg, autostart_scheduler=start_scheduler)
    app.state.scheduler = scheduler
    app.state.get_config = cfg
    if start_scheduler:
        scheduler.start()

    def db() -> sqlite3.Connection:
        conn = connect(cfg().db_path)
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
                "performance": performance_summary(conn, cfg().settings.benchmarks),
                "decisions": queries.decisions_for_run(conn, run_id) if run_id else [],
                "leaderboard": queries.leaderboard(conn, run_id) if run_id else [],
                "health": queries.run_health(conn, run_id) if run_id else None,
                "suggestions": queries.latest_suggestions(conn),
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
                       lambda conn, oid: approve_and_submit(conn, cfg(), oid))

    @app.post("/orders/{order_id}/reject", response_class=HTMLResponse)
    def reject(request: Request, order_id: str):
        return _decide(request, order_id, reject_order)

    # ----- Settings (writable) --------------------------------------------
    def settings_context(flash: str | None = None, flash_ok: bool = True) -> dict:
        return {
            "schema": build_schema(cfg()),
            "watchlist": cfg().watchlist.tickers,
            "secrets": secrets_descriptor(cfg()),
            "schedule_status": scheduler.status(),
            "flash": flash,
            "flash_ok": flash_ok,
        }

    def _settings_partial(request: Request, flash: str | None, ok: bool) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "partials/settings_body.html",
            {"request": request, **settings_context(flash, ok)},
        )

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        return templates.TemplateResponse(
            request, "settings.html", {"request": request, **settings_context()}
        )

    @app.post("/settings/{file_key}", response_class=HTMLResponse)
    async def save_settings(request: Request, file_key: str):
        if file_key not in ("settings", "weights", "risk"):
            return _settings_partial(request, f"unknown section: {file_key}", False)
        form = await request.form()
        index = coercion_map(cfg()).get(file_key, {})
        updates: dict = {}
        errors: list[str] = []
        for path, field in index.items():
            if path in form:
                try:
                    updates[path] = coerce(field, form[path])
                except Exception as exc:  # bad number, etc.
                    errors.append(f"{field.label}: {exc}")
        if errors:
            return _settings_partial(request, "  ·  ".join(errors), False)
        if not updates:
            return _settings_partial(request, "no changes submitted", False)
        try:
            state["config"] = config_writer.apply_edits(state["config_dir"], file_key, updates)
            if file_key == "settings":
                scheduler.reschedule()
            return _settings_partial(request, f"{file_key} saved", True)
        except Exception as exc:  # pydantic validation, etc. — file already rolled back
            return _settings_partial(request, f"rejected: {exc}", False)

    @app.post("/secrets", response_class=HTMLResponse)
    async def save_secret(request: Request):
        form = await request.form()
        env_name = str(form.get("env_name", "")).strip()
        value = str(form.get("value", ""))
        allowed = {d["env"] for d in secrets_descriptor(cfg())}
        if env_name not in allowed:
            return _settings_partial(request, f"unknown credential: {env_name}", False)
        try:
            secrets.write_secret(state["secrets_path"], env_name, value)
            action = "cleared" if value.strip() == "" else "saved"
            return _settings_partial(request, f"{env_name} {action}", True)
        except Exception as exc:
            return _settings_partial(request, str(exc), False)

    @app.post("/watchlist/add", response_class=HTMLResponse)
    async def watchlist_add(request: Request):
        form = await request.form()
        symbol = str(form.get("symbol", "")).strip()
        sector = str(form.get("sector", "")).strip()
        try:
            state["config"] = config_writer.add_ticker(state["config_dir"], symbol, sector)
            return _settings_partial(request, f"added {symbol.upper()}", True)
        except Exception as exc:
            return _settings_partial(request, str(exc), False)

    @app.post("/watchlist/remove", response_class=HTMLResponse)
    async def watchlist_remove(request: Request):
        form = await request.form()
        symbol = str(form.get("symbol", "")).strip()
        try:
            state["config"] = config_writer.remove_ticker(state["config_dir"], symbol)
            return _settings_partial(request, f"removed {symbol.upper()}", True)
        except Exception as exc:
            return _settings_partial(request, str(exc), False)

    @app.post("/run/{which}", response_class=HTMLResponse)
    def run_now(request: Request, which: str):
        try:
            status = scheduler.trigger(which)
            return _settings_partial(request, f"{which}: {status}", True)
        except Exception as exc:
            return _settings_partial(request, str(exc), False)

    @app.get("/reports", response_class=HTMLResponse)
    def reports(request: Request):
        return templates.TemplateResponse(request, "reports.html", {
            "dates": queries.list_report_dates(cfg().reports_dir),
        })

    @app.get("/reports/{report_date}", response_class=PlainTextResponse)
    def report_view(report_date: str):
        # stem-only lookup prevents path traversal
        safe = "".join(c for c in report_date if c.isdigit() or c == "-")
        path = cfg().reports_dir / "daily" / f"{safe}.md"
        if not path.exists():
            return PlainTextResponse("report not found", status_code=404)
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    return app
