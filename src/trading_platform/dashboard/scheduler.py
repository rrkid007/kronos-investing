"""In-app job scheduler for the dashboard (APScheduler, in-process).

Runs the daily analysis and weekly discovery jobs on a cron derived from
``settings.schedule`` while the dashboard process is alive — so scheduling
works on any OS (the DGX deployment still uses systemd timers; this is for
local/dev use). The UI can toggle it, change the times, and trigger either
job immediately ("Run now").

A job never crashes the server: every invocation is wrapped, and the last
outcome (ok/error + timestamp) is recorded for the status panel. Concurrent
duplicate runs of the same job are skipped.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from trading_platform.core.config import AppConfig

logger = logging.getLogger(__name__)


def _default_daily(config: AppConfig) -> str:
    from trading_platform.pipeline import run_daily

    return run_daily(config)


def _default_discovery(config: AppConfig) -> None:
    # Mirror scripts/discover_stocks.py's run_discovery, importing lazily so
    # the dashboard has no hard dependency on the discovery stack at import.
    from trading_platform.core.db import connect, init_db
    from trading_platform.data.market_data import MarketDataService
    from trading_platform.data.universe import refresh_universe
    from trading_platform.discovery.screener import screen

    conn = connect(config.db_path)
    init_db(conn)
    try:
        refresh_universe(conn, max_age_days=config.settings.discovery.universe_max_age_days)
        screen(conn, config, MarketDataService(conn))
    finally:
        conn.close()


def _parse_hhmm(value: str, default: tuple[int, int]) -> tuple[int, int]:
    try:
        hh, mm = str(value).strip().split(":")
        return int(hh), int(mm)
    except Exception:
        return default


class RunScheduler:
    """Owns the APScheduler instance and the two trading jobs."""

    JOBS = ("daily", "discovery")

    def __init__(
        self,
        get_config: Callable[[], AppConfig],
        daily_fn: Callable[[AppConfig], object] | None = None,
        discovery_fn: Callable[[AppConfig], object] | None = None,
        autostart_scheduler: bool = True,
    ) -> None:
        self._get_config = get_config
        self._daily_fn = daily_fn or _default_daily
        self._discovery_fn = discovery_fn or _default_discovery
        self._autostart = autostart_scheduler
        self._sched = BackgroundScheduler(timezone=self._tz())
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self.last: dict[str, dict] = {j: {} for j in self.JOBS}

    # --- lifecycle ---------------------------------------------------------
    def _tz(self) -> str:
        try:
            return self._get_config().settings.timezone
        except Exception:
            return "UTC"

    def start(self) -> None:
        if self._autostart and not self._sched.running:
            self._sched.start()
        self.reschedule()

    def shutdown(self) -> None:
        if self._sched.running:
            self._sched.shutdown(wait=False)

    def reschedule(self) -> None:
        """Rebuild jobs from the current config. Safe to call after any edit."""
        for jid in self.JOBS:
            job = self._sched.get_job(jid)
            if job:
                job.remove()

        cfg = self._get_config()
        sch = cfg.settings.schedule
        tz = cfg.settings.timezone
        if not getattr(sch, "enabled", False):
            return

        hh, mm = _parse_hhmm(sch.run_after_close_local, (17, 30))
        dow = "mon-fri" if sch.weekdays_only else "*"
        self._sched.add_job(
            self._wrap("daily"), CronTrigger(day_of_week=dow, hour=hh, minute=mm, timezone=tz),
            id="daily", replace_existing=True,
        )
        if getattr(sch, "discovery_enabled", False):
            dhh, dmm = _parse_hhmm(sch.discovery_time, (9, 0))
            self._sched.add_job(
                self._wrap("discovery"),
                CronTrigger(day_of_week=sch.discovery_day or "sat", hour=dhh, minute=dmm,
                            timezone=tz),
                id="discovery", replace_existing=True,
            )

    # --- execution ---------------------------------------------------------
    def _run(self, which: str) -> None:
        with self._lock:
            if which in self._running:
                logger.info("scheduler: %s already running, skipping", which)
                return
            self._running.add(which)
        started = datetime.now()
        try:
            cfg = self._get_config()
            fn = self._daily_fn if which == "daily" else self._discovery_fn
            result = fn(cfg)
            self.last[which] = {"status": "ok", "at": started.isoformat(timespec="seconds"),
                                "detail": str(result) if result is not None else "completed"}
            logger.info("scheduler: %s completed", which)
        except Exception as exc:  # never let a job kill the process
            self.last[which] = {"status": "error", "at": started.isoformat(timespec="seconds"),
                                "detail": str(exc)}
            logger.exception("scheduler: %s failed", which)
        finally:
            with self._lock:
                self._running.discard(which)

    def _wrap(self, which: str) -> Callable[[], None]:
        return lambda: self._run(which)

    def trigger(self, which: str) -> str:
        """Fire a job immediately in a background thread. Returns a status word."""
        if which not in self.JOBS:
            raise ValueError(f"unknown job: {which}")
        with self._lock:
            if which in self._running:
                return "already-running"
        threading.Thread(target=self._run, args=(which,), daemon=True).start()
        return "started"

    # --- introspection -----------------------------------------------------
    def status(self) -> dict:
        cfg = self._get_config()
        sch = cfg.settings.schedule
        jobs = {}
        for jid in self.JOBS:
            job = self._sched.get_job(jid)
            nxt = getattr(job, "next_run_time", None) if job else None
            jobs[jid] = {
                "scheduled": job is not None,
                "next_run": nxt.strftime("%Y-%m-%d %H:%M %Z") if nxt else None,
                "running": jid in self._running,
                "last": self.last.get(jid) or None,
            }
        return {
            "enabled": getattr(sch, "enabled", False),
            "timezone": cfg.settings.timezone,
            "jobs": jobs,
        }
