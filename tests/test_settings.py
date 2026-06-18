"""Editable settings: config writer, schema introspection, scheduler, routes."""

import time

import pytest
from fastapi.testclient import TestClient

from trading_platform.core import config_writer
from trading_platform.core.config import load_config
from trading_platform.dashboard.app import create_app
from trading_platform.dashboard.scheduler import RunScheduler
from trading_platform.dashboard.settings_schema import (
    build_schema,
    coerce,
    coercion_map,
)


@pytest.fixture
def config_dir(tmp_config):
    return tmp_config.root / "config"


# ----- config_writer ------------------------------------------------------

def test_apply_edits_scalar_roundtrip_preserves_comments(tmp_config, config_dir):
    new = config_writer.apply_edits(config_dir, "risk", {"max_position_pct": 12.5})
    assert new.risk.max_position_pct == 12.5
    text = (config_dir / "risk_limits.yaml").read_text()
    assert "12.5" in text
    # a human comment from the original file survives the write
    assert "Deterministic risk limits" in text


def test_apply_edits_nested_path(tmp_config, config_dir):
    new = config_writer.apply_edits(config_dir, "settings", {"llm.model": "qwen2.5:7b"})
    assert new.settings.llm.model == "qwen2.5:7b"
    # sibling key untouched
    assert new.settings.llm.base_url == tmp_config.settings.llm.base_url


def test_apply_edits_map_replacement(tmp_config, config_dir):
    new = config_writer.apply_edits(
        config_dir, "weights",
        {"signal_weights": {"fundamentals": 0.4, "technical": 0.2, "kronos": 0.2,
                            "news": 0.1, "sec_filing": 0.1}},
    )
    assert new.weights.signal_weights["fundamentals"] == 0.4


def test_apply_edits_invalid_rolls_back(tmp_config, config_dir):
    before = (config_dir / "weights.yaml").read_text()
    with pytest.raises(Exception):
        config_writer.apply_edits(
            config_dir, "weights",
            {"signal_weights": {"fundamentals": 0.9, "technical": 0.9}},  # sums to 1.8
        )
    # file restored byte-for-byte, and still loads
    assert (config_dir / "weights.yaml").read_text() == before
    assert load_config(config_dir).weights.signal_weights


def test_add_and_remove_ticker(tmp_config, config_dir):
    new = config_writer.add_ticker(config_dir, "tsla", "Consumer Discretionary")
    assert "TSLA" in new.watchlist.symbols
    with pytest.raises(ValueError):
        config_writer.add_ticker(config_dir, "TSLA", "x")  # duplicate
    new = config_writer.remove_ticker(config_dir, "TSLA")
    assert "TSLA" not in new.watchlist.symbols
    with pytest.raises(ValueError):
        config_writer.remove_ticker(config_dir, "TSLA")  # gone


# ----- schema introspection ----------------------------------------------

def test_build_schema_covers_every_section(tmp_config):
    schema = build_schema(tmp_config)
    keys = {g.file_key for g in schema}
    assert keys == {"settings", "weights", "risk"}
    titles = {s.title for g in schema for s in g.sections}
    # nested sub-models surface as their own sections
    assert "Llm" in titles and "Kronos" in titles and "Macro" in titles
    # all settings sub-models present
    paths = {f.path for g in schema for s in g.sections for f in s.fields}
    assert "llm.model" in paths
    assert "kronos.horizon_days" in paths
    assert "macro.sizing_scalars" in paths
    assert "signal_weights" in paths
    assert "sizing.max_positions" in paths


def test_coerce_types():
    index = coercion_map(load_config("config"))
    f = index["risk"]["max_position_pct"]
    assert coerce(f, "13.5") == 13.5
    f = index["settings"]["macro.sizing_scalars"]
    assert coerce(f, "calm: 1.0\nstress: 0.5") == {"calm": 1.0, "stress": 0.5}
    f = index["risk"]["restricted_assets"]
    assert coerce(f, "GME\nAMC") == ["GME", "AMC"]


# ----- scheduler ----------------------------------------------------------

def test_scheduler_trigger_runs_job(tmp_config):
    calls = []
    sched = RunScheduler(
        get_config=lambda: tmp_config,
        daily_fn=lambda cfg: calls.append("daily") or "ok",
        discovery_fn=lambda cfg: calls.append("discovery"),
        autostart_scheduler=False,
    )
    assert sched.trigger("daily") == "started"
    for _ in range(50):
        if sched.last["daily"]:
            break
        time.sleep(0.02)
    assert "daily" in calls
    assert sched.last["daily"]["status"] == "ok"


def test_scheduler_status_shape(tmp_config):
    sched = RunScheduler(get_config=lambda: tmp_config, autostart_scheduler=False)
    sched.reschedule()
    st = sched.status()
    assert set(st["jobs"]) == {"daily", "discovery"}
    assert st["enabled"] is False  # default config has it off


# ----- dashboard routes ---------------------------------------------------

@pytest.fixture
def client(tmp_config):
    app = create_app(tmp_config, start_scheduler=False)
    return TestClient(app)


def test_settings_page_renders(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert "Risk Limits" in r.text
    assert "Scheduler" in r.text
    assert "Watchlist" in r.text


def test_post_settings_persists(client, config_dir):
    r = client.post("/settings/risk", data={"max_position_pct": "11.0"})
    assert r.status_code == 200
    assert "risk saved" in r.text
    assert load_config(config_dir).risk.max_position_pct == 11.0


def test_post_settings_invalid_flashes_error(client, config_dir):
    before = (config_dir / "weights.yaml").read_text()
    r = client.post("/settings/weights", data={"signal_weights": "fundamentals: 0.9\ntechnical: 0.9"})
    assert r.status_code == 200
    assert "rejected" in r.text
    assert (config_dir / "weights.yaml").read_text() == before


def test_watchlist_add_remove_via_http(client, config_dir):
    r = client.post("/watchlist/add", data={"symbol": "NFLX", "sector": "Communication Services"})
    assert "added NFLX" in r.text
    assert "NFLX" in load_config(config_dir).watchlist.symbols
    r = client.post("/watchlist/remove", data={"symbol": "NFLX"})
    assert "removed NFLX" in r.text
    assert "NFLX" not in load_config(config_dir).watchlist.symbols


def test_run_now_endpoint(tmp_config):
    fired = []
    sched = RunScheduler(
        get_config=lambda: tmp_config,
        daily_fn=lambda cfg: fired.append("d") or "ok",
        discovery_fn=lambda cfg: fired.append("disc"),
        autostart_scheduler=False,
    )
    app = create_app(tmp_config, scheduler=sched, start_scheduler=False)
    client = TestClient(app)
    r = client.post("/run/daily")
    assert r.status_code == 200
    assert "daily: started" in r.text
