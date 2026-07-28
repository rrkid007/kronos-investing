"""Paid Kronos service — routing, billing-safety, admission control, receipts.

No GPU and no x402 SDK needed: the forecaster is injected (FakeForecaster) and
payments default to off. The rule these tests exist to protect is that a caller
is charged only when a forecast was actually produced — which in x402 terms
means every failure path must be non-2xx, because settlement follows a 2xx.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from tests.fixtures import FakeForecaster, make_ohlcv
from trading_platform.agents.kronos import KronosAgent
from trading_platform.core.config import KronosSettings
from trading_platform.core.models import Direction
from trading_platform.service import validation
from trading_platform.service.app import create_service_app
from trading_platform.service.config import (
    BASE_MAINNET,
    BASE_SEPOLIA,
    TESTNET_FACILITATOR,
    ServiceConfig,
    load_kronos_settings,
)
from trading_platform.service.payments import PaymentsUnavailable, attach_payments
from trading_platform.service.runner import ForecastRunner, synthetic_frame
from trading_platform.service.tiers import DEEP, STANDARD

SETTINGS = KronosSettings()


def payload(df=None, **extra) -> dict:
    df = make_ohlcv() if df is None else df
    candles = [
        {
            "timestamp": ts.isoformat(),
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "volume": float(r.volume),
        }
        for ts, r in zip(df.index, df.itertuples(index=False))
    ]
    return {"candles": candles, **extra}


def make_client(forecaster=None, config=None, runner=None) -> TestClient:
    app = create_service_app(
        kronos_settings=SETTINGS,
        config=config or ServiceConfig(payments_enabled=False),
        forecaster=forecaster if runner is None else None,
        runner=runner,
    )
    return TestClient(app)


# --- free discovery endpoints ----------------------------------------------

def test_health_is_free_and_reports_payment_state():
    body = make_client(FakeForecaster()).get("/health").json()
    assert body["status"] == "ok"
    assert body["payments_active"] is False
    assert body["queue_limit"] == 8


# --- startup warmup ---------------------------------------------------------

def test_warmup_loads_the_model_and_marks_ready():
    runner = ForecastRunner(SETTINGS, forecaster=FakeForecaster())
    assert asyncio.run(runner.warmup()) is True
    assert runner.model_ready
    assert runner.warmup_error is None
    assert runner.warmup_seconds is not None


def test_warmup_uses_a_coherent_synthetic_frame():
    """The warmup frame must pass the same gate real callers face, or warmup
    would 'succeed' on input we'd reject from a customer."""
    df = synthetic_frame(SETTINGS.context_candles)
    assert len(df) == SETTINGS.context_candles
    assert validation.validate_candles(df) == []


def test_failed_warmup_is_recorded_not_raised():
    """A crash-looping service is harder to diagnose than a degraded one."""
    runner = ForecastRunner(SETTINGS, forecaster=FakeForecaster(fail=RuntimeError("CUDA OOM")))
    assert asyncio.run(runner.warmup()) is False
    assert runner.model_ready is False
    assert "CUDA OOM" in runner.warmup_error


def test_health_reports_503_after_a_failed_warmup():
    runner = ForecastRunner(SETTINGS, forecaster=FakeForecaster(fail=RuntimeError("CUDA OOM")))
    asyncio.run(runner.warmup())
    resp = make_client(runner=runner).get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
    assert resp.json()["model_ready"] is False


def test_health_is_ok_after_a_successful_warmup():
    runner = ForecastRunner(SETTINGS, forecaster=FakeForecaster())
    asyncio.run(runner.warmup())
    resp = make_client(runner=runner).get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_ready"] is True


def test_health_is_ok_when_warmup_was_never_attempted():
    """Lazy loading is a valid mode (--no-warmup), not a fault."""
    resp = make_client(FakeForecaster()).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_ready"] is False


def test_lifespan_runs_warmup_when_enabled():
    cfg = ServiceConfig(payments_enabled=False, warmup_on_startup=True)
    app = create_service_app(
        kronos_settings=SETTINGS, config=cfg, forecaster=FakeForecaster()
    )
    with TestClient(app) as client:  # context manager triggers lifespan
        body = client.get("/health").json()
    assert body["model_ready"] is True
    assert body["warmup_seconds"] is not None


def test_lifespan_skips_warmup_when_disabled():
    cfg = ServiceConfig(payments_enabled=False, warmup_on_startup=False)
    app = create_service_app(
        kronos_settings=SETTINGS, config=cfg, forecaster=FakeForecaster()
    )
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["model_ready"] is False
    assert body["status"] == "ok"


# --- kronos settings loading (decoupled from load_config) --------------------

def test_kronos_settings_read_from_yaml_block(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "kronos:\n  model_id: NeoQuasar/Kronos-base\n  horizon_days: 5\n"
        "news:\n  lookback_days: 7\n",
        encoding="utf-8",
    )
    s = load_kronos_settings(path, env={})
    assert s.model_id == "NeoQuasar/Kronos-base"
    assert s.horizon_days == 5
    assert s.top_p == 0.9  # untouched default


def test_kronos_settings_ignore_the_other_config_files(tmp_path):
    """The whole point: a broken watchlist/weights/risk file must not stop the
    service, because it never reads them."""
    path = tmp_path / "settings.yaml"
    path.write_text("kronos:\n  model_id: X/Y\n", encoding="utf-8")
    (tmp_path / "watchlist.yaml").write_text("this: [is, not, valid: {", encoding="utf-8")
    (tmp_path / "weights.yaml").write_text("garbage: !!!", encoding="utf-8")
    assert load_kronos_settings(path, env={}).model_id == "X/Y"


def test_env_overrides_the_yaml(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("kronos:\n  model_id: from-yaml\n", encoding="utf-8")
    s = load_kronos_settings(path, env={"KRONOS_MODEL_ID": "from-env", "KRONOS_TOP_P": "0.5"})
    assert s.model_id == "from-env"
    assert s.top_p == 0.5


def test_missing_settings_file_falls_back_to_defaults(tmp_path):
    s = load_kronos_settings(tmp_path / "nope.yaml", env={})
    assert s.model_id == KronosSettings().model_id


def test_unreadable_settings_file_falls_back_rather_than_crashing(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("kronos: [unclosed\n", encoding="utf-8")
    assert load_kronos_settings(path, env={}).model_id == KronosSettings().model_id


def test_unknown_yaml_keys_are_ignored(tmp_path):
    """A future addition to the kronos block must not break an older service."""
    path = tmp_path / "settings.yaml"
    path.write_text("kronos:\n  model_id: X/Y\n  some_future_key: 12\n", encoding="utf-8")
    assert load_kronos_settings(path, env={}).model_id == "X/Y"


def test_sample_count_is_not_environment_overridable():
    """It is the priced parameter — it belongs to the tier, not to config."""
    from trading_platform.service.config import KRONOS_ENV_OVERRIDES

    assert "sample_count" not in KRONOS_ENV_OVERRIDES


def test_schema_lists_both_tiers_with_prices():
    body = make_client(FakeForecaster()).get("/v1/kronos/schema").json()
    tiers = {t["name"]: t for t in body["tiers"]}
    assert tiers["standard"]["price"] == "$0.01"
    assert tiers["standard"]["sample_paths"] == 8
    assert tiers["deep"]["price"] == "$0.05"
    assert tiers["deep"]["sample_paths"] == 32
    assert body["payment"]["protocol"] == "x402"


def test_schema_discloses_stochastic_sampling():
    """A buyer must learn before paying that repeat calls differ."""
    body = make_client(FakeForecaster()).get("/v1/kronos/schema").json()
    assert "seed" in body["model"]["stochastic"]


# --- happy path -------------------------------------------------------------

def test_bullish_forecast_scores_and_prices_the_standard_tier():
    client = make_client(FakeForecaster(final_return=0.05, spread=0.01))
    resp = client.post(STANDARD.path, json=payload(symbol="AAPL"))
    assert resp.status_code == 200

    body = resp.json()
    assert body["score"] > 60
    assert body["direction"] == "bullish"
    assert body["confidence"] == 0.9
    assert body["n_samples"] == 8
    assert body["model_info"]["sample_count"] == 8
    assert body["receipt"]["tier"] == "standard"
    assert body["receipt"]["symbol"] == "AAPL"
    assert body["receipt"]["cached"] is False


def test_deep_tier_buys_more_sample_paths():
    client = make_client(FakeForecaster(final_return=0.05, spread=0.01))
    body = client.post(DEEP.path, json=payload()).json()
    assert body["n_samples"] == DEEP.sample_count == 32
    assert body["receipt"]["tier"] == "deep"


def test_direction_matches_the_internal_agent():
    """Buyers and the pipeline must read the same score identically."""
    df = make_ohlcv()
    for final_return in (0.05, -0.05, 0.001):
        agent_result = KronosAgent(
            SETTINGS, forecaster=FakeForecaster(final_return=final_return)
        ).analyze("AAPL", "r1", df)
        body = make_client(
            FakeForecaster(final_return=final_return)
        ).post(STANDARD.path, json=payload(df)).json()

        assert body["score"] == agent_result.score
        assert body["direction"] == agent_result.direction.value
        assert Direction(body["direction"]) == agent_result.direction


def test_percentile_bands_are_ordered():
    body = make_client(FakeForecaster(final_return=0.02, spread=0.03)).post(
        STANDARD.path, json=payload()
    ).json()
    assert body["forecast_p10_pct"] <= body["expected_return_pct"] <= body["forecast_p90_pct"]


# --- billing safety: every failure must be non-2xx ---------------------------

def test_forecast_failure_is_502_not_a_neutral_200():
    """The in-pipeline agent degrades a CUDA OOM to score 50 / confidence 0.
    Selling that would be charging for 'I don't know'."""
    resp = make_client(FakeForecaster(fail=RuntimeError("CUDA OOM"))).post(
        STANDARD.path, json=payload()
    )
    assert resp.status_code == 502
    assert resp.json()["error"] == "forecast_failed"
    assert "CUDA OOM" in resp.json()["detail"]


def test_missing_model_is_503_with_retry_after():
    resp = make_client(FakeForecaster(fail=ImportError("No module named 'torch'"))).post(
        STANDARD.path, json=payload()
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"
    assert resp.headers["Retry-After"] == "60"


def test_capacity_exceeded_sheds_load_before_computing():
    runner = ForecastRunner(SETTINGS, forecaster=FakeForecaster(), max_queue=0)
    resp = make_client(runner=runner).post(STANDARD.path, json=payload())
    assert resp.status_code == 503
    assert resp.json()["error"] == "capacity_exceeded"


def test_timeout_is_504():
    class SlowForecaster:
        def predict_paths(self, df, horizon, sample_count):
            import time
            time.sleep(0.5)
            raise AssertionError("should have timed out")

    runner = ForecastRunner(SETTINGS, forecaster=SlowForecaster(), timeout_seconds=0.05)
    resp = make_client(runner=runner).post(STANDARD.path, json=payload())
    assert resp.status_code == 504
    assert resp.json()["error"] == "forecast_timeout"


# --- input gate (rejected before any GPU work) -------------------------------

def test_too_few_candles_rejected():
    df = make_ohlcv(n_rows=50)
    resp = make_client(FakeForecaster()).post(STANDARD.path, json=payload(df))
    assert resp.status_code == 422


def test_out_of_order_candles_rejected():
    body = payload()
    body["candles"][10], body["candles"][11] = body["candles"][11], body["candles"][10]
    resp = make_client(FakeForecaster()).post(STANDARD.path, json=body)
    assert resp.status_code == 422


def test_incoherent_ohlc_rejected():
    body = payload()
    body["candles"][5]["low"] = body["candles"][5]["high"] * 2  # low above high
    resp = make_client(FakeForecaster()).post(STANDARD.path, json=body)
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_candles"


def test_flat_series_rejected():
    body = payload()
    for c in body["candles"]:
        c["open"] = c["high"] = c["low"] = c["close"] = 100.0
    resp = make_client(FakeForecaster()).post(STANDARD.path, json=body)
    assert resp.status_code == 422
    assert "constant" in resp.json()["detail"]


def test_negative_price_rejected_by_schema():
    body = payload()
    body["candles"][3]["close"] = -1.0
    assert make_client(FakeForecaster()).post(STANDARD.path, json=body).status_code == 422


# --- receipts and caching ----------------------------------------------------

def test_receipt_hash_is_stable_for_identical_input():
    client = make_client(FakeForecaster())
    body = payload()
    a = client.post(STANDARD.path, json=body).json()["receipt"]["input_hash"]
    b = client.post(STANDARD.path, json=body).json()["receipt"]["input_hash"]
    assert a == b


def test_receipt_hash_changes_with_the_candles():
    client = make_client(FakeForecaster())
    body = payload()
    a = client.post(STANDARD.path, json=body).json()["receipt"]["input_hash"]
    # Scale the whole last bar so it stays internally coherent — otherwise the
    # input gate rejects it and we'd be asserting on an error body.
    for field in ("open", "high", "low", "close"):
        body["candles"][-1][field] *= 1.01
    b = client.post(STANDARD.path, json=body).json()["receipt"]["input_hash"]
    assert a != b


def test_receipt_hash_differs_between_tiers():
    """Tier is a priced parameter, so it must be inside the hash."""
    client = make_client(FakeForecaster())
    body = payload()
    std = client.post(STANDARD.path, json=body).json()["receipt"]["input_hash"]
    deep = client.post(DEEP.path, json=body).json()["receipt"]["input_hash"]
    assert std != deep


def test_seeded_requests_are_cached():
    client = make_client(FakeForecaster())
    body = payload(seed=42)
    first = client.post(STANDARD.path, json=body).json()
    second = client.post(STANDARD.path, json=body).json()
    assert first["receipt"]["cached"] is False
    assert second["receipt"]["cached"] is True
    assert first["score"] == second["score"]


def test_unseeded_requests_are_never_cached():
    """Without a seed the caller is buying a fresh draw, not a stored one."""
    client = make_client(FakeForecaster())
    body = payload()
    assert client.post(STANDARD.path, json=body).json()["receipt"]["cached"] is False
    assert client.post(STANDARD.path, json=body).json()["receipt"]["cached"] is False


def test_seed_is_recorded_on_the_receipt():
    body = make_client(FakeForecaster()).post(STANDARD.path, json=payload(seed=7)).json()
    assert body["receipt"]["seed"] == 7


# --- payment configuration ---------------------------------------------------

def test_payments_off_by_default_and_endpoints_are_free():
    client = make_client(FakeForecaster())
    assert client.get("/health").json()["payments_active"] is False
    assert client.post(STANDARD.path, json=payload()).status_code == 200


def test_enabling_payments_without_an_address_fails_fast():
    cfg = ServiceConfig(payments_enabled=True, pay_to=None)
    assert cfg.problems()
    with pytest.raises(PaymentsUnavailable, match="X402_PAY_TO"):
        attach_payments(object(), cfg, [STANDARD])


def test_malformed_evm_address_rejected():
    cfg = ServiceConfig(payments_enabled=True, pay_to="not-an-address")
    assert any("valid EVM address" in p for p in cfg.problems())


def test_mainnet_with_testnet_facilitator_is_rejected():
    """Silently settling mainnet traffic through the testnet facilitator would
    mean serving real forecasts for fake money."""
    cfg = ServiceConfig(
        payments_enabled=True,
        pay_to="0x" + "a" * 40,
        network=BASE_MAINNET,
        facilitator_url=TESTNET_FACILITATOR,
    )
    assert any("mainnet" in p for p in cfg.problems())


def test_valid_testnet_config_has_no_problems():
    cfg = ServiceConfig(
        payments_enabled=True, pay_to="0x" + "b" * 40, network=BASE_SEPOLIA
    )
    assert cfg.problems() == []
    assert cfg.is_testnet


def test_missing_x402_sdk_fails_loudly_rather_than_serving_free():
    """A correctly-configured paywall that cannot load must kill startup — the
    alternative is silently giving away GPU time."""
    try:
        import x402  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("x402 installed; the missing-SDK path cannot be exercised")

    cfg = ServiceConfig(payments_enabled=True, pay_to="0x" + "d" * 40)
    with pytest.raises(PaymentsUnavailable, match="x402 not installed"):
        attach_payments(object(), cfg, [STANDARD])


def _x402_installed() -> bool:
    try:
        import x402  # noqa: F401
    except ImportError:
        return False
    return True


x402_only = pytest.mark.skipif(
    not _x402_installed(), reason="requires uv sync --extra service"
)


@x402_only
def test_paywall_challenges_unpaid_requests():
    """Verified against x402 2.17.0: free routes stay open, paid routes 402."""
    cfg = ServiceConfig(
        payments_enabled=True, pay_to="0x" + "b" * 40, network=BASE_SEPOLIA
    )
    client = make_client(FakeForecaster(), config=cfg)
    assert client.get("/health").json()["payments_active"] is True
    assert client.get("/v1/kronos/schema").status_code == 200
    assert client.post(STANDARD.path, json=payload()).status_code == 402
    assert client.post(DEEP.path, json=payload()).status_code == 402


@x402_only
def test_challenge_advertises_the_right_price_and_deadline():
    """The amounts are in USDC base units (6 dp), and the authorisation window
    must outlast our own compute deadline or settlement races the forecast."""
    import base64
    import json

    cfg = ServiceConfig(
        payments_enabled=True,
        pay_to="0x" + "b" * 40,
        network=BASE_SEPOLIA,
        timeout_seconds=60.0,
    )
    client = make_client(FakeForecaster(), config=cfg)

    for path, expected_units in ((STANDARD.path, "10000"), (DEEP.path, "50000")):
        resp = client.post(path, json=payload())
        challenge = json.loads(base64.b64decode(resp.headers["payment-required"]))
        accepts = challenge["accepts"][0]

        assert accepts["amount"] == expected_units  # $0.01 / $0.05 at 6 decimals
        assert accepts["network"] == BASE_SEPOLIA
        assert accepts["payTo"] == cfg.pay_to
        assert accepts["scheme"] == "exact"
        assert accepts["extra"]["name"] == "USDC"
        assert accepts["maxTimeoutSeconds"] == cfg.payment_timeout_seconds > 60


def test_payment_window_outlasts_the_compute_deadline():
    cfg = ServiceConfig(timeout_seconds=45.0)
    assert cfg.payment_timeout_seconds > cfg.timeout_seconds


def test_config_from_env():
    cfg = ServiceConfig.from_env({
        "X402_ENABLED": "true",
        "X402_PAY_TO": "0x" + "c" * 40,
        "X402_NETWORK": BASE_MAINNET,
        "KRONOS_SERVICE_MAX_QUEUE": "3",
    })
    assert cfg.payments_enabled
    assert cfg.network == BASE_MAINNET
    assert cfg.max_queue == 3


# --- isolation ---------------------------------------------------------------

def test_service_never_imports_execution_or_broker_code():
    """Structural guard, matching the platform's no-real-money pillar: the
    money-receiving surface must not be able to reach the order path."""
    import pkgutil

    import trading_platform.service as pkg

    forbidden = ("execution", "dashboard", "core.db", "core.secrets")
    for mod in pkgutil.iter_modules(pkg.__path__):
        source = (pkg.__path__[0] + "/" + mod.name + ".py")
        with open(source, encoding="utf-8") as fh:
            text = fh.read()
        for name in forbidden:
            assert f"trading_platform.{name}" not in text, f"{mod.name} imports {name}"
