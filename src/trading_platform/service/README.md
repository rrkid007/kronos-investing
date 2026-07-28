# Kronos Forecast Service (x402)

Pay-per-call HTTP API that sells one thing: a Kronos forecast score over OHLCV
candles **the caller supplies**. Metered with [x402](https://docs.x402.org) —
HTTP 402 + USDC micropayments, designed for machine buyers.

This subpackage is **purely additive**. Nothing in `pipeline`, `dashboard`,
`agents` or `execution` imports it. Delete the directory and the platform is
byte-for-byte the system you had before.

---

## Running it

```bash
# Free mode (default) — no payments, no x402 dependency needed
python scripts/run_service.py

# Metered on Base Sepolia testnet
export X402_ENABLED=1
export X402_PAY_TO=0xYourReceivingAddress
export X402_NETWORK=eip155:84532
export X402_FACILITATOR_URL=https://x402.org/facilitator
python scripts/run_service.py
```

Install the payment SDK only when you actually want to meter. Pass both extras
together — `uv sync` installs exactly the extras named, so `--extra service`
alone would uninstall torch:

```bash
uv sync --extra kronos --extra service
```

| Endpoint | Price | Paths |
|---|---|---|
| `GET /health` | free | — |
| `GET /v1/kronos/schema` | free | — |
| `POST /v1/kronos/score` | $0.01 | 8 |
| `POST /v1/kronos/score/deep` | $0.05 | 32 |

Discovery is deliberately free: a buying agent must be able to learn the price,
the chain and the input contract *before* deciding to pay.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `X402_ENABLED` | `0` | Master switch for metering |
| `X402_PAY_TO` | — | Receiving address (required when enabled) |
| `X402_NETWORK` | `eip155:84532` | CAIP-2 chain id |
| `X402_FACILITATOR_URL` | x402.org | Use the CDP URL on mainnet |
| `KRONOS_SERVICE_HOST` / `_PORT` | `127.0.0.1` / `8402` | Bind address |
| `KRONOS_SERVICE_WARMUP` | `1` | Load the model before serving |
| `KRONOS_SERVICE_MAX_QUEUE` | `8` | Load-shed threshold |
| `KRONOS_SERVICE_TIMEOUT` | `60` | Per-request compute deadline |
| `KRONOS_SETTINGS_FILE` | `<repo>/config/settings.yaml` | Where the `kronos:` block lives |
| `KRONOS_MODEL_ID` and friends | from YAML | Per-field overrides — see `config.py` |

`sample_count` is **not** overridable: it is the priced parameter and belongs to
the tier.

---

## The decisions worth knowing

**It never fetches market data.** Callers POST candles. This keeps the service
stateless and asset-agnostic (equity daily bars, 24/7 crypto bars and intraday
bars all work unchanged — which is what makes it reusable by the
`PLAN-crypto.md` fork). It also sidesteps redistributing derived output from
your data vendor to paying third parties, which is a materially different legal
posture than internal research. Adding a ticker mode later is easy; it is a
business decision, not a technical one.

**Failures must not be 2xx.** `agents/kronos.py` degrades any failure into a
neutral score-50 / confidence-0 `AgentResult`. That is right inside the pipeline
— a dead signal should be ignorable — and a billing bug in a shop, because it
means charging for "I don't know." x402 settles after a 2xx, so `runner.py`
raises typed errors that map to 502/503/504 and the caller is never charged.
`tests/test_service.py` pins this.

**Callers cannot set `sample_count`.** It is the dominant GPU cost driver, so
it is fixed per price tier. x402's `exact` scheme prices a route, so tiers are
routes.

**Only seeded requests are cached.** Sampling is stochastic (`T=1.0`,
`top_p=0.9`). Without a seed the caller is buying a fresh draw and serving a
stored one would be selling a different product than advertised. Pass a seed to
get reproducibility and a cache hit. Seeding is best-effort — it seeds torch's
RNG, which is not bit-exact across devices or driver versions.

**One forecast at a time.** `KronosPredictor.predict` is GPU-bound and not
thread-safe, and FastAPI runs sync handlers in a threadpool. `ForecastRunner`
serialises with a semaphore, sheds load at `max_queue` (503 + `Retry-After`)
and hard-stops at `timeout_seconds` (504). **Keep `timeout_seconds` below the
payment authorisation's `maxTimeoutSeconds`**, or you will accept
authorisations that expire before you can settle them.

**The service holds no private key.** x402 settlement pays `X402_PAY_TO`
directly. The box holds a receiving address, not a wallet — which matters on a
host that also runs your trading pipeline. `ServiceConfig` reads payment
settings from the environment only, never the platform YAML, so the dashboard's
Settings page can never redirect where money lands.

**It warms up before serving.** `KronosForecaster` loads lazily, so without a
warmup the first buyer after every restart pays the model-load cost inside
their own timeout and payment window — and `Restart=on-failure` makes that
recur after every crash. The lifespan hook loads the model and runs one
throwaway forecast on synthetic candles (which must pass the same input gate
real callers face) before the port serves traffic.

A failed warmup does **not** crash the process: `/health` returns 503 with
`status: degraded` and the error, so the box stays reachable for diagnosis
instead of crash-looping under systemd. Paid endpoints keep failing closed
through the normal 502/503 paths, so nobody is charged either way. Warmup is
deliberately untimed — the first run on a fresh box may include a HuggingFace
weight download, and killing that would be worse than waiting.

`/health` distinguishes three states: warmup succeeded (`ok`, `model_ready`),
warmup ran and failed (`degraded`, 503), and warmup never attempted
(`ok`, `model_ready: false` — lazy loading via `--no-warmup` is a valid mode,
not a fault).

**It does not read the platform config.** `load_kronos_settings` parses only
the `kronos:` block of `settings.yaml`, with per-field environment overrides.
It deliberately avoids `core.config.load_config`, which also parses and
validates `watchlist.yaml`, `weights.yaml` and `risk_limits.yaml` — none of
which this service reads. A malformed file it never uses must not be able to
stop a paid API from starting. A missing settings file is fine too: the service
runs from environment variables alone.

**Its own input gate.** `data/quality.py::validate_ohlcv` is not reused: it
requires `adj_close` (crypto/FX have none), fails bars older than a few days
against *today* (breaks backtesting customers deliberately scoring 2021), counts
business-day gaps (breaks 24/7 bars), and sets `min_rows=250` for the technical
agent's 200-day MA. `validation.py` keeps only the asset-agnostic coherence
checks.

---

## Receipts

Every 200 carries a receipt:

```json
"receipt": {
  "input_hash": "9f2c…",   // sha256 over candles + tier + model_id + horizon
  "tier": "standard",
  "seed": 42,
  "cached": false,
  "served_at": "2026-07-27T14:02:11Z",
  "compute_seconds": 3.412
}
```

Join `input_hash` against the settlement transaction hash in the
`X-PAYMENT-RESPONSE` header to prove what was paid for. The seed is recorded
separately from the hash, so a buyer can see that two differing answers came
from identical input under different sampling.

---

## Status

`tests/test_service.py` covers the service in 48 tests — no GPU required. Two
are marked x402-only and skip unless the `service` extra is installed.

The real model path is verified too: `NeoQuasar/Kronos-small` loads and produces
a forecast through `ForecastRunner.warmup()` (CPU, x86: 4.0s load, 5.6s total).
Re-measure on the Spark with `Kronos-base` on CUDA — those numbers set
`KRONOS_SERVICE_TIMEOUT`.

`payments.py` is written and verified against **x402 2.17.0**. The published
docs were wrong on three points, so if you upgrade the SDK, re-check these:

- the EVM mechanism lives behind the `evm` marker — `x402[fastapi]` alone
  raises on `import x402.mechanisms.evm.exact`
- `x402ResourceServer` takes `facilitator_clients=`, not `facilitator=`, and
  has no `schemes=` argument; schemes are attached with
  `server.register(network, scheme)` followed by `server.initialize()`
- `RouteConfig` does not take price/network/pay_to directly — those live on a
  nested `PaymentOption(scheme, pay_to, price, network, max_timeout_seconds)`

The generated 402 challenge is asserted in tests: correct USDC base-unit
amounts (10000 / 50000 at 6 decimals), network, payee, and a
`maxTimeoutSeconds` that outlasts the compute deadline.

Still unverified: the **settle** half. A 402 challenge is issued correctly, but
no payment has been signed, verified or settled against a live facilitator.

## Before mainnet

1. End-to-end on Base Sepolia with a funded test wallet — confirm the retry
   with `X-PAYMENT`, verification, and the settlement hash in
   `X-PAYMENT-RESPONSE`. This is the untested half.
2. Confirm no settlement occurs on 502/503/504 (force a CUDA OOM and watch the
   facilitator).
3. Load-test the queue: verify p99 latency stays under `maxTimeoutSeconds`.
4. Put TLS and a rate limiter in front — this app has no auth by design, since
   payment *is* the authorisation.
5. Switch `X402_NETWORK=eip155:8453` and
   `X402_FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402`.
   `ServiceConfig.problems()` refuses mainnet-with-testnet-facilitator, but
   check the receiving address by hand anyway.
