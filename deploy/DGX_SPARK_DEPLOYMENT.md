# DGX Spark (GB10) Deployment Guide

Production deployment guide for the **Local AI Investment Research Platform** on an
**NVIDIA DGX Spark** powered by the **GB10 Grace Blackwell Superchip**.

This guide supersedes the short `deploy/SETUP.md` and adds the hardware-specific
detail that matters on GB10: the Blackwell GPU is compute capability **sm_121**
(CUDA 13), the CPU is **aarch64 (ARM64)**, and CPU+GPU share **128 GB of unified
LPDDR5X memory**. Those three facts drive every non-obvious step below — especially
getting CUDA-enabled PyTorch and Ollama to actually use the GPU rather than silently
falling back to CPU.

Everything assumes the repo lives at `/opt/trading` and runs as a dedicated
`trading` system user.

---

## 0. What you are deploying

A fully local, AI-powered stock research and **paper-trading** system. No real money
ever moves — the Alpaca integration is hard-wired to the paper endpoint and live
trading is not implemented anywhere in the codebase.

The platform runs as two scheduled jobs:

- **trading-daily** — weekdays 17:30 ET. Refreshes market data through a quality
  gate, runs the multi-agent research pipeline (technical, Kronos forecast,
  fundamentals, news, SEC filings, macro regime), aggregates to a decision, sizes
  positions, passes them through a deterministic risk engine, and writes proposed
  orders to a human approval queue. Approved orders fill at the next day's open.
- **trading-discovery** — Saturdays 09:00 ET. Screens ~60 S&P 500 candidates for
  watchlist suggestions.

A read-only **FastAPI dashboard** (localhost:8420) shows the account, approval queue,
decisions, performance, and run health.

### GPU/LLM workload, and why GB10 fits

| Component | What runs on the GPU | Memory footprint |
|-----------|---------------------|------------------|
| **Ollama LLM** (`qwen2.5:32b-instruct`) | News + SEC + (optional local) memo agents | ~20 GB weights, loaded into unified memory |
| **Kronos forecaster** (vendored PyTorch model) | 10-day price-path forecast per ticker | small (25M–102M params) |

GB10's 128 GB of unified memory comfortably holds the 32B LLM and the Kronos model
side by side, which is the whole reason this platform targets the Spark rather than a
typical 24–32 GB discrete GPU. GPU/LLM stages run **serially by design** (one model,
one Ollama instance); only market-data downloads and fundamentals analysis fan out
across CPU threads.

---

## 1. Hardware / OS prerequisites

Confirm the box before installing anything:

```bash
uname -m                       # expect: aarch64
nvidia-smi                     # GPU must be visible; note the driver + CUDA version
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
                               # expect compute_cap 12.1 (sm_121), ~128 GB
cat /etc/nv_tegra_release 2>/dev/null || cat /etc/os-release   # DGX OS / release
free -h                        # unified memory pool
df -h /                        # need ~60 GB free: models (~20 GB) + deps + db/reports
```

If `nvidia-smi` fails, stop here and fix the driver/runtime first — nothing GPU-bound
below will work until it succeeds. The platform *will* still run with the GPU absent
(Kronos degrades to CPU, the LLM agents degrade to neutral scores), but you do not
want to run degraded on a Spark.

> **GB10 gotcha — sm_121 / CUDA 13.** The Blackwell GPU in GB10 is newer than the CUDA
> builds shipped by most upstream Python wheels. Stock `pip install torch` and the
> default Ollama binary may install/run **CPU-only** on this hardware. Sections 2 and 4
> handle this explicitly. Always verify GPU is actually used (the verification steps
> tell you how) rather than assuming.

---

## 2. System packages

Run as a sudo-capable admin user.

```bash
# Dedicated service user + install root
sudo useradd -m -s /bin/bash trading
sudo mkdir -p /opt/trading && sudo chown trading:trading /opt/trading

# Base tooling
sudo apt-get update
sudo apt-get install -y git curl build-essential

# uv (Python project/dependency manager) — install AS the trading user
sudo -u trading bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo ln -sf /home/trading/.local/bin/uv /usr/local/bin/uv
uv --version
```

`uv` manages the Python 3.12+ interpreter and the virtualenv for you — no system
Python changes needed.

---

## 3. Ollama (serves the News / SEC / memo LLM)

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama --version               # GB10 needs a recent build with sm_121/CUDA-13 support
```

> **GB10 gotcha — Ollama GPU detection.** Older Ollama builds (and some pre-installed
> DGX OS images) ship a version that does not recognise the GB10 GPU and serves models
> on CPU, or loops on the model probe. Use a current release. After pulling a model
> (Section 5), confirm GPU placement with `ollama ps` — it must report the model on
> `100% GPU`, not CPU. If it shows CPU, upgrade Ollama and restart the service before
> going further.

The platform talks to Ollama at `http://localhost:11434` (set in
`config/settings.yaml` → `llm.base_url`). Keep Ollama on the same host.

---

## 4. Get the code onto the Spark and set up the project

The application code must end up at **`/opt/trading`**, owned by the `trading` user.
Pick **one** transfer method below depending on what you were handed.

### 4.1 Transfer the code

**Option A — git clone** (you have a repo URL / the box has network access to it):

```bash
sudo -iu trading                 # become the trading user
cd /opt/trading
git clone <repo-url> .           # note the trailing dot: clone INTO /opt/trading, not a subdir
```

**Option B — copy a folder from your machine with `rsync`** (you were handed the
project directory and have SSH to the Spark). Run this **from the machine that has the
code**, not on the Spark:

```bash
# Trailing slash on the source copies the CONTENTS of trading/ into /opt/trading/
rsync -av --delete \
  --exclude '.git/' --exclude '.venv/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
  --exclude 'db/' --exclude 'reports/' --exclude 'data/' --exclude 'logs/' \
  /path/to/trading/  <admin-user>@<spark-host>:/tmp/trading-upload/
# then on the Spark, move it into place with correct ownership:
ssh <admin-user>@<spark-host>
sudo rsync -a /tmp/trading-upload/ /opt/trading/ && rm -rf /tmp/trading-upload
```

Excluding `.venv/`, the caches, and the runtime dirs (`db/ reports/ data/ logs/`) is
deliberate — the virtualenv is rebuilt on the Spark in §4.4 (a Mac/x86 venv will not
run on aarch64), and the runtime dirs are recreated automatically (§4.3).

**Option C — tarball over `scp`** (no rsync). From the machine with the code:

```bash
tar --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    --exclude='.pytest_cache' --exclude='.ruff_cache' \
    --exclude='db' --exclude='reports' --exclude='data' --exclude='logs' \
    -czf trading.tar.gz -C /path/to/trading .
scp trading.tar.gz <admin-user>@<spark-host>:/tmp/
# on the Spark:
ssh <admin-user>@<spark-host>
sudo tar -xzf /tmp/trading.tar.gz -C /opt/trading
```

### 4.2 Expected layout after transfer

`/opt/trading` should contain exactly the version-controlled tree (the runtime dirs
appear later, on first run):

```
/opt/trading
├── config/                 # settings.yaml, watchlist.yaml, weights.yaml, risk_limits.yaml
├── deploy/                 # systemd units, pull_models.sh, this guide, SETUP.md
├── scripts/                # run_daily_analysis.py, approve_trades.py, init_db.py, ...
├── src/trading_platform/   # the application package (agents, risk, execution, dashboard, ...)
├── tests/
├── pyproject.toml
├── uv.lock                 # pinned dependency versions — keep it, it makes installs reproducible
├── README.md  PLAN.md  trading.md
```

Quick sanity check:

```bash
ls /opt/trading/config /opt/trading/scripts /opt/trading/src/trading_platform
```

If any of those are missing, the transfer was incomplete — fix it before continuing.

### 4.3 Fix ownership and runtime directories

If you used Option B or C (which run `sudo`), reset ownership so the `trading` user
owns everything, then create the runtime directories:

```bash
sudo chown -R trading:trading /opt/trading

# Runtime dirs are git-ignored and NOT in the transfer. The app auto-creates each on
# first write (db on init, reports per run, etc.), but creating them up front avoids
# any first-run surprise and lets you set permissions now:
sudo -u trading mkdir -p \
  /opt/trading/db \
  /opt/trading/data \
  /opt/trading/logs \
  /opt/trading/reports/daily \
  /opt/trading/reports/discovery \
  /opt/trading/reports/memos \
  /opt/trading/reports/backtests
```

These four runtime trees hold all generated state — `db/` (the SQLite database and its
WAL files), `reports/` (per-run markdown + JSON), `data/` (backtest scratch DBs), and
`logs/`. They are intentionally excluded from version control; **they are what you back
up** (§15), not the code.

### 4.4 Install dependencies and initialize the database (as `trading`)

```bash
sudo -iu trading            # if not already the trading user
cd /opt/trading

uv sync --extra kronos             # builds .venv on the Spark: core deps + Kronos/PyTorch
uv run python scripts/init_db.py   # create the SQLite schema (idempotent, versioned migrations)
```

`uv sync` reads `uv.lock`, creates `/opt/trading/.venv`, and installs the exact pinned
versions — so the deployed environment matches what was tested. `init_db.py` creates
`db/investment_research.sqlite` (and the `db/` dir if it doesn't exist) and prints the
schema version; re-running it is safe.

> **GB10 gotcha — PyTorch on sm_121.** `uv sync --extra kronos` pulls `torch>=2.0`.
> The default PyPI wheel may be CPU-only or built for an older CUDA that does not
> include Blackwell `sm_121` kernels. If the Kronos verification in Section 7 reports
> `kronos ... on cpu` (instead of `cuda`), or you see a
> `no kernel image is available for execution on the device` error, install the
> CUDA-13 / aarch64 PyTorch build NVIDIA ships for the Spark, then re-run the
> verification. With `uv`, the clean way is to pin torch to NVIDIA's index, e.g.:
>
> ```bash
> # Example — confirm the current CUDA-13 aarch64 index/tag for your DGX OS release
> uv pip install --python .venv torch \
>   --index-url https://download.pytorch.org/whl/cu130
> ```
>
> Kronos still *functions* on CPU (~10s/ticker for Kronos-small), so this is a
> performance/quality fix, not a blocker — but on a Spark you want CUDA.

### Verify the test suite (optional but recommended)

```bash
uv run pytest          # full suite
uv run ruff check .    # lint
```

---

## 5. Prefetch models

So the first scheduled run doesn't pay download time:

```bash
bash deploy/pull_models.sh
```

This pulls `qwen2.5:32b-instruct` (~20 GB) via Ollama and downloads the Kronos
weights from HuggingFace, loading them once to confirm the device. After it finishes:

```bash
ollama ps              # qwen2.5:32b-instruct should sit on 100% GPU when active
```

---

## 6. Configuration review

Everything tunable lives in `config/` and is validated at startup — bad config fails
fast. Review these before the first run:

**`config/settings.yaml`**
- `kronos.model_id` — bump from the dev default `NeoQuasar/Kronos-small` to
  **`NeoQuasar/Kronos-base`** (102M params) on the Spark. Both use the same
  512-context base tokenizer, so only this one line changes.
- `llm.model` — confirm `qwen2.5:32b-instruct` (the DGX target model).
- `sec.identity` — EDGAR requires a descriptive User-Agent with real contact info.
  Currently `"Peak Logic info@peaklogic.ai"`; update if the contact changes.
- `notifications` — see Section 8.

**`config/risk_limits.yaml`**
- Confirm `require_human_approval: true` (locked decision — keep it on).
- Review `max_position_pct`, `max_sector_pct`, `min_cash_reserve_pct`,
  `min_final_score`, `sizing.*`, and `paper_account.starting_cash` (default
  `100000.0`).

**`config/watchlist.yaml`** — tickers and their GICS sector tags. Use official GICS
sector names (e.g. "Information Technology", not "Technology") — mismatches mis-award
the discovery sector-gap bonus.

**`config/weights.yaml`** — signal weights (must sum to 1.0) and entry/exit
thresholds.

---

## 7. Secrets and environment

Secrets are **environment variables only**, never committed to config. They live in a
root-owned env file that the systemd units load:

```bash
sudo tee /etc/trading-platform.env >/dev/null <<'EOF'
FINNHUB_API_KEY=your_finnhub_key_here
EOF
sudo chmod 600 /etc/trading-platform.env
```

Optional additions to the same file:

- `MEMO_LLM_API_KEY` — only if you point pre-approval memos at an external
  OpenAI-compatible endpoint (`memo.base_url` / `memo.model` in settings). Empty =
  memos fall back to local Ollama.
- `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` — only for the Alpaca paper broker
  (Section 10).

### Verify the install end-to-end (as `trading`)

```bash
cd /opt/trading
# Data + 8-check quality gate over the watchlist:
uv run python scripts/refresh_market_data.py
# Ollama reachable with the model present:
curl -s localhost:11434/api/tags | head -c 300
# Kronos loads on the GPU (look for 'on cuda', not 'on cpu'):
bash deploy/pull_models.sh 2>&1 | grep -i kronos
```

A full dry run of the real pipeline:

```bash
uv run python scripts/run_daily_analysis.py
```

Runs are idempotent and resumable — re-invoking the same date resumes an incomplete
run and never duplicates scores or trades.

---

## 8. Notifications (optional but recommended)

You get a push on every run — success summary (decisions, equity, pending approvals)
or failure with the error. Point it at an ntfy topic:

In `config/settings.yaml`:

```yaml
notifications:
  enabled: true
  webhook_url: "https://ntfy.sh/your-private-topic"
```

Any ntfy-compatible endpoint works; pick an unguessable topic name since it's
unauthenticated.

---

## 9. Schedule with systemd

Two timers drive the platform. Install the units shipped in `deploy/`:

```bash
sudo cp deploy/trading-daily.service deploy/trading-daily.timer \
        deploy/trading-discovery.service deploy/trading-discovery.timer \
        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-daily.timer trading-discovery.timer

systemctl list-timers 'trading-*'    # confirm next trigger times
```

| Timer | Schedule | Job |
|-------|----------|-----|
| `trading-daily` | Mon–Fri **17:30 America/New_York** | Full analysis + trading run. `After=ollama.service`, 90-min timeout. |
| `trading-discovery` | Sat **09:00 America/New_York** | S&P 500 watchlist screen, 30-min timeout. |

Both units run as `User=trading`, `WorkingDirectory=/opt/trading`, load
`/etc/trading-platform.env`, and call `uv run`. Both timers use `Persistent=true`:
if the box was off at trigger time, the job runs at next boot. Combined with
idempotent/resumable runs and order expiry, a missed or crashed day degrades safely.

Manual test run + live logs:

```bash
sudo systemctl start trading-daily.service
journalctl -u trading-daily.service -f
```

> **Timezone note.** The `OnCalendar=... America/New_York` suffix needs systemd ≥ 235
> (DGX OS qualifies) and handles US daylight-saving transitions automatically, so the
> job always fires 90 minutes after the 16:00 ET close.

---

## 10. The dashboard

Read-only operational view (account KPIs, equity curve, approval queue with
one-click approve/reject, open positions, decisions with risk verdicts, agent score
leaderboard, run health, report browser).

```bash
uv run python scripts/run_dashboard.py        # http://127.0.0.1:8420
```

It binds to **127.0.0.1 only** and has **no auth layer**. To view it from another
machine, do **not** bind it to `0.0.0.0` on an untrusted network — tunnel over SSH
instead:

```bash
ssh -L 8420:127.0.0.1:8420 trading@<spark-host>
# then browse http://localhost:8420 on your laptop
```

To keep it always-on, wrap it in its own systemd service (a simple `Type=simple`
unit running the command above as `User=trading`).

---

## 11. Optional: Alpaca paper broker

By default, fills are simulated internally. To route approved orders through Alpaca's
**paper** API (real market-on-open executions against fake money):

1. Create a paper account at alpaca.markets and generate paper API keys.
2. Add the keys to `/etc/trading-platform.env`:
   ```
   ALPACA_API_KEY_ID=...
   ALPACA_API_SECRET_KEY=...
   ```
3. Set `execution.broker: alpaca_paper` in `config/settings.yaml`.

Approving an order submits it to Alpaca immediately as market-on-open (an approval on
evening T fills at T+1's open — same convention as local sim). The next daily run
syncs real fill prices into the local ledger and reconciles positions/cash, surfacing
(never auto-fixing) any drift. The base URL is hard-coded to the paper endpoint; a
test enforces that live trading is structurally impossible.

---

## 12. Daily operating workflow

1. The daily timer fires weekdays at 17:30 ET; you get a notification.
2. If trades are proposed, review the queue:
   ```bash
   uv run python scripts/approve_trades.py                 # list pending
   uv run python scripts/approve_trades.py --memo <order>  # read the research memo
   uv run python scripts/approve_trades.py --approve <order_id>   # or --approve-all
   ```
   Unapproved orders expire at the next run.
3. Approved orders fill at the next run at that day's open.
4. Reports land in `reports/daily/` (with a machine-readable JSON twin); the full
   audit trail is in `db/investment_research.sqlite`.
5. Weekly, after the Saturday discovery run: review suggestions in the dashboard or
   `reports/discovery/`, and adopt one with:
   ```bash
   uv run python scripts/discover_stocks.py --add TICKER   # appends to watchlist.yaml
   ```

---

## 13. Health checks & monitoring

```bash
# Data + quality gate pass:
uv run python scripts/refresh_market_data.py

# Ollama up with models, on GPU:
curl -s localhost:11434/api/tags | head -c 300
ollama ps

# Recent run outcomes:
journalctl -u trading-daily.service --since "1 week ago" | grep -E "completed|FAILED"

# GPU utilisation during a run (separate terminal):
watch -n 2 nvidia-smi

# Performance summary once history accumulates:
uv run python scripts/show_performance.py
```

Each run's report includes a **Run Health** section (duration, per-stage failures,
data freshness). The `risk_events` and run-stage tables in SQLite are the durable
record if you need to investigate after the fact.

---

## 14. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Kronos loads `on cpu`, or `no kernel image ... for the device` | PyTorch wheel lacks `sm_121` (Blackwell/CUDA-13) kernels | Install NVIDIA's CUDA-13 aarch64 torch build into the venv (Section 4 gotcha), re-verify. |
| `ollama ps` shows the model on **CPU** | Ollama too old to detect GB10 GPU | Upgrade Ollama, `sudo systemctl restart ollama`, re-pull/re-run. |
| News / SEC agents return neutral scores every run | Ollama unreachable or model missing | `curl localhost:11434/api/tags`; `ollama pull qwen2.5:32b-instruct`; check `llm.base_url`. |
| Run fails immediately at startup | Invalid config (validated fail-fast) | Read the error — it names the bad key/file in `config/`. |
| Timer never fires | Timezone/systemd, or timer not enabled | `systemctl list-timers 'trading-*'`; ensure systemd ≥ 235; `enable --now` the timer. |
| `FINNHUB_API_KEY` warnings / thin news | Env file not loaded or key missing | Check `/etc/trading-platform.env` (mode 600) and that the unit has `EnvironmentFile=`. |
| Missed a day (box was off) | Expected | `Persistent=true` runs the catch-up at next boot; stale orders expire safely. |
| Out-of-memory under load | 32B LLM + Kronos + data all resident | Unusual on 128 GB; if it recurs, confirm nothing else is co-tenant, or quantize the LLM (`qwen2.5:32b-instruct-q4_K_M`). |
| `ModuleNotFoundError: No module named 'trading_platform.data'` | The repo's `.gitignore` had an **unanchored** `data/` rule that also matched `src/trading_platform/data/`, so that source package was never committed and is missing from clones. | Anchor the runtime rules to the repo root (`/db/ /reports/ /data/ /logs/`), `git add` the package, commit, push — then re-clone/pull. Quick unblock: copy `src/trading_platform/data/` onto the box directly. |
| LLM `... validation (attempt N/3)` **looping on every ticker** | Prompt/schema numeric-**scale mismatch**: the model emits values on one scale where the pydantic schema wants another (here, per-headline `sentiment` came back 0–100 but the field is bounded −1.0…1.0). Ollama's structured-output grammar enforces JSON *shape and types* but **not** numeric `ge`/`le` bounds, so every response fails validation. Model-dependent — swapping the LLM can expose it. | Make the scale explicit in the system prompt **and** add a `description` to the constrained field so it reaches the model via the schema. Diagnose by capturing one raw response (see 14.2). Not an infra problem — don't chase Ollama/GPU. |
| `attempt to write a readonly database`, or `Permission denied` writing `.venv`/`.pth` | A runtime dir, the database, or `.venv` is owned by `root`/another user — leftover from a step run under `sudo` or alternating users. | `sudo chown -R trading:trading /opt/trading`. |
| `trading` user can't see files / `cd /opt/trading: No such file or directory` (but admin can) | Either the project lives under `/home/<admin>` (mode `700`, untraversable by the service account) instead of `/opt`, or you're in the wrong cwd after `sudo -iu trading` (it starts in `/home/trading`). | Keep the project at top-level `/opt/trading` (world-traversable); always `cd /opt/trading` after switching users. `namei -l /opt/trading` shows where traversal breaks. |
| `fatal: detected dubious ownership` / `could not lock config file .git/config: Permission denied` | Running git as a different user than owns the tree. | Run git as the tree's owner (or `git config --global --add safe.directory /opt/trading`); `sudo chown -R trading:trading /opt/trading` afterward. |
| `nohup: ignoring input` then it runs | Not an error | Informational — `nohup` just detached stdin; the run continues in the background. |

### 14.1 Deployment-flow gotchas (these cause most first-deploy pain)

These aren't bugs in the platform — they're Linux/`sudo`/git footguns that repeatedly bite during setup. Internalizing them saves hours:

- **Put the project at `/opt/trading`, never under `/home/<user>/`.** Home directories are mode `700`, so the `trading` service account literally cannot traverse into another user's home — admin sees the files, `trading` gets "No such file or directory." `/opt` is world-traversable, which is why the guide and the systemd units hard-code `/opt/trading`.
- **`sudo -iu trading` (and `sudo -i`, and any login) start you in that user's *home*, not the project.** A bare `ls` then shows the home dir and looks empty. Always `cd /opt/trading` first; run `pwd` if unsure.
- **Run `sudo …` commands as your admin user, not from inside the `trading` shell.** `sudo` prompts for the *current* user's password; `trading` is a passwordless, non-sudo service account, so `sudo` from within its shell just fails. Do ownership/copy/systemd steps as admin (`exit` the trading shell first).
- **Do all file transfer and git as one user, then hand ownership over once.** Transfer/clone as admin, then `sudo chown -R trading:trading /opt/trading` at the end. Alternating users mid-stream is what produces the readonly-database, `.venv` permission, and dubious-ownership errors above.
- **A plain file copy is not a git clone.** If `/opt/trading/.git` is absent, `git pull` can't work there; either re-establish the clone (`git init` + `remote add` + `fetch` + `reset --hard origin/<branch>`) or just copy changed files directly. And `git push`/`pull` only move **commits** — uncommitted working-tree edits won't transfer until committed.

### 14.2 Diagnosing LLM validation failures

When an LLM agent fails validation on every ticker, capture one raw response instead of guessing — it shows exactly which field/scale is wrong:

```bash
cd /opt/trading
uv run python - <<'EOF'
import requests
from trading_platform.core.config import load_config
from trading_platform.agents.news import NewsAnalysis, SYSTEM_PROMPT
s = load_config().settings.llm
payload = {"model": s.model,
           "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Stock: AAPL\n\nRecent headlines:\n[0] (2026-06-16, Reuters) Apple shares rise on strong demand"}],
           "stream": False, "format": NewsAnalysis.model_json_schema(),
           "options": {"temperature": 0.1}}
r = requests.post(f"{s.base_url}/api/chat", json=payload, timeout=180).json()
print("done_reason:", r.get("done_reason"), "| RAW:", repr(r["message"]["content"][:600]))
try: NewsAnalysis.model_validate_json(r["message"]["content"]); print("VALID")
except Exception as e: print("INVALID:", str(e)[:300])
EOF
```

Read the result: `done_reason=length` means truncation (raise `num_ctx`); a field/range error (e.g. `Input should be less than or equal to 1, input_value=80`) means a prompt/schema scale mismatch (fix the prompt + field description); prose instead of JSON means the `format` isn't honored (upgrade Ollama).

---

## 15. Backup & recovery

The entire state of the system is the SQLite database — positions, cash, orders,
fills, scores, risk events, and run history all live there.

```bash
# Hot backup (WAL mode — use the SQLite backup API, not a raw cp):
uv run python -c "import sqlite3; \
  src=sqlite3.connect('db/investment_research.sqlite'); \
  dst=sqlite3.connect('/backup/investment_research.$(date +%F).sqlite'); \
  src.backup(dst); dst.close(); src.close()"
```

`config/` is version-controlled with the repo; `reports/`, `data/`, and `db/` are
git-ignored runtime artifacts. Back up `db/` (state) and `/etc/trading-platform.env`
(secrets) regularly. To rebuild a fresh box: clone the repo, restore those two, run
`init_db.py` only if starting clean.

---

### Quick reference — paths & ports

| Item | Value |
|------|-------|
| Install root | `/opt/trading` |
| Service user | `trading` |
| Secrets file | `/etc/trading-platform.env` (root, mode 600) |
| Database | `/opt/trading/db/investment_research.sqlite` |
| Reports | `/opt/trading/reports/{daily,discovery,memos,backtests}` |
| Ollama API | `http://localhost:11434` |
| Dashboard | `http://127.0.0.1:8420` (localhost-only, no auth) |
| systemd units | `trading-daily.{service,timer}`, `trading-discovery.{service,timer}` |
| LLM model | `qwen2.5:32b-instruct` |
| Kronos model (Spark) | `NeoQuasar/Kronos-base` |
