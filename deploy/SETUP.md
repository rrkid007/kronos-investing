# DGX Spark Deployment

Target: NVIDIA DGX Spark (ARM64, DGX OS). Everything below assumes the repo
lives at `/opt/trading` and runs as a dedicated `trading` user.

## 1. System prerequisites

```bash
sudo useradd -m -s /bin/bash trading
sudo mkdir -p /opt/trading && sudo chown trading:trading /opt/trading

# uv (Python manager)
curl -LsSf https://astral.sh/uv/install.sh | sh   # as the trading user
sudo ln -sf ~trading/.local/bin/uv /usr/local/bin/uv

# Ollama (serves the News/SEC LLM)
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

## 2. Project setup (as `trading`)

```bash
cd /opt/trading
git clone <repo-url> .
uv sync --extra kronos          # CUDA torch resolves automatically on ARM64+CUDA
uv run python scripts/init_db.py
bash deploy/pull_models.sh      # ~20GB LLM + Kronos weights
```

Config to review before first run:
- `config/settings.yaml` — switch Kronos to the bigger model on the Spark:
  `kronos.model_id: NeoQuasar/Kronos-base`. Confirm `llm.model: qwen2.5:32b-instruct`.
- `config/risk_limits.yaml` — limits and `require_human_approval: true`.
- `config/watchlist.yaml` — tickers and sectors.

## 3. Secrets and notifications

```bash
sudo tee /etc/trading-platform.env >/dev/null <<'EOF'
FINNHUB_API_KEY=your_key_here
EOF
sudo chmod 600 /etc/trading-platform.env
```

For run notifications, point `notifications.webhook_url` in settings.yaml at
an ntfy topic (e.g. `https://ntfy.sh/your-private-topic`) and set
`notifications.enabled: true`. You'll get a push on every run: success summary
(decisions, equity, pending approvals) or failure with the error.

## 4. Schedule

```bash
sudo cp deploy/trading-daily.service deploy/trading-daily.timer \
        deploy/trading-discovery.service deploy/trading-discovery.timer \
        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-daily.timer trading-discovery.timer

systemctl list-timers 'trading-*'           # confirm next triggers
sudo systemctl start trading-daily.service  # manual test run
journalctl -u trading-daily.service -f      # watch it
```

Two timers:
- **trading-daily** — weekdays 17:30 ET: the full analysis/trading run.
- **trading-discovery** — Saturdays 09:00 ET: screens ~60 S&P 500 candidates
  for watchlist suggestions (rotating through the universe over the weeks).
  Review in the dashboard or `reports/discovery/`, adopt with
  `uv run python scripts/discover_stocks.py --add TICKER`.

Why systemd timers over cron: `Persistent=true` runs a missed trigger at next
boot, and journald captures all logs. Runs are idempotent and resumable, and
unapproved orders expire at the next run — a skipped or crashed day degrades
safely.

## 5. Daily workflow

1. The timer runs weekdays at 17:30 ET; you get a notification.
2. If trades are proposed: `uv run python scripts/approve_trades.py` to review,
   then `--approve <order_id>` (or `--approve-all`). Unapproved orders expire
   at the next run.
3. Approved orders fill at the next run at that day's open.
4. Reports land in `reports/daily/`, the audit trail in
   `db/investment_research.sqlite`.
5. Weekly (after the Saturday discovery run): review watchlist suggestions in
   the dashboard or `reports/discovery/`; adopt one with
   `uv run python scripts/discover_stocks.py --add TICKER`.

## 6. Optional: Alpaca paper broker

By default fills are simulated internally. To route approved orders through
Alpaca's **paper** API instead (real market-on-open executions against fake
money):

1. Create a paper account at alpaca.markets and generate paper API keys.
2. Add to `/etc/trading-platform.env`:
   ```
   ALPACA_API_KEY_ID=...
   ALPACA_API_SECRET_KEY=...
   ```
3. Set `execution.broker: alpaca_paper` in `config/settings.yaml`.

Approving an order then submits it to Alpaca immediately (market-on-open, so
an approval on evening T fills at T+1's open — same convention as local).
The next daily run syncs real fill prices into the local ledger and
reconciles positions/cash, logging any drift. The base URL is hard-coded to
the paper endpoint; live trading is not implemented anywhere in this codebase.

## 7. Health checks

```bash
uv run python scripts/refresh_market_data.py   # data + quality gate pass
curl -s localhost:11434/api/tags | head -c 200 # ollama up with models
journalctl -u trading-daily.service --since "1 week ago" | grep -E "completed|FAILED"
```
