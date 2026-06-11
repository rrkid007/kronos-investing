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
sudo cp deploy/trading-daily.service deploy/trading-daily.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-daily.timer

systemctl list-timers trading-daily.timer   # confirm next trigger
sudo systemctl start trading-daily.service  # manual test run
journalctl -u trading-daily.service -f      # watch it
```

Why systemd timer over cron: `Persistent=true` runs a missed trigger at next
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

## 6. Health checks

```bash
uv run python scripts/refresh_market_data.py   # data + quality gate pass
curl -s localhost:11434/api/tags | head -c 200 # ollama up with models
journalctl -u trading-daily.service --since "1 week ago" | grep -E "completed|FAILED"
```
