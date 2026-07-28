"""Signal skill study — does an agent's score predict forward returns?

Answers a narrower question than `run_backtest.py`, and answers it far cheaper.
The backtest reports P&L, which bundles signal quality together with position
sizing, risk rules and threshold choices — when it loses money you cannot tell
which ingredient failed. This measures the signal alone:

    score on day T   vs   realized return from T to T+horizon

Reported as the information coefficient (IC) — the cross-sectional Spearman
rank correlation between score and forward return, computed per date and then
averaged. Rank correlation, not Pearson, because what matters is whether the
signal *orders* names correctly, not whether the score is calibrated in return
units.

Point-in-time safety: on each sample date the agent sees only bars up to and
including that date, and the forward return is measured strictly after it. No
lookahead is possible by construction.

Cost: one forecast per (date, ticker). Weekly cadence over 2 years with a
10-name watchlist is ~1,040 forecasts; daily is ~5,000. Kronos on CPU runs
~13s each, so use --cadence-days 5 (the default) and a GPU. Rows are flushed to
the CSV as they are produced, so an interrupted run keeps its work.

Examples:
  # Fast control run — no GPU needed, gives you a reference IC
  python scripts/signal_ic.py --agent technical --years 2

  # The real question
  python scripts/signal_ic.py --agent kronos --years 2 --cadence-days 5

  # Cheap probe first: times the run and proves the path
  python scripts/signal_ic.py --agent kronos --years 0.25 --limit 40
"""

import argparse
import csv
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from trading_platform.agents.kronos import KronosAgent
from trading_platform.agents.technical import TechnicalAgent
from trading_platform.backtest.replay import load_frames
from trading_platform.core.config import load_config
from trading_platform.core.db import connect

# Technical needs 200 bars for its slow MA; Kronos wants its full context
# window. Below this a score is technically produced but not comparable.
MIN_BARS = {"technical": 250, "kronos": 400}


def build_agent(name: str, config, sample_count: int | None):
    if name == "technical":
        return TechnicalAgent()
    settings = config.settings.kronos
    if sample_count is not None:
        settings = settings.model_copy(update={"sample_count": sample_count})
    return KronosAgent(settings)


def forward_return(df: pd.DataFrame, ts: pd.Timestamp, horizon: int) -> float | None:
    """Close-to-close return over `horizon` trading bars after `ts`.

    None when the window runs past the end of history — those dates are
    excluded rather than truncated, so late samples can't bias the result.
    """
    idx = df.index.get_indexer([ts])[0]
    if idx < 0 or idx + horizon >= len(df):
        return None
    start = float(df["close"].iloc[idx])
    end = float(df["close"].iloc[idx + horizon])
    if start <= 0:
        return None
    return (end / start - 1.0) * 100.0


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman correlation without scipy — Pearson on the ranks.

    pandas' `method="spearman"` imports scipy, which this project does not
    depend on and should not start depending on for one statistic.
    """
    return a.rank().corr(b.rank())


def summarize(rows: list[dict], horizon: int) -> None:
    if not rows:
        print("\nno observations — widen the window or lower --cadence-days")
        return

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 66}\n  {len(df)} observations | "
          f"{df['date'].nunique()} dates | {df['ticker'].nunique()} tickers | "
          f"horizon {horizon}d\n{'=' * 66}")

    # Per-date cross-sectional rank correlation, then averaged. Dates with
    # fewer than 3 names can't support a meaningful correlation.
    ics = []
    for day, group in df.groupby("date"):
        if len(group) >= 3 and group["score"].nunique() > 1:
            ic = rank_corr(group["score"], group["fwd_return_pct"])
            if pd.notna(ic):
                ics.append(ic)

    if len(ics) >= 2:
        s = pd.Series(ics)
        ir = s.mean() / s.std() if s.std() else float("nan")
        t_stat = ir * (len(s) ** 0.5)
        print(f"\n  mean IC          {s.mean():+.4f}   (per-date rank correlation)")
        print(f"  IC std dev       {s.std():.4f}")
        print(f"  information ratio{ir:+.3f}")
        print(f"  t-statistic      {t_stat:+.2f}   over {len(s)} dates")
        print(f"  IC > 0 on        {(s > 0).mean() * 100:.0f}% of dates")
    else:
        print("\n  too few usable dates for a per-date IC")
        t_stat = float("nan")

    pooled = rank_corr(df["score"], df["fwd_return_pct"])
    print(f"  pooled rank corr {pooled:+.4f}")

    # Top-half vs bottom-half spread, ranked within each date so the split is
    # cross-sectional rather than contaminated by market-wide moves.
    df["rank"] = df.groupby("date")["score"].rank(pct=True)
    top = df[df["rank"] > 0.5]["fwd_return_pct"]
    bottom = df[df["rank"] <= 0.5]["fwd_return_pct"]
    if len(top) and len(bottom):
        print(f"\n  top-half mean    {top.mean():+.3f}%  (n={len(top)})")
        print(f"  bottom-half mean {bottom.mean():+.3f}%  (n={len(bottom)})")
        print(f"  spread           {top.mean() - bottom.mean():+.3f}%")

    directional = df[df["direction"] != "neutral"]
    if len(directional):
        hits = (
            ((directional["direction"] == "bullish") & (directional["fwd_return_pct"] > 0))
            | ((directional["direction"] == "bearish") & (directional["fwd_return_pct"] < 0))
        )
        print(f"\n  direction hit rate {hits.mean() * 100:.1f}%  "
              f"(n={len(directional)}, coin flip = 50%)")

    print(f"\n{'-' * 66}")
    if pd.notna(t_stat) and abs(t_stat) >= 2:
        print("  |t| >= 2: the signal is statistically distinguishable from noise.")
        print("  Worth taking to run_backtest.py to see if it survives costs.")
    else:
        print("  |t| < 2: NOT distinguishable from noise over this sample.")
        print("  More samples per forecast will not fix this — it is a skill")
        print("  question, not a precision one. Try a longer window or a wider")
        print("  universe before concluding, but do not tune thresholds on it.")
    print(f"{'-' * 66}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--agent", default="kronos", choices=["kronos", "technical"])
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--years", type=float, default=2.0,
                        help="window length when --start omitted (default 2)")
    parser.add_argument("--cadence-days", type=int, default=5,
                        help="trading days between samples (5 = weekly)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="forward-return bars (default: kronos horizon_days)")
    parser.add_argument("--sample-count", type=int, default=None,
                        help="override Kronos sample_count (lower = faster)")
    parser.add_argument("--tickers", default=None, help="comma list (default: watchlist)")
    parser.add_argument("--all-cached", action="store_true",
                        help="use every ticker in price_cache instead of the watchlist. "
                             "Cross-sectional IC gains far more power from breadth than "
                             "from extra dates — strongly preferred if you have the compute")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N forecasts — use to time a probe run")
    parser.add_argument("--out", default=None, help="CSV path (default: reports/ic/<name>.csv)")
    args = parser.parse_args()

    config = load_config(args.config_dir)
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = (date.fromisoformat(args.start) if args.start
             else end - timedelta(days=int(args.years * 365.25)))
    horizon = args.horizon or config.settings.kronos.horizon_days
    conn = connect(config.db_path)
    if args.all_cached:
        # Benchmarks are index ETFs — including them in a cross-sectional stock
        # ranking would compare an index against its own constituents.
        benchmarks = set(config.settings.benchmarks)
        tickers = [r[0] for r in
                   conn.execute("SELECT DISTINCT ticker FROM price_cache ORDER BY ticker")
                   if r[0] not in benchmarks]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = config.watchlist.symbols

    frames = load_frames(conn, tickers)
    conn.close()
    if not frames:
        raise SystemExit(
            "no cached price history. Populate it first:\n"
            "  python scripts/refresh_market_data.py\n"
            "or run_backtest.py --refresh"
        )

    missing = sorted(set(tickers) - set(frames))
    if missing:
        print(f"warning: no history for {', '.join(missing)} — excluded")

    # Sample dates: the union of trading days in-window, thinned by cadence.
    all_days = sorted({d.date() for f in frames.values() for d in f.index
                       if start <= d.date() <= end})
    sample_days = all_days[::args.cadence_days]
    if not sample_days:
        raise SystemExit(f"no trading days in {start}..{end}")

    agent = build_agent(args.agent, config, args.sample_count)
    min_bars = MIN_BARS[args.agent]

    out_path = Path(args.out) if args.out else (
        config.root / "reports" / "ic" / f"ic-{args.agent}-{start}-{end}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    planned = len(sample_days) * len(frames)
    if args.limit:
        planned = min(planned, args.limit)
    print(f"agent={args.agent} | {len(sample_days)} dates x {len(frames)} tickers "
          f"= up to {planned} forecasts | horizon {horizon}d")
    print(f"writing {out_path}\n")

    rows: list[dict] = []
    done = skipped = failed = 0
    started = time.perf_counter()

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "date", "ticker", "agent", "score", "confidence", "direction",
            "fwd_return_pct", "horizon",
        ])
        writer.writeheader()

        for day in sample_days:
            ts = pd.Timestamp(day)
            for ticker, df in frames.items():
                if args.limit and done >= args.limit:
                    break

                # Point-in-time: the agent sees nothing after `day`.
                hist = df[df.index <= ts]
                if len(hist) < min_bars or hist.index[-1] != ts:
                    skipped += 1
                    continue
                fwd = forward_return(df, ts, horizon)
                if fwd is None:
                    skipped += 1
                    continue

                result = agent.analyze(ticker, f"ic-{day}", hist)
                # A zero-confidence result is the neutral fallback, not a
                # score — including it would dilute the IC with non-signals.
                if result.confidence == 0.0:
                    failed += 1
                    continue

                row = {
                    "date": day.isoformat(),
                    "ticker": ticker,
                    "agent": args.agent,
                    "score": result.score,
                    "confidence": result.confidence,
                    "direction": result.direction.value,
                    "fwd_return_pct": round(fwd, 4),
                    "horizon": horizon,
                }
                rows.append(row)
                writer.writerow(row)
                fh.flush()  # survive an interrupted multi-hour run
                done += 1

                if done % 25 == 0:
                    elapsed = time.perf_counter() - started
                    rate = elapsed / done
                    remaining = (planned - done) * rate
                    print(f"  {done}/{planned}  {rate:.1f}s/forecast  "
                          f"eta {remaining / 60:.0f}m")
            if args.limit and done >= args.limit:
                break

    elapsed = time.perf_counter() - started
    print(f"\ndone: {done} scored, {skipped} skipped (history/horizon), "
          f"{failed} agent failures, {elapsed / 60:.1f}m "
          f"({elapsed / max(done, 1):.1f}s/forecast)")
    summarize(rows, horizon)
    print(f"\nCSV: {out_path}")


if __name__ == "__main__":
    main()
