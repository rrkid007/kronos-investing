"""Discovery Agent — universe cache, candidate rotation, funnel, ranking,
watchlist mutation. No network."""


import pytest

import trading_platform.data.universe as universe_mod
from tests.fixtures import FakeMarketDataService, make_ohlcv, make_snapshot
from trading_platform.core.db import connect, init_db
from trading_platform.data.universe import (
    load_universe,
    mark_screened,
    refresh_universe,
)
from trading_platform.discovery.screener import (
    Suggestion,
    screen,
    sector_bonus,
    select_candidates,
    weakest_incumbent,
)

FAKE_CONSTITUENTS = [
    {"ticker": "NVO", "name": "Novo Nordisk", "sector": "Health Care"},
    {"ticker": "XOM", "name": "Exxon Mobil", "sector": "Energy"},
    {"ticker": "CAT", "name": "Caterpillar", "sector": "Industrials"},
    {"ticker": "ORCL", "name": "Oracle", "sector": "Information Technology"},
    {"ticker": "JPM", "name": "JPMorgan", "sector": "Financials"},
    {"ticker": "AAPL", "name": "Apple", "sector": "Information Technology"},  # already on watchlist
] + [
    {"ticker": f"T{i:03d}", "name": f"Filler {i}", "sector": "Information Technology"}
    for i in range(395)  # parser sanity check requires >= 400 rows
]


@pytest.fixture
def conn(tmp_config, monkeypatch):
    monkeypatch.setattr(universe_mod, "_fetch_constituents",
                        lambda: list(FAKE_CONSTITUENTS))
    c = connect(tmp_config.db_path)
    init_db(c)
    refresh_universe(c)
    yield c
    c.close()


# --- universe -----------------------------------------------------------------

def test_universe_cached_until_stale(conn, monkeypatch):
    calls = {"n": 0}

    def counting_fetch():
        calls["n"] += 1
        return list(FAKE_CONSTITUENTS)

    monkeypatch.setattr(universe_mod, "_fetch_constituents", counting_fetch)
    refresh_universe(conn)            # fresh -> cache hit
    assert calls["n"] == 0
    refresh_universe(conn, force=True)
    assert calls["n"] == 1
    assert len(load_universe(conn)) == len(FAKE_CONSTITUENTS)


def test_departed_members_removed(conn, monkeypatch):
    smaller = [m for m in FAKE_CONSTITUENTS if m["ticker"] != "XOM"]
    monkeypatch.setattr(universe_mod, "_fetch_constituents", lambda: smaller)
    refresh_universe(conn, force=True)
    assert "XOM" not in {r["ticker"] for r in load_universe(conn)}


# --- candidate selection ---------------------------------------------------------

def test_watchlist_members_excluded(conn, tmp_config):
    candidates = select_candidates(conn, tmp_config, max_candidates=500)
    assert "AAPL" not in {c["ticker"] for c in candidates}


def test_gap_sectors_come_first(conn, tmp_config):
    candidates = select_candidates(conn, tmp_config, max_candidates=5)
    # Watchlist covers Technology/Financials/etc; Health Care, Energy,
    # Industrials are absent -> NVO, XOM, CAT must lead.
    leading = {c["ticker"] for c in candidates[:3]}
    assert leading == {"NVO", "XOM", "CAT"}


def test_rotation_prefers_least_recently_screened(conn, tmp_config):
    mark_screened(conn, ["NVO"])
    candidates = select_candidates(conn, tmp_config, max_candidates=3)
    tickers = [c["ticker"] for c in candidates]
    assert tickers.index("NVO") > tickers.index("CAT")  # screened drops behind


# --- sector bonus ----------------------------------------------------------------

def test_sector_bonus_tiers():
    counts = {"Information Technology": 6, "Financials": 2, "Communication Services": 2}
    assert sector_bonus("Health Care", counts) == 8.0      # absent
    assert sector_bonus("Financials", counts) == 4.0       # below average (3.33)
    assert sector_bonus("Information Technology", counts) == 0.0       # over-weighted


# --- the screen --------------------------------------------------------------------

def make_market(overrides=None, default=None):
    return FakeMarketDataService(
        default_frame=default if default is not None else make_ohlcv(
            n_rows=320, drift=0.002, daily_vol=0.008),
        frames=overrides or {},
    )


def strong_fundamentals(ticker):
    return make_snapshot(ticker=ticker)


def test_screen_ranks_and_persists(conn, tmp_config):
    suggestions = screen(conn, tmp_config, make_market(),
                         fundamentals_fetcher=strong_fundamentals)
    assert suggestions
    assert len(suggestions) <= tmp_config.settings.discovery.top_n
    scores = [s.combined_score for s in suggestions]
    assert scores == sorted(scores, reverse=True)
    # gap-sector candidates with equal quant scores must outrank same-score tech
    assert suggestions[0].sector in {"Health Care", "Energy", "Industrials"}
    assert "fills sector gap" in suggestions[0].rationale

    rows = conn.execute("SELECT * FROM watchlist_suggestions").fetchall()
    assert len(rows) == len(suggestions)
    assert all(r["status"] == "suggested" for r in rows)


def test_fundamentals_floor_blocks_momentum_junk(conn, tmp_config):
    weak = make_snapshot(  # great chart, terrible business
        revenue_growth=-0.05, earnings_growth=-0.1, gross_margin=0.12,
        operating_margin=0.01, profit_margin=0.005, return_on_equity=0.02,
        fcf_margin=-0.02, ocf_margin=0.01, trailing_pe=80.0, forward_pe=70.0,
        ev_to_ebitda=40.0, price_to_fcf=90.0,
    )
    suggestions = screen(conn, tmp_config, make_market(),
                         fundamentals_fetcher=lambda t: weak.model_copy(
                             update={"ticker": t}))
    assert suggestions == []  # nothing clears the floor


def test_gate_failed_candidates_skipped(conn, tmp_config):
    bad = make_ohlcv(n_rows=100)  # insufficient history -> gate fails
    suggestions = screen(
        conn, tmp_config,
        make_market(overrides={"NVO": bad}),
        fundamentals_fetcher=strong_fundamentals,
    )
    assert "NVO" not in {s.ticker for s in suggestions}


def test_screened_candidates_marked_for_rotation(conn, tmp_config):
    screen(conn, tmp_config, make_market(), fundamentals_fetcher=strong_fundamentals)
    n_marked = conn.execute(
        "SELECT COUNT(*) FROM universe WHERE last_screened_at IS NOT NULL"
    ).fetchone()[0]
    assert n_marked > 0


# --- weakest incumbent ---------------------------------------------------------------

def test_weakest_incumbent_excludes_held(conn, tmp_config):
    conn.execute(
        "INSERT INTO runs (run_id, run_date, started_at, status) "
        "VALUES ('r1', '2026-06-10', '2026-06-10T22:00:00', 'completed')"
    )
    for ticker, score in [("COST", 38.0), ("MSFT", 42.0), ("AAPL", 71.0)]:
        conn.execute(
            "INSERT INTO decisions (run_id, ticker, action, final_score, created_at) "
            "VALUES ('r1', ?, 'hold', ?, '')", (ticker, score),
        )
    # COST is weakest but currently held -> MSFT should be flagged
    conn.execute(
        "INSERT INTO positions (ticker, qty, avg_cost, opened_at, updated_at) "
        "VALUES ('COST', 5, 900, '2026-06-01', '2026-06-01')"
    )
    conn.commit()

    weakest = weakest_incumbent(conn, tmp_config)
    assert weakest["ticker"] == "MSFT"
    assert weakest["avg_final_score"] == 42.0


def test_weakest_incumbent_none_without_history(conn, tmp_config):
    assert weakest_incumbent(conn, tmp_config) is None


# --- watchlist mutation -----------------------------------------------------------

def test_add_appends_to_watchlist_yaml(conn, tmp_config):
    from scripts.discover_stocks import add_to_watchlist

    screen(conn, tmp_config, make_market(), fundamentals_fetcher=strong_fundamentals)
    config_dir = tmp_config.root / "config"

    add_to_watchlist(conn, tmp_config, config_dir, "NVO")

    text = (config_dir / "watchlist.yaml").read_text(encoding="utf-8")
    assert "symbol: NVO" in text
    assert "sector: Health Care" in text

    from trading_platform.core.config import load_config
    reloaded = load_config(config_dir)  # must still parse + validate
    assert "NVO" in reloaded.watchlist.symbols
    assert reloaded.watchlist.sector_of("NVO") == "Health Care"

    status = conn.execute(
        "SELECT status FROM watchlist_suggestions WHERE ticker = 'NVO'"
    ).fetchone()
    if status:  # NVO may or may not have made top_n; universe path also valid
        assert status["status"] in ("added", "suggested")


def test_add_duplicate_rejected(conn, tmp_config):
    from scripts.discover_stocks import add_to_watchlist

    with pytest.raises(SystemExit, match="already on the watchlist"):
        add_to_watchlist(conn, tmp_config, tmp_config.root / "config", "AAPL")


def test_add_unknown_ticker_rejected(conn, tmp_config):
    from scripts.discover_stocks import add_to_watchlist

    with pytest.raises(SystemExit, match="not found"):
        add_to_watchlist(conn, tmp_config, tmp_config.root / "config", "ZZZZZ")


def test_suggestion_model_roundtrip():
    s = Suggestion(ticker="NVO", sector="Health Care", technical_score=70.0,
                   fundamental_score=68.0, sector_gap_bonus=8.0,
                   combined_score=77.0, rationale="x")
    assert s.combined_score == 77.0
