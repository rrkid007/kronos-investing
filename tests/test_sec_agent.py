"""SEC Filing Agent — quote grounding, caching, fallbacks. No network/LLM."""

from datetime import date
from pathlib import Path

import pytest

from tests.fixtures import FakeLLM
from trading_platform.agents.sec_filing import (
    FilingAnalysis,
    RiskFinding,
    SECFilingAgent,
    quote_is_grounded,
)
from trading_platform.core.config import load_config
from trading_platform.core.db import connect, init_db
from trading_platform.core.llm import LLMError
from trading_platform.data.sec import FilingDoc

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO_ROOT / "config")

RISK_TEXT = (
    "We face substantial litigation in multiple jurisdictions regarding patent "
    "infringement claims. Our indebtedness could limit cash flow available for "
    "operations. Increased regulatory scrutiny of our data practices may result "
    "in significant fines."
)


def make_doc(ticker="TEST", accession="0001-26-000001"):
    return FilingDoc(
        ticker=ticker,
        form_type="10-Q",
        filing_date=date(2026, 5, 1),
        accession_no=accession,
        risk_factors=RISK_TEXT,
        mdna="Revenue increased due to strong demand for our products.",
    )


def make_analysis(score=65.0, findings=None):
    return FilingAnalysis(
        filing_score=score,
        summary="Routine risk profile with moderate litigation exposure.",
        findings=findings if findings is not None else [
            RiskFinding(category="litigation", severity="medium",
                        quote="substantial litigation in multiple jurisdictions"),
            RiskFinding(category="debt", severity="low",
                        quote="indebtedness could limit cash flow"),
        ],
    )


def make_agent(llm, doc=..., conn=None, fetch_fail=None):
    def fetcher(ticker):
        if fetch_fail is not None:
            raise fetch_fail
        return make_doc(ticker) if doc is ... else doc

    return SECFilingAgent(CONFIG, llm=llm, fetcher=fetcher, conn=conn)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.sqlite")
    init_db(c)
    yield c
    c.close()


# --- grounding ---------------------------------------------------------------

def test_quote_grounding_normalizes_whitespace_and_case():
    assert quote_is_grounded("SUBSTANTIAL   litigation in\nmultiple jurisdictions", RISK_TEXT)
    assert not quote_is_grounded("completely invented quote about fraud", RISK_TEXT)
    assert not quote_is_grounded("short", RISK_TEXT)  # too short to be meaningful


def test_grounded_findings_kept_ungrounded_dropped():
    findings = [
        RiskFinding(category="litigation", severity="medium",
                    quote="substantial litigation in multiple jurisdictions"),
        RiskFinding(category="going_concern", severity="high",
                    quote="substantial doubt about our ability to continue"),  # invented
    ]
    agent = make_agent(FakeLLM(response=make_analysis(findings=findings)))
    result = agent.analyze("TEST", "r1", None)
    assert result.details["risk_terms"] == 1
    assert result.details["ungrounded_findings_dropped"] == 1
    assert result.details["findings"][0]["category"] == "litigation"


def test_majority_ungrounded_cuts_confidence():
    findings = [
        RiskFinding(category="litigation", severity="high", quote="not in the text at all"),
        RiskFinding(category="debt", severity="high", quote="also completely invented"),
    ]
    agent = make_agent(FakeLLM(response=make_analysis(findings=findings)))
    result = agent.analyze("TEST", "r1", None)
    assert result.confidence == 0.4


def test_well_grounded_analysis_gets_full_confidence():
    agent = make_agent(FakeLLM(response=make_analysis()))
    result = agent.analyze("TEST", "r1", None)
    assert result.confidence == 0.7
    assert result.score == 65.0
    assert result.details["form_type"] == "10-Q"


# --- caching -----------------------------------------------------------------

def test_same_filing_hits_llm_exactly_once(conn):
    llm = FakeLLM(response=make_analysis())
    agent = make_agent(llm, conn=conn)
    first = agent.analyze("TEST", "r1", None)
    second = agent.analyze("TEST", "r2", None)

    assert llm.calls == 1
    assert second.score == first.score
    assert second.details["from_cache"] is True
    assert second.run_id == "r2"  # rebuilt for the new run


def test_new_accession_triggers_fresh_analysis(conn):
    llm = FakeLLM(response=make_analysis())
    agent = SECFilingAgent(
        CONFIG, llm=llm, conn=conn,
        fetcher=lambda t: make_doc(t, accession=f"acc-{llm.calls}"),
    )
    agent.analyze("TEST", "r1", None)
    agent.analyze("TEST", "r2", None)
    assert llm.calls == 2


# --- fallbacks ---------------------------------------------------------------

def test_no_filing_returns_neutral():
    agent = make_agent(FakeLLM(response=make_analysis()), doc=None)
    result = agent.analyze("TEST", "r1", None)
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert result.details["fallback_reason"] == "no 10-K/10-Q found"


def test_fetch_failure_returns_neutral():
    agent = make_agent(FakeLLM(response=make_analysis()), fetch_fail=TimeoutError("edgar slow"))
    result = agent.analyze("TEST", "r1", None)
    assert result.confidence == 0.0
    assert "filing fetch failed" in result.details["fallback_reason"]


def test_llm_failure_returns_neutral():
    agent = make_agent(FakeLLM(fail=LLMError("ollama unreachable")))
    result = agent.analyze("TEST", "r1", None)
    assert result.score == 50.0
    assert result.confidence == 0.0
