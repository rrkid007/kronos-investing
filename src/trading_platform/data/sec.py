"""SEC filing retrieval via edgartools: latest 10-K/10-Q with key sections.

Section extraction is defensive — edgartools item access varies by form type
and version — and falls back to the head of the full filing text when the
structured sections can't be located, so the agent always has something
grounded to analyze.
"""

from __future__ import annotations

import logging
from datetime import date

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FilingDoc(BaseModel):
    ticker: str
    form_type: str
    filing_date: date
    accession_no: str
    risk_factors: str = ""
    mdna: str = ""

    @property
    def combined_text(self) -> str:
        return self.risk_factors + "\n" + self.mdna


def _extract_section(obj, keys: list[str]) -> str:
    for key in keys:
        try:
            text = obj[key]
        except Exception:
            continue
        if text and len(str(text).strip()) > 200:
            return str(text)
    return ""


def fetch_latest_filing(
    ticker: str,
    identity: str,
    max_risk_chars: int = 10_000,
    max_mdna_chars: int = 6_000,
) -> FilingDoc | None:
    """Latest 10-K or 10-Q for the ticker, or None when nothing is found."""
    from edgar import Company, set_identity

    set_identity(identity)
    filings = Company(ticker).get_filings(form=["10-K", "10-Q"])
    if filings is None or len(filings) == 0:
        return None
    filing = filings.latest(1)

    risk, mdna = "", ""
    try:
        obj = filing.obj()
        # 10-K: Item 1A risk factors, Item 7 MD&A.
        # 10-Q: Part II Item 1A risk factors, Part I Item 2 MD&A.
        risk = _extract_section(obj, ["Item 1A", "Part II Item 1A"])
        mdna = _extract_section(obj, ["Item 7", "Part I Item 2", "Item 2"])
    except Exception as exc:
        logger.warning("structured section extraction failed for %s: %s", ticker, exc)

    if not risk and not mdna:
        try:
            risk = (filing.text() or "")[: max_risk_chars + max_mdna_chars]
        except Exception as exc:
            logger.warning("full-text fallback failed for %s: %s", ticker, exc)
            return None

    return FilingDoc(
        ticker=ticker,
        form_type=filing.form,
        filing_date=filing.filing_date,
        accession_no=filing.accession_no,
        risk_factors=risk[:max_risk_chars],
        mdna=mdna[:max_mdna_chars],
    )
