"""The quality gate must catch each corruption mode on otherwise-clean data."""

from datetime import date

import numpy as np
import pandas as pd

from tests.fixtures import make_ohlcv
from trading_platform.data.quality import validate_ohlcv

AS_OF = date(2026, 6, 11)


def _failed_names(report):
    return {c.name for c in report.failures}


def test_clean_data_passes_all_checks():
    report = validate_ohlcv(make_ohlcv(), "TEST", as_of=AS_OF)
    assert report.passed, report.summary()


def test_insufficient_history():
    report = validate_ohlcv(make_ohlcv(n_rows=100), "TEST", as_of=AS_OF)
    assert "sufficient_history" in _failed_names(report)


def test_empty_frame_fails_without_crashing():
    report = validate_ohlcv(make_ohlcv(n_rows=300).iloc[0:0], "TEST", as_of=AS_OF)
    assert not report.passed


def test_nan_prices_detected():
    df = make_ohlcv()
    df.iloc[50, df.columns.get_loc("close")] = np.nan
    assert "no_nan" in _failed_names(validate_ohlcv(df, "TEST", as_of=AS_OF))


def test_nonpositive_prices_detected():
    df = make_ohlcv()
    df.iloc[10, df.columns.get_loc("low")] = -1.0
    assert "positive_prices" in _failed_names(validate_ohlcv(df, "TEST", as_of=AS_OF))


def test_high_low_violation_detected():
    df = make_ohlcv()
    df.iloc[20, df.columns.get_loc("high")] = df.iloc[20]["close"] * 0.5
    assert "high_low_consistency" in _failed_names(validate_ohlcv(df, "TEST", as_of=AS_OF))


def test_date_gap_detected():
    df = make_ohlcv(n_rows=320)
    df = pd.concat([df.iloc[:150], df.iloc[165:]])  # 15-bar hole
    assert "no_date_gaps" in _failed_names(validate_ohlcv(df, "TEST", as_of=AS_OF))


def test_stale_data_detected():
    df = make_ohlcv(end=date(2026, 5, 15))  # ~4 weeks before as_of
    assert "fresh" in _failed_names(validate_ohlcv(df, "TEST", as_of=AS_OF))


def test_unadjusted_split_artifact_detected():
    df = make_ohlcv()
    # Simulate a missed 2:1 split adjustment: adj_close halves overnight.
    half = df.index[200:]
    df.loc[half, "adj_close"] = df.loc[half, "adj_close"] * 0.5
    assert "no_split_artifacts" in _failed_names(validate_ohlcv(df, "TEST", as_of=AS_OF))


def test_recent_zero_volume_detected():
    df = make_ohlcv()
    df.iloc[-5:, df.columns.get_loc("volume")] = 0
    assert "recent_volume" in _failed_names(validate_ohlcv(df, "TEST", as_of=AS_OF))
