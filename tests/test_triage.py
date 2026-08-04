"""Tests for the ML/NLP triage engine: clustering, exception labels, Wilson, flakiness."""
from __future__ import annotations

import pandas as pd

from triage_engine import (
    _exception_type,
    _wilson_lower_bound,
    cluster_failures,
    device_anomaly,
    flakiness_index,
)


def test_clustering_groups_similar_traces(sample_df):
    clusters, annotated = cluster_failures(sample_df)

    def cid(sid):
        return annotated.loc[annotated.session_id == sid, "cluster_id"].iloc[0]

    # The two NoSuchElement failures share a cluster; the two SocketTimeouts share
    # another; and the two groups are distinct.
    assert cid("s2") == cid("s3")
    assert cid("s4") == cid("s5")
    assert cid("s2") != cid("s4")
    assert len(clusters) == 2


def test_exception_type_extraction():
    assert _exception_type("org.openqa.selenium.NoSuchElementException: x") == "NoSuchElementException"
    assert _exception_type("java.net.SocketTimeoutException: y") == "SocketTimeoutException"
    assert _exception_type("no exception here") == ""


def test_wilson_downweights_small_samples():
    # A single unlucky failure must not outrank a sustained one.
    assert _wilson_lower_bound(25, 50, 1.96) > _wilson_lower_bound(1, 1, 1.96)
    assert _wilson_lower_bound(0, 10, 1.96) == 0.0


def test_device_anomaly_ranks_by_risk_score():
    df = pd.DataFrame({
        "session_id": [f"x{i}" for i in range(11)],
        "device": ["A"] + ["B"] * 10,
        "os_version": ["13.0"] * 11,
        "is_failure": [True] + [True] * 5 + [False] * 5,
    })
    result = device_anomaly(df)
    # B (5/10, larger sample) outranks A (1/1 fluke) despite A's 100% raw rate.
    assert result.iloc[0]["device"] == "B"
    assert bool(result.loc[result.device == "A", "low_sample"].iloc[0]) is True
    assert bool(result.loc[result.device == "B", "low_sample"].iloc[0]) is False


def test_flakiness_flip_rate_from_history():
    hist = pd.DataFrame({
        "name": ["FlakyTest"] * 4,
        "is_failure": [False, True, False, True],   # alternates every build
        "duration": [10.0, 10.0, 10.0, 10.0],
        "build_id": ["b1", "b2", "b3", "b4"],
        "created_at": ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"],
    })
    result = flakiness_index(hist, history=hist)
    row = result.loc[result.name == "FlakyTest"].iloc[0]
    assert row["flip_rate"] == 1.0          # pass/fail/pass/fail = every transition flips
    assert row["flakiness_index"] > 0.5


def test_flakiness_falls_back_to_in_build(sample_df):
    # No history -> in-build proxy; flip_rate column present but zero.
    result = flakiness_index(sample_df, history=None)
    assert "flakiness_index" in result.columns
    assert (result["flip_rate"] == 0.0).all()
