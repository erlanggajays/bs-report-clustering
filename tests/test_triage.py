"""Tests for the ML/NLP triage engine: clustering, exception labels, Wilson, flakiness."""
from __future__ import annotations

import pandas as pd

from triage_engine import (
    _exception_type,
    _fingerprint,
    _top_frame,
    _wilson_lower_bound,
    cluster_failures,
    device_anomaly,
    flakiness_index,
)


def test_clustering_groups_by_signature(sample_df):
    clusters, annotated = cluster_failures(sample_df)

    def cid(sid):
        return annotated.loc[annotated.session_id == sid, "cluster_id"].iloc[0]

    # Same exception + same app frame -> one cluster, despite differing element ids.
    assert cid("s2") == cid("s3")
    assert cid("s4") == cid("s5")
    assert cid("s2") != cid("s4")
    assert len(clusters) == 2
    # Clusters are self-labeled by their root-cause signature.
    labels = {c.label for c in clusters}
    assert "NoSuchElementException @ PaymentConfirmScreen.tapConfirm" in labels
    assert "SocketTimeoutException @ WalletApiClient.getBalance" in labels


def test_fingerprint_prefers_app_frame_and_deepest_cause():
    assert _top_frame("boom\n\tat com.gopay.a.B.c(F:1)") == "B.c"
    # Caused-by chain: deepest cause wins.
    fp = _fingerprint(
        "org.openqa.selenium.WebDriverException: session broke\n"
        "Caused by: java.net.SocketTimeoutException: read timed out\n"
        "\tat com.gopay.net.WalletApiClient.getBalance(WalletApiClient.java:88)"
    )
    assert fp == "SocketTimeoutException @ WalletApiClient.getBalance"


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
    assert row["flip_rate"] == 1.0          # pass/fail/pass/fail = every build flips
    assert row["builds"] == 4
    assert row["raw_flakiness"] > 0.5       # strong raw signal
    # 4 builds clears flakiness_min_builds, but scores are still shrunk.
    assert bool(row["low_sample"]) is False
    assert row["flakiness_index"] < row["raw_flakiness"]


def test_repeats_within_one_build_are_not_counted_as_flips():
    """A scenario repeated inside a single build (multi-device, or a data-driven
    test) must not register as run-over-run flakiness — only as 'mixed'."""
    rows = [
        {"name": "Parameterised", "is_failure": f, "duration": 60.0 + i,
         "build_id": "onlybuild", "created_at": f"2026-08-05T03:{i:02d}:00Z"}
        # alternating pass/fail across 8 sessions in ONE build
        for i, f in enumerate([False, True, False, True, False, True, False, True])
    ]
    hist = pd.DataFrame(rows)

    row = flakiness_index(hist, history=hist).iloc[0]
    assert row["runs"] == 8
    assert row["builds"] == 1
    assert row["flip_rate"] == 0.0      # undefined across a single build
    assert row["mixed_rate"] == 1.0     # but it did both pass and fail in that build
    assert bool(row["low_sample"]) is True


def test_small_sample_cannot_outrank_well_sampled_flaky():
    """A 2-session pass->fail flip maxes the flip rate; it must not outrank a
    scenario proven flaky over many sessions."""
    rows = []
    for i, f in enumerate([False, True]):                      # 2 sessions
        rows.append({"name": "TinySample", "is_failure": f, "duration": 40.0 + i,
                     "build_id": f"b{i}", "created_at": f"2026-08-{i + 1:02d}"})
    for i in range(20):                                        # 20 sessions, alternating
        rows.append({"name": "ProvenFlaky", "is_failure": i % 2 == 0, "duration": 30.0 + (i % 5),
                     "build_id": f"c{i}", "created_at": f"2026-07-{i + 1:02d}"})
    hist = pd.DataFrame(rows)

    result = flakiness_index(hist, history=hist)
    assert result.iloc[0]["name"] == "ProvenFlaky"
    assert bool(result.iloc[0]["low_sample"]) is False
    tiny = result.loc[result.name == "TinySample"].iloc[0]
    assert bool(tiny["low_sample"]) is True
    assert tiny["flakiness_index"] < result.iloc[0]["flakiness_index"]


def test_flakiness_falls_back_to_in_build(sample_df):
    # No history -> in-build proxy; flip_rate column present but zero.
    result = flakiness_index(sample_df, history=None)
    assert "flakiness_index" in result.columns
    assert (result["flip_rate"] == 0.0).all()
