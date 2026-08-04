"""Tests for executive/ROI metrics."""
from __future__ import annotations

from exec_metrics import compute_exec_metrics
from triage_engine import cluster_failures, device_anomaly


def test_metrics_and_mttr_formula(sample_df):
    clusters, _ = cluster_failures(sample_df)
    device_risk = device_anomaly(sample_df)
    m = compute_exec_metrics(sample_df, clusters, device_risk)

    assert m.total_tests == 5
    assert m.failed == 4
    assert m.passed == 1
    assert m.num_root_causes == len(clusters)
    # MTTR saved = (failures - clusters) * per-failure triage minutes.
    from config import settings
    expected = (m.failed - m.num_root_causes) * settings.manual_triage_minutes_per_failure
    assert m.mttr_minutes_saved == expected
    assert 0.0 <= m.suite_health_index <= 100.0


def test_summary_cards_shape(sample_df):
    clusters, _ = cluster_failures(sample_df)
    m = compute_exec_metrics(sample_df, clusters, device_anomaly(sample_df))
    cards = m.summary_cards()
    for key in ("total_tests", "passed", "failed", "pass_rate_pct",
                "suite_health_index", "num_root_causes", "mttr_hours_saved"):
        assert key in cards
