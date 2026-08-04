"""Tests for SQLite history persistence."""
from __future__ import annotations

import history


def _build_meta():
    return {
        "hashed_id": "build_abc",
        "project": "Finserv - Gopay Android",
        "name": "erlangga.jaya triggered from Mac OS X",
        "status": "done",
        "created_at": "2026-08-04T11:00:00.000Z",
    }


def test_persist_and_load_roundtrip(tmp_path, sample_df):
    db = tmp_path / "hist.db"
    history.persist_build(_build_meta(), sample_df, db_path=db)

    loaded = history.load_recent_sessions("Finserv - Gopay Android", db_path=db)
    assert len(loaded) == 5
    assert loaded["is_failure"].dtype == bool
    assert int(loaded["is_failure"].sum()) == 4
    assert set(loaded["session_id"]) == {"s1", "s2", "s3", "s4", "s5"}


def test_persist_is_idempotent(tmp_path, sample_df):
    db = tmp_path / "hist.db"
    history.persist_build(_build_meta(), sample_df, db_path=db)
    history.persist_build(_build_meta(), sample_df, db_path=db)  # re-run
    loaded = history.load_recent_sessions("Finserv - Gopay Android", db_path=db)
    assert len(loaded) == 5  # INSERT OR REPLACE, no duplicates


def test_suite_health_trend(tmp_path, sample_df):
    db = tmp_path / "hist.db"
    history.persist_build(_build_meta(), sample_df, db_path=db)
    trend = history.suite_health_trend("Finserv - Gopay Android", db_path=db)
    assert len(trend) == 1
    assert trend.iloc[0]["total"] == 5
    assert trend.iloc[0]["failed"] == 4
    assert abs(trend.iloc[0]["pass_rate"] - 0.2) < 1e-9
