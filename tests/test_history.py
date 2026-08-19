"""Tests for SQLite history persistence."""
from __future__ import annotations

import pandas as pd

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


# --- multi-project labels and empty builds ---------------------------------
def test_project_names_splits_a_combined_label():
    from history import project_names
    assert project_names("A + B") == ["A", "B"]
    assert project_names(["X", "Y"]) == ["X", "Y"]
    assert project_names("Solo") == ["Solo"]
    assert project_names("") == []


def test_trend_excludes_builds_with_no_sessions(tmp_path, sample_df):
    """A build row left with zero sessions has no pass rate. Treating it as 0%
    drew a phantom collapse to zero on the trend chart."""
    db = tmp_path / "h.db"
    history.persist_build(
        {"hashed_id": "real", "project": "P", "name": "real",
         "created_at": "2026-08-01T00:00:00.000Z"}, sample_df, db_path=db)
    # A build with no sessions at all, as left behind by an earlier bug.
    history.persist_build(
        {"hashed_id": "empty", "project": "P", "name": "empty",
         "created_at": "2026-08-02T00:00:00.000Z"},
        sample_df.iloc[0:0], db_path=db)
    trend = history.suite_health_trend("P", db_path=db)
    assert list(trend["build_id"]) == ["real"]
    assert 0.0 < float(trend["pass_rate"].iloc[0]) <= 1.0


def _two_project_db(tmp_path, sample_df):
    """Two builds in two projects, each with its OWN session ids.

    Distinct ids matter: session_id is the primary key, so persisting the same
    sessions under a second build re-parents them away from the first.
    """
    db = tmp_path / "h.db"
    a = sample_df.copy()
    b = sample_df.copy()
    b["session_id"] = b["session_id"] + "-b"
    history.persist_build({"hashed_id": "a1", "project": "Proj A", "name": "a",
                           "created_at": "2026-08-01T00:00:00.000Z"}, a, db_path=db)
    history.persist_build({"hashed_id": "b1", "project": "Proj B", "name": "b",
                           "created_at": "2026-08-02T00:00:00.000Z"}, b, db_path=db)
    return db


def test_trend_spans_both_projects_of_a_cross_platform_run(tmp_path, sample_df):
    """Builds are stored per project, so the combined "A + B" label must be split
    or the trend finds nothing that was actually ingested."""
    db = _two_project_db(tmp_path, sample_df)
    trend = history.suite_health_trend("Proj A + Proj B", db_path=db)
    assert list(trend["build_id"]) == ["a1", "b1"]      # oldest -> newest


def test_load_recent_sessions_spans_both_projects(tmp_path, sample_df):
    db = _two_project_db(tmp_path, sample_df)
    rows = history.load_recent_sessions("Proj A + Proj B", db_path=db)
    assert set(rows["build_id"]) == {"a1", "b1"}


def test_reusing_session_ids_reparents_them(tmp_path, sample_df):
    """Documents the primary-key behaviour behind the phantom trend points: the
    same sessions stored under a second build leave the first one empty."""
    db = tmp_path / "h.db"
    for build in ("first", "second"):
        history.persist_build(
            {"hashed_id": build, "project": "P", "name": build,
             "created_at": f"2026-08-0{1 if build == 'first' else 2}T00:00:00.000Z"},
            sample_df, db_path=db)
    trend = history.suite_health_trend("P", db_path=db)
    assert list(trend["build_id"]) == ["second"]   # "first" is now empty, so excluded


def test_latest_mode_persists_each_build_separately(tmp_path, sample_df):
    """The regression: a cross-platform latest run stored the combined frame under
    one synthetic build, re-parenting every session and flattening the trend."""
    from cli import _persist_latest
    db = tmp_path / "h.db"
    a = sample_df.copy(); a["build_id"] = "and1"
    b = sample_df.copy(); b["build_id"] = "ios1"; b["session_id"] = b["session_id"] + "-i"
    combined = pd.concat([a, b], ignore_index=True)
    build_meta = {
        "hashed_id": "SYNTHETIC", "project": "Proj A + Proj B", "name": "2 projects",
        "created_at": "2026-08-02T00:00:00.000Z",
        "builds": [
            {"hashed_id": "and1", "project": "Proj A", "name": "android build",
             "created_at": "2026-08-01T00:00:00.000Z"},
            {"hashed_id": "ios1", "project": "Proj B", "name": "ios build",
             "created_at": "2026-08-02T00:00:00.000Z"},
        ],
    }
    _persist_latest(build_meta, combined, db)
    trend = history.suite_health_trend("Proj A + Proj B", db_path=db)
    assert list(trend["build_id"]) == ["and1", "ios1"]
    assert "SYNTHETIC" not in set(trend["build_id"])


def test_single_project_latest_still_persists_normally(tmp_path, sample_df):
    from cli import _persist_latest
    db = tmp_path / "h.db"
    meta = {"hashed_id": "solo", "project": "P", "name": "b",
            "created_at": "2026-08-01T00:00:00.000Z"}
    _persist_latest(meta, sample_df, db)
    assert list(history.suite_health_trend("P", db_path=db)["build_id"]) == ["solo"]
