"""Tests for ingestion: parsing, token scrubbing, project/build resolution, pagination."""
from __future__ import annotations

import pytest

import ingestor
from ingestor import (
    _strip_token,
    get_project_id_by_name,
    get_recent_builds_for_project,
    IngestionError,
)


def test_parse_basic_fields(sample_df):
    assert len(sample_df) == 5
    assert int(sample_df["is_failure"].sum()) == 4       # 1 passed, 4 failed
    assert set(["session_id", "device", "os_version", "reason", "app_version"]).issubset(sample_df.columns)
    assert sample_df.loc[sample_df.session_id == "s2", "device"].iloc[0] == "Samsung Galaxy S23"
    assert sample_df.loc[sample_df.session_id == "s1", "app_version"].iloc[0] == "6.7.0"


def test_strip_token_removes_query():
    url = "https://x.browserstack.com/a/b?auth_token=SECRET&z=1"
    assert _strip_token(url) == "https://x.browserstack.com/a/b"
    assert _strip_token("") == ""


def test_tokens_never_persisted(sample_df):
    # session_url keeps the token-free path; tokened public_url is discarded.
    s1 = sample_df.loc[sample_df.session_id == "s1"].iloc[0]
    assert s1["session_url"] == "https://app-automate.browserstack.com/builds/b/sessions/s1"
    joined = " ".join(str(v) for v in sample_df["session_url"].tolist() + sample_df["logs"].tolist())
    assert "auth_token" not in joined and "SECRET" not in joined and "X-Amz-Signature" not in joined


def test_get_project_id_by_name(monkeypatch):
    projects = [
        {"id": 2256311, "name": "Finserv - Gopay Android"},
        {"id": 999, "name": "Other Project"},
    ]
    monkeypatch.setattr(ingestor, "_get", lambda url, params=None: projects)
    assert get_project_id_by_name("Finserv - Gopay Android") == 2256311
    assert get_project_id_by_name("finserv - gopay ANDROID") == 2256311  # case-insensitive
    with pytest.raises(IngestionError):
        get_project_id_by_name("Nonexistent")


def test_get_recent_builds_filters_and_orders(monkeypatch):
    detail = {"project": {"id": 1, "name": "P", "builds": [
        {"hashed_id": "new", "status": "done", "created_at": "2026-08-04T11:00:00.000Z"},
        {"hashed_id": "mid", "status": "failed", "created_at": "2026-08-03T11:00:00.000Z"},
        {"hashed_id": "old", "status": "timeout", "created_at": "2026-08-02T11:00:00.000Z"},
        {"hashed_id": "running", "status": "running", "created_at": "2026-08-05T11:00:00.000Z"},
    ]}}
    monkeypatch.setattr(ingestor, "_get", lambda url, params=None: detail)
    builds = get_recent_builds_for_project(1, last_n=2)
    # "running" is excluded by BUILD_STATUS_FILTER; newest eligible first.
    assert [b["hashed_id"] for b in builds] == ["new", "mid"]


def test_fetch_sessions_pagination_dedupes(monkeypatch):
    # Simulate an API that returns a short second page (last page).
    page0 = [{"automation_session": {"hashed_id": f"s{i}"}} for i in range(100)]
    page1 = [{"automation_session": {"hashed_id": f"s{i}"}} for i in range(100, 130)]

    def fake_get(url, params=None):
        return page0 if params["offset"] == 0 else page1

    monkeypatch.setattr(ingestor, "_get", fake_get)
    sessions = ingestor._fetch_sessions_for_build("build123")
    assert len(sessions) == 130
    assert len({s["hashed_id"] for s in sessions}) == 130  # no duplicates


def test_fetch_sessions_stops_when_offset_ignored(monkeypatch):
    # An API ignoring offset would return the same full page forever; the
    # dedupe guard must stop instead of looping.
    same_page = [{"automation_session": {"hashed_id": f"s{i}"}} for i in range(100)]
    monkeypatch.setattr(ingestor, "_get", lambda url, params=None: same_page)
    sessions = ingestor._fetch_sessions_for_build("build123")
    assert len(sessions) == 100
