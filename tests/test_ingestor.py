"""Tests for ingestion: parsing, token scrubbing, project/build resolution, pagination."""
from __future__ import annotations

import pytest

import ingestor
from ingestor import (
    _looks_like_crash,
    _strip_token,
    get_project_id_by_name,
    get_recent_builds_for_project,
    IngestionError,
)


def test_extract_error_finds_buried_failure_not_teardown_noise():
    """Regression: Appium logs a lot of ADB/proxy noise *after* a failure. The
    extractor must return the real error, not the bookkeeping that follows it."""
    from ingestor import _extract_error

    noise = ("2026-08-05 03:48:18 - [x][ADB] Running '/usr/local/.browserstack/android-sdk/"
             "platform-tools/adb -P 5037 -s RZCWC0TCQMP shell dumpsys window displays'")
    error = ("2026-08-05 03:48:25 - [x][W3C] Encountered internal error running command: "
             "NoSuchElementException: An element could not be located on the page")
    proxy = ('2026-08-05 03:48:30 - [x] Proxying [POST /element] to [POST http://127.0.0.1:8201/'
             'session/abc/element] with body: {"strategy":"xpath","selector":"//View"}')
    raw = "\n".join([noise] * 20 + [error] + [proxy, noise] * 60)

    extracted = _extract_error(raw)
    assert "NoSuchElementException" in extracted
    assert "dumpsys" not in extracted          # noise excluded
    assert "Proxying" not in extracted


def test_extract_error_prefers_most_specific_signal():
    from ingestor import _extract_error

    raw = "\n".join([
        "some generic Error: whatever",
        "Caused by: java.net.SocketTimeoutException: read timed out",
        "2026-08-05 - [ADB] Getting focused package and activity",
    ])
    assert "SocketTimeoutException" in _extract_error(raw)


def test_enrichment_is_parallel_and_fault_tolerant():
    """Log enrichment is network-bound; a range run is hundreds of round-trips, so
    fetches run concurrently. One failing session must not abort the rest."""
    import time

    from ingestor import _sessions_to_dataframe, enrich_failure_reasons

    raw = [
        {"hashed_id": f"s{i}", "name": f"t{i}", "status": "failed", "duration": 30,
         "os_version": "14.0", "device": "S24", "reason": "TIMEOUT",
         "created_at": "t", "build_hashed_id": "b1"}
        for i in range(16)
    ]
    df = _sessions_to_dataframe(raw)

    def fake(build_id, session_id, terminal_url):
        time.sleep(0.05)
        if session_id == "s5":
            raise RuntimeError("simulated fetch failure")
        return f"[W3C] NoSuchElementError: could not locate element {session_id}", False

    original = ingestor._fetch_session_logs
    ingestor._fetch_session_logs = fake
    try:
        started = time.time()
        out = enrich_failure_reasons(df.copy())
        elapsed = time.time() - started
    finally:
        ingestor._fetch_session_logs = original

    sequential = 16 * 0.05
    assert elapsed < sequential / 2, f"not parallel: {elapsed:.2f}s vs {sequential:.2f}s"
    assert out["reason"].str.contains("NoSuchElementError").sum() == 15
    # The session that raised keeps its original reason rather than losing data.
    assert out.loc[out.session_id == "s5", "reason"].iloc[0] == "TIMEOUT"


def test_enrichment_never_overwrites_a_usable_reason():
    """Regression: enrichment replaced BrowserStack's own verdict ("Element not
    found", "Header value cannot be null") with Appium log chatter, leaving no error
    signal — so real, diagnosable failures were filed as "no diagnostic logs"."""
    from ingestor import _sessions_to_dataframe, enrich_failure_reasons
    from taxonomy import classify

    raw = [
        {"hashed_id": "keep1", "name": "expenseChart", "status": "failed", "duration": 90,
         "os_version": "14.0", "device": "S24", "created_at": "t", "build_hashed_id": "b1",
         "reason": "Element not found"},
        {"hashed_id": "keep2", "name": "diraWidget", "status": "failed", "duration": 90,
         "os_version": "14.0", "device": "S24", "created_at": "t", "build_hashed_id": "b1",
         "reason": "No element found with text/description/locator matching 'Show more insights'"},
        {"hashed_id": "keep3", "name": "addAccounts", "status": "failed", "duration": 90,
         "os_version": "14.0", "device": "S24", "created_at": "t", "build_hashed_id": "b1",
         "reason": "Header value cannot be null"},
        # An uninformative reason SHOULD be filled in from the log.
        {"hashed_id": "fill", "name": "other", "status": "failed", "duration": 90,
         "os_version": "14.0", "device": "S24", "created_at": "t", "build_hashed_id": "b1",
         "reason": "TIMEOUT"},
    ]
    df = _sessions_to_dataframe(raw)
    noise = "2026-08-05 04:51:22 - [abc][HTTP] <-- POST /wd/hub/session/x/element"
    useful = "2026-08-05 04:51:25 - [abc][W3C] NoSuchElementError: could not be located"

    def fake(build_id, session_id, terminal_url):
        return (useful if session_id == "fill" else noise), False

    original = ingestor._fetch_session_logs
    ingestor._fetch_session_logs = fake
    try:
        out = enrich_failure_reasons(df.copy())
    finally:
        ingestor._fetch_session_logs = original

    by_id = out.set_index("session_id")["reason"].to_dict()
    assert by_id["keep1"] == "Element not found"
    assert "Show more insights" in by_id["keep2"]
    assert by_id["keep3"] == "Header value cannot be null"
    assert "NoSuchElementError" in by_id["fill"]

    # And they land in the right categories rather than "no-diagnostic-logs".
    cats = [classify(r.reason, r.log_text, r.duration, r.status)[0] for r in out.itertuples()]
    assert cats == ["element-not-found", "element-not-found",
                    "missing-auth-header", "element-not-found"]


def test_looks_like_crash():
    assert _looks_like_crash("FATAL EXCEPTION: main\n\tat com.gopay.Foo.bar(Foo.java:1)") is True
    assert _looks_like_crash("No crashes were detected for this session.") is False
    assert _looks_like_crash("") is False
    assert _looks_like_crash("   ") is False


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
