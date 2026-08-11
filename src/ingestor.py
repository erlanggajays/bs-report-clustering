"""Dynamic data ingestion from BrowserStack (or a local mock payload).

The ingestor is source-agnostic: it can hit the live BrowserStack REST API
(credentials from the environment) or load an identically-shaped mock JSON file
produced by ``scripts/generate_mock_data.py``. Both paths yield the same
normalized Pandas DataFrame so the rest of the pipeline never knows the
difference.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
from profiling import PROFILE_COLUMNS, parse_profile
from taxonomy import HAS_ERROR_SIGNAL
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    PROJECT_DETAIL_ENDPOINT,
    PROJECTS_ENDPOINT,
    SESSION_APPIUM_LOGS_ENDPOINT,
    SESSION_CRASH_LOGS_ENDPOINT,
    SESSION_DEVICE_LOGS_ENDPOINT,
    SESSION_LOGS_ENDPOINT,
    SESSION_PROFILING_ENDPOINT,
    SESSIONS_ENDPOINT,
    settings,
)

# Log-source name -> endpoint template. Enrichment fetches settings.log_sources.
_LOG_ENDPOINTS = {
    "crash": SESSION_CRASH_LOGS_ENDPOINT,
    "appium": SESSION_APPIUM_LOGS_ENDPOINT,
    "device": SESSION_DEVICE_LOGS_ENDPOINT,
    "text": SESSION_LOGS_ENDPOINT,
}
# Injected into log_text when a crashlog is present, so a taxonomy rule can match
# it deterministically ("presence of crashlog => app crashed").
CRASH_MARKER = "BROWSERSTACK_CRASHLOG_PRESENT"
# Per-source cap when fetching logs. Generous on purpose: the failure must stay
# inside the window for _extract_error to find it (see _fetch_session_logs).
_LOG_FETCH_CAP = 200_000
# How much of the log to persist on the row for the taxonomy classifier.
_LOG_STORE_CHARS = 8_000

logger = logging.getLogger(__name__)


def _owner_from_build_name(name: str) -> str:
    """Derive a display owner from a BrowserStack build name.

    Build names look like "erlangga.jaya triggered from Mac OS X at ...";
    we take the leading token before " triggered". Empty if not parseable.
    """
    if not name:
        return ""
    head = name.split(" triggered", 1)[0].strip()
    # Only accept a short, name-like token (avoid returning the whole string).
    return head if head and len(head) <= 40 and " " not in head else ""

# Canonical schema every downstream module can rely on.
COLUMNS = [
    "session_id",
    "name",
    "status",
    "reason",
    "duration",
    "platform",          # android / ios — free from the session payload's "os"
    "project",           # owning BrowserStack project, for multi-project runs
    "os_version",
    "device",
    "app_version",
    "app_filename",      # kept separate: mixing it into app_version broke version analysis
    "browserstack_seconds",  # from insights: platform overhead
    "user_seconds",          # from insights: time spent in the test itself
    "session_url",       # token-free deep link to the BrowserStack session
    "logs",
    "created_at",
    "build_id",          # build hashed_id, for constructing the /logs endpoint
    "log_text",          # enriched Appium log text (for taxonomy + clustering)
    "terminal_logs_url",  # signed URL, kept in-memory for enrichment fallback
]


class IngestionError(RuntimeError):
    """Raised when a build for the target user cannot be located or parsed."""


def _strip_token(url: str) -> str:
    """Drop the query string so signed tokens/auth params never get persisted."""
    if not url:
        return ""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _sessions_to_dataframe(sessions: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize a list of raw session dicts into the canonical DataFrame.

    Security: ``browser_url``/``public_url``/``video_url`` carry auth tokens; we
    only keep the token-free ``browser_url`` path as ``session_url`` for report
    deep-links. ``public_url``/``video_url`` are deliberately discarded.
    """
    rows = []
    for s in sessions:
        app = s.get("app_details") or {}
        # Prefer the token-free browser_url; strip query params defensively.
        session_url = _strip_token(s.get("browser_url") or s.get("public_url", ""))
        # BrowserStack "insights" splits the runtime into platform vs user time.
        totals = (
            ((s.get("insights") or {}).get("summary") or {}).get("totals") or {}
        )
        rows.append(
            {
                "session_id": s.get("hashed_id") or s.get("session_id", ""),
                "name": (s.get("name") or "").strip(),
                "status": (s.get("status") or "unknown").lower(),
                "reason": (s.get("reason") or "").strip(),
                "duration": float(s.get("duration") or 0.0),
                "platform": str(s.get("os") or "unknown").lower(),
                "project": str(s.get("project_name") or ""),
                "os_version": str(s.get("os_version") or "unknown"),
                "device": s.get("device") or "unknown",
                # Keep these distinct — falling back to the filename previously
                # polluted app_version and made version correlation meaningless.
                "app_version": str(app.get("app_version") or ""),
                "app_filename": str(app.get("app_filename") or ""),
                "browserstack_seconds": float(totals.get("browserstack_time") or 0.0),
                "user_seconds": float(totals.get("user_time") or 0.0),
                "session_url": session_url,
                "logs": _strip_token(s.get("logs", "")),
                "created_at": s.get("created_at", ""),
                "build_id": s.get("build_hashed_id", ""),
                "log_text": "",  # populated by enrichment when --enrich-logs is set
                "terminal_logs_url": s.get("session_terminal_logs_url", ""),
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS)
    # Normalize BrowserStack's "error"/"timeout" states into a binary view while
    # preserving the raw status for reporting.
    df["is_failure"] = df["status"].isin({"failed", "error", "timeout"})
    return df


# --- Live API path ----------------------------------------------------------
_LOCAL = threading.local()


def _http() -> requests.Session:
    """A per-thread requests session with retry/backoff on 429 and 5xx.

    Thread-local because enrichment fetches concurrently and ``requests.Session``
    is not guaranteed thread-safe. The connection pool is sized to the worker count
    so concurrent requests do not thrash a too-small pool.
    """
    sess = getattr(_LOCAL, "session", None)
    if sess is None:
        retry = Retry(
            total=settings.http_max_retries,
            backoff_factor=settings.http_backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        sess = requests.Session()
        pool = max(10, settings.enrich_workers)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _LOCAL.session = sess
    return sess


def _get(url: str, params: dict[str, Any] | None = None) -> Any:
    if not settings.has_credentials:
        raise IngestionError(
            "BrowserStack credentials missing. Set BROWSERSTACK_USERNAME and "
            "BROWSERSTACK_ACCESS_KEY, or use source='file' with mock data."
        )
    resp = _http().get(
        url,
        params=params,
        auth=settings.auth,
        timeout=settings.request_timeout_seconds,
    )
    resp.raise_for_status()
    return resp.json()


def _get_text(url: str) -> str:
    """GET a plain-text resource (e.g. the session /logs endpoint)."""
    if not settings.has_credentials:
        raise IngestionError("BrowserStack credentials missing.")
    resp = _http().get(url, auth=settings.auth, timeout=settings.request_timeout_seconds)
    resp.raise_for_status()
    return resp.text


def get_project_id_by_name(project_name: str) -> int:
    """Resolve a BrowserStack project name to its numeric id.

    ``projects.json`` returns a flat list of project dicts, e.g.
    ``{"id": 2256311, "name": "Finserv - Gopay Android", ...}``.
    """
    payload = _get(PROJECTS_ENDPOINT)
    projects = [item.get("project", item) for item in payload]  # tolerate nesting
    target = project_name.strip().lower()

    for proj in projects:
        if str(proj.get("name", "")).strip().lower() == target:
            return int(proj["id"])

    available = ", ".join(sorted(str(p.get("name", "?")) for p in projects))
    raise IngestionError(
        f"Project '{project_name}' not found. Available projects: {available}"
    )


def get_latest_build_for_project(project_id: int) -> dict[str, Any]:
    """Return the newest eligible build within a project.

    ``projects/{id}.json`` returns the project with a nested ``builds`` array.
    We keep only builds whose status is in ``settings.build_status_filter``
    (default ``("done",)``) and return the newest by ``created_at``. If none
    match, we fall back to the newest build overall so the run still proceeds.
    """
    payload = _get(PROJECT_DETAIL_ENDPOINT.format(project_id=project_id))
    project = payload.get("project", payload)
    builds = project.get("builds", [])
    if not builds:
        raise IngestionError(f"Project {project_id} has no builds.")

    eligible = [
        b for b in builds
        if str(b.get("status", "")).lower() in settings.build_status_filter
    ]
    pool = eligible or builds
    pool.sort(key=lambda b: b.get("created_at", ""), reverse=True)
    chosen = pool[0]
    if not eligible:
        logger.warning(
            "No build with status in %s; falling back to newest (status=%s).",
            settings.build_status_filter, chosen.get("status"),
        )
    return chosen


def as_project_list(project: str | list[str] | None) -> list[str]:
    """Normalise the --project input into a list of project names.

    Accepts a single name, a comma-separated string, or a list (repeated flags).
    """
    if project is None:
        project = settings.target_project
    if isinstance(project, str):
        project = project.split(",")
    return [p.strip() for p in project if str(p).strip()]


def _resolve_project_id(name: str) -> int:
    """Project name -> id, honouring TARGET_PROJECT_ID for a single-project run."""
    if settings.target_project_id and len(as_project_list(settings.target_project)) == 1:
        return int(settings.target_project_id)
    return get_project_id_by_name(name)


def _combined_meta(metas: list[dict[str, Any]], projects: list[str]) -> dict[str, Any]:
    """Fold several per-project build metas into one report header."""
    if len(metas) == 1:
        return metas[0]
    return {
        "project": " + ".join(projects),
        "projects": projects,
        "is_multi_project": True,
        "name": f"{len(metas)} projects",
        "hashed_id": ", ".join(str(m.get("hashed_id", ""))[:8] for m in metas),
        "status": "combined",
        "created_at": max((m.get("created_at", "") for m in metas), default=""),
        "user_name": next((m.get("user_name", "") for m in metas if m.get("user_name")), ""),
    }


def get_recent_builds_for_project(project_id: int, last_n: int) -> list[dict[str, Any]]:
    """Return the most recent ``last_n`` eligible builds (newest first, capped)."""
    payload = _get(PROJECT_DETAIL_ENDPOINT.format(project_id=project_id))
    project = payload.get("project", payload)
    builds = project.get("builds", [])
    if not builds:
        raise IngestionError(f"Project {project_id} has no builds.")
    eligible = [
        b for b in builds
        if str(b.get("status", "")).lower() in settings.build_status_filter
    ] or builds
    eligible.sort(key=lambda b: b.get("created_at", ""), reverse=True)
    cap = max(1, min(last_n, settings.max_range_builds))
    return eligible[:cap]


def _fetch_sessions_for_build(build_id: str) -> list[dict[str, Any]]:
    """Fetch all sessions for a build, paginating via limit/offset.

    Guards against an API that ignores pagination params (which would loop
    forever) by stopping as soon as a page yields no *new* session ids, and by
    an absolute page cap.
    """
    url = SESSIONS_ENDPOINT.format(build_id=build_id)
    limit = settings.sessions_page_size
    offset = 0
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for _ in range(settings.max_session_pages):
        payload = _get(url, params={"limit": limit, "offset": offset})
        page = [item.get("automation_session", item) for item in payload]
        if not page:
            break
        new = [s for s in page if (s.get("hashed_id") or id(s)) not in seen]
        for s in new:
            seen.add(s.get("hashed_id") or id(s))
        collected.extend(new)
        # Stop when the page is short (last page) or brought nothing new
        # (API ignored offset -> avoid infinite loop).
        if len(page) < limit or not new:
            break
        offset += limit

    return collected


_ERROR_LINE = re.compile(r"(Exception|Error|assert|FAIL|Traceback|Caused by)", re.I)

# Appium/BrowserStack logs are mostly bookkeeping. These lines carry no failure
# information and would otherwise be mistaken for the error.
_LOG_NOISE = re.compile(
    r"Proxying \[|Got response with status 200|Responding to client|Running '/usr/local"
    r"|^\s*REQUEST \[|Plugin \w+ is now handling|jdwp-control|dumpsys"
    r"|\[ADB\] Getting focused|Executing default handling|Waiting up to|Matched '/"
    # HTTP wire traffic and device-log hex dumps carry no diagnostic value and
    # previously became the "error", producing one junk cluster per session.
    r"|\[HTTP\] (?:<--|-->)|/wd/hub|^0{8,}:\s|\[HTTP\] Request idempotency",
    re.I | re.M,
)

# Ordered by specificity: the first group with a hit wins, and within it we take
# the LAST occurrence, because the failure that ended the test is at the end.
_ERROR_SIGNALS = [
    re.compile(r"FATAL EXCEPTION|\bANR\b|SIGSEGV|signal 11", re.I),
    re.compile(r"Caused by:", re.I),
    re.compile(r"NoSuchElementException|no such element|could not be located"
               r"|Element not found|can(?:no|')?t locate an element", re.I),
    re.compile(r"AssertionError|AssertionFailedError|expected .*but was", re.I),
    re.compile(r"SocketTimeoutException|ECONNRESET|Connection refused"
               r"|internal server error|read timed out", re.I),
    re.compile(r"EADDRINUSE|Could not configure Appium server", re.I),
    re.compile(r"is still running after \d+ms", re.I),
    re.compile(r"Encountered internal error", re.I),
    re.compile(r"Could not start a new session|session not created", re.I),
    re.compile(r"\b\w*(?:Exception|Error)\b:", re.I),
]


def _extract_error(log_text: str, context_lines: int = 6, max_chars: int = 2000) -> str:
    """Pull the actual failure out of a raw Appium/terminal log.

    Strategy: drop bookkeeping noise, then find the most *specific* error signal
    (ordered list) and take its last occurrence plus a little following context.
    Searching from the end matters — the failure that ended the test is there,
    while the top of the log is session setup.
    """
    lines = [ln for ln in (log_text or "").splitlines() if ln.strip()]
    signal_lines = [ln for ln in lines if not _LOG_NOISE.search(ln)]
    if not signal_lines:
        signal_lines = lines

    for pattern in _ERROR_SIGNALS:
        hits = [i for i, ln in enumerate(signal_lines) if pattern.search(ln)]
        if hits:
            start = hits[-1]
            chosen = signal_lines[start : start + context_lines]
            return "\n".join(chosen)[:max_chars]

    # Nothing recognisable: the tail is still the best guess.
    return "\n".join(signal_lines[-context_lines:])[:max_chars]


def _looks_like_crash(text: str) -> bool:
    """A crashlog endpoint returns content only when the app actually crashed;
    guard against empty / "no crash" placeholder responses."""
    t = (text or "").strip()
    if len(t) < 20:
        return False
    low = t.lower()
    return not any(x in low for x in ("no crash", "not available", "no data", "no logs found"))


def _fetch_session_logs(build_id: str, session_id: str, terminal_url: str) -> tuple[str, bool]:
    """Fetch the configured log sources for a failed session.

    Returns (combined_text, crash_present). Each source is best-effort — a failed
    fetch is skipped, not fatal. Falls back to the signed terminal-log S3 URL only
    if no configured API source returned anything.

    The text is kept whole (bounded by ``_LOG_FETCH_CAP``) rather than tail-trimmed:
    Appium emits a lot of ADB/teardown noise *after* a failure, so a small tail
    window would cut the actual error out and leave only bookkeeping.
    """
    parts: list[str] = []
    crash_present = False
    if build_id and session_id:
        for src in settings.log_sources:
            endpoint = _LOG_ENDPOINTS.get(src)
            if not endpoint:
                continue
            try:
                text = _get_text(endpoint.format(build_id=build_id, session_id=session_id))
            except (requests.RequestException, IngestionError):
                continue
            if not text or not text.strip():
                continue
            if src == "crash":
                if not _looks_like_crash(text):
                    continue
                crash_present = True
            parts.append(f"--- {src} ---\n{text[-_LOG_FETCH_CAP:]}")

    if not parts and terminal_url:
        try:
            resp = requests.get(terminal_url, timeout=settings.request_timeout_seconds)
            resp.raise_for_status()
            if resp.text.strip():
                parts.append(resp.text[-_LOG_FETCH_CAP:])
        except requests.RequestException:
            pass

    return "\n".join(parts), crash_present


def fetch_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-session performance figures from the app-profiling endpoint.

    Runs for **every** session, not just failures: a performance regression does not
    fail a test, so passing sessions carry the signal. Fetches concurrently and
    tolerates sessions with profiling disabled (which return nothing).
    """
    targets = [i for i in df.index if df.at[i, "session_id"] and df.at[i, "build_id"]]
    if not targets:
        return df

    def _fetch(idx: int) -> tuple[int, dict]:
        url = SESSION_PROFILING_ENDPOINT.format(
            build_id=df.at[idx, "build_id"], session_id=df.at[idx, "session_id"]
        )
        try:
            payload = _get(url)
        except (requests.RequestException, IngestionError, ValueError):
            return idx, {}
        if not isinstance(payload, list):
            return idx, {}
        return idx, parse_profile(payload)

    workers = min(settings.enrich_workers, len(targets))
    logger.info("      profiling %d sessions with %d workers…", len(targets), workers)
    results: list[tuple[int, dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_fetch, i) for i in targets]):
            try:
                results.append(future.result())
            except Exception as exc:
                logger.debug("profiling fetch failed: %s", exc)

    for col in PROFILE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    got = 0
    for idx, metrics in results:
        if not metrics:
            continue
        got += 1
        for col, value in metrics.items():
            df.at[idx, col] = value
    logger.info("      profiling data returned for %d/%d sessions", got, len(targets))
    return df


def enrich_failure_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: fetch each failed session's logs to (1) recover the real stack
    trace into ``reason`` for clustering and (2) store the log tail in ``log_text``
    for the taxonomy classifier. A present crashlog injects CRASH_MARKER so the
    classifier can tag it as an app crash. Network failures are swallowed per
    session so one bad fetch never aborts the run.

    Fetches run concurrently because the work is purely network-bound — a range
    run over several builds is otherwise hundreds of sequential round-trips. Only
    the fetching is threaded: results are applied to the DataFrame afterwards on
    this thread, since pandas is not safe for concurrent writes.
    """
    targets = list(df.index[df["is_failure"]])
    if not targets:
        return df

    def _fetch(idx: int) -> tuple[int, str, bool]:
        combined, crash = _fetch_session_logs(
            df.at[idx, "build_id"], df.at[idx, "session_id"], df.at[idx, "terminal_logs_url"]
        )
        return idx, combined, crash

    results: list[tuple[int, str, bool]] = []
    workers = min(settings.enrich_workers, len(targets))
    logger.info("      enriching %d failed sessions with %d workers…", len(targets), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, i) for i in targets]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as exc:                    # one bad session is not fatal
                logger.debug("log fetch failed: %s", exc)
            if done % 25 == 0:
                logger.debug("  fetched logs for %d/%d sessions", done, len(targets))

    for idx, combined, crash in results:
        if not combined:
            continue
        marker = f"{CRASH_MARKER}\n" if crash else ""
        # Extract from the WHOLE log, but persist only a bounded tail. The error
        # snippet is prepended so the classifier still sees it even if the stored
        # tail is all teardown noise.
        snippet = _extract_error(combined)
        df.at[idx, "log_text"] = marker + snippet + "\n" + combined[-_LOG_STORE_CHARS:]

        # Enrichment must AUGMENT, never replace a usable reason. BrowserStack often
        # already carries the framework's own verdict ("Element not found", "Header
        # value cannot be null"), which is a better cluster key and classification
        # input than anything extracted from Appium chatter. Overwriting it
        # unconditionally destroyed that text and pushed real, diagnosable failures
        # into "no diagnostic logs". Only fill in when the original says nothing
        # useful (empty, "COMPLETED", "TIMEOUT", a "Multiple Failures" wrapper).
        original = str(df.at[idx, "reason"] or "")
        if snippet and not HAS_ERROR_SIGNAL.search(original):
            df.at[idx, "reason"] = snippet
    return df


# --- Public entry point -----------------------------------------------------
def ingest(
    username: str | None = None,
    source: str = "file",
    mock_path: str | Path | None = None,
    project: str | None = None,
    enrich_logs: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest the latest build for a project and return (sessions_df, build_meta).

    Args:
        username: Target user, used for report display only.
        source:   "file" to load mock JSON, "api" to query BrowserStack live.
        mock_path: Path to the mock JSON when ``source='file'``.
        project:  Target project name; defaults to ``settings.target_project``.
    """
    username = username or settings.target_user
    project = project or settings.target_project

    if source == "file":
        path = Path(
            mock_path
            or Path(__file__).resolve().parents[1] / "data" / "mock_browserstack_build.json"
        )
        if not path.exists():
            raise IngestionError(
                f"Mock file not found: {path}. Run scripts/generate_mock_data.py first."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        build_meta = payload["build"]
        sessions = payload["sessions"]

    elif source == "api":
        # One latest build per requested project, then combine.
        projects = as_project_list(project)
        frames: list[pd.DataFrame] = []
        metas: list[dict[str, Any]] = []
        for name in projects:
            meta = get_latest_build_for_project(_resolve_project_id(name))
            meta.setdefault("project", name)
            meta.setdefault(
                "user_name", username or _owner_from_build_name(meta.get("name", ""))
            )
            build_id = meta.get("hashed_id") or meta.get("build_id")
            if not build_id:
                raise IngestionError(f"Build for '{name}' has no hashed_id.")
            part = _sessions_to_dataframe(_fetch_sessions_for_build(build_id))
            if part.empty:
                logger.warning("Project '%s' latest build had no sessions; skipping.", name)
                continue
            part["build_id"] = build_id
            part.loc[part["project"] == "", "project"] = name
            frames.append(part)
            metas.append(meta)
        if not frames:
            raise IngestionError("No sessions found for the requested project(s).")
        df = pd.concat(frames, ignore_index=True)
        build_meta = _combined_meta(metas, projects)
        sessions = []  # already normalised above

    else:
        raise ValueError(f"Unknown source '{source}'. Use 'file' or 'api'.")

    if source != "api":
        df = _sessions_to_dataframe(sessions)
    if df.empty:
        raise IngestionError("Build contained zero sessions.")

    # Optional: recover real stack traces from terminal logs (API source only).
    do_enrich = settings.enrich_failure_logs if enrich_logs is None else enrich_logs
    if do_enrich and source == "api":
        df = enrich_failure_reasons(df)
    if settings.enrich_profiling and source == "api":
        df = fetch_profiles(df)

    return df, build_meta


def ingest_range(
    username: str | None = None,
    source: str = "api",
    project: str | None = None,
    last_n: int | None = None,
    enrich_logs: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest the last ``last_n`` builds and return (combined_df, window_meta).

    - source='api':  walks recent builds, fetches+persists each build's sessions,
      then aggregates them (complete data, even on a fresh DB).
    - source='file': aggregates from the seeded sample history DB (local testing;
      note: token-free session links are not stored, so cluster replay links are
      unavailable in this path).
    """
    import history  # local import: history is a lower-level storage module

    username = username or settings.target_user
    project = project or settings.target_project
    last_n = last_n or settings.default_last_n
    do_enrich = settings.enrich_failure_logs if enrich_logs is None else enrich_logs

    projects = as_project_list(project)

    if source == "api":
        frames: list[pd.DataFrame] = []
        for name in projects:
            builds = get_recent_builds_for_project(_resolve_project_id(name), last_n)
            logger.info("  project '%s': %d builds", name, len(builds))
            for b in builds:
                b.setdefault("project", name)
                b.setdefault("user_name", username)
                bid = b.get("hashed_id") or b.get("build_id")
                if not bid:
                    continue
                part = _sessions_to_dataframe(_fetch_sessions_for_build(bid))
                if do_enrich:
                    part = enrich_failure_reasons(part)
                if settings.enrich_profiling:
                    part = fetch_profiles(part)
                if part.empty:
                    continue
                part["build_id"] = bid
                part.loc[part["project"] == "", "project"] = name
                history.persist_build(b, part, db_path=settings.history_db_path)
                frames.append(part)
                logger.info("    + build %s… (%d sessions)", bid[:12], len(part))
        if not frames:
            raise IngestionError("No sessions found across the selected builds.")
        combined = pd.concat(frames, ignore_index=True)

    elif source == "file":
        parts = [
            history.load_recent_sessions(
                p, builds_window=last_n, db_path=settings.sample_history_db_path
            )
            for p in projects
        ]
        parts = [p for p in parts if p is not None and not p.empty]
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if combined.empty:
            raise IngestionError(
                "No stored history. Seed it first: "
                "python scripts/generate_mock_data.py --seed-history 8"
            )
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'file' or 'api'.")

    if combined.empty:
        raise IngestionError("Range contained zero sessions.")

    n_builds = int(combined["build_id"].nunique())
    dates = sorted(d for d in combined["created_at"].tolist() if d)
    date_from = dates[0][:10] if dates else ""
    date_to = dates[-1][:10] if dates else ""
    window_meta = {
        "project": " + ".join(projects),
        "projects": projects,
        "is_multi_project": len(projects) > 1,
        "user_name": username,
        "name": f"Last {n_builds} builds",
        "hashed_id": f"{n_builds} builds · {date_from} → {date_to}",
        "status": "range",
        "created_at": date_to,
        "is_range": True,
        "n_builds": n_builds,
        "date_from": date_from,
        "date_to": date_to,
    }
    return combined, window_meta


if __name__ == "__main__":
    frame, meta = ingest(source="file")
    print(f"Build '{meta.get('name')}' by {meta.get('user_name')}")
    print(frame[["name", "device", "os_version", "status"]].head(10).to_string())
