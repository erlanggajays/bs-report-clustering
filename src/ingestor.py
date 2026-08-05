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
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    PROJECT_DETAIL_ENDPOINT,
    PROJECTS_ENDPOINT,
    SESSION_APPIUM_LOGS_ENDPOINT,
    SESSION_CRASH_LOGS_ENDPOINT,
    SESSION_DEVICE_LOGS_ENDPOINT,
    SESSION_LOGS_ENDPOINT,
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
    "os_version",
    "device",
    "app_version",
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
        rows.append(
            {
                "session_id": s.get("hashed_id") or s.get("session_id", ""),
                "name": (s.get("name") or "").strip(),
                "status": (s.get("status") or "unknown").lower(),
                "reason": (s.get("reason") or "").strip(),
                "duration": float(s.get("duration") or 0.0),
                "os_version": str(s.get("os_version") or "unknown"),
                "device": s.get("device") or "unknown",
                "app_version": str(app.get("app_version") or app.get("app_filename") or ""),
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
_SESSION: requests.Session | None = None


def _http() -> requests.Session:
    """A shared requests session with retry/backoff on 429 and 5xx."""
    global _SESSION
    if _SESSION is None:
        retry = Retry(
            total=settings.http_max_retries,
            backoff_factor=settings.http_backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        sess = requests.Session()
        adapter = HTTPAdapter(max_retries=retry)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _SESSION = sess
    return _SESSION


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


def _extract_error(log_text: str, max_lines: int = 15, max_chars: int = 2000) -> str:
    """Pull the most error-like slice out of a raw terminal log."""
    lines = [ln for ln in log_text.splitlines() if ln.strip()]
    hits = [ln for ln in lines if _ERROR_LINE.search(ln)]
    chosen = hits[:max_lines] if hits else lines[-max_lines:]
    return "\n".join(chosen)[:max_chars]


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
            parts.append(f"--- {src} ---\n{text[-6000:]}")

    if not parts and terminal_url:
        try:
            resp = requests.get(terminal_url, timeout=settings.request_timeout_seconds)
            resp.raise_for_status()
            if resp.text.strip():
                parts.append(resp.text[-6000:])
        except requests.RequestException:
            pass

    return "\n".join(parts), crash_present


def enrich_failure_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: fetch each failed session's logs to (1) recover the real stack
    trace into ``reason`` for clustering and (2) store the log tail in ``log_text``
    for the taxonomy classifier. A present crashlog injects CRASH_MARKER so the
    classifier can tag it as an app crash. Network failures are swallowed per
    session so one bad fetch never aborts the run.
    """
    for idx in df.index[df["is_failure"]]:
        combined, crash = _fetch_session_logs(
            df.at[idx, "build_id"], df.at[idx, "session_id"], df.at[idx, "terminal_logs_url"]
        )
        if not combined:
            continue
        marker = f"{CRASH_MARKER}\n" if crash else ""
        df.at[idx, "log_text"] = marker + combined[-8000:]  # marker kept at front
        snippet = _extract_error(combined)
        if snippet:
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
        # Resolve the project id (env-configured id short-circuits the lookup).
        if settings.target_project_id:
            project_id = int(settings.target_project_id)
        else:
            project_id = get_project_id_by_name(project)

        build_meta = get_latest_build_for_project(project_id)
        build_meta.setdefault("project", project)
        build_meta.setdefault(
            "user_name", username or _owner_from_build_name(build_meta.get("name", ""))
        )
        # Sessions are keyed by the build's hashed_id.
        build_id = build_meta.get("hashed_id") or build_meta.get("build_id")
        if not build_id:
            raise IngestionError("Resolved build has no hashed_id; cannot fetch sessions.")
        sessions = _fetch_sessions_for_build(build_id)

    else:
        raise ValueError(f"Unknown source '{source}'. Use 'file' or 'api'.")

    df = _sessions_to_dataframe(sessions)
    if df.empty:
        raise IngestionError("Build contained zero sessions.")

    # Optional: recover real stack traces from terminal logs (API source only).
    do_enrich = settings.enrich_failure_logs if enrich_logs is None else enrich_logs
    if do_enrich and source == "api":
        df = enrich_failure_reasons(df)

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

    if source == "api":
        project_id = (
            int(settings.target_project_id)
            if settings.target_project_id
            else get_project_id_by_name(project)
        )
        builds = get_recent_builds_for_project(project_id, last_n)
        frames: list[pd.DataFrame] = []
        for b in builds:
            b.setdefault("project", project)
            b.setdefault("user_name", username)
            bid = b.get("hashed_id") or b.get("build_id")
            if not bid:
                continue
            df = _sessions_to_dataframe(_fetch_sessions_for_build(bid))
            if do_enrich:
                df = enrich_failure_reasons(df)
            if df.empty:
                continue
            df["build_id"] = bid
            history.persist_build(b, df, db_path=settings.history_db_path)
            frames.append(df)
            logger.info("  + build %s… (%d sessions)", bid[:12], len(df))
        if not frames:
            raise IngestionError("No sessions found across the selected builds.")
        combined = pd.concat(frames, ignore_index=True)

    elif source == "file":
        combined = history.load_recent_sessions(
            project, builds_window=last_n, db_path=settings.sample_history_db_path
        )
        if combined is None or combined.empty:
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
        "project": project,
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
