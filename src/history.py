"""SQLite persistence for cross-build history.

Each pipeline run persists its build + sessions here. Accumulated history powers
true run-over-run flakiness (see ``triage_engine._flakiness_from_history``) and
the suite-health trend chart. Storage is deliberately lightweight — one file,
no server — keeping Phase 1 self-contained.

Security: only non-sensitive fields are stored. Token-bearing URLs are never
persisted (the ingestor strips them before this layer ever sees a row).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS builds (
    build_id    TEXT PRIMARY KEY,
    project     TEXT,
    name        TEXT,
    status      TEXT,
    created_at  TEXT,
    ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    build_id    TEXT,
    project     TEXT,
    name        TEXT,
    status      TEXT,
    is_failure  INTEGER,
    reason      TEXT,
    duration    REAL,
    os_version  TEXT,
    device      TEXT,
    app_version TEXT,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
CREATE INDEX IF NOT EXISTS idx_sessions_build   ON sessions(build_id);
"""

# Columns persisted from the sessions DataFrame (sensitive URLs excluded).
_SESSION_COLS = [
    "session_id", "name", "status", "reason",
    "duration", "os_version", "device", "app_version", "created_at",
]


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or settings.history_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    return conn


def persist_build(
    build_meta: dict[str, Any],
    df: pd.DataFrame,
    db_path: str | Path | None = None,
) -> None:
    """Upsert one build and its sessions. Idempotent on re-runs (INSERT OR REPLACE)."""
    build_id = build_meta.get("hashed_id") or build_meta.get("build_id") or ""
    project = build_meta.get("project", "")
    now = datetime.now(timezone.utc).isoformat()

    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO builds "
            "(build_id, project, name, status, created_at, ingested_at) "
            "VALUES (?,?,?,?,?,?)",
            (build_id, project, build_meta.get("name", ""),
             build_meta.get("status", ""), build_meta.get("created_at", ""), now),
        )
        rows = [
            (
                r["session_id"], build_id, project, r["name"], r["status"],
                int(bool(r["is_failure"])), r["reason"], float(r["duration"]),
                r["os_version"], r["device"], r.get("app_version", ""), r["created_at"],
            )
            for _, r in df.iterrows()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, build_id, project, name, status, is_failure, reason, "
            " duration, os_version, device, app_version, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )


def load_recent_sessions(
    project: str,
    builds_window: int | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return sessions from the most recent N builds of a project (for flakiness).

    Empty frame if the DB has no history yet (e.g. very first run).
    """
    window = builds_window or settings.history_builds_window
    with closing(_connect(db_path)) as conn:
        recent_builds = pd.read_sql_query(
            "SELECT build_id FROM builds WHERE project = ? "
            "ORDER BY created_at DESC LIMIT ?",
            conn, params=(project, window),
        )
        if recent_builds.empty:
            return pd.DataFrame()
        ids = tuple(recent_builds["build_id"].tolist())
        placeholders = ",".join("?" * len(ids))
        sessions = pd.read_sql_query(
            f"SELECT * FROM sessions WHERE build_id IN ({placeholders})",
            conn, params=ids,
        )
    sessions["is_failure"] = sessions["is_failure"].astype(bool)
    return sessions


def suite_health_trend(
    project: str,
    builds_window: int | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Per-build pass rate over time, oldest→newest, for the trend chart."""
    window = builds_window or settings.history_builds_window
    with closing(_connect(db_path)) as conn:
        df = pd.read_sql_query(
            "SELECT b.build_id, b.name, b.created_at, "
            "  COUNT(s.session_id) AS total, "
            "  SUM(s.is_failure)   AS failed "
            "FROM builds b LEFT JOIN sessions s ON s.build_id = b.build_id "
            "WHERE b.project = ? "
            "GROUP BY b.build_id ORDER BY b.created_at DESC LIMIT ?",
            conn, params=(project, window),
        )
    if df.empty:
        return df
    df = df.iloc[::-1].reset_index(drop=True)  # oldest -> newest
    df["total"] = df["total"].fillna(0).astype(int)
    df["failed"] = df["failed"].fillna(0).astype(int)
    df["pass_rate"] = (
        (df["total"] - df["failed"]) / df["total"].where(df["total"] > 0, 1)
    ).round(3)
    return df


if __name__ == "__main__":
    trend = suite_health_trend(settings.target_project)
    print(f"History has {len(trend)} builds for '{settings.target_project}'")
    if not trend.empty:
        print(trend[["created_at", "total", "failed", "pass_rate"]].to_string())
