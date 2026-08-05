"""Central configuration for the Test Execution & Analytics Engine.

Credentials are read exclusively from the environment. Never hardcode secrets.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# --- BrowserStack REST API endpoints ----------------------------------------
# App Automate (mobile app tests). Hierarchy is Projects -> Builds -> Sessions:
#   projects.json                    -> list projects (id + name)
#   projects/{project_id}.json       -> project detail WITH nested builds[]
#   builds/{build_id}/sessions.json  -> sessions within a build (build_id = hashed_id)
BROWSERSTACK_BASE_URL = "https://api-cloud.browserstack.com/app-automate"
PROJECTS_ENDPOINT = f"{BROWSERSTACK_BASE_URL}/projects.json"
PROJECT_DETAIL_ENDPOINT = f"{BROWSERSTACK_BASE_URL}/projects/{{project_id}}.json"
SESSIONS_ENDPOINT = f"{BROWSERSTACK_BASE_URL}/builds/{{build_id}}/sessions.json"


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Credentials are injected from the environment only."""

    # TODO: Securely load this value from an environment variable or secrets vault. Do not hardcode.
    browserstack_username: str = field(
        default_factory=lambda: os.environ.get("BROWSERSTACK_USERNAME", "")
    )
    # TODO: Securely load this value from an environment variable or secrets vault. Do not hardcode.
    browserstack_access_key: str = field(
        default_factory=lambda: os.environ.get("BROWSERSTACK_ACCESS_KEY", "")
    )

    # Selection is by project. Set TARGET_PROJECT_ID to skip the name->id lookup
    # (fastest); otherwise the name is resolved via projects.json.
    target_project: str = field(
        default_factory=lambda: os.environ.get("TARGET_PROJECT", "Finserv - Gopay Android")
    )
    target_project_id: str = field(
        default_factory=lambda: os.environ.get("TARGET_PROJECT_ID", "")
    )
    # Display-only owner label. Empty by default (no hardcoded name); when unset
    # the ingestor derives it from the build name, and the report hides it if
    # still unknown. Override with TARGET_USER.
    target_user: str = field(
        default_factory=lambda: os.environ.get("TARGET_USER", "")
    )
    suite_name: str = field(
        default_factory=lambda: os.environ.get(
            "SUITE_NAME", "gopay_consumer_app_android_tests"
        )
    )

    # Which build-level statuses are eligible for selection. We consider
    # completed builds regardless of outcome so failures/timeouts are analyzed,
    # then pick the newest. Falls back to newest overall if none match.
    # Override via BUILD_STATUS_FILTER (comma-separated).
    build_status_filter: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            s.strip().lower()
            for s in os.environ.get(
                "BUILD_STATUS_FILTER", "done,failed,timeout"
            ).split(",")
            if s.strip()
        )
    )
    # Fetch terminal logs for failed sessions to recover real stack traces for
    # clustering (off by default: it is N extra HTTP calls). ENRICH_LOGS=1.
    enrich_failure_logs: bool = field(
        default_factory=lambda: os.environ.get("ENRICH_LOGS", "").lower()
        in {"1", "true", "yes"}
    )

    # HTTP behaviour
    request_timeout_seconds: int = 30
    max_builds_to_scan: int = 50

    # Range mode: analyze the last N builds instead of only the latest.
    default_last_n: int = 10           # default window size for --mode range
    max_range_builds: int = 25         # hard cap on builds fetched per range run
    http_max_retries: int = 4          # retries on 429/5xx with backoff
    http_backoff_factor: float = 0.5
    sessions_page_size: int = 100      # pagination page size for sessions.json
    max_session_pages: int = 100       # safety cap against non-paginating loops

    # Clustering tunables. Failures are grouped by a fingerprint (exception +
    # top application stack frame); DBSCAN is the fallback for free-text failures
    # with no parseable trace.
    dbscan_eps: float = 0.55
    dbscan_min_samples: int = 2
    # Package hints identifying *your* stack frames (the actionable ones), so the
    # fingerprint anchors on your code, not framework frames. Comma-separated.
    app_package_hints: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            s.strip().lower()
            for s in os.environ.get("APP_PACKAGE_HINTS", "gopay,gojek").split(",")
            if s.strip()
        )
    )

    # Device risk: below this many runs, a cell is flagged low-confidence and
    # ranked by the Wilson lower bound rather than the raw failure rate.
    device_min_sample_size: int = 5
    wilson_z: float = 1.96             # 95% confidence

    # Historical persistence (SQLite) for cross-build flakiness and trends.
    # Paths are relative to the current working directory (resolved at use time),
    # so they are independent of where this module is installed.
    history_db_path: str = field(
        default_factory=lambda: os.environ.get("HISTORY_DB_PATH", "data/history.db")
    )
    # Sample/mock runs persist to a SEPARATE db so they never pollute real history.
    sample_history_db_path: str = "data/history_sample.db"
    history_builds_window: int = 20    # builds of history used for trend/flakiness

    def history_db_for(self, source: str) -> str:
        """Real API runs use the real DB; file/mock runs use the sample DB."""
        return self.history_db_path if source == "api" else self.sample_history_db_path

    # Business assumption for MTTR calculation (minutes a human spends triaging
    # one failure before clustering collapses duplicates).
    manual_triage_minutes_per_failure: float = 12.0

    @property
    def has_credentials(self) -> bool:
        return bool(self.browserstack_username and self.browserstack_access_key)

    @property
    def auth(self) -> tuple[str, str]:
        """Basic-auth tuple for `requests`."""
        return (self.browserstack_username, self.browserstack_access_key)


settings = Settings()
