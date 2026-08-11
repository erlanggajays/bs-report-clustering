"""Report on (and optionally clear) failure reasons corrupted by old enrichment.

An earlier version of ``enrich_failure_reasons`` overwrote each session's reason with
a snippet extracted from Appium logs. When that snippet was bookkeeping noise, the
real message BrowserStack had supplied ("Element not found", "Header value cannot be
null") was lost, and the failure was later filed as "no diagnostic logs".

The originals are not recoverable from the database — only a fresh enriched run can
restore them, and only for builds still inside the range window. What this script does
is *identify* the damaged rows so the numbers can be trusted, and optionally blank them
so the noise stops polluting clustering and categories when history is analysed.

Dry-run by default; nothing is modified without ``--apply``.

    python scripts/repair_history.py                      # report only
    python scripts/repair_history.py --apply               # blank the noise reasons
    python scripts/repair_history.py --db data/history.db  # pick a database
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taxonomy import HAS_ERROR_SIGNAL  # noqa: E402

# Hallmarks of a reason that is an Appium/BrowserStack log dump rather than an error.
_LOG_DUMP_MARKERS = (
    "[HTTP]", "/wd/hub", "Proxying [", "Responding to client",
    "[ADB]", "dumpsys", "jdwp-control", "REQUEST [", "POST /session",
    "Calling AppiumDriver", "Clearing new command timeout",
)


def looks_corrupted(reason: str) -> bool:
    """A reason is suspect when it reads like a log dump and carries no error signal.

    Both conditions are required: a genuine stack trace can mention an ADB command,
    and a terse-but-valid reason ("TIMEOUT") is not corruption.
    """
    if not reason:
        return False
    if HAS_ERROR_SIGNAL.search(reason):
        return False
    return any(marker in reason for marker in _LOG_DUMP_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="data/history.db")
    parser.add_argument("--apply", action="store_true",
                        help="Blank the corrupted reasons (default is a dry run).")
    args = parser.parse_args()

    path = Path(args.db)
    if not path.exists():
        print(f"No database at {path} — nothing to repair.")
        return 0

    conn = sqlite3.connect(str(path))
    rows = conn.execute(
        "SELECT session_id, build_id, name, reason FROM sessions WHERE is_failure = 1"
    ).fetchall()

    suspect = [r for r in rows if looks_corrupted(r[3] or "")]
    print(f"database : {path}")
    print(f"failures : {len(rows)}")
    print(f"corrupted: {len(suspect)}  "
          f"({len(suspect) / len(rows) * 100:.0f}%)" if rows else "corrupted: 0")

    if suspect:
        builds = {r[1] for r in suspect}
        print(f"\naffected builds: {len(builds)}")
        print("\nexamples:")
        for session_id, _build, name, reason in suspect[:5]:
            print(f"  {(name or '?')[:52]:<54} {reason[:60].replace(chr(10), ' ')}…")

    if not args.apply:
        print("\nDry run. Re-run with --apply to blank these reasons.")
        print("To actually restore them, re-run the pipeline over those builds:")
        print("  python main.py --source api --mode range --last 10 --enrich-logs")
        return 0

    with conn:
        conn.executemany(
            "UPDATE sessions SET reason = '' WHERE session_id = ?",
            [(r[0],) for r in suspect],
        )
    print(f"\nBlanked {len(suspect)} corrupted reasons. A fresh enriched run over those "
          "builds will repopulate them correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
