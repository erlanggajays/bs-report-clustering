"""Report on (and optionally remove) phantom build rows left by an earlier bug.

Until this was fixed, a cross-platform ``--mode latest`` run persisted its combined
frame under one *synthetic* build whose project read like "A + B". Because
``sessions.session_id`` is the primary key, that rewrote the ``build_id`` of rows
already stored, re-parenting real builds' sessions onto the synthetic one.

Two kinds of damage are left behind:

* **empty builds** — a real build whose sessions were re-parented away, or a
  synthetic build whose sessions a later run reclaimed. Either way it has no
  sessions, so it has no pass rate; the trend chart used to plot it as 0%.
* **synthetic builds** — a row whose project names two projects at once. Its
  sessions belong to two real builds that can no longer be told apart from the
  database alone, so they are removed with it and must be re-ingested.

Re-ingest afterwards to restore real per-build history, e.g.

    python main.py --source api --build <android-build> --build <ios-build>

Dry-run by default; nothing is modified without ``--apply``.

    python scripts/repair_phantom_builds.py                      # report only
    python scripts/repair_phantom_builds.py --apply               # remove them
    python scripts/repair_phantom_builds.py --db data/history.db  # pick a database
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

# A synthetic multi-project row: persist_build joined the project names with " + ".
_COMBINED_MARKER = " + "


def _rows(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT b.build_id, b.project, b.name, b.created_at, "
        "       COUNT(s.session_id) AS total "
        "FROM builds b LEFT JOIN sessions s ON s.build_id = b.build_id "
        "GROUP BY b.build_id ORDER BY b.created_at"
    ).fetchall()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="data/history.db")
    parser.add_argument("--apply", action="store_true",
                        help="Delete the phantom rows (default is a dry run).")
    args = parser.parse_args()

    path = Path(args.db)
    if not path.exists():
        print(f"No database at {path} — nothing to repair.")
        return 0

    conn = sqlite3.connect(str(path))
    rows = _rows(conn)
    synthetic = [r for r in rows if _COMBINED_MARKER in (r[1] or "")]
    empty = [r for r in rows if r[4] == 0 and r not in synthetic]

    print(f"database : {path}")
    print(f"builds   : {len(rows)}")
    print(f"synthetic: {len(synthetic)}  (multi-project rows, sessions unattributable)")
    print(f"empty    : {len(empty)}  (no sessions, plotted as 0% on the trend)")

    for label, group in (("synthetic", synthetic), ("empty", empty)):
        if not group:
            continue
        print(f"\n{label}:")
        for build_id, project, name, created_at, total in group:
            print(f"  {created_at[:10]}  {total:>4} sessions  {project[:44]:<46} {(name or '')[:34]}")

    doomed_sessions = sum(r[4] for r in synthetic)
    if not args.apply:
        print(f"\nDry run. --apply would delete {len(synthetic) + len(empty)} build rows "
              f"and {doomed_sessions} sessions attached to synthetic builds.")
        print("Re-ingest afterwards to restore real per-build history:")
        print("  python main.py --source api --build <android-build> --build <ios-build>")
        return 0

    with conn:
        for build_id, *_ in synthetic:
            conn.execute("DELETE FROM sessions WHERE build_id = ?", (build_id,))
        for build_id, *_ in synthetic + empty:
            conn.execute("DELETE FROM builds WHERE build_id = ?", (build_id,))
    print(f"\nRemoved {len(synthetic) + len(empty)} build rows and {doomed_sessions} "
          "sessions. Re-run the pipeline to repopulate real per-build history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
