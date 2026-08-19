"""Command-line entry point for the Test Execution & Analytics Engine.

Installed as the ``bsa-report`` console script (see pyproject.toml), or run via
``python main.py`` from the repo root.

Flow: ingest -> triage (cluster / anomaly / flakiness) -> exec metrics -> HTML.
"""
from __future__ import annotations

import argparse
import logging
import sys

from config import settings
from ingestor import (
    as_project_list,
    ingest,
    ingest_builds,
    ingest_range,
    IngestionError,
)
from triage_engine import run_triage
from exec_metrics import compute_exec_metrics
from report_generator import generate_report
import history

logger = logging.getLogger("bsanalytics")


def _persist_latest(build_meta: dict, df, db_path) -> None:
    """Persist a latest-mode run, one row per real build.

    A cross-platform run has one build per project. Persisting the combined frame
    under the synthetic combined id would rewrite every session's build_id — and
    because session_id is the primary key, it also re-parents rows stored by
    earlier runs, collapsing the per-build history that the trend chart and
    cross-build flakiness are computed from.
    """
    for meta in build_meta.get("builds") or []:
        build_id = meta.get("hashed_id") or meta.get("build_id")
        part = df[df["build_id"] == build_id]
        if not part.empty:
            history.persist_build(meta, part, db_path=db_path)
    if not build_meta.get("builds"):
        history.persist_build(build_meta, df, db_path=db_path)


def project_label(build_meta: dict, args) -> str:
    """Project string used for history lookups and the report header."""
    return build_meta.get("project") or " + ".join(args.project)


def _configure_logging(verbose: bool) -> None:
    """INFO by default, DEBUG with --verbose. Errors carry a timestamp."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s" if not verbose else "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BrowserStack test analytics pipeline.")
    parser.add_argument("--user", default=settings.target_user)
    parser.add_argument(
        "--project",
        action="append",
        default=None,
        help=("BrowserStack project name. Repeat the flag (or pass a comma-separated "
              "list) to combine platforms into one cross-platform report, e.g. "
              "--project 'Finserv - Gopay Android' --project 'Finserv - Gopay iOS'."),
    )
    parser.add_argument("--source", choices=["file", "api"], default="file")
    parser.add_argument(
        "--mode",
        choices=["latest", "range"],
        default="latest",
        help="'latest' analyzes the newest build; 'range' aggregates the last N builds.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=settings.default_last_n,
        help="Number of recent builds to analyze in --mode range.",
    )
    parser.add_argument(
        "--build",
        action="append",
        default=None,
        metavar="NAME",
        help="Analyze specific builds by name, e.g. --build "
             "'gopay_consumer_app_android_tests - 174456408'. Repeat per platform. "
             "The trailing run number alone is enough if it is unambiguous. "
             "--project is optional here: without it every project is searched for "
             "the name, and the owning project is read off the build. Overrides "
             "--mode/--last, and requires --source api. Use this to compare "
             "platforms on equivalent builds when their newest runs differ in scope.",
    )
    parser.add_argument("--mock-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--enrich-logs",
        action="store_true",
        help="Fetch terminal logs for failed sessions to recover real stack traces (API only).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Also fetch per-session performance profiling (CPU/memory/battery). "
             "Covers passing sessions too, so it is one fetch per session.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    # Whether --project was actually passed, kept before normalisation: with
    # --build, omitting it means "search every project", which is not the same as
    # falling back to the TARGET_PROJECT default that as_project_list() applies.
    explicit_projects = args.project
    # Normalise --project (repeatable and/or comma-separated) into a list.
    args.project = as_project_list(args.project)

    enrich = True if args.enrich_logs else None
    if args.profile:
        object.__setattr__(settings, 'enrich_profiling', True)
    db_path = settings.history_db_for(args.source)

    try:
        if args.build:
            logger.info(
                "[1/4] Ingesting %d named build(s) from %s…",
                len(args.build),
                f"project '{' + '.join(args.project)}'" if explicit_projects
                else "all visible projects",
            )
            df, build_meta = ingest_builds(
                args.user, source=args.source, project=explicit_projects,
                build_names=args.build, enrich_logs=enrich,
            )
            logger.info(
                "      %d builds (%s → %s)",
                build_meta["n_builds"], build_meta["date_from"], build_meta["date_to"],
            )
            logger.info("      %d sessions · %d failures", len(df), int(df["is_failure"].sum()))
            # Each build was persisted individually inside ingest_builds, and the
            # combined frame is the flakiness history for this window.
            hist = df
        elif args.mode == "range":
            logger.info(
                "[1/4] Ingesting last %d builds for project '%s' (source=%s)…",
                args.last, " + ".join(args.project), args.source,
            )
            df, build_meta = ingest_range(
                args.user, source=args.source, project=args.project,
                last_n=args.last, enrich_logs=enrich,
            )
            logger.info(
                "      window: %d builds (%s → %s)",
                build_meta["n_builds"], build_meta["date_from"], build_meta["date_to"],
            )
            logger.info("      %d sessions · %d failures", len(df), int(df["is_failure"].sum()))
            hist = df  # the combined multi-build frame IS the history for flakiness
        else:
            logger.info(
                "[1/4] Ingesting latest build for project '%s' (source=%s)…",
                " + ".join(args.project), args.source,
            )
            df, build_meta = ingest(
                args.user, source=args.source, mock_path=args.mock_path,
                project=args.project, enrich_logs=enrich,
            )
            logger.info(
                "      project '%s' · build %s (status=%s)",
                build_meta.get("project"), build_meta.get("hashed_id", "n/a"),
                build_meta.get("status", "n/a"),
            )
            logger.info("      %d sessions · %d failures", len(df), int(df["is_failure"].sum()))
            _persist_latest(build_meta, df, db_path)
            hist = history.load_recent_sessions(project_label(build_meta, args), db_path=db_path)
            if hist is not None and not hist.empty:
                logger.info("      history: %d builds available", hist["build_id"].nunique())

        project_name = project_label(build_meta, args)
        trend = history.suite_health_trend(project_name, db_path=db_path)

        logger.info("[2/4] Running ML/NLP triage…")
        triage = run_triage(df, history=hist)
        logger.info("      %d root-cause clusters discovered", len(triage["clusters"]))
        for f in triage["findings"].head(3).itertuples():
            logger.info(
                "      finding: %s=%s %.0f%% vs %.0f%% baseline (OR %.1fx, p=%.4f)",
                f.dimension, f.level, f.failure_rate * 100, f.baseline_rate * 100,
                f.odds_ratio, f.p_value,
            )

        logger.info("[3/4] Computing executive metrics…")
        metrics = compute_exec_metrics(df, triage["clusters"], triage["device_anomaly"])
        c = metrics.summary_cards()
        logger.info("      health=%s · MTTR saved=%sh", c["suite_health_index"], c["mttr_hours_saved"])

        logger.info("[4/4] Rendering standalone HTML report…")
        path = generate_report(
            build_meta, metrics, triage["clusters"], triage["device_anomaly"],
            triage["flakiness"], output_path=args.output,
            is_sample=(args.source == "file"), trend=trend,
            categories=triage["categories"],
            classified=triage["classified_failures"],
            triage=triage,
        )
        logger.info("\n✅ Report: %s", path)
        return 0

    except IngestionError as exc:
        logger.error("\n❌ Ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
