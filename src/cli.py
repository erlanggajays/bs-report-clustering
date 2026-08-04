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
from ingestor import ingest, ingest_range, IngestionError
from triage_engine import run_triage
from exec_metrics import compute_exec_metrics
from report_generator import generate_report
import history

logger = logging.getLogger("bsanalytics")


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
    parser.add_argument("--project", default=settings.target_project)
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
    parser.add_argument("--mock-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--enrich-logs",
        action="store_true",
        help="Fetch terminal logs for failed sessions to recover real stack traces (API only).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    enrich = True if args.enrich_logs else None
    db_path = settings.history_db_for(args.source)

    try:
        if args.mode == "range":
            logger.info(
                "[1/4] Ingesting last %d builds for project '%s' (source=%s)…",
                args.last, args.project, args.source,
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
                args.project, args.source,
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
            history.persist_build(build_meta, df, db_path=db_path)
            hist = history.load_recent_sessions(build_meta.get("project", args.project), db_path=db_path)
            if hist is not None and not hist.empty:
                logger.info("      history: %d builds available", hist["build_id"].nunique())

        project_name = build_meta.get("project", args.project)
        trend = history.suite_health_trend(project_name, db_path=db_path)

        logger.info("[2/4] Running ML/NLP triage…")
        triage = run_triage(df, history=hist)
        logger.info("      %d root-cause clusters discovered", len(triage["clusters"]))

        logger.info("[3/4] Computing executive metrics…")
        metrics = compute_exec_metrics(df, triage["clusters"], triage["device_anomaly"])
        c = metrics.summary_cards()
        logger.info("      health=%s · MTTR saved=%sh", c["suite_health_index"], c["mttr_hours_saved"])

        logger.info("[4/4] Rendering standalone HTML report…")
        path = generate_report(
            build_meta, metrics, triage["clusters"], triage["device_anomaly"],
            triage["flakiness"], output_path=args.output,
            is_sample=(args.source == "file"), trend=trend,
        )
        logger.info("\n✅ Report: %s", path)
        return 0

    except IngestionError as exc:
        logger.error("\n❌ Ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
