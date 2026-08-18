"""Executive / ROI metrics derived from the ingested build and triage output.

These are the numbers a QA lead or EM cares about: is the suite healthy, how
much manual triage effort did clustering save, and which device/OS cells carry
the most risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config import settings
from triage_engine import FailureCluster


@dataclass
class ExecMetrics:
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    suite_health_index: float          # 0-100 composite reliability score
    num_root_causes: int
    mttr_minutes_saved: float
    mttr_hours_saved: float
    device_risk_matrix: pd.DataFrame = field(repr=False)
    top_risk_cell: dict[str, Any] = field(default_factory=dict)

    def summary_cards(self) -> dict[str, Any]:
        """Flat dict of the headline numbers for the report's summary cards."""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate_pct": round(self.pass_rate * 100, 1),
            "suite_health_index": round(self.suite_health_index, 1),
            "num_root_causes": self.num_root_causes,
            "mttr_hours_saved": round(self.mttr_hours_saved, 1),
        }


def _suite_health_index(pass_rate: float, device_risk: pd.DataFrame) -> float:
    """Composite reliability score in [0, 100].

    Anchored on pass rate but penalized by concentration of failures on any one
    environment — a suite that is 90% green but collapses on one device is less
    healthy than the raw pass rate suggests.
    """
    base = pass_rate * 100.0
    # Use the Wilson lower bound (risk_score) so a 1/1 fluke can't tank the score.
    col = "risk_score" if "risk_score" in device_risk else "failure_rate"
    worst = float(device_risk[col].max()) if not device_risk.empty else 0.0
    concentration_penalty = worst * 15.0  # up to -15 points
    return max(0.0, min(100.0, base - concentration_penalty))


def _mttr_saved(num_failures: int, num_clusters: int) -> float:
    """Estimated triage minutes saved by root-cause grouping.

    Without clustering an engineer inspects every failure; with clustering they
    inspect one representative per root cause. Saved effort is therefore the
    number of duplicate failures collapsed, times the per-failure triage cost.
    """
    duplicates_collapsed = max(0, num_failures - num_clusters)
    return duplicates_collapsed * settings.manual_triage_minutes_per_failure


def compute_exec_metrics(
    df: pd.DataFrame,
    clusters: list[FailureCluster],
    device_risk: pd.DataFrame,
) -> ExecMetrics:
    total = len(df)
    failed = int(df["is_failure"].sum())
    passed = total - failed
    pass_rate = (passed / total) if total else 0.0

    num_clusters = len(clusters)
    health = _suite_health_index(pass_rate, device_risk)
    minutes_saved = _mttr_saved(failed, num_clusters)

    top_cell: dict[str, Any] = {}
    if not device_risk.empty:
        worst = device_risk.iloc[0]
        top_cell = {
            "device": worst["device"],
            "os_version": worst["os_version"],
            "os_label": str(worst.get("os_label") or worst["os_version"]),
            "failure_rate": float(worst["failure_rate"]),
            "risk_score": float(worst.get("risk_score", worst["failure_rate"])),
            "failed": int(worst["failed"]),
            "total": int(worst["total"]),
            "low_sample": bool(worst.get("low_sample", False)),
        }

    return ExecMetrics(
        total_tests=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        suite_health_index=health,
        num_root_causes=num_clusters,
        mttr_minutes_saved=minutes_saved,
        mttr_hours_saved=minutes_saved / 60.0,
        device_risk_matrix=device_risk,
        top_risk_cell=top_cell,
    )


if __name__ == "__main__":
    from ingestor import ingest
    from triage_engine import run_triage

    frame, _ = ingest(source="file")
    triage = run_triage(frame)
    metrics = compute_exec_metrics(frame, triage["clusters"], triage["device_anomaly"])
    print(metrics.summary_cards())
