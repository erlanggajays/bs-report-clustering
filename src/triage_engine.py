"""ML / NLP triage engine.

Three responsibilities:
  1. Failure clustering  -> group stack traces into root causes (TF-IDF + DBSCAN).
  2. Device anomaly       -> failure distribution across device/OS combinations.
  3. Flakiness metric     -> per-scenario instability score.

The vectorizer/clusterer are intentionally simple and dependency-light. TF-IDF
over word + character n-grams captures the templated structure of stack traces
well; DBSCAN discovers the number of root causes on its own and isolates one-off
failures as noise (which we surface as singleton "unique" clusters).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

# Matches fully-qualified exception/error class names in a stack trace.
_EXCEPTION_RE = re.compile(r"([A-Za-z_][\w.]*(?:Exception|Error|Failure|Timeout))")

# Volatile tokens (ids, timings, memory addresses) are masked so that two
# failures with the same root cause but different runtime values still cluster.
_NOISE_PATTERNS = [
    (re.compile(r"0x[0-9a-fA-F]+"), " HEX "),
    (re.compile(r"\b\d+ms\b"), " DUR "),
    (re.compile(r"\b\d[\d.,]*\b"), " NUM "),
    (re.compile(r"[0-9a-f]{16,}"), " HASH "),
    (re.compile(r"\s+"), " "),
]


@dataclass
class FailureCluster:
    """One root-cause group discovered among the failures."""

    cluster_id: int
    label: str                       # human-readable representative message
    size: int
    confidence: float                # mean intra-cluster cosine similarity (0-1)
    exception_type: str = ""         # extracted exception class, if any
    session_ids: list[str] = field(default_factory=list)
    session_urls: list[str] = field(default_factory=list)  # token-free deep links
    app_versions: list[str] = field(default_factory=list)
    example_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "size": self.size,
            "confidence": round(self.confidence, 3),
            "exception_type": self.exception_type,
            "session_ids": self.session_ids,
            "session_urls": self.session_urls,
            "app_versions": self.app_versions,
            "example_reason": self.example_reason,
        }


def _normalize(reason: str) -> str:
    text = reason.lower()
    for pattern, repl in _NOISE_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


def _exception_type(reason: str) -> str:
    """Extract the short exception class name from a stack trace, if present."""
    m = _EXCEPTION_RE.search(reason or "")
    return m.group(1).split(".")[-1] if m else ""


def _cluster_label(reason: str, max_len: int = 100) -> str:
    """Readable cluster label: exception class prefixed onto the first line."""
    first = reason.strip().splitlines()[0] if reason.strip() else "Unknown failure"
    exc = _exception_type(reason)
    if exc and exc.lower() not in first.lower():
        first = f"{exc}: {first}"
    return (first[: max_len - 1] + "…") if len(first) > max_len else first


def _cluster_extras(members: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Collect token-free session links and distinct app versions for a cluster."""
    urls = [u for u in members.get("session_url", pd.Series(dtype=str)).tolist() if u]
    if "app_version" in members:
        versions = sorted({v for v in members["app_version"].tolist() if v})
    else:
        versions = []
    return urls, versions


def cluster_failures(
    df: pd.DataFrame, eps: float | None = None, min_samples: int | None = None
) -> tuple[list[FailureCluster], pd.DataFrame]:
    """Cluster failed sessions by stack-trace similarity.

    Returns the discovered clusters (largest first) and a copy of the failed-
    sessions frame annotated with a ``cluster_id`` column. ``eps`` /
    ``min_samples`` default to the tunables in ``config.settings``.
    """
    eps = settings.dbscan_eps if eps is None else eps
    min_samples = settings.dbscan_min_samples if min_samples is None else min_samples

    failures = df[df["is_failure"] & df["reason"].str.len().gt(0)].copy()
    if failures.empty:
        return [], failures

    normalized = failures["reason"].map(_normalize).tolist()

    # Single failure -> no clustering needed.
    if len(normalized) == 1:
        failures["cluster_id"] = 0
        row = failures.iloc[0]
        urls, versions = _cluster_extras(failures)
        return (
            [
                FailureCluster(
                    cluster_id=0,
                    label=_cluster_label(row["reason"]),
                    size=1,
                    confidence=1.0,
                    exception_type=_exception_type(row["reason"]),
                    session_ids=[row["session_id"]],
                    session_urls=urls,
                    app_versions=versions,
                    example_reason=row["reason"],
                )
            ],
            failures,
        )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(normalized)

    # Cosine distance = 1 - cosine similarity; DBSCAN with metric="cosine".
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    raw_labels = model.fit_predict(matrix)

    # DBSCAN marks singletons as -1 (noise). Re-map each noise point to its own
    # unique cluster so nothing is silently dropped from the report.
    labels = raw_labels.copy()
    next_id = (labels.max() + 1) if labels.max() >= 0 else 0
    for i, lab in enumerate(labels):
        if lab == -1:
            labels[i] = next_id
            next_id += 1

    failures = failures.copy()
    failures["cluster_id"] = labels

    clusters: list[FailureCluster] = []
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        members = failures.iloc[idx]
        # Confidence = mean pairwise cosine similarity within the cluster.
        if len(idx) > 1:
            sim = cosine_similarity(matrix[idx])
            iu = np.triu_indices_from(sim, k=1)
            confidence = float(sim[iu].mean())
        else:
            confidence = 1.0  # a singleton is trivially self-consistent
        representative = members.iloc[0]["reason"]
        urls, versions = _cluster_extras(members)
        clusters.append(
            FailureCluster(
                cluster_id=int(cid),
                label=_cluster_label(representative),
                size=int(len(idx)),
                confidence=confidence,
                exception_type=_exception_type(representative),
                session_ids=members["session_id"].tolist(),
                session_urls=urls,
                app_versions=versions,
                example_reason=representative,
            )
        )

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters, failures


def _wilson_lower_bound(failures: int, total: int, z: float) -> float:
    """Lower bound of the Wilson score interval for a failure proportion.

    This down-weights small samples: 1/1 failures yields a far lower score than
    50/50, so a single unlucky run cannot masquerade as the top risk cell.
    """
    if total == 0:
        return 0.0
    p = failures / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denom)


def device_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Failure distribution per (device, os_version), ranked with statistical rigor.

    Adds a Wilson lower-bound ``risk_score`` and a ``low_sample`` flag so cells
    with too few runs don't dominate the ranking. Sorted by ``risk_score``.
    """
    grouped = (
        df.groupby(["device", "os_version"])
        .agg(total=("session_id", "count"), failed=("is_failure", "sum"))
        .reset_index()
    )
    grouped["failure_rate"] = (grouped["failed"] / grouped["total"]).round(3)
    grouped["risk_score"] = grouped.apply(
        lambda r: round(_wilson_lower_bound(int(r["failed"]), int(r["total"]), settings.wilson_z), 3),
        axis=1,
    )
    grouped["low_sample"] = grouped["total"] < settings.device_min_sample_size
    return grouped.sort_values(
        ["risk_score", "failure_rate"], ascending=False
    ).reset_index(drop=True)


def _duration_cov(durations: np.ndarray) -> float:
    mean_d = durations.mean() if durations.size else 0.0
    cov = (durations.std() / mean_d) if mean_d > 0 else 0.0
    return float(np.clip(cov, 0.0, 1.0))


def _flakiness_in_build(df: pd.DataFrame) -> pd.DataFrame:
    """In-build flakiness proxy (single build, no history).

    Blends status variance (peaks at a 50/50 pass/fail split) and duration
    coefficient of variation. Used as a fallback until history accumulates.
    """
    def _score(group: pd.DataFrame) -> pd.Series:
        fail_rate = group["is_failure"].mean()
        status_var = 4.0 * fail_rate * (1.0 - fail_rate)
        duration_var = _duration_cov(group["duration"].to_numpy(dtype=float))
        score = 0.65 * status_var + 0.35 * duration_var
        return pd.Series(
            {
                "runs": len(group),
                "failure_rate": round(float(fail_rate), 3),
                "duration_cov": round(duration_var, 3),
                "flip_rate": 0.0,
                "flakiness_index": round(float(np.clip(score, 0.0, 1.0)), 3),
            }
        )

    result = df.groupby("name").apply(_score, include_groups=False).reset_index()
    return result.sort_values("flakiness_index", ascending=False).reset_index(drop=True)


def _flakiness_from_history(history: pd.DataFrame) -> pd.DataFrame:
    """True run-over-run flakiness using cross-build history.

    The dominant signal is the *flip rate*: how often a scenario alternates
    pass<->fail across consecutive builds. A test that flips a lot is flaky even
    if its overall failure rate is moderate.
    """
    def _score(group: pd.DataFrame) -> pd.Series:
        ordered = group.sort_values("created_at")
        fails = ordered["is_failure"].astype(int).to_numpy()
        n = len(fails)
        flips = int(np.abs(np.diff(fails)).sum()) if n > 1 else 0
        flip_rate = flips / (n - 1) if n > 1 else 0.0
        fail_rate = float(fails.mean()) if n else 0.0
        status_var = 4.0 * fail_rate * (1.0 - fail_rate)
        duration_var = _duration_cov(ordered["duration"].to_numpy(dtype=float))
        score = 0.55 * flip_rate + 0.30 * status_var + 0.15 * duration_var
        return pd.Series(
            {
                "runs": n,
                "builds": int(ordered["build_id"].nunique()),
                "failure_rate": round(fail_rate, 3),
                "duration_cov": round(duration_var, 3),
                "flip_rate": round(float(flip_rate), 3),
                "flakiness_index": round(float(np.clip(score, 0.0, 1.0)), 3),
            }
        )

    result = history.groupby("name").apply(_score, include_groups=False).reset_index()
    return result.sort_values("flakiness_index", ascending=False).reset_index(drop=True)


def flakiness_index(
    df: pd.DataFrame, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-scenario flakiness in [0, 1].

    Uses cross-build history when available (>1 build) for a true flip-rate
    signal; otherwise falls back to the single-build proxy.
    """
    if (
        history is not None
        and not history.empty
        and "build_id" in history
        and history["build_id"].nunique() > 1
    ):
        return _flakiness_from_history(history)
    return _flakiness_in_build(df)


def run_triage(
    df: pd.DataFrame, history: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Convenience wrapper returning every triage artifact in one dict."""
    clusters, annotated = cluster_failures(df)
    return {
        "clusters": clusters,
        "annotated_failures": annotated,
        "device_anomaly": device_anomaly(df),
        "flakiness": flakiness_index(df, history),
    }


if __name__ == "__main__":
    from ingestor import ingest

    frame, _ = ingest(source="file")
    out = run_triage(frame)
    print(f"Discovered {len(out['clusters'])} root-cause clusters:")
    for c in out["clusters"]:
        print(f"  [{c.size:>3}]  conf={c.confidence:.2f}  {c.label}")
