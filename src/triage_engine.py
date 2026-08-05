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
from scipy.sparse import hstack
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

# Matches fully-qualified exception/error class names in a stack trace.
_EXCEPTION_RE = re.compile(r"([A-Za-z_][\w.]*(?:Exception|Error|Failure|Timeout))")
# Deepest "Caused by:" is usually the true root cause.
_CAUSED_BY_RE = re.compile(r"Caused by:\s*([A-Za-z_][\w.]*(?:Exception|Error|Failure|Timeout))")
# A stack frame: `at pkg.Class.method(File.java:123)` -> class path, method.
_FRAME_RE = re.compile(r"\bat\s+([\w.$]+)\.([\w<>$]+)\s*\(")

# Volatile tokens (ids, timings, locators, urls, literals) are masked so that
# two failures with the same root cause but different runtime values still
# cluster. Order matters: broader/structured patterns first.
_NOISE_PATTERNS = [
    (re.compile(r"https?://\S+"), " URL "),
    (re.compile(r"\{\{?[^{}]*\}?\}"), " LOC "),         # {{id=btn}} / {id=btn} locators
    (re.compile(r"'[^']*'|\"[^\"]*\""), " STR "),        # quoted literals
    (re.compile(r"0x[0-9a-fA-F]+"), " HEX "),
    (re.compile(r"\b[0-9a-f]{16,}\b"), " HASH "),        # long hex / uuid-ish ids
    (re.compile(r"\b\d+ms\b"), " DUR "),
    (re.compile(r":\d+\b"), " LINE "),                   # File.java:213 line numbers
    (re.compile(r"\b\d[\d.,]*\b"), " NUM "),             # numbers, currency amounts
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


def _root_exception(reason: str) -> str:
    """The most informative exception class: the deepest 'Caused by:' if present,
    else the first thrown exception."""
    causes = _CAUSED_BY_RE.findall(reason or "")
    if causes:
        return causes[-1].split(".")[-1]
    return _exception_type(reason)


def _top_frame(reason: str) -> str:
    """The most actionable stack frame as 'Class.method'.

    Prefers the first frame in an application package (settings.app_package_hints);
    falls back to the first frame of any kind. Empty if no frame is present.
    """
    first = ""
    for m in _FRAME_RE.finditer(reason or ""):
        cls_path, method = m.group(1), m.group(2)
        label = f"{cls_path.split('.')[-1]}.{method}"
        if not first:
            first = label
        if any(h in cls_path.lower() for h in settings.app_package_hints):
            return label
    return first


def _fingerprint(reason: str) -> str:
    """Stable root-cause signature: 'Exception @ Class.method'.

    Robust to volatile details (element ids, line numbers, timings) because it is
    built from structure, not raw text. Empty when there is no parseable
    exception (such failures fall back to text clustering).
    """
    exc = _root_exception(reason)
    frame = _top_frame(reason)
    if exc and frame:
        return f"{exc} @ {frame}"
    return exc  # exception-only, or "" when nothing parseable


def _build_matrix(normalized: list[str]):
    """Combined word + char n-gram TF-IDF (char n-grams are robust to identifiers
    and CamelCase). Returns None for a single document."""
    if len(normalized) < 2:
        return None
    word = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1, sublinear_tf=True)
    char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=1, sublinear_tf=True)
    return hstack([word.fit_transform(normalized), char.fit_transform(normalized)]).tocsr()


def cluster_failures(
    df: pd.DataFrame, eps: float | None = None, min_samples: int | None = None
) -> tuple[list[FailureCluster], pd.DataFrame]:
    """Group failed sessions by root-cause **signature**, with a text fallback.

    Stage 1 (deterministic): failures with a parseable stack trace are grouped by
    fingerprint = ``Exception @ Class.method``. This is interpretable and robust
    to volatile details (element ids, line numbers, timings).
    Stage 2 (fallback): failures with no parseable trace are clustered by
    TF-IDF + DBSCAN on their text. ``eps`` / ``min_samples`` default to settings.

    Returns clusters (largest first) and the failed-sessions frame annotated with
    a ``cluster_id`` column.
    """
    eps = settings.dbscan_eps if eps is None else eps
    min_samples = settings.dbscan_min_samples if min_samples is None else min_samples

    failures = df[df["is_failure"] & df["reason"].str.len().gt(0)].copy()
    if failures.empty:
        return [], failures
    failures = failures.reset_index(drop=True)

    reasons = failures["reason"].tolist()
    normalized = [_normalize(r) for r in reasons]
    fingerprints = [_fingerprint(r) for r in reasons]
    # Combined word+char TF-IDF: powers both confidence and the text fallback.
    matrix = _build_matrix(normalized)

    cluster_key: list[int] = [-1] * len(failures)
    next_id = 0

    # --- Stage 1: exact fingerprint groups ---
    sig_groups: dict[str, list[int]] = {}
    fallback_idx: list[int] = []
    for i, fp in enumerate(fingerprints):
        if fp:
            sig_groups.setdefault(fp, []).append(i)
        else:
            fallback_idx.append(i)
    for _fp, idxs in sig_groups.items():
        for i in idxs:
            cluster_key[i] = next_id
        next_id += 1

    # --- Stage 2: DBSCAN fallback for un-fingerprinted (free-text) failures ---
    if len(fallback_idx) == 1:
        cluster_key[fallback_idx[0]] = next_id
        next_id += 1
    elif fallback_idx:
        sub_labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(
            matrix[fallback_idx]
        )
        local: dict[int, int] = {}
        for j, lab in zip(fallback_idx, sub_labels):
            lab = int(lab)
            if lab == -1:  # DBSCAN noise -> its own singleton cluster
                cluster_key[j] = next_id
                next_id += 1
            elif lab in local:
                cluster_key[j] = local[lab]
            else:
                local[lab] = next_id
                cluster_key[j] = next_id
                next_id += 1

    failures["cluster_id"] = cluster_key

    clusters: list[FailureCluster] = []
    for cid in sorted(set(cluster_key)):
        idx = [i for i, k in enumerate(cluster_key) if k == cid]
        members = failures.iloc[idx]
        representative = members.iloc[0]["reason"]
        if matrix is not None and len(idx) > 1:
            sim = cosine_similarity(matrix[idx])
            iu = np.triu_indices_from(sim, k=1)
            confidence = float(sim[iu].mean())
        else:
            confidence = 1.0
        fp = fingerprints[idx[0]]
        urls, versions = _cluster_extras(members)
        clusters.append(
            FailureCluster(
                cluster_id=int(cid),
                label=fp or _cluster_label(representative),
                size=len(idx),
                confidence=confidence,
                exception_type=_root_exception(representative),
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
