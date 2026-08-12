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
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import inference
from config import settings
from features import extract_locator, extract_screen, extract_selector, feature_area
from taxonomy import HAS_ERROR_SIGNAL, category_description, classify

# Exception/error CLASS names. The final segment must start with a capital, so
# helper functions such as "Object.errorWithException" are not mistaken for the
# exception type (a real bug seen in Appium logs).
# The (?<![\w$]) guard is essential: without it the capital inside
# "errorWithException" matches and yields the nonsense class "WithException".
_EXCEPTION_RE = re.compile(
    r"(?<![\w$])(?:[A-Za-z_][\w.$]*\.)?([A-Z]\w*(?:Exception|Error|Failure|Timeout))\b"
)
# Appium/BrowserStack log lines start with a timestamp and bracketed tags, e.g.
# "2026-08-05 04:51:22:421 - [cdea8165][HTTP] ". Stripping that prefix before
# building a signature stops masked timestamps ("NUM - NUM LINE") from becoming
# the cluster label.
_LOG_PREFIX_RE = re.compile(
    r"^\s*(?:\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:[:.]\d+)?)?\s*-?\s*"
    r"(?:\[[^\]]{1,48}\]\s*)*"
)
# Presence of an error signal is defined once, in taxonomy, so the cluster view and
# the category view can never disagree about the same failure. Testing for a signal
# beats blocklisting noise: Appium emits endless bookkeeping variants and each new
# one would otherwise become its own bogus cluster.
_HAS_ERROR_SIGNAL = HAS_ERROR_SIGNAL


def _strip_log_prefix(line: str) -> str:
    return _LOG_PREFIX_RE.sub("", line or "").strip()
# Deepest "Caused by:" is usually the true root cause.
_CAUSED_BY_RE = re.compile(
    r"Caused by:\s*(?:[A-Za-z_][\w.$]*\.)?([A-Z]\w*(?:Exception|Error|Failure|Timeout))\b"
)
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
    # Most failures are framework assertion messages with no exception class, so the
    # taxonomy category is the label that is always present — and it is the one that
    # says who should act.
    category: str = ""
    owner: str = ""
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
            "category": self.category,
            "owner": self.owner,
            "session_ids": self.session_ids,
            "session_urls": self.session_urls,
            "app_versions": self.app_versions,
            "example_reason": self.example_reason,
        }


def _mask(reason: str) -> str:
    """Mask volatile tokens, preserving case (used for human-facing labels)."""
    text = reason or ""
    for pattern, repl in _NOISE_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


def _normalize(reason: str) -> str:
    """Masked and lower-cased — the form fed to the TF-IDF vectorizer."""
    return _mask(reason).lower()


def _exception_type(reason: str) -> str:
    """Extract the short exception class name from a stack trace, if present."""
    m = _EXCEPTION_RE.search(reason or "")
    return m.group(1).split(".")[-1] if m else ""


def _cluster_label(reason: str, max_len: int = 100) -> str:
    """Readable cluster label: exception class prefixed onto the first line.

    The log-line prefix (timestamp + [tags]) is removed first, otherwise labels for
    text-clustered failures read as "2026-08-05 04:30:06:968 - [93a1e2e6][...]".
    """
    first = _strip_log_prefix(
        reason.strip().splitlines()[0] if reason.strip() else ""
    ) or "Unknown failure"
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


def _app_frame(reason: str) -> str:
    """The top frame that belongs to an application package, as 'Class.method'.

    Strict (unlike ``_top_frame``): returns "" rather than an arbitrary framework
    frame, because framework frames vary between otherwise-identical failures and
    would split a single root cause into several clusters.
    """
    for m in _FRAME_RE.finditer(reason or ""):
        cls_path, method = m.group(1), m.group(2)
        if any(h in cls_path.lower() for h in settings.app_package_hints):
            return f"{cls_path.split('.')[-1]}.{method}"
    return ""


def _message_signature(reason: str, words: int = 8, max_len: int = 64) -> str:
    """A stable template from the message's first line.

    Runs through ``_normalize`` first, so locators/ids/numbers are masked — meaning
    "Can't locate an element ... accessibilityId: Lifetime interest" and the same
    error on a different locator share one signature.
    """
    first = (reason or "").strip().splitlines()[0] if (reason or "").strip() else ""
    if not first:
        return ""
    # Remove the log-line prefix (timestamp + [tags]) so the signature describes the
    # error rather than a masked timestamp.
    first = _strip_log_prefix(first)
    # Prefer the text *after* the exception class, which is the actual message.
    m = re.search(r"(?:[A-Za-z_][\w.$]*\.)?[A-Z]\w*(?:Exception|Error|Failure|Timeout)\s*:\s*(.+)",
                  first)
    if m:
        first = m.group(1)
    else:
        first = re.sub(r"^[\w.$]*(?:Exception|Error|Failure|Timeout)\s*:\s*", "", first)
    # Case-preserving mask, so the label stays readable in the report.
    sig = " ".join(_mask(first).split()[:words])[:max_len]
    # Trim ragged trailing punctuation left by truncation (e.g. "By.chained(").
    return sig.rstrip(" ([{:,-").strip()


def _fingerprint(reason: str) -> str:
    """Stable root-cause signature, robust to volatile details.

    Preference order:
      1. ``Exception @ AppClass.method`` — the most actionable form.
      2. ``Exception: <masked message template>`` — when only framework frames
         exist, the templated message discriminates root causes without splitting
         on element ids or numbers.
      3. Empty — nothing parseable; the text-clustering fallback handles it.
    """
    exc = _root_exception(reason)
    frame = _app_frame(reason)
    if exc and frame:
        return f"{exc} @ {frame}"
    if exc:
        sig = _message_signature(reason)
        return f"{exc}: {sig}" if sig else exc
    return ""


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
    # Failures whose text is pure HTTP/device-log noise carry no root cause. Left
    # to the text clusterer they become one singleton each (unique timestamps and
    # session ids), burying the real clusters — so they get a single shared bucket.
    sig_groups: dict[str, list[int]] = {}
    fallback_idx: list[int] = []
    low_signal_idx: list[int] = []
    for i, fp in enumerate(fingerprints):
        if fp:
            sig_groups.setdefault(fp, []).append(i)
        elif not _HAS_ERROR_SIGNAL.search(reasons[i]):
            low_signal_idx.append(i)
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

    # --- Stage 3: one shared bucket for diagnostically-empty text ---
    low_signal_cid = None
    if low_signal_idx:
        low_signal_cid = next_id
        for i in low_signal_idx:
            cluster_key[i] = next_id
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
        if cid == low_signal_cid:
            fp = "No diagnostic error in logs (HTTP/device-log noise only)"
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


def _shrink(raw_score: float, runs: int) -> float:
    """Shrink a flakiness score toward 0 for small samples.

    With 2 runs, one pass->fail flip already yields a 100% flip rate, which would
    otherwise outrank a test proven flaky over 40 runs. The factor
    ``runs / (runs + smoothing)`` discounts thin evidence without hiding it.
    """
    factor = runs / (runs + settings.flakiness_smoothing) if runs > 0 else 0.0
    return float(np.clip(raw_score * factor, 0.0, 1.0))


def _rank_flakiness(result: pd.DataFrame) -> pd.DataFrame:
    """Well-evidenced scenarios first, then low-confidence ones; both by score.

    Cross-build history is judged on the number of builds (too few builds means
    run-over-run flakiness simply cannot be established); the single-build proxy
    falls back to the session count.
    """
    if "builds" in result.columns:
        result["low_sample"] = result["builds"] < settings.flakiness_min_builds
    else:
        result["low_sample"] = result["runs"] < settings.flakiness_min_runs
    return result.sort_values(
        ["low_sample", "flakiness_index"], ascending=[True, False]
    ).reset_index(drop=True)


def _flakiness_in_build(df: pd.DataFrame) -> pd.DataFrame:
    """In-build flakiness proxy (single build, no history).

    Blends status variance (peaks at a 50/50 pass/fail split) and duration
    coefficient of variation. Used as a fallback until history accumulates.
    """
    def _score(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        fail_rate = group["is_failure"].mean()
        status_var = 4.0 * fail_rate * (1.0 - fail_rate)
        duration_var = _duration_cov(group["duration"].to_numpy(dtype=float))
        raw = float(np.clip(0.65 * status_var + 0.35 * duration_var, 0.0, 1.0))
        return pd.Series(
            {
                "runs": n,
                "failure_rate": round(float(fail_rate), 3),
                "duration_cov": round(duration_var, 3),
                "flip_rate": 0.0,
                "raw_flakiness": round(raw, 3),
                "flakiness_index": round(_shrink(raw, n), 3),
            }
        )

    result = df.groupby("name").apply(_score, include_groups=False).reset_index()
    # A mixed-dtype Series coerces counts to float; restore integers for display.
    result["runs"] = result["runs"].astype(int)
    return _rank_flakiness(result)


def _flakiness_from_history(history: pd.DataFrame) -> pd.DataFrame:
    """True run-over-run flakiness using cross-build history.

    A scenario often produces *several* sessions inside one build — one per device,
    or one per case in a data-driven test. Those are not sequential re-runs, so
    counting pass<->fail transitions across raw sessions would invent "flips" that
    are really just different cases. Instead we:

      1. collapse each (scenario, build) to one outcome (failed if any session
         failed), ordered by when the build ran;
      2. measure ``flip_rate`` — pass<->fail alternation across *builds*, the
         genuine run-over-run flakiness signal;
      3. measure ``mixed_rate`` — the share of builds where the same scenario both
         passed and failed, which flags device-specific or parameter-specific
         instability rather than time-based flakiness.
    """
    def _score(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        # 1. one row per build, chronologically.
        per_build = (
            group.groupby("build_id")
            .agg(
                any_fail=("is_failure", "max"),
                all_fail=("is_failure", "min"),
                first_seen=("created_at", "min"),
            )
            .sort_values("first_seen")
        )
        n_builds = len(per_build)
        outcomes = per_build["any_fail"].astype(int).to_numpy()

        # 2. cross-build alternation.
        if n_builds > 1:
            flips = int(np.abs(np.diff(outcomes)).sum())
            flip_rate = flips / (n_builds - 1)
        else:
            flip_rate = 0.0  # undefined with a single build

        # 3. within-build disagreement (some sessions passed, some failed).
        mixed = (per_build["any_fail"].astype(int) - per_build["all_fail"].astype(int)) > 0
        mixed_rate = float(mixed.mean()) if n_builds else 0.0

        fail_rate = float(group["is_failure"].mean()) if n else 0.0
        status_var = 4.0 * fail_rate * (1.0 - fail_rate)
        duration_var = _duration_cov(group["duration"].to_numpy(dtype=float))
        raw = float(np.clip(
            0.45 * flip_rate + 0.25 * mixed_rate + 0.20 * status_var + 0.10 * duration_var,
            0.0, 1.0,
        ))
        # Shrink on the number of BUILDS: that is the evidence flakiness needs.
        build_factor = n_builds / (n_builds + settings.flakiness_build_smoothing)
        return pd.Series(
            {
                "runs": n,
                "builds": n_builds,
                "failure_rate": round(fail_rate, 3),
                "duration_cov": round(duration_var, 3),
                "flip_rate": round(float(flip_rate), 3),
                "mixed_rate": round(mixed_rate, 3),
                "raw_flakiness": round(raw, 3),
                "flakiness_index": round(float(np.clip(raw * build_factor, 0.0, 1.0)), 3),
            }
        )

    result = history.groupby("name").apply(_score, include_groups=False).reset_index()
    # A mixed-dtype Series coerces counts to float; restore integers for display.
    result[["runs", "builds"]] = result[["runs", "builds"]].astype(int)
    return _rank_flakiness(result)


def flakiness_index(
    df: pd.DataFrame, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-scenario flakiness in [0, 1].

    Uses the per-build model whenever build ids are available — even for a single
    build, where it honestly reports ``flip_rate`` 0 (run-over-run flakiness is
    undefined) while still surfacing ``mixed_rate``. Only with no history at all
    does it fall back to the single-build proxy.
    """
    if history is not None and not history.empty and "build_id" in history:
        return _flakiness_from_history(history)
    return _flakiness_in_build(df)


def classify_failures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each failure a taxonomy (category, owner) and return a breakdown.

    Returns (classified_failures, breakdown). The breakdown aggregates counts per
    (category, owner) so the report can answer "how much is our bug vs test flake
    vs infra?" at a glance.
    """
    failures = df[df["is_failure"]].copy()
    cols = ["category", "owner", "count"]
    if failures.empty:
        return failures.assign(category=[], owner=[]), pd.DataFrame(columns=cols)

    results = [
        classify(
            r.get("reason", ""), r.get("log_text", ""),
            float(r.get("duration", 0.0)), r.get("status", ""),
        )
        for _, r in failures.iterrows()
    ]
    failures["category"] = [c for c, _ in results]
    failures["owner"] = [o for _, o in results]
    # Actionable atoms: which locator/screen the failure was looking for.
    failures["locator"] = [
        extract_locator(f"{r.get('reason','')}\n{r.get('log_text','')}")
        for _, r in failures.iterrows()
    ]
    # The verbatim selector, kept alongside the normalised value: grouping needs
    # the element, fixing needs the exact expression.
    failures["selector"] = [
        extract_selector(f"{r.get('reason','')}\n{r.get('log_text','')}")
        for _, r in failures.iterrows()
    ]
    failures["screen"] = [
        extract_screen(f"{r.get('reason','')}\n{r.get('log_text','')}")
        for _, r in failures.iterrows()
    ]

    breakdown = (
        failures.groupby(["category", "owner"]).size().reset_index(name="count")
        .sort_values("count", ascending=False).reset_index(drop=True)
    )
    breakdown["description"] = breakdown["category"].map(category_description)
    return failures, breakdown


def _label_clusters_with_category(
    clusters: list[FailureCluster], classified: pd.DataFrame
) -> None:
    """Give each cluster the dominant taxonomy category among its sessions.

    Reuses the already-computed classification rather than re-running the rules, and
    fills the one label that is present for every cluster: an exception class exists
    only when the framework threw one, whereas a category always resolves.
    """
    if classified.empty or "category" not in classified:
        return
    by_session = dict(zip(classified["session_id"], classified["category"]))
    owner_by_category = dict(zip(classified["category"], classified["owner"]))
    for cluster in clusters:
        cats = [by_session[s] for s in cluster.session_ids if s in by_session]
        if not cats:
            continue
        dominant = Counter(cats).most_common(1)[0][0]
        cluster.category = dominant
        cluster.owner = owner_by_category.get(dominant, "")


def locator_hotspots(
    classified: pd.DataFrame, top: int = 6, min_tests: int = 2
) -> tuple[pd.DataFrame, int]:
    """Rank failing UI locators by blast radius, and count the ones left out.

    Returns (hotspots, omitted). A locator that breaks a *single* test is not a
    hotspot: it carries no more information than the test itself, which the
    categories section already lists. Requiring ``min_tests`` keeps the panel to
    locators whose fix pays off across several tests, and the omitted count is
    returned so the report can say so rather than silently hiding them.
    """
    if classified.empty or "locator" not in classified:
        return pd.DataFrame(), 0
    hits = classified[classified["locator"].astype(str).str.len() > 1]
    if hits.empty:
        return pd.DataFrame(), 0

    def _selectors(group: pd.Series) -> list[str]:
        """The single most specific selector for this element.

        Listing every variant was the main driver of the panel's height; the longest
        is the most specific, and so the most useful one to act on.
        """
        seen = {s for s in group.tolist() if s}
        return sorted(seen, key=len, reverse=True)[:1]

    out = (
        hits.groupby("locator")
        .agg(
            failures=("session_id", "count"),
            tests_affected=("name", "nunique"),
            devices=("device", "nunique"),
            example_test=("name", "first"),
            selectors=("selector", _selectors) if "selector" in hits else ("name", "first"),
        )
        .reset_index()
        .sort_values(["tests_affected", "failures"], ascending=False)
        .reset_index(drop=True)
    )
    omitted = int((out["tests_affected"] < min_tests).sum())
    out = out[out["tests_affected"] >= min_tests].head(top).reset_index(drop=True)
    return out, omitted


def run_triage(
    df: pd.DataFrame, history: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Convenience wrapper returning every triage artifact in one dict."""
    # Derive the business dimension before any inference runs on it.
    df = df.copy()
    df["feature_area"] = df["name"].map(feature_area)

    clusters, annotated = cluster_failures(df)
    classified, categories = classify_failures(df)
    _label_clusters_with_category(clusters, classified)

    _locator_rows, _locator_omitted = locator_hotspots(classified)
    findings = inference.significant_findings(df)
    cat_table, cat_p, cat_resid = inference.category_by_dimension(classified, "os_version")

    return {
        "clusters": clusters,
        "annotated_failures": annotated,
        "device_anomaly": device_anomaly(df),
        "flakiness": flakiness_index(df, history),
        "classified_failures": classified,
        "categories": categories,
        # --- data-science layer ---
        "feature_area_health": _feature_area_health(df),
        "findings": findings,
        "category_by_os": (cat_table, cat_p, cat_resid),
        "locator_hotspots": _locator_rows,
        "locator_omitted": _locator_omitted,
        "duration_outliers": inference.duration_outliers(df),
        "time_split": inference.time_split(df),
        "perf_vs_failure": inference.performance_vs_failure(df),
        "perf_hotspots": _perf_hotspots(df),
        # Always present on a multi-platform run so a combined pass rate can never
        # hide one platform's regression behind the other's green.
        "platform_breakdown": _platform_breakdown(df),
        "area_by_platform": _area_by_platform(df),
    }


def _platform_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per-platform (and per-project) pass/fail summary. Empty for a single platform."""
    if "platform" not in df or df["platform"].nunique() < 2:
        return pd.DataFrame()
    group = ["platform"] + (["project"] if "project" in df else [])
    out = (
        df.groupby(group)
        .agg(sessions=("session_id", "count"), failures=("is_failure", "sum"),
             tests=("name", "nunique"))
        .reset_index()
    )
    out["failure_rate"] = (out["failures"] / out["sessions"]).round(3)
    return out.sort_values("failure_rate", ascending=False).reset_index(drop=True)


def _area_by_platform(df: pd.DataFrame) -> pd.DataFrame:
    """Failure rate per business area x platform — 'is this a product bug or a
    platform bug?' answered for every area at once. Empty for a single platform.
    """
    if "platform" not in df or df["platform"].nunique() < 2 or "feature_area" not in df:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index="feature_area", columns="platform", values="is_failure", aggfunc="mean"
    ).round(3)
    counts = df.pivot_table(
        index="feature_area", columns="platform", values="session_id", aggfunc="count"
    )
    # Drop cells with too little data to be worth showing.
    pivot = pivot.where(counts >= settings.inference_min_group)
    return pivot.dropna(how="all").reset_index()


def _perf_hotspots(df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Per-scenario performance summary, most CPU-hungry first.

    Empty unless profiling was fetched (--profile), and app CPU is preferred over
    device CPU because the device figure includes the platform's own work.
    """
    if "app_cpu_mean" not in df.columns:
        return pd.DataFrame()
    numeric = df.assign(
        app_cpu_mean=pd.to_numeric(df["app_cpu_mean"], errors="coerce"),
        app_cpu_max=pd.to_numeric(df.get("app_cpu_max"), errors="coerce"),
        app_mem_max_mb=pd.to_numeric(df.get("app_mem_max_mb"), errors="coerce"),
    ).dropna(subset=["app_cpu_mean"])
    if numeric.empty:
        return pd.DataFrame()
    out = (
        numeric.groupby("name")
        .agg(sessions=("session_id", "count"),
             cpu_mean=("app_cpu_mean", "mean"),
             cpu_max=("app_cpu_max", "max"),
             mem_max_mb=("app_mem_max_mb", "max"))
        .reset_index()
        .round(1)
        .sort_values("cpu_mean", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
    return out


def _feature_area_health(df: pd.DataFrame) -> pd.DataFrame:
    """Per-business-area pass/fail summary, worst first."""
    if "feature_area" not in df:
        return pd.DataFrame()
    out = (
        df.groupby("feature_area")
        .agg(sessions=("session_id", "count"), failures=("is_failure", "sum"),
             tests=("name", "nunique"))
        .reset_index()
    )
    out["failure_rate"] = (out["failures"] / out["sessions"]).round(3)
    return out.sort_values("failure_rate", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from ingestor import ingest

    frame, _ = ingest(source="file")
    out = run_triage(frame)
    print(f"Discovered {len(out['clusters'])} root-cause clusters:")
    for c in out["clusters"]:
        print(f"  [{c.size:>3}]  conf={c.confidence:.2f}  {c.label}")
