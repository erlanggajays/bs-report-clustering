"""Statistical inference: turn descriptive counts into defensible claims.

A raw failure rate per device is only an observation — it may be noise. This module
tests whether an association is real, and how large it is:

* ``attribute`` — for one dimension (device, os_version, feature_area, …), compare
  each level against the rest with **Fisher's exact test** (exact for the small
  samples typical of a test suite) and report an **odds ratio with a confidence
  interval**, so findings carry both significance and effect size.
* ``significant_findings`` — run every dimension and return only what clears the
  significance threshold, ranked by effect size.
* ``duration_outliers`` — robust (median/MAD) outlier detection on runtimes, which
  catches hangs and retries that a mean-based rule misses.
* ``time_split`` — attribute runtime to BrowserStack infrastructure vs user code
  using the ``insights`` figures BrowserStack already returns.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from config import settings


def _odds_ratio_ci(a: int, b: int, c: int, d: int, z: float = 1.96) -> tuple[float, float, float]:
    """Odds ratio for [[a,b],[c,d]] with a log-scale Woolf confidence interval.

    Haldane-Anscombe correction (+0.5) keeps the estimate finite when a cell is 0.
    """
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a_ * d_) / (b_ * c_)
    se = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    return or_, or_ * math.exp(-z * se), or_ * math.exp(z * se)


def attribute(
    df: pd.DataFrame, dimension: str, min_group: int | None = None
) -> pd.DataFrame:
    """Test each level of ``dimension`` for association with failure.

    Every level is compared against all other levels combined (one-vs-rest), which
    keeps each test a clean 2x2 and avoids the multi-way sparsity that would make a
    single chi-square unusable on a small suite.
    """
    min_group = settings.inference_min_group if min_group is None else min_group
    if dimension not in df.columns or df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    total_fail = int(df["is_failure"].sum())
    total_pass = len(df) - total_fail

    for level, group in df.groupby(dimension):
        n = len(group)
        if n < min_group or not level:
            continue
        fail = int(group["is_failure"].sum())
        others_fail = total_fail - fail
        others_pass = total_pass - (n - fail)
        if (others_fail + others_pass) < min_group:
            continue  # nothing to compare against

        table = [[fail, n - fail], [others_fail, others_pass]]
        try:
            _, p = fisher_exact(table)
        except ValueError:
            continue
        or_, lo, hi = _odds_ratio_ci(fail, n - fail, others_fail, others_pass,
                                     settings.wilson_z)
        rows.append({
            "dimension": dimension,
            "level": str(level),
            "sessions": n,
            "failures": fail,
            "failure_rate": round(fail / n, 3),
            "baseline_rate": round(others_fail / max(1, others_fail + others_pass), 3),
            "odds_ratio": round(or_, 2),
            "ci_low": round(lo, 2),
            "ci_high": round(hi, 2),
            "p_value": p,
            "significant": bool(p < settings.inference_alpha),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("p_value").reset_index(drop=True)


def _confounded_with(df: pd.DataFrame, dimension: str, level: str,
                     others: list[str]) -> list[str]:
    """Dimensions whose levels select exactly the same sessions as this one.

    In a small device lab a device often maps 1:1 to an OS version, so "S24 Ultra"
    and "Android 14" describe the *same* rows. Reporting both as separate findings
    would imply two independent problems, so we mark the confound instead.
    """
    mask = df[dimension].astype(str) == level
    confounded = []
    for other in others:
        if other == dimension or other not in df.columns:
            continue
        for other_level, group in df.groupby(other):
            other_mask = df[other].astype(str) == str(other_level)
            if other_mask.equals(mask) and len(group) > 0:
                confounded.append(f"{other}={other_level}")
    return confounded


def significant_findings(
    df: pd.DataFrame, dimensions: list[str] | None = None
) -> pd.DataFrame:
    """Significant failure associations across all dimensions, strongest first.

    Only levels whose failure rate is *above* baseline are returned — a level that
    is significantly *better* than average is not a finding to act on. Findings that
    describe an identical set of sessions are collapsed to one row, with the
    equivalent dimensions listed in ``confounded_with``, so a single environment
    is not double-counted as several problems.
    """
    # platform first: on a cross-platform run it is usually the strongest split.
    dimensions = dimensions or [
        "platform", "device", "os_version", "feature_area", "app_version",
    ]
    frames = [attribute(df, d) for d in dimensions]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    allf = pd.concat(frames, ignore_index=True)
    hits = allf[allf["significant"] & (allf["failure_rate"] > allf["baseline_rate"])].copy()
    if hits.empty:
        return hits

    hits["confounded_with"] = [
        ", ".join(_confounded_with(df, r.dimension, r.level, dimensions))
        for r in hits.itertuples()
    ]
    # Collapse rows describing the same session set: keep the first, note the rest.
    seen: set[frozenset] = set()
    keep: list[int] = []
    for idx, r in hits.iterrows():
        key = frozenset(df.index[df[r["dimension"]].astype(str) == r["level"]])
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    hits = hits.loc[keep]
    return hits.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


def category_by_dimension(
    failures: pd.DataFrame, dimension: str = "os_version"
) -> tuple[pd.DataFrame, float | None, pd.DataFrame]:
    """Cross failure categories with a dimension.

    Returns (contingency table, chi-square p-value, standardised residuals). A
    residual beyond +/-2 marks a cell that is over- or under-represented well
    beyond chance, which localises a failure type to an environment.
    """
    if failures.empty or "category" not in failures or dimension not in failures:
        return pd.DataFrame(), None, pd.DataFrame()
    table = pd.crosstab(failures["category"], failures[dimension])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return table, None, pd.DataFrame()
    try:
        _, p, _, expected = chi2_contingency(table)
    except ValueError:
        return table, None, pd.DataFrame()
    residuals = ((table - expected) / np.sqrt(expected)).round(2)
    return table, float(p), residuals


def duration_outliers(df: pd.DataFrame, z_threshold: float = 3.5) -> pd.DataFrame:
    """Robust per-scenario runtime outliers using the median/MAD modified z-score.

    Mean and standard deviation are themselves distorted by the outlier we are
    hunting, so we use the median absolute deviation. 3.5 is the conventional cut.
    """
    if df.empty or "duration" not in df:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for name, group in df.groupby("name"):
        d = group["duration"].to_numpy(dtype=float)
        if len(d) < settings.inference_min_group:
            continue
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        if mad > 0:
            scores = 0.6745 * (d - med) / mad          # modified z-score
        else:
            # MAD collapses to 0 when most runtimes are identical — precisely when
            # a single hang is most obvious. Fall back to the scaled mean absolute
            # deviation; if that is 0 too there is genuinely no variation.
            mean_ad = float(np.mean(np.abs(d - med)))
            if mean_ad <= 0:
                continue
            scores = (d - med) / (1.253314 * mean_ad)
        worst = int(np.argmax(scores))
        if scores[worst] < z_threshold:
            continue
        rows.append({
            "name": name,
            "sessions": len(d),
            "median_seconds": round(med, 1),
            "slowest_seconds": round(float(d[worst]), 1),
            "ratio": round(float(d[worst]) / med, 1) if med else 0.0,
            "modified_z": round(float(scores[worst]), 1),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("modified_z", ascending=False).reset_index(drop=True)


def time_split(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate BrowserStack-infrastructure time vs user (test code) time.

    Answers "is the suite slow because of our tests or the platform?" using the
    ``insights`` figures already present in the session payload.
    """
    if "browserstack_seconds" not in df or "user_seconds" not in df:
        return {}
    bs = float(df["browserstack_seconds"].fillna(0).sum())
    us = float(df["user_seconds"].fillna(0).sum())
    total = bs + us
    if total <= 0:
        return {}
    return {
        "browserstack_seconds": round(bs, 1),
        "user_seconds": round(us, 1),
        "browserstack_pct": round(bs / total * 100, 1),
        "user_pct": round(us / total * 100, 1),
    }
