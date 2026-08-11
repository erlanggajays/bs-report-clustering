"""Device/app performance profiling from BrowserStack's app-profiling endpoint.

The endpoint returns one sample every ~1-2s for the life of a session::

    {"ts": 1785843415, "cpu": 80, "mem": 11548.8, "mema": 5739.79,
     "batt": 79, "temp": 32.2,
     "comgojekgopaynightlystaging_cpu": 24,      # app CPU %
     "comgojekgopaynightlystaging_mem": 466.434, # app memory MB
     "comgojekgopaynightlystaging_netr": null, "..._nets": null}

Notes that shape this module:

* The app-specific keys are **named after the package with punctuation stripped**,
  so they are discovered at runtime rather than hardcoded.
* ``mem`` is the device's *total* RAM (constant), so memory in use is ``mem - mema``.
* App samples are sparse — ``_mem`` may appear only a few times per session — so
  every statistic reports how many samples it is based on and slope-based figures
  refuse to answer below a minimum count.
* ``batt`` and ``temp`` barely move across a 60-90s test. They are summarised as a
  delta and left to longer sessions to make meaningful.
* There is no frame/FPS series here, so rendering smoothness cannot be measured.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

# Device-level keys, as opposed to the dynamically named per-app ones.
_DEVICE_KEYS = {"ts", "cpu", "mem", "mema", "batt", "temp"}
_APP_KEY_RE = re.compile(r"^([a-z0-9]+)_(cpu|mem|netr|nets)$", re.I)

# Slope-based figures (memory growth) need enough points to mean anything.
MIN_SLOPE_SAMPLES = 8


def app_key_prefix(rows: list[dict[str, Any]]) -> str:
    """Discover the per-app column prefix (e.g. "comgojekgopaynightlystaging")."""
    for row in rows:
        for key in row:
            if key in _DEVICE_KEYS:
                continue
            m = _APP_KEY_RE.match(key)
            if m:
                return m.group(1)
    return ""


def _series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    """Numeric series for a key, dropping the frequent nulls."""
    vals = [r.get(key) for r in rows]
    return np.array([float(v) for v in vals if v is not None], dtype=float)


def _slope_per_minute(rows: list[dict[str, Any]], key: str) -> float | None:
    """Least-squares slope of a metric in units per minute, or None if too sparse.

    Used for memory growth: a clearly positive slope within a single session is the
    signature of a leak, whereas a flat line is healthy.
    """
    pairs = [(r.get("ts"), r.get(key)) for r in rows]
    pairs = [(float(t), float(v)) for t, v in pairs if t is not None and v is not None]
    if len(pairs) < MIN_SLOPE_SAMPLES:
        return None
    ts = np.array([p[0] for p in pairs])
    vals = np.array([p[1] for p in pairs])
    span = ts.max() - ts.min()
    if span <= 0:
        return None
    ts = (ts - ts.min()) / 60.0                      # minutes since session start
    slope = float(np.polyfit(ts, vals, 1)[0])
    return round(slope, 2)


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "p95": None, "max": None}
    return {
        "mean": round(float(values.mean()), 1),
        "p95": round(float(np.percentile(values, 95)), 1),
        "max": round(float(values.max()), 1),
    }


def parse_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce one session's profiling series to comparable per-session figures.

    Returns ``{}`` for an empty/unusable payload so callers can treat profiling as
    optional. Every ``*_samples`` field records the evidence behind a figure —
    important because app metrics are sampled sparsely.
    """
    if not rows:
        return {}
    prefix = app_key_prefix(rows)

    device_cpu = _series(rows, "cpu")
    app_cpu = _series(rows, f"{prefix}_cpu") if prefix else np.array([])
    app_mem = _series(rows, f"{prefix}_mem") if prefix else np.array([])
    total = _series(rows, "mem")
    avail = _series(rows, "mema")
    batt = _series(rows, "batt")
    temp = _series(rows, "temp")

    # Device memory in use; "mem" is total RAM, so it only means something as a pair.
    used = np.array([])
    if total.size and avail.size:
        n = min(total.size, avail.size)
        used = total[:n] - avail[:n]

    ts = _series(rows, "ts")
    duration = float(ts.max() - ts.min()) if ts.size > 1 else 0.0

    dev = _stats(device_cpu)
    acpu = _stats(app_cpu)
    amem = _stats(app_mem)
    umem = _stats(used)

    return {
        "profile_samples": len(rows),
        "profile_seconds": round(duration, 1),
        # Device CPU includes the platform's own work, so the app figure is the
        # trustworthy one; both are kept so they can be compared.
        "device_cpu_mean": dev["mean"],
        "device_cpu_max": dev["max"],
        "app_cpu_mean": acpu["mean"],
        "app_cpu_p95": acpu["p95"],
        "app_cpu_max": acpu["max"],
        "app_cpu_samples": int(app_cpu.size),
        "app_mem_max_mb": amem["max"],
        "app_mem_samples": int(app_mem.size),
        "app_mem_growth_mb_per_min": _slope_per_minute(rows, f"{prefix}_mem") if prefix else None,
        # Device memory is reported as context only. Its *slope* is deliberately not
        # exposed: available RAM swings with Android's cache management and every
        # other process on the device, so over a 60-90s test it produces
        # alarming-looking figures that say nothing about the app under test. The
        # app-level slope above is the trustworthy leak signal.
        "device_mem_used_max_mb": umem["max"],
        # Flat over a short test; meaningful only for long-running sessions.
        "battery_delta_pct": round(float(batt[0] - batt[-1]), 1) if batt.size > 1 else None,
        "temp_max_c": round(float(temp.max()), 1) if temp.size else None,
        "temp_delta_c": round(float(temp.max() - temp.min()), 1) if temp.size else None,
    }



PROFILE_COLUMNS = [
    "profile_samples", "profile_seconds",
    "device_cpu_mean", "device_cpu_max",
    "app_cpu_mean", "app_cpu_p95", "app_cpu_max", "app_cpu_samples",
    "app_mem_max_mb", "app_mem_samples", "app_mem_growth_mb_per_min",
    "device_mem_used_max_mb",
    "battery_delta_pct", "temp_max_c", "temp_delta_c",
]
