"""Tests for app-profiling parsing, using the real BrowserStack payload shape."""
from __future__ import annotations

import pandas as pd

from inference import performance_vs_failure
from profiling import MIN_SLOPE_SAMPLES, app_key_prefix, parse_profile

# Abridged from a real /appprofiling/v2 response. Note the dynamically named app
# keys, the frequent nulls, and that "mem" is total device RAM (constant).
REAL_ROWS = [
    {"ts": 1785843415, "cpu": 80, "mem": 11548.8, "mema": 5739.79, "batt": 79, "temp": 32.2,
     "comgojekgopaynightlystaging_cpu": None, "comgojekgopaynightlystaging_mem": None,
     "comgojekgopaynightlystaging_netr": None, "comgojekgopaynightlystaging_nets": None},
    {"ts": 1785843419, "cpu": 94, "mem": 11548.8, "mema": 5516.37, "batt": 79, "temp": 32.2,
     "comgojekgopaynightlystaging_cpu": 24, "comgojekgopaynightlystaging_mem": None},
    {"ts": 1785843421, "cpu": 94, "mem": 11548.8, "mema": 5312.93, "batt": 79, "temp": 32.2,
     "comgojekgopaynightlystaging_cpu": 22, "comgojekgopaynightlystaging_mem": None},
    {"ts": 1785843438, "cpu": 26, "mem": 11548.8, "mema": 5403.40, "batt": 79, "temp": 32.2,
     "comgojekgopaynightlystaging_cpu": 0, "comgojekgopaynightlystaging_mem": 466.434},
    {"ts": 1785843448, "cpu": 84, "mem": 11548.8, "mema": 5393.25, "batt": 79, "temp": 32.2,
     "comgojekgopaynightlystaging_cpu": 39, "comgojekgopaynightlystaging_mem": None},
    {"ts": 1785843458, "cpu": 58, "mem": 11548.8, "mema": 5310.94, "batt": 79, "temp": 32.2,
     "comgojekgopaynightlystaging_cpu": 20, "comgojekgopaynightlystaging_mem": 667.796},
    {"ts": 1785843467, "cpu": 39, "mem": 11548.8, "mema": 5306.63, "batt": 79, "temp": 32.2,
     "comgojekgopaynightlystaging_cpu": 14, "comgojekgopaynightlystaging_mem": 667.796},
]


def test_app_key_prefix_is_discovered_not_hardcoded():
    """Columns are named after the app package with punctuation stripped, so they
    differ per app and must be found at runtime."""
    assert app_key_prefix(REAL_ROWS) == "comgojekgopaynightlystaging"
    assert app_key_prefix([{"ts": 1, "cpu": 2}]) == ""


def test_parse_real_payload():
    p = parse_profile(REAL_ROWS)
    assert p["profile_samples"] == len(REAL_ROWS)
    assert p["profile_seconds"] == 52.0
    # App CPU is taken from the app column, ignoring nulls.
    assert p["app_cpu_max"] == 39.0
    assert p["app_cpu_samples"] == 6
    assert p["app_mem_max_mb"] == 667.8
    assert p["app_mem_samples"] == 3
    # "mem" is total RAM; memory in use is total minus available.
    assert p["device_mem_used_max_mb"] > 5000


def test_empty_payload_is_optional_not_an_error():
    assert parse_profile([]) == {}


def test_sparse_series_refuses_a_growth_slope():
    """App memory is sampled rarely; a slope from 3 points would be noise dressed
    up as a leak, so it is withheld below MIN_SLOPE_SAMPLES."""
    p = parse_profile(REAL_ROWS)
    assert p["app_mem_samples"] < MIN_SLOPE_SAMPLES
    assert p["app_mem_growth_mb_per_min"] is None


def test_growth_slope_reported_when_samples_allow():
    rows = [
        {"ts": 1785843400 + i * 2, "cpu": 50, "mem": 11548.8, "mema": 5000,
         "batt": 79, "temp": 32.2, "app_cpu": 10, "app_mem": 400 + i * 10}
        for i in range(12)
    ]
    p = parse_profile(rows)
    assert p["app_mem_samples"] == 12
    # +10 MB every 2s == 300 MB/min.
    assert p["app_mem_growth_mb_per_min"] == 300.0


def test_flat_battery_and_temperature_report_zero_not_noise():
    """Over a 60-90s test these barely move; the figures must say so plainly."""
    p = parse_profile(REAL_ROWS)
    assert p["battery_delta_pct"] == 0.0
    assert p["temp_delta_c"] == 0.0
    assert p["temp_max_c"] == 32.2


def test_device_memory_slope_is_not_exposed():
    """Device-wide available RAM swings with Android's cache management, so its
    slope looks alarming while saying nothing about the app under test."""
    assert "device_mem_growth_mb_per_min" not in parse_profile(REAL_ROWS)


def test_performance_vs_failure_detects_a_difference():
    rows = []
    for i in range(20):
        rows.append({"session_id": f"p{i}", "is_failure": False, "app_cpu_mean": 10 + i % 3})
    for i in range(20):
        rows.append({"session_id": f"f{i}", "is_failure": True, "app_cpu_mean": 60 + i % 3})
    result = performance_vs_failure(pd.DataFrame(rows), ["app_cpu_mean"])
    row = result.iloc[0]
    assert bool(row["significant"]) is True
    assert row["failed_median"] > row["passed_median"]


def test_performance_vs_failure_needs_both_groups():
    rows = [{"session_id": f"p{i}", "is_failure": False, "app_cpu_mean": 10} for i in range(20)]
    assert performance_vs_failure(pd.DataFrame(rows), ["app_cpu_mean"]).empty
