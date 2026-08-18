"""Tests for the presentation layer: template data shaping and chart labelling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from report_generator import _device_heatmap, _records
from triage_engine import device_anomaly, platform_comparability


def _mixed_platform_sessions() -> pd.DataFrame:
    """One iOS device and one Android device, each with its own version numbers."""
    rows = []
    for i in range(6):
        rows.append({"session_id": f"i{i}", "device": "iPhone 15 Pro", "platform": "ios",
                     "os_version": "17.3", "is_failure": i < 3})
        rows.append({"session_id": f"a{i}", "device": "Samsung Galaxy S24 Ultra",
                     "platform": "android", "os_version": "14.0", "is_failure": i < 1})
    return pd.DataFrame(rows)


def test_records_converts_nan_to_none():
    """Jinja's `is not none` is true for a float NaN, so a pivot with no data for
    a combination rendered a literal "nan%". NaN must arrive as None instead."""
    frame = pd.DataFrame({"feature_area": ["accounts"], "android": [0.62], "ios": [np.nan]})
    record = _records(frame)[0]
    assert record["ios"] is None
    assert record["android"] == 0.62


def test_records_preserves_empty_and_limit():
    assert _records(None) == []
    assert _records(pd.DataFrame()) == []
    frame = pd.DataFrame({"a": [1, 2, 3]})
    assert len(_records(frame, limit=2)) == 2


def test_device_anomaly_labels_each_platform_correctly():
    """The iPhone row must not be labelled with an Android version."""
    risk = device_anomaly(_mixed_platform_sessions())
    labels = dict(zip(risk["device"], risk["os_label"]))
    assert labels["iPhone 15 Pro"] == "iOS 17.3"
    assert labels["Samsung Galaxy S24 Ultra"] == "Android 14.0"
    # The raw version is left untouched, so persisted history stays comparable.
    assert set(risk["os_version"]) == {"17.3", "14.0"}


def test_heatmap_axis_uses_platform_qualified_labels():
    html = _device_heatmap(device_anomaly(_mixed_platform_sessions()))
    assert "iOS 17.3" in html
    assert "Android 17.3" not in html   # the bug: one family hardcoded for all


def test_device_anomaly_without_platform_column_still_works():
    """History rows written before `platform` was persisted have no such column."""
    df = _mixed_platform_sessions().drop(columns=["platform"])
    risk = device_anomaly(df)
    assert set(risk["os_label"]) == {"17.3", "14.0"}   # unqualified, never guessed


# --- platform comparability -------------------------------------------------
def _two_platform_run(android_version: str, ios_version: str,
                      android_day: str, ios_day: str) -> pd.DataFrame:
    rows = []
    for i in range(5):
        rows.append({"session_id": f"a{i}", "platform": "android", "is_failure": False,
                     "app_version": android_version,
                     "created_at": f"2026-08-{android_day}T10:0{i}:00.000Z"})
        rows.append({"session_id": f"i{i}", "platform": "ios", "is_failure": True,
                     "app_version": ios_version,
                     "created_at": f"2026-08-{ios_day}T10:0{i}:00.000Z"})
    return pd.DataFrame(rows)


def test_same_build_same_day_is_comparable():
    result = platform_comparability(_two_platform_run("2.18.0", "2.18.0", "13", "13"))
    assert result["comparable"] is True
    assert result["reasons"] == []


def test_different_app_builds_are_flagged():
    result = platform_comparability(
        _two_platform_run("2.16.0-staging", "2.18.0", "13", "13"))
    assert result["comparable"] is False
    assert any("different app builds" in r for r in result["reasons"])


def test_runs_far_apart_are_flagged():
    """Android on Aug 5 against iOS on Aug 13 confounds platform with time, so a
    gap between the columns is equally explicable as a regression between builds."""
    result = platform_comparability(_two_platform_run("2.18.0", "2.18.0", "05", "13"))
    assert result["comparable"] is False
    assert any("8 days apart" in r for r in result["reasons"])


def test_single_platform_run_is_never_flagged():
    df = pd.DataFrame([{"session_id": "a", "platform": "android", "is_failure": False,
                        "app_version": "2.18.0", "created_at": "2026-08-13T10:00:00.000Z"}])
    assert platform_comparability(df)["comparable"] is True


def test_stale_platform_is_named_so_the_reader_knows_which_column_to_distrust():
    """A platform whose last run is days older reports an older build's failures,
    so a green cell there is not evidence of a green build today."""
    result = platform_comparability(_two_platform_run("2.18.0", "2.18.0", "05", "13"))
    assert len(result["stale"]) == 1
    assert "android is 8 days behind" in result["stale"][0]
    assert "2026-08-05" in result["stale"][0]


def test_platforms_run_together_have_no_stale_warning():
    assert platform_comparability(
        _two_platform_run("2.18.0", "2.18.0", "13", "13"))["stale"] == []


def test_area_by_platform_carries_counts_beside_each_rate():
    """A bare 0% hides whether it is 0-of-7-passed or a rounding artifact."""
    from triage_engine import _area_by_platform
    rows = []
    for i in range(6):
        rows.append({"session_id": f"a{i}", "platform": "android",
                     "feature_area": "autosave", "is_failure": False})
        rows.append({"session_id": f"i{i}", "platform": "ios",
                     "feature_area": "autosave", "is_failure": i < 5})
    out = _area_by_platform(pd.DataFrame(rows))
    row = out.iloc[0]
    assert row["android"] == 0.0 and row["n__android"] == 6 and row["f__android"] == 0
    assert row["n__ios"] == 6 and row["f__ios"] == 5


def test_platform_names_excludes_the_count_companions():
    """Header generation must not turn n__/f__ helper columns into platforms."""
    from report_generator import _platform_names
    frame = pd.DataFrame({"feature_area": ["gold"], "android": [0.1], "ios": [0.2],
                          "n__android": [7], "f__android": [1],
                          "n__ios": [7], "f__ios": [2]})
    assert _platform_names(frame) == ["android", "ios"]
