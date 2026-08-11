"""Tests for the statistical inference layer and feature extraction."""
from __future__ import annotations

import pandas as pd

from features import extract_locator, extract_screen, extract_selector, feature_area
from inference import (
    attribute,
    duration_outliers,
    significant_findings,
    time_split,
)


def _sessions(spec: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """spec: (device, feature_name, n_pass, n_fail) -> a sessions frame."""
    rows = []
    i = 0
    for device, name, n_pass, n_fail in spec:
        for _ in range(n_pass):
            rows.append({"session_id": f"s{i}", "device": device, "name": name,
                         "os_version": "14.0", "is_failure": False, "duration": 60.0})
            i += 1
        for _ in range(n_fail):
            rows.append({"session_id": f"s{i}", "device": device, "name": name,
                         "os_version": "14.0", "is_failure": True, "duration": 60.0})
            i += 1
    return pd.DataFrame(rows)


# --- feature areas ---------------------------------------------------------
def test_feature_area_mapping():
    assert feature_area("testGoldHomeTopSectionVisible: Verify gold home top section") == "gold"
    assert feature_area("testProspectusShownOnBuyFromMFL2: ...") == "mutual-fund"
    assert feature_area("testFilterOrderHistoryByTermDepositMethod: Deposito by Jago") == "deposito"
    assert feature_area("testUserCompletesKycSelfie: upload a valid ID") == "kyc"
    assert feature_area("somethingCompletelyUnrelated") == "unmapped"
    assert feature_area("") == "unmapped"


def test_feature_area_precedence():
    """A test naming both a product and a cross-cutting step belongs to the product:
    here the Tabungan Usaha creation flow is under test and KYC is just where it
    lands. Ordering in DEFAULT_FEATURE_AREAS encodes that precedence."""
    name = "verifyUserNavigatesToKycSelfieWhenTappingCreateTabunganUsahaButton"
    assert feature_area(name) == "tabungan"


def test_locator_and_screen():
    reason = ("Can't locate an element by this strategy: "
              "By.chained({AppiumBy.accessibilityId: Lifetime interest})")
    assert extract_locator(reason) == "Lifetime interest"
    assert extract_locator("") == ""
    assert extract_screen("at com.gopay.home.WalletHomeActivity.onCreate") == "WalletHomeActivity"
    assert extract_screen("no screen here") == ""


def test_locator_escaped_quotes():
    """Appium embeds selectors in JSON, so quotes arrive as \\". This previously
    returned nothing even though a locator was present."""
    raw = r'view.View[contains(@content-desc, \"Rp\")][1]","SESSION_ID_PLACEHOLDER"'
    assert extract_locator(raw) == "Rp"


def test_locator_rejects_ids():
    """A UUID/hex value differs every session; grouping on it is meaningless."""
    raw = r'getAttribute() with args: ["content-desc","deadbeef-1111-2222-3333-444"]'
    assert extract_locator(raw) == ""


def test_locator_semantic_target():
    """Grouping must key on the element, not the selector string — otherwise
    unrelated elements sharing an xpath prefix are merged."""
    raw = ('selector":"//android.view.View[@content-desc=\'Estimated final balance (net)\']'
           '/following-sibling::android.view.View')
    # Closing parenthesis of "(net)" must survive delimiter trimming.
    assert extract_locator(raw) == "Estimated final balance (net)"


def test_same_element_2_strategies():
    a = extract_locator(r'//*[contains(@content-desc, "Mutual Fund")]')
    b = extract_locator(r'//android.widget.Button[@content-desc="Mutual Fund"]')
    assert a == b == "Mutual Fund"


def test_full_selector_kept_for_display():
    """Grouping normalises to the element, but fixing needs the exact expression,
    so the verbatim selector is preserved alongside it."""
    raw = (r'{"strategy":"xpath","selector":"//android.view.View[@content-desc='
           r'\"Estimated final balance (net)\"]/following-sibling::android.view.View'
           r'[contains(@content-desc, \"Rp\")]","context":""}')
    selector = extract_selector(raw)
    # The whole expression survives — not cut at the first inner quote.
    assert selector.startswith("//android.view.View[@content-desc=")
    assert selector.endswith("]")
    assert "following-sibling" in selector


def test_selector_allows_arbitrary_label_characters():
    """Enumerating allowed xpath characters loses labels containing & or %."""
    raw = r'//android.widget.TextView[@text="Renew principal & interest"]'
    assert extract_selector(raw) == raw


def test_grouping_key_and_selector_are_independent():
    a = r'//android.widget.Button[@content-desc="Mutual Fund"]'
    b = r'//*[contains(@content-desc, "Mutual Fund")]'
    # One row (same element) …
    assert extract_locator(a) == extract_locator(b)
    # … but both selectors are retained for display.
    assert extract_selector(a) != extract_selector(b)


# --- attribution -----------------------------------------------------------
def test_attribute_detects_real():
    # Device B fails far more often than A, with adequate sample sizes.
    df = _sessions([("A", "t1", 45, 5), ("B", "t2", 20, 30)])
    result = attribute(df, "device")
    b = result.loc[result.level == "B"].iloc[0]
    assert bool(b["significant"]) is True
    assert b["odds_ratio"] > 1
    assert b["ci_low"] > 1          # CI excludes 1 -> genuinely elevated


def test_attribute_no_association():
    df = _sessions([("A", "t1", 40, 10), ("B", "t2", 40, 10)])
    result = attribute(df, "device")
    assert not result["significant"].any()


def test_attribute_skips_tiny_groups():
    # Group B has 2 sessions: below inference_min_group, so no claim is made about
    # it, while A and C (large enough, and large enough to compare against) are.
    df = _sessions([("A", "t1", 45, 5), ("B", "t2", 0, 2), ("C", "t3", 20, 30)])
    result = attribute(df, "device")
    assert "B" not in set(result["level"])
    assert {"A", "C"} <= set(result["level"])


def test_findings_collapse_confound():
    """A device that maps 1:1 to an OS version must not be reported twice."""
    rows = []
    for i in range(50):
        rows.append({"session_id": f"a{i}", "device": "A", "os_version": "15.0",
                     "name": "t1", "is_failure": i < 5, "duration": 60.0})
    for i in range(50):
        rows.append({"session_id": f"b{i}", "device": "B", "os_version": "14.0",
                     "name": "t2", "is_failure": i < 30, "duration": 60.0})
    df = pd.DataFrame(rows)

    findings = significant_findings(df, ["device", "os_version"])
    assert len(findings) == 1                       # collapsed, not double-counted
    assert findings.iloc[0]["confounded_with"]      # confound recorded


def test_findings_skip_better():
    """A level that fails *less* than baseline is not an actionable finding."""
    rows = []
    for i in range(60):
        rows.append({"session_id": f"a{i}", "device": "Good", "os_version": "15.0",
                     "name": "t1", "is_failure": False, "duration": 60.0})
    for i in range(60):
        rows.append({"session_id": f"b{i}", "device": "Bad", "os_version": "14.0",
                     "name": "t2", "is_failure": i < 40, "duration": 60.0})
    df = pd.DataFrame(rows)
    findings = significant_findings(df, ["device"])
    assert set(findings["level"]) == {"Bad"}


# --- duration --------------------------------------------------------------
def test_duration_outliers_flags_hang():
    durations = [60.0] * 10 + [600.0]
    df = pd.DataFrame({
        "name": ["slowTest"] * 11,
        "duration": durations,
        "is_failure": [False] * 11,
        "session_id": [f"s{i}" for i in range(11)],
    })
    out = duration_outliers(df)
    assert len(out) == 1
    assert out.iloc[0]["slowest_seconds"] == 600.0
    assert out.iloc[0]["ratio"] == 10.0


def test_duration_no_false_outlier():
    df = pd.DataFrame({
        "name": ["steady"] * 10,
        "duration": [60.0, 61.0, 59.0, 60.0, 62.0, 58.0, 60.0, 61.0, 59.0, 60.0],
        "is_failure": [False] * 10,
        "session_id": [f"s{i}" for i in range(10)],
    })
    assert duration_outliers(df).empty


def test_time_split():
    df = pd.DataFrame({"browserstack_seconds": [30.0, 20.0], "user_seconds": [70.0, 80.0]})
    split = time_split(df)
    assert split["browserstack_pct"] == 25.0
    assert split["user_pct"] == 75.0
    assert time_split(pd.DataFrame({"a": [1]})) == {}
