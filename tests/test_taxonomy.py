"""Tests for the rule-based failure taxonomy."""
from __future__ import annotations

from taxonomy import classify


def test_native_permission_dialog_wins_over_element_not_found():
    # The reason looks like element-not-found, but the permission-dialog rule
    # (listed first) is the real root cause and takes precedence.
    reason = ("NoSuchElementException: Unable to locate element btn_next — a native "
              "dialog 'Allow this app? While using the app' was blocking")
    cat, owner = classify(reason, duration=42, status="failed")
    assert cat == "native-permission-dialog"
    assert owner == "Test automation"


def test_backend_and_assertion_and_element():
    assert classify("java.net.SocketTimeoutException: read timed out")[0] == "backend-error"
    assert classify("junit.framework.AssertionFailedError: expected <1> but was <2>")[0] == "assertion-failure"
    assert classify("org.openqa.selenium.NoSuchElementException: x")[0] == "element-not-found"
    assert classify("Header value cannot be null")[0] == "missing-auth-header"


def test_short_duration_is_did_not_run_only_without_real_signal():
    # No recognizable error + very short -> aborted before running.
    cat, owner = classify("", duration=12, status="failed")
    assert cat == "test-did-not-run"
    assert owner == "Infra / re-run"
    # But a short session WITH a real assertion is a genuine (fast) failure.
    assert classify("AssertionError: expected true", duration=12, status="failed")[0] == "assertion-failure"


def test_timeout_status_is_did_not_run():
    assert classify("", duration=300, status="timeout")[0] == "test-did-not-run"


def test_real_appium_locator_message_is_element_not_found():
    """Regression: this real Appium phrasing used to fall through to
    'test-did-not-run' via the short-duration heuristic."""
    reason = ("Can't locate an element by this strategy: "
              "By.chained({AppiumBy.accessibilityId: Lifetime interest})")
    # Even at 20s (under min_valid_test_seconds) it must stay element-not-found.
    assert classify(reason, duration=20, status="failed")[0] == "element-not-found"


def test_selenium_wait_timeout_is_not_infra():
    reason = ("org.openqa.selenium.TimeoutException: Expected condition failed: "
              "waiting for visibility of element located by id: btn_next")
    assert classify(reason, duration=35, status="failed")[0] == "element-not-found"


def test_backend_errors_are_not_test_did_not_run():
    for reason in (
        "API error: HTTP 500 Internal Server Error from /v1/wallet/balance",
        "java.net.SocketTimeoutException: read timed out",
        "Connection refused while calling the ledger service",
        "status code: 503 Service Unavailable",
    ):
        assert classify(reason, duration=25, status="failed")[0] == "backend-error", reason


def test_currency_amount_is_not_mistaken_for_http_5xx():
    # "Rp 500" must not look like a 5xx server error.
    reason = "junit.framework.AssertionFailedError: expected balance <Rp 500> but was <Rp 0>"
    assert classify(reason, duration=30, status="failed")[0] == "assertion-failure"


def test_genuine_infra_abort_still_classified():
    assert classify("Could not start a new session: app is not installed",
                    duration=8, status="failed")[0] == "test-did-not-run"
    # No error text at all + very short -> genuinely did not run.
    assert classify("", duration=9, status="failed")[0] == "test-did-not-run"


def test_crashlog_marker_is_app_crash():
    # A present crashlog injects the marker; it wins even over a surface symptom.
    log = "BROWSERSTACK_CRASHLOG_PRESENT\n--- crash ---\nsignal 11 (SIGSEGV)"
    cat, owner = classify("NoSuchElementException: btn not found", log, 50, "failed")
    assert cat == "app-crash"
    assert owner == "Dev"


def test_signal_free_text_is_no_diagnostic_logs():
    """Text with no error signal means we could not SEE the failure — a different
    problem (and owner) from having an error we lack a rule for."""
    noise = ('Calling AppiumDriver.execute() with args: ["mobile: getCurrentActivity"]\n'
             "Clearing new command timeout pre-emptively")
    cat, owner = classify(noise, duration=120, status="failed")
    assert cat == "no-diagnostic-logs"
    assert owner == "Infra / re-run"


def test_uncategorized_needs_a_rule_not_triage_of_missing_logs():
    """A real error with no matching rule stays 'uncategorized' — that is the
    signal to add a rule, and its owner must not read like the category name."""
    cat, owner = classify("SomethingWeirdException: a brand new failure mode",
                          duration=120, status="failed")
    assert cat == "uncategorized"
    assert owner == "Needs triage"
