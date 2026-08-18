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
    assert owner == "Product bug"


def test_junit_colon_form_is_an_assertion_not_infra():
    """Regression, and the costliest kind: JUnit writes 'expected: "x" but was: "y"',
    so a pattern requiring a space after 'expected' missed it. The reason also begins
    with the JUnit "Multiple Failures" wrapper, which used to be an explicit
    test-did-not-run pattern — so a real balance defect was filed as an infra abort
    that someone would simply re-run."""
    reason = ('Multiple Failures (1 failure) -- failure 1 -- '
              '[Estimated final balance (net) mismatch] expected: "Rp5.210.000" '
              'but was: "Rp5.220.000" at DepositoAssertions.lambda$'
              'assertReviewPrincipalAndEstimatedBalanceVisible$11(DepositoAssertions.java:202)')
    cat, owner = classify(reason, duration=90, status="failed")
    assert cat == "assertion-failure"
    assert owner == "Product bug"


def test_multiple_failures_wrapper_classifies_on_its_contents():
    """The wrapper says nothing about the cause, so what it wraps must decide."""
    assert classify("Multiple Failures (1 failure) -- failure 1 -- Element not found",
                    duration=90, status="failed")[0] == "element-not-found"
    # A wrapper with no detail at all: admit we cannot see the failure.
    assert classify("Multiple Failures (1 failure) -- failure 1 -- [Gopay]",
                    duration=90, status="failed")[0] == "no-diagnostic-logs"


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


# --- iOS / framework-assertion vocabulary -----------------------------------
def test_ios_driver_failures_are_infra_not_element_problems():
    """WebDriverAgent is the XCUITest counterpart of the uiautomator2 server, so
    its startup failures are host problems. Note there is deliberately no bare
    'XCUITestDriver' rule: that string prefixes every iOS log line."""
    wda = ("Encountered internal error running command: Error: Unable to start "
           "WebDriverAgent session. Original error: Could not proxy command to the "
           "remote server. Original error: socket hang up")
    assert classify(wda, duration=120, status="failed") == (
        "appium-server-error", "Infra / re-run")
    stale_source = ("Execution failed during attempt 3; Failed to generate view "
                    "elements after 3 attempts. Page source may be invalid or the "
                    "Appium driver server is not responding.")
    assert classify(stale_source, duration=120, status="failed")[0] == "appium-server-error"


def test_xcuitest_element_errors_stay_element_not_found():
    """The iOS driver prefix must not divert a genuine element failure to infra."""
    reason = ("2026-08-13 11:40:38:882 - [132232de][XCUITestDriver@ce39] Encountered "
              "internal error running command: NoSuchElementError: An element could "
              "not be located on the page using the given search parameters.")
    assert classify(reason, duration=120, status="failed")[0] == "element-not-found"


def test_unsupported_driver_command_is_test_automation_not_infra():
    """hideKeyboard is unimplemented on iOS, so re-running can never help — the
    test has to change, which makes the owner Test automation, not Infra."""
    reason = ("Failed to hide keyboard: The requested resource could not be found, "
              "or a request was received using an HTTP method that is not supported "
              "by the mapped resource.")
    assert classify(reason, duration=120, status="failed") == (
        "unsupported-driver-command", "Test automation")


def test_framework_visibility_assertions_are_assertion_failures():
    """The suite's assertion helpers report an unmet screen expectation in prose.
    That is an assertion the app failed, not a driver lookup error."""
    for reason in (
        "Multiple Failures (1 failure) -- failure 1 --Tabungan icon is not visible "
        "at TabunganAssertions.lambda$assertTabunganHomePage$0(TabunganAssertions.java:16)",
        "[Download order history button is not visible] Expecting value to be true but was false",
        "Gopay Saldo widget info icon is not displayed at FinanceAssertion.lambda$x$5",
        "Set as Default Payment Method toggle is not ON at TabunganAssertions.lambda$y$8",
    ):
        assert classify(reason, duration=120, status="failed") == (
            "assertion-failure", "Product bug"), reason


def test_assertj_participle_phrasing_is_matched():
    """AssertJ writes the participle ("Expecting"), which a pattern anchored on
    "expected" silently misses — the same class of bug as the JUnit colon."""
    assert classify('Expecting actual: "Time 19:21" to contain: "20:07"',
                    duration=120, status="failed")[0] == "assertion-failure"
    assert classify("Expecting value to be true but was false",
                    duration=120, status="failed")[0] == "assertion-failure"


def test_context_switch_failure_is_webview_context():
    reason = ("Context switching failed. Check server logs for more details.; The app "
              "associated with webview context WEBVIEW_IN_APP is either not focused "
              "or not present.")
    assert classify(reason, duration=120, status="failed") == (
        "webview-context", "Test automation")


def test_ambiguous_step_is_not_a_missing_element():
    """Nothing was looked for, so this is a test-authoring problem."""
    reason = ("Request lacks specificity - 'home page' cannot be mapped to a single "
              "UI element or text string for verification.; Ambiguous target element")
    assert classify(reason, duration=120, status="failed") == (
        "ambiguous-test-step", "Test automation")
