"""Rule-based failure taxonomy.

Classifies each failure into a **known category** with an **owner**, so triage
routes automatically ("is this our bug, a test-automation gap, or infra flake?").
This complements clustering: rules label known modes with high precision; whatever
no rule matches falls through to the fingerprint clustering to discover new ones.

Rules come from code defaults, optionally overridden by a JSON file at
``settings.taxonomy_config_path``. File rules are evaluated **first**, so QA can
add or pre-empt categories without touching code. A rule matches if any of its
configured conditions hold (regex over reason+log, session status, or a max
duration). Rules are evaluated in order — first match wins — so specific
categories are listed before generic ones, and the duration-based "did not run"
rule sits late so real error signals win first.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from config import settings


@dataclass
class Rule:
    category: str
    owner: str
    patterns: tuple[str, ...] = ()
    # If any of these match, the rule's *heuristic* conditions (status / duration)
    # are suppressed. Used so "short session" or "timeout status" cannot claim a
    # failure whose text shows a genuine in-test error.
    exclude_patterns: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    max_duration_seconds: float | None = None
    description: str = ""
    _res: list[re.Pattern] = field(default_factory=list, init=False, repr=False)
    _excl: list[re.Pattern] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._res = [re.compile(p, re.I) for p in self.patterns]
        self._excl = [re.compile(p, re.I) for p in self.exclude_patterns]

    def matches(self, text: str, duration: float, status: str) -> bool:
        # An explicit pattern hit is a confident signal.
        if self._res and any(r.search(text) for r in self._res):
            return True
        # Status/duration are weak heuristics: back off when the text carries a
        # real error signal (that means the test ran and genuinely failed).
        if self._excl and any(r.search(text) for r in self._excl):
            return False
        if self.statuses and status.lower() in self.statuses:
            return True
        if self.max_duration_seconds is not None and 0 < duration <= self.max_duration_seconds:
            return True
        return False


UNCATEGORIZED = Rule("uncategorized", "Unassigned",
                     description="No rule matched — see the failure clusters below")


def _default_rules() -> list[Rule]:
    """Ordered defaults. Specific first; the duration-based catch-all last."""
    return [
        # Crash first: a present crashlog (CRASH_MARKER) or crash text is the root
        # cause regardless of any surface symptom (e.g. a later element-not-found).
        Rule("app-crash", "Dev",
             patterns=(r"BROWSERSTACK_CRASHLOG_PRESENT", r"FATAL EXCEPTION", r"\bANR\b",
                       r"application not responding", r"has stopped", r"SIGSEGV",
                       r"native crash", r"signal 11"),
             description="App crashed or became unresponsive (crashlog / fatal exception)"),
        Rule("native-permission-dialog", "Test automation",
             patterns=(r"while using the app", r"allow (only )?this time",
                       r"permissioncontroller", r"permission[_-]?allow", r"\ballow\b.*\bpermission\b"),
             description="Unhandled native OS permission dialog blocked the flow"),
        # Appium could not stop/reset the app between tests. Seen in real GoPay
        # logs as "'<package>' is still running after 500ms timeout".
        Rule("app-lifecycle", "Infra / re-run",
             patterns=(r"is still running after \d+\s*ms",
                       r"(?:unable|failed) to (?:terminate|force.?stop|activate) .*app",
                       r"app(?:lication)? did not (?:start|stop|terminate)",
                       r"Encountered internal error running command.*still running"),
             description="Appium could not stop/reset the app between tests"),
        # Appium/driver server problems on the host — not the app under test.
        Rule("appium-server-error", "Infra / re-run",
             patterns=(r"EADDRINUSE", r"address already in use",
                       r"Could not configure Appium server",
                       r"a driver or plugin tried to update the server",
                       r"uiautomator2 server .*(?:crash|not responding)",
                       r"instrumentation process (?:crash|cannot be initialized)"),
             description="Appium/driver server failure on the host (port clash, driver crash)"),
        Rule("webview-context", "Test automation",
             patterns=(r"NoSuchContextException", r"WEBVIEW_\S* is not available",
                       r"context .*not available", r"failed to (switch|attach).*context",
                       r"no such context"),
             description="WebView / native context bridge unavailable"),
        # Auth before generic backend: a 401/403 is a test-setup problem, not a 5xx.
        Rule("missing-auth-header", "Test setup",
             patterns=(r"header value cannot be null", r"\b401\b", r"\b403\b",
                       r"unauthorized", r"forbidden", r"missing .*(header|token)",
                       r"invalid token", r"token (has )?expired", r"auth.*failed"),
             description="Missing or invalid authentication header/token"),
        Rule("backend-error", "Backend",
             patterns=(r"SocketTimeoutException", r"read timed out",
                       # 5xx only in an HTTP context, so amounts like "Rp 500" don't match.
                       r"\b(?:HTTP|status(?:\s*code)?|response(?:\s*code)?)\s*[:=]?\s*5\d{2}\b",
                       r"internal server error", r"bad gateway", r"service unavailable",
                       r"gateway timeout", r"connection (reset|refused|closed)",
                       r"ECONNRESET", r"UnknownHostException", r"SSL\w*Exception",
                       r"graphql error", r"failed to fetch", r"upstream .*error",
                       r"api .*(error|failure)", r"empty response from server"),
             description="Backend/API timeout, 5xx, or connectivity failure"),
        Rule("element-not-found", "Test automation",
             patterns=(r"NoSuchElementException", r"unable to locate element",
                       # Real Appium/Selenium phrasings:
                       r"can(?:no|')?t locate an element", r"could not be located",
                       r"by this strategy", r"AppiumBy\.", r"By\.chained",
                       r"StaleElementReferenceException", r"ElementNotVisible\w*",
                       r"ElementNotInteractable\w*", r"click intercepted",
                       r"element .*not (found|visible|clickable|interactable)",
                       # A Selenium wait that expired on an element is a locator issue,
                       # NOT an infra timeout.
                       r"Expected condition failed.*waiting for",
                       r"TimeoutException.*(element|visib|presen|clickable)",
                       r"waiting for (element|visib|presen)"),
             description="UI element not found / locator drift / overlay"),
        Rule("assertion-failure", "Dev",
             patterns=(r"AssertionFailedError", r"AssertionError", r"expected .*but was",
                       r"expected .*to (be|equal|contain)", r"assertEquals", r"assertTrue",
                       r"assertion failed", r"\bshould (be|equal|contain)\b",
                       r"did not match expected"),
             description="Functional assertion failed (likely a real bug)"),
        # Last resort. Explicit infra phrases only; the duration/status heuristics
        # are suppressed whenever the text shows a genuine in-test error.
        Rule("test-did-not-run", "Infra / re-run",
             patterns=(r"session not created", r"session timed out", r"idle timeout",
                       r"BROWSERSTACK_IDLE_TIMEOUT", r"could not start a new session",
                       r"app(?:lication)? (?:is )?not installed",
                       r"unable to (install|launch)", r"failed to (start|launch)",
                       r"device (?:not|un)available", r"no session (found|active)",
                       r"did not (start|run)", r"Multiple Failures"),
             exclude_patterns=(r"Exception\b", r"\bError\b", r"assert",
                               r"locate an element", r"could not be located",
                               r"by this strategy", r"expected .*but was"),
             statuses=("timeout",),
             max_duration_seconds=settings.min_valid_test_seconds,
             description="Aborted before the test ran properly (infra / setup)"),
    ]


def category_description(category: str) -> str:
    """Human-readable meaning of a category (shown in the report)."""
    for rule in _rules():
        if rule.category == category:
            return rule.description
    return UNCATEGORIZED.description


def _load_rules() -> list[Rule]:
    """File rules (if any) first, then code defaults."""
    rules: list[Rule] = []
    path = Path(settings.taxonomy_config_path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for r in data.get("rules", []):
                rules.append(Rule(
                    category=r["category"],
                    owner=r.get("owner", "Unassigned"),
                    patterns=tuple(r.get("patterns", [])),
                    statuses=tuple(s.lower() for s in r.get("statuses", [])),
                    max_duration_seconds=r.get("max_duration_seconds"),
                    description=r.get("description", ""),
                ))
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # a malformed override must not break the pipeline
    rules.extend(_default_rules())
    return rules


_RULES: list[Rule] | None = None


def _rules() -> list[Rule]:
    global _RULES
    if _RULES is None:
        _RULES = _load_rules()
    return _RULES


def classify(reason: str, log_text: str = "", duration: float = 0.0, status: str = "") -> tuple[str, str]:
    """Return (category, owner) for a failure. 'uncategorized' if no rule matches."""
    text = f"{reason}\n{log_text}"
    for rule in _rules():
        if rule.matches(text, duration, status):
            return rule.category, rule.owner
    return UNCATEGORIZED.category, UNCATEGORIZED.owner
