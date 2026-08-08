"""Feature engineering: derive analysable dimensions from raw test data.

Two extractors:

* ``feature_area`` — maps a test name to a **business domain** (gold, deposito,
  tabungan, …). This gives the attribution engine a dimension that product owners
  care about, without anyone hand-tagging 226 tests. The keyword map was derived
  from the actual vocabulary in the suite and can be overridden by a JSON file.
* ``extract_locator`` / ``extract_screen`` — pull the failing UI locator and screen
  out of a failure message. A locator is the most actionable atom in a failure:
  one broken ``accessibilityId`` often explains many failing tests.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from config import settings

# Ordered: FIRST MATCH WINS. Attribution needs exactly one area per session (a
# clean partition), so a test naming several domains must resolve to one. The
# ordering encodes precedence: concrete product areas (gold, deposito, tabungan)
# come before cross-cutting steps (kyc, onboarding, help), because a test that
# reaches KYC while creating a Tabungan account is a Tabungan test.
# Patterns are matched against the *camelCase-split* name (see _split_camel), which
# is why word boundaries work and multi-word phrases allow optional whitespace.
DEFAULT_FEATURE_AREAS: list[tuple[str, str]] = [
    ("mutual-fund", r"\bMutual\s*Fund|\bMFL\d?\b|\bProspectus|\bCAGR\b|Assets?\s*To\s*Sell"),
    ("gold", r"\bGold\b"),
    ("deposito", r"\bDeposito|\bJago\b|\bTerm\s*Deposit"),
    ("tabungan", r"\bTabungan|\bUsaha\b"),
    ("simpanan", r"\bSimpanan|\bInterest\s*(?:Rate|Summary|Calculator)"),
    ("investment", r"\bInvestment|\bInvest\s*Here"),
    ("kyc", r"\bKyc\b|\bSelfie\b|\bLicense\b|\bConsent\b"),
    ("pinjaman", r"\bPinjaman"),
    ("order-history", r"\bOrder\s*History|\bOrder\b|\bTransaction"),
    ("money-landing", r"\bMoney\s*Landing|\bPayment\s*Method|\bSavings\s*(?:And|Product)"),
    ("financial-report", r"\bExpense|\bFinancial\s*Report"),
    ("accounts", r"\bAdd\s*More\s*Accounts|\bSource\s*Of\s*Fund|\bBalance\s*Card|\bDefault\s*Balance"),
    ("dira", r"\bDira\b"),
    ("help", r"\bHelp\b"),
    ("onboarding", r"\bOnboarding"),
]

# Insert a space at camelCase boundaries so \b-anchored patterns can match tokens
# inside identifiers like "testGoldHomeTopSectionVisible". Runs of capitals such
# as "MFL2" are preserved.
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_camel(text: str) -> str:
    return _CAMEL_SPLIT.sub(" ", text or "")

# Locator extraction targets the SEMANTIC value a test was looking for (the
# accessibility label, text, or resource id) — never the xpath scaffolding around
# it. Two tests hunting "Mutual Fund" via different strategies describe the same
# broken element, so they must group together; grouping on the raw selector string
# would instead merge unrelated elements that share an xpath prefix.
_LOCATOR_PATTERNS = [
    # xpath predicates: [@content-desc="X"], [contains(@text, "X")], [@resource-id="X"]
    re.compile(r"(?:contains\s*\(\s*)?@(?:content-desc|text|resource-id|name|label)"
               r"\s*(?:,|=)\s*[\"']([^\"']{2,80})[\"']"),
    # Appium accessibility id, incl. By.chained({AppiumBy.accessibilityId: X})
    re.compile(r"accessibilityId\s*[:=]\s*[\"']?([^\"'\]\}\),]{2,80})"),
    re.compile(r"accessibility\s+id\s*[:=,]\s*[\"']([^\"']{2,80})[\"']"),
    # resource/element id references
    re.compile(r"\bid\s*[:=]\s*[\"']?([A-Za-z_][\w./]{2,60})"),
    # last resort: a quoted string inside an "unable to locate element" message
    re.compile(r"locate element[^\"']{0,40}[\"']([^\"']{2,80})[\"']"),
]
# Values that are identifiers rather than human-meaningful locators. Grouping on
# these is worthless: every session has a different one.
_LOCATOR_REJECT = re.compile(
    r"^(?:[0-9a-f]{8}-|[0-9a-f]{12,}$|0x|\d+$|null$|undefined$|true$|false$)", re.I
)
_SCREEN_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Activity|Screen|Page|Fragment))\b")

_AREAS: list[tuple[str, re.Pattern]] | None = None


def _load_areas() -> list[tuple[str, re.Pattern]]:
    """Compile the feature-area map; an optional JSON file overrides the defaults.

    File format: ``{"feature_areas": [["area-name", "regex"], ...]}``. File entries
    are evaluated first so teams can add or pre-empt areas without code changes.
    """
    entries: list[tuple[str, str]] = []
    path = Path(settings.feature_areas_path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries.extend((a, p) for a, p in data.get("feature_areas", []))
        except (json.JSONDecodeError, ValueError, OSError):
            pass  # a malformed override must not break the pipeline
    entries.extend(DEFAULT_FEATURE_AREAS)
    return [(area, re.compile(pat, re.I)) for area, pat in entries]


def _areas() -> list[tuple[str, re.Pattern]]:
    global _AREAS
    if _AREAS is None:
        _AREAS = _load_areas()
    return _AREAS


def feature_area(test_name: str) -> str:
    """Business domain for a test, or ``"unmapped"`` when no keyword matches."""
    if not test_name:
        return "unmapped"
    haystack = _split_camel(test_name)
    for area, pattern in _areas():
        if pattern.search(haystack):
            return area
    return "unmapped"


_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}


def _trim_delimiters(value: str) -> str:
    """Strip surrounding delimiters without mangling the value.

    Only *unbalanced* brackets are removed, so a legitimate label such as
    "Estimated final balance (net)" keeps its closing parenthesis while scaffolding
    like "Mutual Fund)]" loses it.
    """
    value = value.strip().strip("'\"[]{},:; \\")
    while value and value[-1] in _BRACKET_PAIRS:
        closer, opener = value[-1], _BRACKET_PAIRS[value[-1]]
        if value.count(closer) <= value.count(opener):
            break
        value = value[:-1]
    return value.strip()


def extract_locator(text: str, max_len: int = 70) -> str:
    """The semantic UI target a failure was looking for, or "" if not identifiable.

    Returns the accessibility label / text / id itself — not the surrounding xpath —
    so the same broken element groups together regardless of locator strategy.
    Identifier-like values (UUIDs, hex, bare numbers) are rejected because they
    differ every session and would fragment the grouping.
    """
    if not text:
        return ""
    # Appium logs embed selectors in JSON, so quotes arrive backslash-escaped.
    unescaped = text.replace('\\"', '"').replace("\\'", "'")
    for pattern in _LOCATOR_PATTERNS:
        for m in pattern.finditer(unescaped):
            value = _trim_delimiters(" ".join(m.group(1).split()))
            if len(value) < 2 or _LOCATOR_REJECT.search(value):
                continue
            return value[:max_len]
    return ""


def extract_screen(text: str) -> str:
    """Screen/Activity name mentioned in a failure, or "" if absent."""
    m = _SCREEN_RE.search(text or "")
    return m.group(1) if m else ""


if __name__ == "__main__":
    samples = [
        "testAssetProspectusSection: Verify user able to see prospectus details visible",
        "testUserAbleToSeeExpenseChart: Verify user able to see the expense chart in Financial Report",
        "verifySavingsAndInsuranceOnMoneyLandingPagePreActivated: ...",
        "testFilterOrderHistoryByTermDepositMethod: Verify filter order history by Deposito by Jago",
    ]
    for s in samples:
        print(f"{feature_area(s):<18} {s[:60]}")
