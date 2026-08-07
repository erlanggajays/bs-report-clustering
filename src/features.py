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

# Locator strategies seen in real Appium/Selenium failures.
_LOCATOR_PATTERNS = [
    re.compile(r"accessibilityId\s*[:=]\s*['\"]?([^'\"\]\}\)]{2,60})"),
    re.compile(r"content-desc\s*[,=]\s*['\"]([^'\"]{2,60})['\"]"),
    re.compile(r"@content-desc\s*=\s*['\"]([^'\"]{2,60})['\"]"),
    re.compile(r"\bid\s*[:=]\s*['\"]?([A-Za-z_][\w./]{2,60})"),
    re.compile(r"selector\"\s*:\s*\"([^\"]{4,120})\""),
    re.compile(r"locate element\s*[:\-]?\s*\{?\s*([^\s\}]{3,60})"),
]
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


def extract_locator(text: str, max_len: int = 70) -> str:
    """The UI locator a failure was looking for, or "" if not identifiable."""
    if not text:
        return ""
    for pattern in _LOCATOR_PATTERNS:
        m = pattern.search(text)
        if m:
            value = " ".join(m.group(1).split())  # collapse whitespace
            value = value.strip("'\"[]{}(),:; ")
            if len(value) >= 2:
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
