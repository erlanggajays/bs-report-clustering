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
    # Gold is the only product priced in grams. "Denomination" alone is NOT a gold
    # marker — deposit products use denomination pills for amounts too.
    ("gold", r"\bGold\b|\bGram\b"),
    ("deposito", r"\bDeposito|\bJago\b|\bTerm\s*Deposit|\bTD\b|\bMaturity|\bPrincipal"
                 r"|\bDeposit\s*Period|\bEstimated\s*(?:Net|Final)\s*Balance"
                 r"|\bProjected\s*Interest"),
    ("tabungan", r"\bTabungan|\bUsaha\b"),
    ("simpanan", r"\bSimpanan|\bInterest\s*(?:Rate|Summary|Calculator)"),
    ("autosave", r"\bAuto\s*(?:Debit|Save|Sweep)|\bAutosave|\bAutosweep|\bFrequenc(?:y|ies)"),
    ("budget", r"\bBudget"),
    ("investment", r"\bInvestment|\bInvest\s*Here"),
    # Viewing and selling what the user already owns, as opposed to buying it. Kept
    # after the product areas above so a named product still wins, and the Asset
    # alternatives are narrow so the bare word "asset" is not pulled out of them.
    ("portfolio", r"\bPortfolio|\bAsset\s*(?:Detail|Holding|Screen)|\bFund\s*Fact\s*Sheet"
                  r"|\bSell\s*(?:Cta|Enter\s*Amount)"),
    ("kyc", r"\bKyc\b|\bSelfie\b|\bLicense\b|\bConsent\b"),
    ("pinjaman", r"\bPinjam|\bRepay"),
    ("order-history", r"\bOrder\s*History|\bOrder\b|\bTransaction|\bFilter\s*Chip"
                      r"|\bMulti\s*(?:Service|Method)|\bMultiservice"),
    ("money-landing", r"\bMoney\s*Landing|\bPayment\s*Method|\bSavings\s*(?:And|Product)"),
    ("financial-report", r"\bExpense|\bFinancial\s*Report"),
    ("accounts", r"\bAdd\s*More\s*Accounts?|\bSource\s*Of\s*Fund|\bBalance\s*Card"
                 r"|\bDefault\s*Balance|\bLinking\b|\bOneklik|\bDirect\s*Debit"),
    # The GoPay Saldo surface itself: pockets, coins, tiles, top-up and transfer.
    ("balance", r"\bGopay\s*Saldo|\bSaldo\b|\b(?:Main\s*)?Pocket\b|\bCoins\b|\bPayment\s*Tiles"
                r"|\bAdd\s*Money|\bTopup\b|\bTransfer\b|\bReceiving\s*Method"
                r"|\bBalance\s*Page|\bLow\s*Balance"),
    ("rewards", r"\bReward|\bBenefits?\b|\bUpgrade\s*Now"),
    ("dira", r"\bDira\b"),
    # help before content: a help *article* is a help test, not a content test.
    ("help", r"\bHelp\b"),
    ("content", r"\bArticle|\bEducation|\bBanner"),
    ("onboarding", r"\bOnboarding|\bPersonal\s*Details|\bSetup\s*Now|\bFresh\s*(?:Gps\s*)?User"),
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

# The COMPLETE selector expression, kept verbatim for display. Grouping still uses
# the semantic value above (so one element is one row), but fixing a locator needs
# the exact strategy and path, not just the label it was hunting.
# Matched against the RAW text (escapes intact), because the backslashes are what
# delimit the JSON string — unescaping first would end the match at the first inner
# quote and truncate the selector.
_JSON_SELECTOR_RE = re.compile(r'"selector"\s*:\s*"((?:[^"\\]|\\.)+)"')

# Matched against unescaped text. The xpath pattern deliberately allows spaces and
# quotes and anchors on a closing bracket, since real xpaths contain both.
_SELECTOR_PATTERNS = [
    re.compile(r"by this strategy\s*:\s*(.+?)(?:\s*$|\n)", re.M),
    re.compile(r"((?:AppiumBy|By)\.\w+\(.*?\)\s*\}?\)?)"),
    # Any xpath up to a closing bracket that ends the expression. Enumerating the
    # allowed characters is a losing game (real labels contain "&", "%", "/", ...),
    # so accept anything but a newline and anchor on a terminator instead.
    re.compile(r"(//[^\n]{4,300}?\])(?=[\s,\"']|$)"),
]

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


# BrowserStack reports the family in the session's "os" field and the release in
# "os_version". The two must be joined for display: a bare "17.3" is a valid iOS
# release *and* a plausible Android one, so labelling versions with a hardcoded
# family mislabels every session on the other platform.
_PLATFORM_LABELS = {"ios": "iOS", "android": "Android", "windows": "Windows", "os x": "macOS"}


def os_label(platform: str, os_version: str) -> str:
    """Display label for an OS, e.g. ``iOS 17.3``.

    Falls back to the bare version when the platform is missing or unrecognised —
    the family cannot be inferred from the version number, so an unqualified
    version is preferable to a guess.
    """
    version = str(os_version or "").strip()
    if version.lower() == "unknown":
        version = ""
    family = _PLATFORM_LABELS.get(str(platform or "").strip().lower(), "")
    if not family:
        return version or "unknown"
    return f"{family} {version}".strip()


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


def extract_selector(text: str, max_len: int = 300) -> str:
    """The full selector expression a failure used, verbatim, or "".

    Complements ``extract_locator``: that one normalises for grouping, this one
    preserves the exact strategy and path so the locator is actually fixable.
    """
    if not text:
        return ""

    def _clean(raw: str) -> str:
        return " ".join(raw.split()).rstrip(",;")

    # JSON form first, on the raw text, then unescape the captured value.
    m = _JSON_SELECTOR_RE.search(text)
    if m:
        value = _clean(m.group(1)).replace('\\"', '"').replace("\\'", "'")
        if len(value) >= 4:
            return value[:max_len]

    unescaped = text.replace('\\"', '"').replace("\\'", "'")
    for pattern in _SELECTOR_PATTERNS:
        m = pattern.search(unescaped)
        if m:
            value = _clean(m.group(1))
            if len(value) >= 4:
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
