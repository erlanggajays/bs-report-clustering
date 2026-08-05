"""Generate a realistic mock BrowserStack payload for local testing.

Produces a JSON file shaped like the App Automate `builds.json` + per-build
`sessions.json` responses, seeded with recurring stack traces so the clustering
engine has real structure to find. No BrowserStack credentials required.

Usage:
    python scripts/generate_mock_data.py
    python scripts/generate_mock_data.py --sessions 120 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_USER = "Erlangga Jaya"
DEFAULT_PROJECT = "Finserv - Gopay Android"
DEFAULT_SUITE = "gopay_consumer_app_android_tests"
APP_VERSIONS = ["6.7.0-staging", "6.8.0-staging", "6.9.0-staging"]

DEVICES = [
    ("Samsung Galaxy S23", "13.0"),
    ("Samsung Galaxy S23", "14.0"),
    ("Google Pixel 7", "13.0"),
    ("Google Pixel 8", "14.0"),
    ("OnePlus 11R", "13.0"),
    ("Xiaomi Redmi Note 12", "12.0"),
    ("Samsung Galaxy A54", "13.0"),
]

TEST_SCENARIOS = [
    "TopUp via bank transfer completes successfully",
    "Send money to phone number shows success screen",
    "QRIS scan-and-pay debits correct amount",
    "PayLater activation flow reaches confirmation",
    "Transaction history loads and paginates",
    "Biometric login unlocks the wallet",
    "Promo voucher applies discount at checkout",
    "Split bill request sends to selected contacts",
    "Profile KYC upload accepts a valid ID",
    "Balance refreshes after successful top-up",
]

# Recurring failure signatures: (message, app stack frame). Each root cause has a
# FIXED frame, so the same buggy code path recurs across many test scenarios and
# the fingerprinting engine groups them — as it would in real BrowserStack data.
FAILURE_TEMPLATES = [
    (
        "org.openqa.selenium.NoSuchElementException: Unable to locate element "
        "{{id=btn_confirm_payment}} within 20000ms. Page may not have finished "
        "rendering after the network call to /v1/payments returned.",
        "com.gopay.payments.PaymentConfirmScreen.tapConfirm",
    ),
    (
        "java.net.SocketTimeoutException: timeout while calling backend "
        "https://api.internal.example.com/wallet/balance after 30000ms. "
        "Retry budget exhausted.",
        "com.gopay.net.WalletApiClient.getBalance",
    ),
    (
        "org.openqa.selenium.StaleElementReferenceException: stale element "
        "reference: element is not attached to the page document. Element "
        "{{class=RecyclerView.row}} was recycled during scroll.",
        "com.gopay.common.RecyclerScroller.scrollToItem",
    ),
    (
        "junit.framework.AssertionFailedError: expected balance <Rp 150.000> "
        "but was <Rp 100.000>. Ledger did not reflect the top-up within the "
        "polling window.",
        "com.gopay.tests.assertions.LedgerAssertions.assertBalance",
    ),
    (
        "io.appium.java_client.NoSuchContextException: The context WEBVIEW_com."
        "gopay.consumer is not available. WebView bridge failed to attach on "
        "this device build.",
        "com.gopay.webview.WebViewBridge.attachContext",
    ),
    (
        "org.openqa.selenium.ElementClickInterceptedException: element click "
        "intercepted: other element would receive the click. A promo bottom-"
        "sheet overlay was still visible.",
        "com.gopay.promo.PromoBottomSheet.dismiss",
    ),
]


def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat()


def _make_session(
    rng: random.Random, base_time: datetime, index: int, app_version: str
) -> dict:
    device, os_version = rng.choice(DEVICES)
    scenario = rng.choice(TEST_SCENARIOS)
    sid = uuid.uuid4().hex[:24]

    # ~28% failure rate overall; some devices are deliberately flakier.
    device_bias = 0.18 if "Redmi" in device or "A54" in device else 0.0
    failed = rng.random() < (0.24 + device_bias)

    duration = round(rng.uniform(8.0, 45.0), 2)
    if failed:
        duration = round(duration * rng.uniform(1.1, 2.4), 2)  # failures run longer

    reason = ""
    if failed:
        message, frame = rng.choice(FAILURE_TEMPLATES)
        # Real traces list the throw site first, test entry point last. The fixed
        # root-cause frame therefore comes before the scenario test frame.
        reason = (
            f"{message}\n\tat {frame}(SourceFile:{rng.randint(40, 500)})"
            f"\n\tat com.gopay.tests.{scenario.split()[0]}Test.run(Test.java:{rng.randint(40, 320)})"
        )

    created = base_time + timedelta(seconds=index * rng.randint(20, 90))
    base = f"https://app-automate.browserstack.com/builds/mock/sessions/{sid}"
    return {
        "name": scenario,
        "duration": duration,
        "os": "android",
        "os_version": os_version,
        "device": device,
        "status": "failed" if failed else "passed",
        "reason": reason if failed else "COMPLETED",
        "hashed_id": sid,
        "created_at": _iso(created),
        "build_name": DEFAULT_SUITE,
        "user_name": DEFAULT_USER,
        # Token-free URL the report deep-links to:
        "browser_url": base,
        # Tokened URL that MUST be scrubbed (never persisted/rendered):
        "public_url": f"{base}?auth_token=FAKE_SHOULD_BE_STRIPPED_{sid}",
        "session_terminal_logs_url": f"{base}/terminal-logs.txt?X-Amz-Signature=FAKE",
        "logs": f"{base}/logs",
        "app_details": {
            "app_version": app_version,
            "app_filename": f"universal-staging-release-{sid[:8]}.apk",
            "app_name": "com.gojek.gopay.nightly.staging",
        },
    }


def generate(
    user: str = DEFAULT_USER,
    suite: str = DEFAULT_SUITE,
    num_sessions: int = 80,
    seed: int | None = 42,
    project: str = DEFAULT_PROJECT,
) -> dict:
    """Build a mock payload: one 'latest' build plus its sessions."""
    rng = random.Random(seed)
    build_time = datetime.utcnow() - timedelta(hours=1)
    app_version = APP_VERSIONS[-1] if seed is None else APP_VERSIONS[seed % len(APP_VERSIONS)]
    sessions = [
        _make_session(rng, build_time, i, app_version) for i in range(num_sessions)
    ]

    failed = sum(1 for s in sessions if s["status"] == "failed")
    build = {
        "hashed_id": uuid.uuid4().hex[:24],
        "name": f"{user} triggered from Mac OS X (mock)",
        "project": project,
        "suite": suite,
        "user_name": user,
        "status": "failed" if failed else "done",
        "duration": int(sum(s["duration"] for s in sessions)),
        "created_at": _iso(build_time),
        "sessions_count": len(sessions),
    }
    return {"build": build, "sessions": sessions}


def seed_history(n_builds: int, base_seed: int = 100) -> int:
    """Persist N synthetic historical builds to the history DB via the real
    ingestion path, so trend + cross-build flakiness have data to work with.
    Returns the number of builds written.
    """
    # Import here to avoid a hard dependency for plain payload generation.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))          # for `config`
    sys.path.insert(0, str(root / "src"))  # for `ingestor`, `history`
    import ingestor  # noqa: E402
    import history    # noqa: E402
    from config import settings  # noqa: E402

    db_path = settings.sample_history_db_path  # keep mock data out of real history
    now = datetime.utcnow()
    for k in range(n_builds):
        payload = generate(seed=base_seed + k, num_sessions=60)
        # Space builds out over the past days for a believable trend/window.
        bt = now - timedelta(days=(n_builds - k))
        payload["build"]["created_at"] = _iso(bt)
        payload["build"]["hashed_id"] = uuid.uuid4().hex[:40]
        # Align session timestamps with the build's day so the date window is realistic.
        for i, s in enumerate(payload["sessions"]):
            s["created_at"] = _iso(bt + timedelta(seconds=i * 30))
        df = ingestor._sessions_to_dataframe(payload["sessions"])
        history.persist_build(payload["build"], df, db_path=db_path)
    return n_builds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock BrowserStack data.")
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--sessions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-history",
        type=int,
        default=0,
        metavar="N",
        help="Also persist N synthetic prior builds to the history DB (for trend/flakiness).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "mock_browserstack_build.json"),
    )
    args = parser.parse_args()

    payload = generate(args.user, args.suite, args.sessions, args.seed, args.project)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    failed = sum(1 for s in payload["sessions"] if s["status"] == "failed")
    print(f"Wrote {len(payload['sessions'])} sessions ({failed} failed) -> {out_path}")

    if args.seed_history > 0:
        n = seed_history(args.seed_history)
        print(f"Seeded {n} historical builds into the history DB.")


if __name__ == "__main__":
    main()
