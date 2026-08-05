"""Shared pytest fixtures. `pythonpath = ["src"]` (pyproject) puts the modules
on the import path, so tests import them directly (e.g. `import ingestor`).
"""
from __future__ import annotations

import pytest

from ingestor import _sessions_to_dataframe


@pytest.fixture
def raw_sessions() -> list[dict]:
    """Unwrapped session dicts shaped like BrowserStack's real `sessions.json`.

    One pass + two recurring failure signatures (each appearing twice) so the
    clustering engine has real structure to recover. Includes a tokened
    public_url / signed terminal URL to exercise scrubbing.
    """
    return [
        {
            "hashed_id": "s1", "name": "deeplink lands on money page",
            "status": "passed", "reason": "COMPLETED", "duration": 10,
            "os_version": "15.0", "device": "Google Pixel 9",
            "browser_url": "https://app-automate.browserstack.com/builds/b/sessions/s1",
            "public_url": "https://app-automate.browserstack.com/builds/b/sessions/s1?auth_token=SECRET",
            "session_terminal_logs_url": "https://s3.aws.com/x-terminal.txt?X-Amz-Signature=SIG",
            "app_details": {"app_version": "6.7.0"},
            "created_at": "2026-08-01T10:00:00.000Z",
        },
        {
            "hashed_id": "s2", "name": "topup confirm",
            "status": "failed", "duration": 22, "os_version": "14.0",
            "device": "Samsung Galaxy S23",
            # Same exception + same app frame as s3 (different element id) -> one cluster.
            "reason": "org.openqa.selenium.NoSuchElementException: Unable to locate element {{id=btn_confirm}} within 20000ms\n\tat com.gopay.payments.PaymentConfirmScreen.tapConfirm(SourceFile:88)",
            "app_details": {"app_version": "6.7.0"},
            "created_at": "2026-08-01T10:01:00.000Z",
        },
        {
            "hashed_id": "s3", "name": "send money",
            "status": "failed", "duration": 24, "os_version": "14.0",
            "device": "Samsung Galaxy S23",
            "reason": "org.openqa.selenium.NoSuchElementException: Unable to locate element {{id=field_amount}} within 20000ms\n\tat com.gopay.payments.PaymentConfirmScreen.tapConfirm(SourceFile:91)",
            "app_details": {"app_version": "6.7.0"},
            "created_at": "2026-08-01T10:02:00.000Z",
        },
        {
            "hashed_id": "s4", "name": "balance refresh",
            "status": "failed", "duration": 40, "os_version": "13.0",
            "device": "Xiaomi Redmi Note 12",
            # Different exception + frame from s2/s3 -> a separate cluster.
            "reason": "java.net.SocketTimeoutException: timeout while calling backend after 30000ms\n\tat com.gopay.net.WalletApiClient.getBalance(SourceFile:142)",
            "app_details": {"app_version": "6.7.0"},
            "created_at": "2026-08-01T10:03:00.000Z",
        },
        {
            "hashed_id": "s5", "name": "history loads",
            "status": "failed", "duration": 41, "os_version": "13.0",
            "device": "Xiaomi Redmi Note 12",
            "reason": "java.net.SocketTimeoutException: timeout while calling backend after 12000ms\n\tat com.gopay.net.WalletApiClient.getBalance(SourceFile:142)",
            "app_details": {"app_version": "6.7.0"},
            "created_at": "2026-08-01T10:04:00.000Z",
        },
    ]


@pytest.fixture
def sample_df(raw_sessions):
    return _sessions_to_dataframe(raw_sessions)
