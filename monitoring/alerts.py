"""
monitoring/alerts.py — Alert dispatch with rate limiting.

Alert types
-----------
REGIME_CHANGE     HMM detected a new confirmed regime
CIRCUIT_BREAKER   RiskManager triggered a halt
ORDER_FAILURE     Order rejected or fill timed out
DRAWDOWN_WARNING  Daily/weekly drawdown threshold crossed
DATA_FEED_DOWN    WebSocket / bar feed stopped delivering
API_LOST          Alpaca REST connection failed
HMM_RETRAINED     HMM model was retrained
FLICKER_EXCEEDED  Regime flicker rate above threshold
SYSTEM_ERROR      Unhandled exception in main loop

Rate limiting: same alert type is suppressed for 15 minutes after it fires.
force=True bypasses the limit (used for SYSTEM_ERROR and circuit breakers).
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum, auto
from typing import Any, Optional

try:
    import urllib.request
    import urllib.error
    import json as _json
    _URLLIB_OK = True
except ImportError:
    _URLLIB_OK = False

logger = logging.getLogger(__name__)


class AlertType(Enum):
    REGIME_CHANGE = auto()
    CIRCUIT_BREAKER = auto()
    ORDER_FAILURE = auto()
    DRAWDOWN_WARNING = auto()
    DATA_FEED_DOWN = auto()
    API_LOST = auto()
    HMM_RETRAINED = auto()
    FLICKER_EXCEEDED = auto()
    SYSTEM_ERROR = auto()


class AlertManager:
    """
    Dispatches alerts via configured channels with rate limiting.

    Channels (always-on first):
      1. Python logger (console + alerts.log)
      2. Email via SMTP (optional, call configure_email())
      3. HTTP webhook (optional, call configure_webhook())
    """

    def __init__(self, rate_limit_minutes: int = 15) -> None:
        self._rate_limit = timedelta(minutes=rate_limit_minutes)
        self._last_sent: dict[AlertType, datetime] = {}
        self._recent: list[str] = []           # last 50 alert messages (for dashboard)
        self._email_cfg: Optional[dict] = None
        self._webhook_cfg: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fire(
        self,
        alert_type: AlertType,
        title: str,
        body: str,
        force: bool = False,
    ) -> bool:
        """
        Fire an alert if not rate-limited.

        Parameters
        ----------
        alert_type : category used for rate limiting
        title      : short subject line
        body       : full message body
        force      : bypass rate limiting (SYSTEM_ERROR, CIRCUIT_BREAKER)

        Returns True if the alert was dispatched, False if suppressed.
        """
        if not force and self._is_rate_limited(alert_type):
            logger.debug("Alert %s suppressed (rate limited)", alert_type.name)
            return False

        self._last_sent[alert_type] = datetime.now()

        ts = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{ts}] [{alert_type.name}] {title}: {body}"

        # 1. Always log
        log_level = logging.ERROR if alert_type in (
            AlertType.CIRCUIT_BREAKER, AlertType.SYSTEM_ERROR
        ) else logging.WARNING
        logger.log(log_level, "ALERT %s — %s: %s", alert_type.name, title, body)

        # Store for dashboard
        self._recent.append(full_msg)
        if len(self._recent) > 50:
            self._recent = self._recent[-50:]

        # 2. Email
        if self._email_cfg:
            try:
                self._send_email(title, body)
            except Exception as exc:
                logger.error("Alert email failed: %s", exc)

        # 3. Webhook
        if self._webhook_cfg:
            try:
                self._send_webhook(title, body)
            except Exception as exc:
                logger.error("Alert webhook failed: %s", exc)

        return True

    def get_recent(self) -> list[str]:
        """Return the last 50 alert messages for display."""
        return list(self._recent)

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        recipient: str,
        password: str,
    ) -> None:
        self._email_cfg = {
            "host": smtp_host,
            "port": smtp_port,
            "sender": sender,
            "recipient": recipient,
            "password": password,
        }
        logger.info("Email alerts configured → %s", recipient)

    def configure_webhook(self, url: str, headers: Optional[dict] = None) -> None:
        self._webhook_cfg = {"url": url, "headers": headers or {}}
        logger.info("Webhook alerts configured → %s", url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_rate_limited(self, alert_type: AlertType) -> bool:
        last = self._last_sent.get(alert_type)
        if last is None:
            return False
        return (datetime.now() - last) < self._rate_limit

    def _send_email(self, title: str, body: str) -> None:
        cfg = self._email_cfg
        msg = MIMEText(body, "plain")
        msg["Subject"] = f"[regime-trader] {title}"
        msg["From"] = cfg["sender"]
        msg["To"] = cfg["recipient"]

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context) as server:
            server.login(cfg["sender"], cfg["password"])
            server.sendmail(cfg["sender"], cfg["recipient"], msg.as_string())

    def _send_webhook(self, title: str, body: str) -> None:
        if not _URLLIB_OK:
            return
        cfg = self._webhook_cfg
        payload = _json.dumps({
            "text": f"*{title}*\n{body}",
            "username": "regime-trader",
        }).encode("utf-8")
        req = urllib.request.Request(
            cfg["url"],
            data=payload,
            headers={"Content-Type": "application/json", **cfg["headers"]},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"Webhook returned HTTP {resp.status}")
