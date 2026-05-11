"""
monitoring/logger.py — Structured JSON logging for regime-trader.

Provides four rotating log files (10 MB each, 30-day retention):
  logs/main.log    — general system events
  logs/trades.log  — order lifecycle events
  logs/alerts.log  — alerts fired
  logs/regime.log  — regime changes and HMM state

Every record includes: timestamp, level, module, message, and an optional
`extra` dict for structured fields (regime, probability, equity, etc.).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

LOG_DIR = Path("logs")
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
_BACKUP_COUNT = 30               # ~30 days at one file per day of heavy use


# ------------------------------------------------------------------ #
# JSON formatter                                                       #
# ------------------------------------------------------------------ #

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    SKIP_ATTRS = frozenset({
        "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno",
        "funcName", "created", "msecs", "relativeCreated", "thread",
        "threadName", "processName", "process", "message",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Any extra fields attached via logger.info(..., extra={...})
        for key, val in record.__dict__.items():
            if key not in self.SKIP_ATTRS and not key.startswith("_"):
                payload[key] = val

        return json.dumps(payload, default=str)


# ------------------------------------------------------------------ #
# Logger factory                                                       #
# ------------------------------------------------------------------ #

def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Return a named logger with JSON formatting.

    If log_file is given the logger writes JSON to that rotating file
    in addition to the root handler (stdout plain text).
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    if log_file is not None and not _has_file_handler(log, log_file):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setFormatter(JSONFormatter())
        fh.setLevel(level)
        log.addHandler(fh)

    return log


def _has_file_handler(log: logging.Logger, path: Path) -> bool:
    for h in log.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            if Path(h.baseFilename).resolve() == path.resolve():
                return True
    return False


# ------------------------------------------------------------------ #
# Pre-built loggers                                                    #
# ------------------------------------------------------------------ #

def _make(name: str, filename: str) -> logging.Logger:
    return get_logger(f"regime_trader.{name}", log_file=LOG_DIR / filename)


def get_main_logger() -> logging.Logger:
    return _make("main", "main.log")


def get_trade_logger() -> logging.Logger:
    return _make("trades", "trades.log")


def get_alert_logger() -> logging.Logger:
    return _make("alerts", "alerts.log")


def get_regime_logger() -> logging.Logger:
    return _make("regime", "regime.log")


# ------------------------------------------------------------------ #
# Structured event helpers                                             #
# ------------------------------------------------------------------ #

def log_order_event(
    log: logging.Logger,
    event: str,
    order_id: str,
    symbol: str,
    side: str,
    qty: int,
    price: Optional[float] = None,
    **extra: Any,
) -> None:
    """Emit a structured order lifecycle event."""
    log.info(
        "%s %s %s %d@%.4f",
        event, side.upper(), symbol, qty, price or 0.0,
        extra={
            "event": event,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            **extra,
        },
    )


def log_regime_change(
    log: logging.Logger,
    symbol: str,
    old_regime: str,
    new_regime: str,
    confidence: float,
    vol_environment: str = "",
    equity: Optional[float] = None,
) -> None:
    """Emit a structured regime-change event."""
    log.info(
        "REGIME_CHANGE %s: %s → %s (p=%.2f)",
        symbol, old_regime, new_regime, confidence,
        extra={
            "event": "regime_change",
            "symbol": symbol,
            "old_regime": old_regime,
            "new_regime": new_regime,
            "confidence": confidence,
            "vol_environment": vol_environment,
            "equity": equity,
        },
    )


def log_bar_summary(
    log: logging.Logger,
    regime: str,
    probability: float,
    equity: float,
    positions: dict,
    daily_pnl: float,
) -> None:
    """Log end-of-bar summary with full context (required by spec)."""
    log.info(
        "BAR regime=%s p=%.2f equity=%.2f daily_pnl=%.2f",
        regime, probability, equity, daily_pnl,
        extra={
            "event": "bar_summary",
            "regime": regime,
            "probability": probability,
            "equity": equity,
            "positions": positions,
            "daily_pnl": daily_pnl,
        },
    )
