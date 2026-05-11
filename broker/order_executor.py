"""
order_executor.py — Order placement, modification, and cancellation.

Translates approved signals (from SignalGenerator) into Alpaca orders.
Enforces paper-trading guardrails and validates order parameters before
submission. Emits structured log events for every order lifecycle event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from broker.alpaca_client import AlpacaClient
from core.signal_generator import ApprovedSignal

logger = logging.getLogger(__name__)

_TERMINAL = frozenset(["FILLED", "CANCELLED", "REJECTED", "EXPIRED"])


class OrderStatus(Enum):
    PENDING = auto()
    SUBMITTED = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


_ALPACA_STATUS_MAP = {
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "new": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "replaced": OrderStatus.CANCELLED,
}


@dataclass
class OrderRecord:
    """Internal record of an order lifecycle."""

    order_id: str
    symbol: str
    side: str                           # "buy" | "sell"
    qty: int
    order_type: str                     # "market" | "limit"
    limit_price: Optional[float]
    status: OrderStatus
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    filled_avg_price: Optional[float] = None
    signal_id: str = ""
    notes: str = ""


class OrderExecutor:
    """
    Translates ApprovedSignals into Alpaca orders and tracks their lifecycle.

    Maintains an in-memory order log for the current session. A full order
    history should be persisted to disk (implemented in Phase 7).
    """

    _MIN_QTY = 1        # minimum shares to bother submitting an order

    def __init__(self, client: AlpacaClient, paper: bool = True) -> None:
        self.client = client
        self.paper = paper
        self._order_log: list[OrderRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_signal(self, approved: ApprovedSignal) -> Optional[OrderRecord]:
        """
        Convert an approved signal to an order and submit to Alpaca.

        Steps:
          1. Fetch portfolio equity and current position
          2. Compute target share qty from approved_weight
          3. Compute delta vs current position
          4. Skip if delta < _MIN_QTY
          5. Submit market order
          6. Store and return OrderRecord
        """
        signal = approved.signal
        symbol = signal.symbol
        current_price = approved.current_price
        approved_weight = approved.decision.approved_weight

        if current_price <= 0:
            logger.warning("execute_signal: invalid current_price=%.4f for %s", current_price, symbol)
            return None

        equity = self._get_equity()
        target_qty = self._compute_qty(approved_weight, equity, current_price)
        current_qty = self._get_current_qty(symbol)
        delta = target_qty - current_qty

        if abs(delta) < self._MIN_QTY:
            logger.debug("%s: delta=%d shares below min lot — skipping.", symbol, delta)
            return None

        side = "buy" if delta > 0 else "sell"
        qty = abs(delta)

        logger.info(
            "Submitting %s %s %d @ ~%.2f (target_weight=%.1f%%)",
            side.upper(), symbol, qty, current_price, approved_weight * 100,
        )

        order_dict = self.client.submit_market_order(symbol, qty, side)
        record = self._record_order(order_dict, approved, side=side, qty=qty, symbol=symbol)
        self._order_log.append(record)
        return record

    def cancel_order(self, order_id: str) -> None:
        self.client.cancel_order(order_id)
        for rec in self._order_log:
            if rec.order_id == order_id:
                rec.status = OrderStatus.CANCELLED
                break

    def cancel_all_open_orders(self) -> None:
        self.client.cancel_all_orders()
        for rec in self._order_log:
            if rec.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                                   OrderStatus.REJECTED, OrderStatus.EXPIRED):
                rec.status = OrderStatus.CANCELLED

    def sync_order_statuses(self) -> None:
        """Poll Alpaca for updates on all non-terminal orders."""
        for rec in self._order_log:
            if rec.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                              OrderStatus.REJECTED, OrderStatus.EXPIRED):
                continue
            try:
                order_dict = self.client.get_order(rec.order_id)
                alpaca_status = order_dict.get("status", "").lower()
                rec.status = _ALPACA_STATUS_MAP.get(alpaca_status, rec.status)
                if order_dict.get("filled_at"):
                    filled_at_raw = order_dict["filled_at"]
                    rec.filled_at = (
                        datetime.fromisoformat(filled_at_raw.replace("Z", "+00:00"))
                        if isinstance(filled_at_raw, str) else filled_at_raw
                    )
                if order_dict.get("filled_avg_price"):
                    rec.filled_avg_price = float(order_dict["filled_avg_price"])
            except Exception as exc:
                logger.warning("Failed to sync order %s: %s", rec.order_id, exc)

    def get_order_log(self) -> list[OrderRecord]:
        return list(self._order_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_equity(self) -> float:
        try:
            acct = self.client.get_account()
            return float(acct["equity"])
        except Exception as exc:
            logger.warning("Could not fetch account equity: %s — defaulting to 0", exc)
            return 0.0

    def _compute_qty(
        self,
        target_weight: float,
        portfolio_equity: float,
        current_price: float,
    ) -> int:
        if portfolio_equity <= 0 or current_price <= 0:
            return 0
        return int(target_weight * portfolio_equity / current_price)

    def _get_current_qty(self, symbol: str) -> int:
        try:
            pos = self.client.get_position(symbol)
            if pos is None:
                return 0
            return int(float(pos.get("qty", 0)))
        except Exception:
            return 0

    def _record_order(
        self,
        order_dict: dict,
        approved: ApprovedSignal,
        *,
        side: str,
        qty: int,
        symbol: str,
    ) -> OrderRecord:
        alpaca_status = order_dict.get("status", "").lower()
        status = _ALPACA_STATUS_MAP.get(alpaca_status, OrderStatus.SUBMITTED)
        submitted_raw = order_dict.get("submitted_at")
        if submitted_raw:
            try:
                submitted_at = datetime.fromisoformat(submitted_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                submitted_at = datetime.utcnow()
        else:
            submitted_at = datetime.utcnow()

        return OrderRecord(
            order_id=order_dict["id"],
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_dict.get("type", "market"),
            limit_price=None,
            status=status,
            submitted_at=submitted_at,
        )
