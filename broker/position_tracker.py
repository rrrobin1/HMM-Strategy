"""
position_tracker.py — Track open positions and real-time P&L.

Builds PortfolioSnapshot objects (consumed by RiskManager and monitoring)
by combining Alpaca position data with internal order records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from broker.alpaca_client import AlpacaClient
from core.risk_manager import PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass
class PositionRecord:
    """Single open position."""

    symbol: str
    qty: int                            # shares held (negative = short)
    avg_entry_price: float
    current_price: float
    market_value: float                 # qty * current_price
    unrealized_pnl: float
    unrealized_pnl_pct: float
    opened_at: Optional[datetime] = None
    regime_at_entry: str = ""           # vol_environment when position was opened


class PositionTracker:
    """
    Maintains a live view of all open positions and account equity.

    Syncs with Alpaca on each call to sync(); also accepts price updates
    from the market data feed to avoid extra API calls.
    """

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client
        self._positions: dict[str, PositionRecord] = {}
        self._peak_equity: float = 0.0
        self._daily_start_equity: float = 0.0
        self._weekly_start_equity: float = 0.0
        self._daily_trade_count: int = 0
        self._last_equity: float = 0.0
        self._last_cash: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self) -> None:
        """Pull latest positions and account data from Alpaca."""
        acct = self.client.get_account()
        equity = float(acct["equity"])
        cash = float(acct["cash"])

        self._last_equity = equity
        self._last_cash = cash
        self._update_peak(equity)

        # Seed start-of-day / start-of-week equity on first sync
        if self._daily_start_equity == 0.0:
            self._daily_start_equity = equity
        if self._weekly_start_equity == 0.0:
            self._weekly_start_equity = equity

        positions_raw = self.client.get_all_positions()
        self._positions = {}
        for p in positions_raw:
            qty = int(float(p.get("qty", 0)))
            if qty == 0:
                continue
            entry = float(p.get("avg_entry_price", 0.0))
            price = float(p.get("current_price", 0.0))
            mkt_val = float(p.get("market_value", qty * price))
            upnl = float(p.get("unrealized_pl", 0.0))
            upnl_pct = float(p.get("unrealized_plpc", 0.0))
            self._positions[p["symbol"]] = PositionRecord(
                symbol=p["symbol"],
                qty=qty,
                avg_entry_price=entry,
                current_price=price,
                market_value=mkt_val,
                unrealized_pnl=upnl,
                unrealized_pnl_pct=upnl_pct,
            )

        logger.debug(
            "sync: equity=%.2f cash=%.2f positions=%d",
            equity, cash, len(self._positions),
        )

    def get_snapshot(self) -> PortfolioSnapshot:
        """Return the current PortfolioSnapshot for the risk manager."""
        return PortfolioSnapshot(
            equity=self._last_equity,
            cash=self._last_cash,
            positions={sym: rec.market_value for sym, rec in self._positions.items()},
            peak_equity=self._peak_equity,
            daily_start_equity=self._daily_start_equity,
            weekly_start_equity=self._weekly_start_equity,
            open_trade_count_today=self._daily_trade_count,
            gross_exposure=self._compute_gross_exposure(),
        )

    def update_prices(self, prices: dict[str, float]) -> None:
        """Apply real-time price updates without a full API round-trip."""
        for sym, price in prices.items():
            if sym in self._positions and price > 0:
                rec = self._positions[sym]
                rec.current_price = price
                rec.market_value = rec.qty * price
                rec.unrealized_pnl = rec.market_value - rec.avg_entry_price * rec.qty
                if rec.avg_entry_price > 0:
                    rec.unrealized_pnl_pct = (
                        (price - rec.avg_entry_price) / rec.avg_entry_price
                    )
        # Recompute equity estimate from updated prices
        total_pos_value = sum(r.market_value for r in self._positions.values())
        self._last_equity = self._last_cash + total_pos_value
        self._update_peak(self._last_equity)

    def record_trade(self, symbol: str) -> None:
        """Increment the daily trade counter (called on confirmed order submission)."""
        self._daily_trade_count += 1

    def reset_daily_counters(self) -> None:
        """Called at market open each day."""
        self._daily_start_equity = self._last_equity
        self._daily_trade_count = 0
        logger.info("Daily counters reset. Start equity: %.2f", self._daily_start_equity)

    def reset_weekly_counters(self) -> None:
        """Called at market open each Monday."""
        self._weekly_start_equity = self._last_equity
        logger.info("Weekly counters reset. Start equity: %.2f", self._weekly_start_equity)

    def get_positions(self) -> dict[str, PositionRecord]:
        return dict(self._positions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_peak(self, equity: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity

    def _compute_gross_exposure(self) -> float:
        return sum(abs(r.market_value) for r in self._positions.values())
