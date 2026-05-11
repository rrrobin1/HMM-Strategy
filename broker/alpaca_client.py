"""
alpaca_client.py — Thin wrapper around the Alpaca REST/WebSocket APIs.

Handles authentication, paper vs live routing, and low-level HTTP calls.
All higher-level logic (order placement, position tracking) lives in
order_executor.py and position_tracker.py respectively.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AlpacaClient:
    """
    Authenticates with Alpaca and exposes raw API methods.

    Reads credentials from environment variables:
        ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
    """

    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_BASE_URL = "https://api.alpaca.markets"
    DATA_BASE_URL = "https://data.alpaca.markets"

    def __init__(self, paper: bool = True) -> None:
        self.paper = paper
        self._api = None        # alpaca_trade_api.REST instance (set in connect())
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Initialize the Alpaca REST client from environment variables.

        Raises
        ------
        EnvironmentError
            If ALPACA_API_KEY or ALPACA_SECRET_KEY are not set.
        ValueError
            If credentials are invalid (401 from Alpaca).
        """
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment."
            )

        try:
            import alpaca_trade_api as tradeapi  # type: ignore
        except ImportError:
            raise ImportError(
                "alpaca-trade-api is not installed. Run: pip install alpaca-trade-api"
            )

        base_url = self.PAPER_BASE_URL if self.paper else self.LIVE_BASE_URL
        self._api = tradeapi.REST(api_key, secret_key, base_url=base_url, api_version="v2")
        # Verify credentials by fetching account
        try:
            self._api.get_account()
        except Exception as exc:
            raise ValueError(f"Alpaca credential check failed: {exc}") from exc
        self._connected = True
        logger.info("Alpaca client connected (%s mode).", "paper" if self.paper else "LIVE")

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        acct = self._api.get_account()
        return {
            "id": acct.id,
            "equity": str(acct.equity),
            "cash": str(acct.cash),
            "buying_power": str(acct.buying_power),
            "portfolio_value": str(acct.portfolio_value),
            "status": acct.status,
        }

    def get_clock(self) -> dict:
        clock = self._api.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars.

        Returns DataFrame with DatetimeIndex and columns
        [open, high, low, close, volume].
        """
        kwargs: dict = {"timeframe": timeframe, "start": start}
        if end:
            kwargs["end"] = end
        if limit:
            kwargs["limit"] = limit
        bars = self._api.get_bars(symbol, feed="iex", **kwargs).df
        if bars.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        bars.index = pd.to_datetime(bars.index, utc=True).tz_convert("America/New_York")
        bars.columns = [c.lower() if isinstance(c, str) else str(c) for c in bars.columns]
        # Alpaca sometimes returns abbreviated names (o/h/l/c/v)
        bars = bars.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in bars.columns]
        return bars[cols]

    def get_latest_bar(self, symbol: str) -> dict:
        bar = self._api.get_latest_bar(symbol, feed="iex")
        return {
            "open": float(bar.o),
            "high": float(bar.h),
            "low": float(bar.l),
            "close": float(bar.c),
            "volume": float(bar.v),
            "timestamp": str(bar.t),
        }

    def get_latest_quote(self, symbol: str) -> dict:
        quote = self._api.get_latest_quote(symbol, feed="iex")
        return {
            "ask": float(quote.ap),
            "bid": float(quote.bp),
            "ask_size": int(quote.as_),
            "bid_size": int(quote.bs),
            "timestamp": str(quote.t),
        }

    # ------------------------------------------------------------------
    # Orders (low-level)
    # ------------------------------------------------------------------

    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        time_in_force: str = "day",
    ) -> dict:
        order = self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force=time_in_force,
        )
        return self._order_to_dict(order)

    def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> dict:
        order = self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="limit",
            time_in_force=time_in_force,
            limit_price=str(round(limit_price, 2)),
        )
        return self._order_to_dict(order)

    def cancel_order(self, order_id: str) -> None:
        self._api.cancel_order(order_id)

    def cancel_all_orders(self) -> None:
        self._api.cancel_all_orders()

    def get_order(self, order_id: str) -> dict:
        return self._order_to_dict(self._api.get_order(order_id))

    def list_open_orders(self) -> list[dict]:
        return [self._order_to_dict(o) for o in self._api.list_orders(status="open")]

    def list_recent_orders(self, limit: int = 20) -> list[dict]:
        return [self._order_to_dict(o) for o in self._api.list_orders(status="all", limit=limit)]

    # ------------------------------------------------------------------
    # Positions (low-level)
    # ------------------------------------------------------------------

    def get_all_positions(self) -> list[dict]:
        return [self._position_to_dict(p) for p in self._api.list_positions()]

    def get_position(self, symbol: str) -> Optional[dict]:
        try:
            pos = self._api.get_position(symbol)
            return self._position_to_dict(pos)
        except Exception:
            return None

    def close_position(self, symbol: str) -> dict:
        order = self._api.close_position(symbol)
        return self._order_to_dict(order)

    def close_all_positions(self) -> None:
        self._api.close_all_positions()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _order_to_dict(order) -> dict:
        return {
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "type": order.type,
            "status": order.status,
            "submitted_at": str(order.submitted_at) if order.submitted_at else None,
            "filled_at": str(order.filled_at) if order.filled_at else None,
            "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
        }

    @staticmethod
    def _position_to_dict(pos) -> dict:
        return {
            "symbol": pos.symbol,
            "qty": pos.qty,
            "avg_entry_price": pos.avg_entry_price,
            "current_price": pos.current_price,
            "market_value": pos.market_value,
            "unrealized_pl": pos.unrealized_pl,
            "unrealized_plpc": pos.unrealized_plpc,
        }
