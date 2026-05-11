"""
market_data.py — Historical and real-time market data fetching.

Wraps AlpacaClient to provide DataFrames with consistent column names and
DatetimeIndex. Also handles data caching to reduce redundant API calls.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetches OHLCV bars from Alpaca with optional local disk caching.

    Cache files are stored in data/cache/ as Parquet. The cache key is
    a hash of (symbol, timeframe, start, end); invalidated by any date change.
    """

    CACHE_DIR = Path("data/cache")

    def __init__(self, client: AlpacaClient, use_cache: bool = True) -> None:
        self.client = client
        self.use_cache = use_cache

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return OHLCV bars as DataFrame with DatetimeIndex.

        Checks local Parquet cache before hitting the API. Columns are
        lowercase: open, high, low, close, volume.
        """
        end_key = end or "latest"
        cache_path = self._cache_path(symbol, timeframe, start, end_key)

        if self.use_cache:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                logger.debug("Cache hit: %s %s %s→%s", symbol, timeframe, start, end_key)
                return cached

        logger.debug("Fetching %s %s %s→%s from Alpaca", symbol, timeframe, start, end_key)
        df = self.client.get_bars(symbol, timeframe, start, end)

        if self.use_cache and not df.empty:
            self._save_to_cache(df, cache_path)

        return df

    def get_bars_multi(
        self,
        symbols: list[str],
        timeframe: str,
        start: str,
        end: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols; returns {symbol: DataFrame}."""
        result = {}
        for sym in symbols:
            try:
                df = self.get_bars(sym, timeframe, start, end)
                if not df.empty:
                    result[sym] = df
                else:
                    logger.warning("No bars returned for %s", sym)
            except Exception as exc:
                logger.error("Failed to fetch bars for %s: %s", sym, exc)
        return result

    def refresh(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetch only the most recent bar and return it.

        Does NOT append to the cache (cache is for full historical windows).
        Use this to get the latest bar at end-of-day without re-downloading
        all history.
        """
        try:
            bar = self.client.get_latest_bar(symbol)
            return pd.DataFrame([{
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            }], index=pd.to_datetime([bar["timestamp"]]))
        except Exception as exc:
            logger.error("Failed to refresh %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Real-time
    # ------------------------------------------------------------------

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Return {symbol: latest_price} using Alpaca latest bar close."""
        prices: dict[str, float] = {}
        for sym in symbols:
            try:
                bar = self.client.get_latest_bar(sym)
                prices[sym] = float(bar["close"])
            except Exception as exc:
                logger.warning("Could not get price for %s: %s", sym, exc)
        return prices

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, symbol: str, timeframe: str, start: str, end: str) -> Path:
        key = f"{symbol}_{timeframe}_{start}_{end}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.CACHE_DIR / f"{symbol}_{timeframe}_{h}.parquet"

    def _load_from_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Cache read failed (%s): %s", path.name, exc)
            return None

    def _save_to_cache(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path)
        except Exception as exc:
            logger.warning("Cache write failed (%s): %s", path.name, exc)
