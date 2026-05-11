"""
feature_engineering.py — Technical indicators and HMM feature construction.

All indicators are computed without look-ahead: only data up to the current
bar is used. The test_look_ahead.py test suite verifies this invariant.

HMM features (14 total, returned by build_hmm_features()):
  Returns      : log_ret_1, log_ret_5, log_ret_20
  Volatility   : realized_vol_20, vol_ratio (5-day / 20-day)
  Volume       : vol_norm (z-score vs 50-day SMA), vol_trend (slope of 10-day SMA)
  Trend        : adx (14-day), sma50_slope
  Mean-revert  : rsi14_zscore, dist_sma200
  Momentum     : roc_10, roc_20
  Range        : norm_atr (14-day ATR / close)

All features are standardized with rolling z-scores (252-bar lookback,
min 100 bars) before being passed to the HMM.

Strategy indicators (appended by enrich_bars()):
  ema_50, ema_200, atr, rsi, adx, bb_width
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Stateless helper — all methods are pure functions of input data."""

    # Feature names in the order they appear in the HMM matrix.
    FEATURE_COLS: list[str] = [
        "log_ret_1", "log_ret_5", "log_ret_20",
        "realized_vol_20", "vol_ratio",
        "vol_norm", "vol_trend",
        "adx", "sma50_slope",
        "rsi14_zscore", "dist_sma200",
        "roc_10", "roc_20",
        "norm_atr",
    ]

    # ------------------------------------------------------------------
    # HMM feature set
    # ------------------------------------------------------------------

    @staticmethod
    def build_hmm_features(bars: pd.DataFrame) -> pd.DataFrame:
        """
        Build the standardized feature matrix fed to HMMEngine.

        Parameters
        ----------
        bars:
            OHLCV DataFrame with DatetimeIndex and lowercase column names
            (open, high, low, close, volume). Chronological order required.

        Returns
        -------
        DataFrame with columns == FEATURE_COLS. NaN rows (initial window)
        are dropped so the caller can pass the result directly to fit/predict.
        Each column is a rolling z-score with a 252-bar lookback.
        """
        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        volume = bars["volume"]

        out = pd.DataFrame(index=bars.index)

        # ---- Returns ------------------------------------------------
        log_ret = np.log(close / close.shift(1))
        out["log_ret_1"] = log_ret
        out["log_ret_5"] = np.log(close / close.shift(5))
        out["log_ret_20"] = np.log(close / close.shift(20))

        # ---- Volatility ---------------------------------------------
        out["realized_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)
        vol_5d = log_ret.rolling(5).std()
        vol_20d = log_ret.rolling(20).std()
        out["vol_ratio"] = vol_5d / (vol_20d + 1e-10)

        # ---- Volume -------------------------------------------------
        vol_sma50 = volume.rolling(50).mean()
        vol_std50 = volume.rolling(50).std()
        out["vol_norm"] = (volume - vol_sma50) / (vol_std50 + 1.0)
        vol_sma10 = volume.rolling(10).mean()
        # 5-bar change of volume SMA, normalized by lagged SMA
        out["vol_trend"] = vol_sma10.diff(5) / (vol_sma10.shift(5) + 1.0)

        # ---- Trend --------------------------------------------------
        out["adx"] = FeatureEngineer.adx(high, low, close, 14)
        sma50 = close.rolling(50).mean()
        out["sma50_slope"] = sma50.diff(5) / (close + 1e-10)

        # ---- Mean reversion -----------------------------------------
        rsi14 = FeatureEngineer.rsi(close, 14)
        rsi_mu = rsi14.rolling(252, min_periods=50).mean()
        rsi_sd = rsi14.rolling(252, min_periods=50).std()
        out["rsi14_zscore"] = (rsi14 - rsi_mu) / (rsi_sd + 1e-10)
        sma200 = close.rolling(200).mean()
        out["dist_sma200"] = (close - sma200) / (close + 1e-10)

        # ---- Momentum -----------------------------------------------
        out["roc_10"] = close.pct_change(10)
        out["roc_20"] = close.pct_change(20)

        # ---- Range --------------------------------------------------
        atr14 = FeatureEngineer.atr(high, low, close, 14)
        out["norm_atr"] = atr14 / (close + 1e-10)

        # ---- Rolling z-score standardization (252-bar) --------------
        for col in FeatureEngineer.FEATURE_COLS:
            mu = out[col].rolling(252, min_periods=100).mean()
            sd = out[col].rolling(252, min_periods=100).std()
            out[col] = (out[col] - mu) / (sd + 1e-10)

        return out[FeatureEngineer.FEATURE_COLS].dropna()

    # ------------------------------------------------------------------
    # Strategy indicators
    # ------------------------------------------------------------------

    @staticmethod
    def enrich_bars(bars: pd.DataFrame) -> pd.DataFrame:
        """
        Append strategy indicator columns to a bars DataFrame.

        Returns a copy with additional columns:
        ema_50, ema_200, atr, rsi, adx, bb_width.
        """
        df = bars.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        df["ema_50"] = FeatureEngineer.ema(close, 50)
        df["ema_200"] = FeatureEngineer.ema(close, 200)
        df["atr"] = FeatureEngineer.atr(high, low, close, 14)
        df["rsi"] = FeatureEngineer.rsi(close, 14)
        df["adx"] = FeatureEngineer.adx(high, low, close, 14)
        df["bb_width"] = FeatureEngineer.bb_width(close, 20, 2.0)

        return df

    # ------------------------------------------------------------------
    # Individual indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def log_returns(close: pd.Series) -> pd.Series:
        return np.log(close / close.shift(1))

    @staticmethod
    def realized_vol(log_returns: pd.Series, window: int = 21) -> pd.Series:
        """Annualized realized volatility: rolling std * sqrt(252)."""
        return log_returns.rolling(window).std() * np.sqrt(252)

    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range using Wilder smoothing (EWM alpha=1/period)."""
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """RSI using Wilder smoothing (EWM alpha=1/period)."""
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average Directional Index using Wilder smoothing."""
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        up = high - high.shift(1)
        down = low.shift(1) - low

        pos_dm = pd.Series(
            np.where((up > down) & (up > 0), up.values, 0.0),
            index=close.index,
        )
        neg_dm = pd.Series(
            np.where((down > up) & (down > 0), down.values, 0.0),
            index=close.index,
        )

        alpha = 1.0 / period
        atr_w = tr.ewm(alpha=alpha, adjust=False).mean()
        pos_di = 100.0 * pos_dm.ewm(alpha=alpha, adjust=False).mean() / (atr_w + 1e-10)
        neg_di = 100.0 * neg_dm.ewm(alpha=alpha, adjust=False).mean() / (atr_w + 1e-10)

        dx = 100.0 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-10)
        return dx.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def bb_width(
        close: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.Series:
        """Bollinger Band width: (upper - lower) / middle."""
        mid = close.rolling(window).mean()
        std = close.rolling(window).std()
        return (2.0 * num_std * std) / (mid + 1e-10)
