"""
regime_strategies.py — Volatility-based allocation strategies.

DESIGN: The HMM detects volatility environments. Strategy is simple:
  Low vol  → be fully invested (calm markets drift upward)
  Mid vol  → stay invested if trend intact, reduce if not
  High vol → stay 60% invested (catches V-shaped rebounds)

ALWAYS LONG. NEVER SHORT.
Shorting destroys returns: markets have long-term upward drift and
V-shaped recoveries arrive faster than the HMM can detect them.
The correct response to high vol is REDUCING allocation, not reversing.

The edge comes from avoiding large drawdowns. Cut the worst drawdown
in half and compounding does the rest over time.

Vol-rank mapping (position = vol_rank / (n_states - 1)):
  position <= 0.33  →  LowVolBullStrategy
  0.33 < pos < 0.67 →  MidVolCautiousStrategy
  position >= 0.67  →  HighVolDefensiveStrategy

This sort is independent of the HMM's return-based label sort. "BULL"
label does NOT imply low volatility. The orchestrator uses vol_rank only.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Signal dataclass                                                     #
# ------------------------------------------------------------------ #

@dataclass
class Signal:
    """Allocation signal produced by a strategy for one symbol."""

    symbol: str
    direction: str                          # "LONG" | "FLAT"
    confidence: float                       # regime probability
    entry_price: float                      # last close at signal time
    stop_loss: float                        # stop price (absolute, not pct)
    position_size_pct: float                # base allocation fraction (e.g. 0.95)
    leverage: float                         # 1.0 or 1.25
    regime_id: int                          # return-sorted state rank
    regime_name: str                        # "BULL", "CRASH", etc.
    regime_probability: float               # P(state | obs_1:t)
    strategy_name: str
    timestamp: Optional[pd.Timestamp] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Backward-compatible aliases (used by risk_manager, order_executor)
    # ------------------------------------------------------------------

    @property
    def target_weight(self) -> float:
        """Base allocation fraction (position_size_pct, without leverage)."""
        return self.position_size_pct

    @property
    def stop_loss_pct(self) -> float:
        """Stop distance as a fraction of entry price."""
        if self.entry_price > 0:
            return max(0.0, (self.entry_price - self.stop_loss) / self.entry_price)
        return 0.0

    @property
    def vol_environment(self) -> str:
        return self.metadata.get("vol_environment", "unknown")

    @property
    def notes(self) -> str:
        return self.reasoning


# ------------------------------------------------------------------ #
# Base class                                                           #
# ------------------------------------------------------------------ #

class BaseStrategy(abc.ABC):
    """Abstract base for all regime-trader strategies."""

    NAME: ClassVar[str] = ""
    VOL_ENVIRONMENT: ClassVar[str] = ""          # "low" | "mid" | "high"
    REQUIRED_INDICATORS: ClassVar[list[str]] = ["ema_50", "atr"]

    def __init__(self, config: dict) -> None:
        self.config = config

    @abc.abstractmethod
    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """
        Produce an allocation signal for `symbol`.

        Parameters
        ----------
        symbol:
            Ticker being evaluated.
        bars:
            OHLCV + enriched indicator columns (ema_50, atr at minimum).
            Must be chronological with no future data.
        regime_state:
            Current regime from HMMEngine (forward algorithm output).

        Returns
        -------
        Signal, or None if bars are insufficient to compute indicators.
        """

    def get_stop_loss(self, signal: Signal, bars: pd.DataFrame) -> float:
        """Return stop-loss price given signal and bar data."""
        return signal.stop_loss

    def get_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
    ) -> int:
        """Return number of shares to hold (0 when direction is FLAT)."""
        if signal.direction == "FLAT" or current_price <= 0:
            return 0
        notional = portfolio_value * signal.position_size_pct * signal.leverage
        return max(0, int(notional / current_price))

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_indicators(
        self, bars: pd.DataFrame
    ) -> Optional[tuple[float, float, float]]:
        """
        Extract (close, ema_50, atr) from the last bar.

        Returns None if required columns are missing or NaN.
        """
        for col in self.REQUIRED_INDICATORS:
            if col not in bars.columns:
                return None
        row = bars.iloc[-1]
        close = float(row["close"])
        ema_50 = float(row["ema_50"])
        atr = float(row["atr"])
        if any(np.isnan(v) for v in (close, ema_50, atr)) or atr <= 0:
            return None
        return close, ema_50, atr

    def _apply_uncertainty(
        self,
        signal: Signal,
        is_uncertain: bool,
    ) -> Signal:
        """Halve position size and force leverage=1.0 when uncertain."""
        if is_uncertain:
            signal.position_size_pct = signal.position_size_pct * 0.5
            signal.leverage = 1.0
            signal.reasoning += " [UNCERTAINTY — size halved]"
        return signal

    def _flat_signal(
        self,
        symbol: str,
        regime_state: RegimeState,
        reason: str = "",
    ) -> Signal:
        return Signal(
            symbol=symbol,
            direction="FLAT",
            confidence=regime_state.probability,
            entry_price=0.0,
            stop_loss=0.0,
            position_size_pct=0.0,
            leverage=1.0,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            strategy_name=self.NAME,
            reasoning=reason,
        )


# ------------------------------------------------------------------ #
# Strategy registry                                                    #
# ------------------------------------------------------------------ #

class StrategyRegistry:
    """Central registry mapping strategy names to classes."""

    _registry: dict[str, type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator: @StrategyRegistry.register('my_strategy')."""
        def decorator(strategy_cls: type[BaseStrategy]) -> type[BaseStrategy]:
            strategy_cls.NAME = name
            cls._registry[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseStrategy]:
        if name not in cls._registry:
            raise KeyError(
                f"Strategy '{name}' not registered. "
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def list_names(cls) -> list[str]:
        return list(cls._registry.keys())


# ------------------------------------------------------------------ #
# Concrete strategies                                                  #
# ------------------------------------------------------------------ #

@StrategyRegistry.register("low_vol_bull")
class LowVolBullStrategy(BaseStrategy):
    """
    High-conviction long allocation in low-volatility environments.

    Targets 95% equity allocation with mild 1.25x leverage when the HMM
    identifies a stable, calm market state. This is where compounding
    earns most of its returns.

    Stop: max(price - 3*ATR, EMA50 - 0.5*ATR)
    """

    VOL_ENVIRONMENT = "low"
    REQUIRED_INDICATORS = ["ema_50", "atr"]

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        indicators = self._get_indicators(bars)
        if indicators is None:
            return None
        close, ema_50, atr = indicators

        base_alloc = float(self.config.get("low_vol_allocation", 0.95))
        leverage = float(self.config.get("low_vol_leverage", 1.25))

        stop = max(close - 3.0 * atr, ema_50 - 0.5 * atr)
        stop = min(stop, close * 0.92)   # hard cap: never more than 8% below close

        reasoning = (
            f"Low-vol regime '{regime_state.label}' "
            f"(p={regime_state.probability:.2f}). "
            f"Fully invested with {leverage}x leverage. "
            f"Stop @ {stop:.2f} (3-ATR floor)."
        )

        return Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=close,
            stop_loss=stop,
            position_size_pct=base_alloc,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            strategy_name=self.NAME,
            timestamp=regime_state.timestamp,
            reasoning=reasoning,
            metadata={"vol_environment": "low", "ema_50": ema_50, "atr": atr},
        )


@StrategyRegistry.register("mid_vol_cautious")
class MidVolCautiousStrategy(BaseStrategy):
    """
    Moderate allocation in transitional / mid-volatility regimes.

    Adjusts between 60% and 95% based on trend confirmation:
      - Price > EMA50 (trend intact) → 95%, 1.0x
      - Price < EMA50 (trend broken)  → 60%, 1.0x

    Stop: EMA50 - 0.5*ATR
    """

    VOL_ENVIRONMENT = "mid"
    REQUIRED_INDICATORS = ["ema_50", "atr"]

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        indicators = self._get_indicators(bars)
        if indicators is None:
            return None
        close, ema_50, atr = indicators

        trend_intact = close > ema_50
        if trend_intact:
            alloc = float(self.config.get("mid_vol_allocation_trend", 0.95))
            reason_suffix = "Trend intact (price > EMA50). Full mid-vol allocation."
        else:
            alloc = float(self.config.get("mid_vol_allocation_no_trend", 0.60))
            reason_suffix = "Trend broken (price < EMA50). Reduced allocation."

        stop = ema_50 - 0.5 * atr
        stop = min(stop, close * 0.94)   # hard cap: never more than 6% below close

        reasoning = (
            f"Mid-vol regime '{regime_state.label}' "
            f"(p={regime_state.probability:.2f}). "
            f"{reason_suffix} "
            f"Stop @ {stop:.2f} (EMA50 - 0.5 ATR)."
        )

        return Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=close,
            stop_loss=stop,
            position_size_pct=alloc,
            leverage=1.0,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            strategy_name=self.NAME,
            timestamp=regime_state.timestamp,
            reasoning=reasoning,
            metadata={
                "vol_environment": "mid",
                "trend_intact": trend_intact,
                "ema_50": ema_50,
                "atr": atr,
            },
        )


@StrategyRegistry.register("high_vol_defensive")
class HighVolDefensiveStrategy(BaseStrategy):
    """
    Capital-preservation allocation in high-volatility / stress regimes.

    Stays 60% invested (NOT flat, NOT short) to capture V-shaped rebounds.
    The HMM is 2-3 bars late detecting recoveries; going to cash/short
    means missing the sharpest up-days and destroying the strategy's edge.

    Stop: EMA50 - 1.0*ATR  (wider to avoid whipsaws in volatile conditions)
    """

    VOL_ENVIRONMENT = "high"
    REQUIRED_INDICATORS = ["ema_50", "atr"]

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        indicators = self._get_indicators(bars)
        if indicators is None:
            return None
        close, ema_50, atr = indicators

        alloc = float(self.config.get("high_vol_allocation", 0.60))

        stop = ema_50 - 1.0 * atr
        stop = min(stop, close * 0.90)   # hard cap: never more than 10% below close

        reasoning = (
            f"High-vol regime '{regime_state.label}' "
            f"(p={regime_state.probability:.2f}). "
            f"Defensive allocation {alloc:.0%}, leverage 1.0x. "
            f"Staying long to catch rebounds. "
            f"Stop @ {stop:.2f} (EMA50 - 1.0 ATR)."
        )

        return Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=close,
            stop_loss=stop,
            position_size_pct=alloc,
            leverage=1.0,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            strategy_name=self.NAME,
            timestamp=regime_state.timestamp,
            reasoning=reasoning,
            metadata={"vol_environment": "high", "ema_50": ema_50, "atr": atr},
        )


# ------------------------------------------------------------------ #
# Backward-compatible aliases                                          #
# ------------------------------------------------------------------ #

CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

# Maps every possible HMM label → the strategy class that handles it.
# The orchestrator uses vol_rank (not this dict) — this is a convenience
# lookup for ad-hoc signal generation and backward compat.
LABEL_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    "CRASH": HighVolDefensiveStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
    "WEAK_BEAR": MidVolCautiousStrategy,
    "BEAR": MidVolCautiousStrategy,
    "NEUTRAL": MidVolCautiousStrategy,
    "WEAK_BULL": MidVolCautiousStrategy,
    "BULL": LowVolBullStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "EUPHORIA": LowVolBullStrategy,
}


# ------------------------------------------------------------------ #
# Strategy Orchestrator                                                #
# ------------------------------------------------------------------ #

class StrategyOrchestrator:
    """
    Routes each bar to the correct strategy based on volatility rank.

    The orchestrator is constructed once from HMMEngine.get_regime_infos()
    and rebuilt after each HMM retrain via update_regime_infos().

    Vol-rank mapping (position = rank / (n - 1)):
      <= 0.33 → LowVolBullStrategy
      >= 0.67 → HighVolDefensiveStrategy
      else    → MidVolCautiousStrategy

    This mapping is INDEPENDENT of the HMM's return-based label sort.
    "BULL" does not imply low volatility.
    """

    def __init__(
        self,
        config: dict,
        regime_infos: dict[int, RegimeInfo],
    ) -> None:
        self.config = config
        self.rebalance_threshold = float(config.get("rebalance_threshold", 0.10))
        self.min_confidence = float(config.get("min_confidence", 0.55))
        self._strategy_map: dict[int, BaseStrategy] = {}
        self.update_regime_infos(regime_infos)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def update_regime_infos(self, regime_infos: dict[int, RegimeInfo]) -> None:
        """
        Rebuild the regime_id → strategy mapping.

        Called on construction and after each HMM retrain. Sorts regimes
        by expected_volatility (ascending) to assign vol_rank, then maps
        each rank to a strategy class via the 0.33 / 0.67 thresholds.
        """
        if not regime_infos:
            return

        sorted_by_vol = sorted(
            regime_infos.values(),
            key=lambda r: r.expected_volatility,
        )
        n = len(sorted_by_vol)

        self._strategy_map = {}
        for vol_rank, info in enumerate(sorted_by_vol):
            position = vol_rank / max(n - 1, 1)
            if position <= 0.33:
                strategy_cls = LowVolBullStrategy
            elif position >= 0.67:
                strategy_cls = HighVolDefensiveStrategy
            else:
                strategy_cls = MidVolCautiousStrategy

            self._strategy_map[info.regime_id] = strategy_cls(self.config)
            logger.debug(
                "Regime %d ('%s') vol_rank=%d → %s",
                info.regime_id, info.regime_name, vol_rank, strategy_cls.__name__,
            )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool = False,
        current_weights: dict[str, float] | None = None,
    ) -> list[Signal]:
        """
        Produce signals for all symbols given the current regime state.

        Steps for each symbol:
          1. Look up strategy by regime_state.state_id (vol_rank-based)
          2. Enrich bars with indicators if not already present
          3. Call strategy.generate_signal()
          4. Apply uncertainty mode (halve size, force lev=1.0) if needed
          5. Apply rebalancing threshold (skip if drift <= 10%)
          6. Collect approved signals

        Parameters
        ----------
        symbols:
            Tickers to evaluate this bar.
        bars_by_symbol:
            {symbol: OHLCV DataFrame}. Enriched columns (ema_50, atr)
            are added automatically if missing.
        regime_state:
            Current regime from HMMEngine.predict().
        is_flickering:
            Set True when HMMEngine.is_flickering() returns True.
        current_weights:
            {symbol: current_allocation_fraction}. Used for rebalancing
            threshold. Defaults to all zeros if None.

        Returns
        -------
        List of Signal objects approved for execution.
        """
        strategy = self._strategy_map.get(regime_state.state_id)
        if strategy is None:
            logger.warning(
                "No strategy mapped for regime_id=%d; falling back to HighVol.",
                regime_state.state_id,
            )
            strategy = HighVolDefensiveStrategy(self.config)

        is_uncertain = (
            regime_state.probability < self.min_confidence
            or not regime_state.is_confirmed
            or is_flickering
        )

        weights = current_weights or {}
        signals: list[Signal] = []

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol)
            if bars is None or len(bars) < 60:
                logger.debug("Skipping %s — insufficient bars.", symbol)
                continue

            bars = self._ensure_enriched(bars)

            sig = strategy.generate_signal(symbol, bars, regime_state)
            if sig is None:
                logger.debug("Skipping %s — strategy returned None (missing indicators).", symbol)
                continue

            # Uncertainty mode
            if is_uncertain:
                sig.position_size_pct *= 0.5
                sig.leverage = 1.0
                sig.reasoning += " [UNCERTAINTY — size halved]"

            # Rebalancing threshold — skip if change is too small
            current_wt = weights.get(symbol, 0.0)
            if abs(sig.position_size_pct - current_wt) <= self.rebalance_threshold:
                logger.debug(
                    "%s: target=%.2f current=%.2f — within rebalance threshold, skipping.",
                    symbol, sig.position_size_pct, current_wt,
                )
                continue

            signals.append(sig)

        if signals:
            logger.info(
                "Regime '%s' (p=%.2f) → %d/%d signals generated%s.",
                regime_state.label,
                regime_state.probability,
                len(signals),
                len(symbols),
                " [UNCERTAIN]" if is_uncertain else "",
            )

        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_enriched(bars: pd.DataFrame) -> pd.DataFrame:
        """Add strategy indicators if missing (ema_50, atr at minimum)."""
        if "ema_50" not in bars.columns or "atr" not in bars.columns:
            from data.feature_engineering import FeatureEngineer
            bars = FeatureEngineer.enrich_bars(bars)
        return bars

    def get_strategy_for_regime(self, state_id: int) -> Optional[BaseStrategy]:
        """Return the strategy instance mapped to this regime state_id."""
        return self._strategy_map.get(state_id)

    def describe_mapping(self) -> dict[int, str]:
        """Return {state_id: strategy_name} for logging / dashboard display."""
        return {
            regime_id: strategy.NAME
            for regime_id, strategy in self._strategy_map.items()
        }
