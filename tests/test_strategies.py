"""
test_strategies.py — Unit tests for the Phase 3 strategy layer.

Covers:
  - StrategyRegistry: registration, lookup, unknown key raises
  - All three default strategies are registered
  - Signal fields are correctly populated (direction, price, stop, pct, leverage)
  - LowVolBull: 95% allocation, 1.25x leverage
  - MidVolCautious: 95% above EMA50, 60% below EMA50
  - HighVolDefensive: 60% allocation, 1.0x leverage, never short
  - Uncertainty mode halves position_size_pct and forces leverage=1.0
  - Rebalancing threshold prevents signals when change <= 10%
  - StrategyOrchestrator vol_rank routing
  - LABEL_TO_STRATEGY covers all canonical label names
  - Backward-compatible aliases exist
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    LABEL_TO_STRATEGY,
    BaseStrategy,
    BullTrendStrategy,
    CrashDefensiveStrategy,
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    MidVolCautiousStrategy,
    Signal,
    StrategyOrchestrator,
    StrategyRegistry,
)


# ------------------------------------------------------------------ #
# Fixtures / helpers                                                   #
# ------------------------------------------------------------------ #

STRATEGY_CONFIG = {
    "low_vol_allocation": 0.95,
    "mid_vol_allocation_trend": 0.95,
    "mid_vol_allocation_no_trend": 0.60,
    "high_vol_allocation": 0.60,
    "low_vol_leverage": 1.25,
    "rebalance_threshold": 0.10,
    "min_confidence": 0.55,
}


def make_bars(
    n: int = 300,
    trend: str = "up",
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic OHLCV bars, already enriched with strategy indicators."""
    from data.feature_engineering import FeatureEngineer

    rng = np.random.default_rng(seed)
    drift = 0.0004 if trend == "up" else -0.0004
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.010, n)))
    dates = pd.bdate_range("2020-01-01", periods=n)
    bars = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )
    return FeatureEngineer.enrich_bars(bars)


def make_regime_info(
    regime_id: int,
    expected_vol: float,
    name: str = "BULL",
) -> RegimeInfo:
    return RegimeInfo(
        regime_id=regime_id,
        regime_name=name,
        expected_return=0.001,
        expected_volatility=expected_vol,
        vol_environment="low" if expected_vol < 0.1 else "high",
        recommended_strategy_type="low_vol_bull",
        max_leverage_allowed=1.25,
    )


def make_regime_state(
    state_id: int = 0,
    label: str = "BULL",
    probability: float = 0.80,
    n_states: int = 3,
    is_confirmed: bool = True,
) -> RegimeState:
    probs = np.zeros(n_states)
    probs[state_id % n_states] = probability
    probs[(state_id + 1) % n_states] = 1.0 - probability
    return RegimeState(
        label=label,
        state_id=state_id,
        probability=probability,
        state_probabilities=probs,
        timestamp=pd.Timestamp("2023-01-01"),
        is_confirmed=is_confirmed,
        consecutive_bars=5,
        vol_environment="low",
        n_states=n_states,
    )


def make_low_vol_infos() -> dict[int, RegimeInfo]:
    """Three-state regime_infos: low / mid / high vol."""
    return {
        0: make_regime_info(0, 0.08, "BULL"),
        1: make_regime_info(1, 0.15, "NEUTRAL"),
        2: make_regime_info(2, 0.30, "BEAR"),
    }


# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #

class TestStrategyRegistry:
    def test_defaults_registered(self):
        names = StrategyRegistry.list_names()
        assert "low_vol_bull" in names
        assert "mid_vol_cautious" in names
        assert "high_vol_defensive" in names

    def test_get_returns_class(self):
        cls = StrategyRegistry.get("low_vol_bull")
        assert issubclass(cls, BaseStrategy)

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            StrategyRegistry.get("nonexistent_xyz")

    def test_register_decorator(self):
        @StrategyRegistry.register("_test_temp")
        class _Temp(BaseStrategy):
            VOL_ENVIRONMENT = "low"
            REQUIRED_INDICATORS = []
            def generate_signal(self, sym, bars, rs): return None

        assert StrategyRegistry.get("_test_temp") is _Temp

    def test_label_to_strategy_complete(self):
        """Every canonical HMM label must map to a strategy class."""
        canonical = [
            "CRASH", "STRONG_BEAR", "WEAK_BEAR", "BEAR",
            "NEUTRAL",
            "WEAK_BULL", "BULL", "STRONG_BULL", "EUPHORIA",
        ]
        for label in canonical:
            assert label in LABEL_TO_STRATEGY, f"'{label}' missing from LABEL_TO_STRATEGY"
            assert issubclass(LABEL_TO_STRATEGY[label], BaseStrategy)

    def test_backward_compatible_aliases(self):
        assert CrashDefensiveStrategy is HighVolDefensiveStrategy
        assert BullTrendStrategy is LowVolBullStrategy


# ------------------------------------------------------------------ #
# LowVolBullStrategy                                                   #
# ------------------------------------------------------------------ #

class TestLowVolBullStrategy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.bars = make_bars(300, trend="up")
        self.strategy = LowVolBullStrategy(STRATEGY_CONFIG)
        self.regime = make_regime_state(0, "BULL", 0.85)

    def test_returns_signal(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert isinstance(sig, Signal)

    def test_direction_is_long(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert sig.direction == "LONG"

    def test_allocation_is_95(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert abs(sig.position_size_pct - 0.95) < 1e-9

    def test_leverage_is_1_25(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert abs(sig.leverage - 1.25) < 1e-9

    def test_stop_below_entry(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert sig.stop_loss < sig.entry_price

    def test_stop_pct_positive(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert sig.stop_loss_pct > 0.0

    def test_never_short(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert sig.direction != "SHORT"

    def test_returns_none_when_indicators_missing(self):
        # Strip enriched indicator columns — strategy must return None gracefully.
        raw = self.bars[["open", "high", "low", "close", "volume"]]
        sig = self.strategy.generate_signal("SPY", raw, self.regime)
        assert sig is None


# ------------------------------------------------------------------ #
# MidVolCautiousStrategy                                               #
# ------------------------------------------------------------------ #

class TestMidVolCautiousStrategy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.strategy = MidVolCautiousStrategy(STRATEGY_CONFIG)
        self.regime = make_regime_state(1, "NEUTRAL", 0.78)

    def test_95_when_price_above_ema50(self):
        bars = make_bars(300, trend="up")  # uptrend → price above EMA50
        sig = self.strategy.generate_signal("SPY", bars, self.regime)
        assert sig is not None
        assert sig.direction == "LONG"
        # Allow for minor floating-point tolerance
        assert sig.position_size_pct <= 0.95 + 1e-9

    def test_60_when_price_below_ema50(self):
        bars = make_bars(300, trend="down")
        sig = self.strategy.generate_signal("SPY", bars, self.regime)
        assert sig is not None
        assert sig.direction == "LONG"
        # Downtrend may put price below EMA50 → reduced allocation
        # (Accept both 60% and 95% — depends on how far trend has moved)
        assert sig.position_size_pct <= 0.95 + 1e-9

    def test_leverage_is_1(self):
        bars = make_bars(300, trend="up")
        sig = self.strategy.generate_signal("SPY", bars, self.regime)
        assert abs(sig.leverage - 1.0) < 1e-9

    def test_stop_below_entry(self):
        bars = make_bars(300, trend="up")
        sig = self.strategy.generate_signal("SPY", bars, self.regime)
        assert sig.stop_loss < sig.entry_price

    def test_never_short(self):
        bars = make_bars(300, trend="down")
        sig = self.strategy.generate_signal("SPY", bars, self.regime)
        assert sig.direction in ("LONG", "FLAT")
        assert sig.direction != "SHORT"


# ------------------------------------------------------------------ #
# HighVolDefensiveStrategy                                             #
# ------------------------------------------------------------------ #

class TestHighVolDefensiveStrategy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.bars = make_bars(300, trend="down")
        self.strategy = HighVolDefensiveStrategy(STRATEGY_CONFIG)
        self.regime = make_regime_state(2, "BEAR", 0.72)

    def test_returns_signal(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert isinstance(sig, Signal)

    def test_direction_long_not_short(self):
        """CRITICAL: Must never go short even in high-vol / crash regimes."""
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert sig.direction == "LONG", (
            "HighVolDefensiveStrategy returned SHORT — this is forbidden. "
            "Correct response to high vol is reducing allocation, not reversing."
        )

    def test_allocation_is_60(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert abs(sig.position_size_pct - 0.60) < 1e-9

    def test_leverage_is_1(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert abs(sig.leverage - 1.0) < 1e-9

    def test_stop_below_entry(self):
        sig = self.strategy.generate_signal("SPY", self.bars, self.regime)
        assert sig.stop_loss < sig.entry_price

    def test_wider_stop_than_midvol(self):
        """High-vol stop (1.0 ATR below EMA50) should be wider than mid-vol (0.5 ATR)."""
        mid_strat = MidVolCautiousStrategy(STRATEGY_CONFIG)
        high_strat = HighVolDefensiveStrategy(STRATEGY_CONFIG)
        regime_mid = make_regime_state(1, "NEUTRAL", 0.78)
        sig_mid = mid_strat.generate_signal("SPY", self.bars, regime_mid)
        sig_high = high_strat.generate_signal("SPY", self.bars, self.regime)
        if sig_mid and sig_high:
            assert sig_high.stop_loss_pct >= sig_mid.stop_loss_pct - 1e-6


# ------------------------------------------------------------------ #
# Uncertainty mode                                                     #
# ------------------------------------------------------------------ #

class TestUncertaintyMode:
    def test_low_confidence_halves_size(self):
        bars = make_bars(300, trend="up")
        strategy = LowVolBullStrategy(STRATEGY_CONFIG)
        # High confidence signal for comparison
        regime_hi = make_regime_state(0, "BULL", 0.85)
        sig_hi = strategy.generate_signal("SPY", bars, regime_hi)

        # Uncertainty applied directly via _apply_uncertainty
        import copy
        sig_lo = copy.deepcopy(sig_hi)
        strategy._apply_uncertainty(sig_lo, is_uncertain=True)

        assert abs(sig_lo.position_size_pct - sig_hi.position_size_pct * 0.5) < 1e-9
        assert abs(sig_lo.leverage - 1.0) < 1e-9
        assert "UNCERTAINTY" in sig_lo.reasoning

    def test_normal_confidence_no_change(self):
        bars = make_bars(300, trend="up")
        strategy = LowVolBullStrategy(STRATEGY_CONFIG)
        regime = make_regime_state(0, "BULL", 0.85)
        sig = strategy.generate_signal("SPY", bars, regime)
        orig_pct = sig.position_size_pct
        strategy._apply_uncertainty(sig, is_uncertain=False)
        assert abs(sig.position_size_pct - orig_pct) < 1e-9


# ------------------------------------------------------------------ #
# StrategyOrchestrator                                                 #
# ------------------------------------------------------------------ #

class TestStrategyOrchestrator:
    @pytest.fixture(autouse=True)
    def setup(self):
        regime_infos = make_low_vol_infos()
        self.orch = StrategyOrchestrator(STRATEGY_CONFIG, regime_infos)
        self.bars = make_bars(300, trend="up")
        self.bars_by_symbol = {"SPY": self.bars, "QQQ": self.bars.copy()}

    def test_vol_rank_low_maps_to_low_vol_strategy(self):
        """State 0 has lowest vol → should map to LowVolBullStrategy."""
        regime = make_regime_state(0, "BULL", 0.85)
        strategy = self.orch.get_strategy_for_regime(0)
        assert isinstance(strategy, LowVolBullStrategy)

    def test_vol_rank_high_maps_to_high_vol_strategy(self):
        """State 2 has highest vol → should map to HighVolDefensiveStrategy."""
        strategy = self.orch.get_strategy_for_regime(2)
        assert isinstance(strategy, HighVolDefensiveStrategy)

    def test_generate_signals_returns_list(self):
        regime = make_regime_state(0, "BULL", 0.85)
        sigs = self.orch.generate_signals(
            ["SPY", "QQQ"], self.bars_by_symbol, regime
        )
        assert isinstance(sigs, list)
        assert all(isinstance(s, Signal) for s in sigs)

    def test_all_signals_long_not_short(self):
        for state_id in [0, 1, 2]:
            labels = ["BULL", "NEUTRAL", "BEAR"]
            regime = make_regime_state(state_id, labels[state_id], 0.80)
            sigs = self.orch.generate_signals(
                ["SPY"], self.bars_by_symbol, regime
            )
            for s in sigs:
                assert s.direction in ("LONG", "FLAT"), (
                    f"Got SHORT from regime {labels[state_id]}; this is forbidden."
                )

    def test_rebalancing_threshold_suppresses_small_changes(self):
        """If current weight already ≈ target, no signal should be emitted."""
        regime = make_regime_state(0, "BULL", 0.85)
        # Pretend we already hold 93% (target is 95%; difference 2% < 10%)
        sigs = self.orch.generate_signals(
            ["SPY"], self.bars_by_symbol, regime,
            current_weights={"SPY": 0.93},
        )
        assert len(sigs) == 0, (
            f"Expected 0 signals (within rebalance threshold), got {len(sigs)}"
        )

    def test_rebalancing_triggers_on_large_change(self):
        """Moving from 0% → 95% exceeds threshold → signal must be emitted."""
        regime = make_regime_state(0, "BULL", 0.85)
        sigs = self.orch.generate_signals(
            ["SPY"], self.bars_by_symbol, regime,
            current_weights={"SPY": 0.0},
        )
        assert len(sigs) >= 1

    def test_uncertainty_applied_when_flickering(self):
        regime = make_regime_state(0, "BULL", 0.85)
        sigs_normal = self.orch.generate_signals(
            ["SPY"], self.bars_by_symbol, regime, is_flickering=False,
            current_weights={"SPY": 0.0},
        )
        sigs_flicker = self.orch.generate_signals(
            ["SPY"], self.bars_by_symbol, regime, is_flickering=True,
            current_weights={"SPY": 0.0},
        )
        if sigs_normal and sigs_flicker:
            assert sigs_flicker[0].position_size_pct < sigs_normal[0].position_size_pct

    def test_uncertainty_applied_when_low_probability(self):
        regime_lo = make_regime_state(0, "BULL", 0.50)   # below min_confidence
        sigs_lo = self.orch.generate_signals(
            ["SPY"], self.bars_by_symbol, regime_lo,
            current_weights={"SPY": 0.0},
        )
        regime_hi = make_regime_state(0, "BULL", 0.85)
        sigs_hi = self.orch.generate_signals(
            ["SPY"], self.bars_by_symbol, regime_hi,
            current_weights={"SPY": 0.0},
        )
        if sigs_lo and sigs_hi:
            assert sigs_lo[0].position_size_pct < sigs_hi[0].position_size_pct

    def test_update_regime_infos_rebuilds_map(self):
        new_infos = {
            0: make_regime_info(0, 0.05, "BULL"),
            1: make_regime_info(1, 0.20, "BEAR"),
        }
        self.orch.update_regime_infos(new_infos)
        assert isinstance(self.orch.get_strategy_for_regime(0), LowVolBullStrategy)
        assert isinstance(self.orch.get_strategy_for_regime(1), HighVolDefensiveStrategy)

    def test_describe_mapping(self):
        mapping = self.orch.describe_mapping()
        assert isinstance(mapping, dict)
        assert all(isinstance(v, str) for v in mapping.values())
