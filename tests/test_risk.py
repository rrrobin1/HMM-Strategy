"""
test_risk.py — Unit tests for RiskManager.

Tests cover:
  - Signal allowed when all limits are clear
  - Signal blocked when system is halted
  - Signal blocked when daily trade limit reached
  - Signal reduced when daily drawdown hits reduce threshold
  - Signal blocked when daily drawdown hits halt threshold
  - Signal blocked when weekly drawdown hits halt threshold
  - Signal blocked when peak drawdown exceeds max_dd_from_peak
  - Signal blocked when gross exposure exceeds max_exposure
  - Signal blocked when single position would exceed max_single_position
  - Signal blocked when concurrent positions exceed max_concurrent
"""

from __future__ import annotations

import pytest

from core.regime_strategies import Signal
from core.risk_manager import (
    PortfolioSnapshot,
    RiskAction,
    RiskConfig,
    RiskManager,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def make_config(**overrides) -> RiskConfig:
    defaults = dict(
        max_risk_per_trade=0.01,
        max_exposure=0.80,
        max_leverage=1.25,
        max_single_position=0.15,
        max_concurrent=5,
        max_daily_trades=20,
        daily_dd_reduce=0.02,
        daily_dd_halt=0.03,
        weekly_dd_reduce=0.05,
        weekly_dd_halt=0.07,
        max_dd_from_peak=0.10,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def make_portfolio(
    equity: float = 100_000,
    positions: dict | None = None,
    daily_start: float | None = None,
    weekly_start: float | None = None,
    peak: float | None = None,
    trade_count: int = 0,
    gross_exposure: float = 0.0,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        equity=equity,
        cash=equity - gross_exposure,
        positions=positions or {},
        peak_equity=peak or equity,
        daily_start_equity=daily_start or equity,
        weekly_start_equity=weekly_start or equity,
        open_trade_count_today=trade_count,
        gross_exposure=gross_exposure,
    )


def make_signal(symbol: str = "SPY", weight: float = 0.10) -> Signal:
    entry = 100.0
    return Signal(
        symbol=symbol,
        direction="LONG",
        confidence=0.80,
        entry_price=entry,
        stop_loss=entry * 0.98,   # 2% stop → stop_loss_pct ≈ 0.02
        position_size_pct=weight,
        leverage=1.0,
        regime_id=1,
        regime_name="BULL",
        regime_probability=0.80,
        strategy_name="low_vol_bull",
    )


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestRiskManagerAllow:
    def test_clean_signal_is_allowed(self):
        rm = RiskManager(make_config())
        portfolio = make_portfolio()
        decision = rm.validate(make_signal(), portfolio, stop_loss_price=95.0, current_price=100.0)
        assert decision.action == RiskAction.ALLOW

    def test_approved_weight_near_target(self):
        rm = RiskManager(make_config())
        portfolio = make_portfolio()
        signal = make_signal(weight=0.10)
        decision = rm.validate(signal, portfolio, stop_loss_price=98.0, current_price=100.0)
        assert decision.approved_weight <= signal.target_weight + 1e-6


class TestCircuitBreakers:
    def test_halted_system_blocks_all(self):
        rm = RiskManager(make_config())
        rm._halted = True
        rm._halt_reason = "manual halt"
        decision = rm.validate(make_signal(), make_portfolio(), 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK

    def test_daily_trade_limit_blocks(self):
        rm = RiskManager(make_config(max_daily_trades=5))
        portfolio = make_portfolio(trade_count=5)
        decision = rm.validate(make_signal(), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK

    def test_daily_dd_reduce(self):
        rm = RiskManager(make_config(daily_dd_reduce=0.02))
        # equity dropped 2.5% today → should REDUCE
        portfolio = make_portfolio(equity=97_500, daily_start=100_000)
        decision = rm.validate(make_signal(weight=0.15), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.REDUCE
        assert decision.approved_weight < 0.15

    def test_daily_dd_halt(self):
        rm = RiskManager(make_config(daily_dd_halt=0.03))
        portfolio = make_portfolio(equity=96_500, daily_start=100_000)
        decision = rm.validate(make_signal(), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK

    def test_weekly_dd_halt(self):
        rm = RiskManager(make_config(weekly_dd_halt=0.07))
        portfolio = make_portfolio(equity=92_500, weekly_start=100_000)
        decision = rm.validate(make_signal(), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK

    def test_peak_drawdown_halt(self):
        rm = RiskManager(make_config(max_dd_from_peak=0.10))
        portfolio = make_portfolio(equity=88_000, peak=100_000)
        rm.update_snapshot(portfolio)
        decision = rm.validate(make_signal(), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK


class TestExposureLimits:
    def test_gross_exposure_limit(self):
        rm = RiskManager(make_config(max_exposure=0.80))
        # Already at 79% exposure; new signal would push to 89%
        portfolio = make_portfolio(equity=100_000, gross_exposure=79_000)
        decision = rm.validate(make_signal(weight=0.10), portfolio, 95.0, 100.0)
        # Should reduce or block
        assert decision.action in (RiskAction.REDUCE, RiskAction.BLOCK)

    def test_single_position_limit(self):
        rm = RiskManager(make_config(max_single_position=0.15))
        signal = make_signal(weight=0.20)   # exceeds limit
        decision = rm.validate(signal, make_portfolio(), 95.0, 100.0)
        assert decision.approved_weight <= 0.15 + 1e-6

    def test_concurrent_position_limit(self):
        rm = RiskManager(make_config(max_concurrent=3))
        positions = {"AAPL": 10_000, "MSFT": 10_000, "AMZN": 10_000}
        portfolio = make_portfolio(positions=positions, gross_exposure=30_000)
        decision = rm.validate(make_signal("NVDA"), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK
