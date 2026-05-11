"""
test_risk_stress.py — Risk manager stress tests.

Test (c) from Phase 9 spec:
  - Extreme signals are capped (not passed through at full size)
  - Rapid-fire signals are blocked when daily trade count exceeded
  - No-stop signals are accepted (stop_loss_price=None is valid)
  - Compound drawdown scenarios trigger the right action
  - System halt blocks everything including previously-allowed symbols
"""

from __future__ import annotations

import pytest

from core.regime_strategies import Signal
from core.risk_manager import (
    PortfolioSnapshot, RiskAction, RiskConfig, RiskDecision, RiskManager,
)


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

def make_signal(
    symbol: str = "SPY",
    weight: float = 0.10,
    entry: float = 100.0,
    stop: float = 95.0,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction="LONG",
        confidence=0.80,
        entry_price=entry,
        stop_loss=stop,
        position_size_pct=weight,
        leverage=1.0,
        regime_id=1,
        regime_name="BULL",
        regime_probability=0.80,
        strategy_name="low_vol_bull",
    )


def clean_portfolio(
    equity: float = 100_000.0,
    positions: dict | None = None,
    peak: float | None = None,
    daily_start: float | None = None,
    weekly_start: float | None = None,
    gross_exposure: float = 0.0,
    trade_count: int = 0,
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


# ------------------------------------------------------------------ #
# c-1. Extreme signals capped                                         #
# ------------------------------------------------------------------ #

class TestExtremeSignalsCapped:

    def test_200pct_weight_capped_at_max_single_position(self):
        rm = RiskManager(RiskConfig(max_single_position=0.15))
        sig = make_signal(weight=2.00)           # 200% — absurdly large
        decision = rm.validate(sig, clean_portfolio(), 95.0, 100.0)
        assert decision.approved_weight <= 0.15 + 1e-9
        assert decision.action in (RiskAction.REDUCE, RiskAction.BLOCK)

    def test_110pct_weight_capped_at_exposure_limit(self):
        rm = RiskManager(RiskConfig(max_exposure=0.80, max_single_position=1.0))
        # Portfolio already at 75% exposure; signal wants 40% more → push to 115%
        portfolio = clean_portfolio(equity=100_000, gross_exposure=75_000)
        sig = make_signal(weight=0.40)
        decision = rm.validate(sig, portfolio, 95.0, 100.0)
        assert decision.approved_weight <= 0.05 + 1e-9, (
            f"Should cap at 5% headroom, got {decision.approved_weight:.2%}"
        )

    def test_risk_adjusted_weight_limits_stop_distance(self):
        """Wide stop → tiny position (max_risk_per_trade=1% of equity)."""
        rm = RiskManager(RiskConfig(max_risk_per_trade=0.01))
        # entry=100, stop=50 → risk/share = $50, risk budget = $1000 → max 20 shares
        # weight = 20 * 100 / 100_000 = 2%
        sig = make_signal(weight=0.50, entry=100.0, stop=50.0)
        decision = rm.validate(sig, clean_portfolio(), 50.0, 100.0)
        expected_max = 0.01 * 100.0 / (100.0 - 50.0)   # = 0.02
        assert decision.approved_weight <= expected_max + 1e-9

    def test_tight_stop_allows_larger_weight(self):
        """Tight stop → position can be full-size (stop-loss sizing is not binding)."""
        rm = RiskManager(RiskConfig(max_risk_per_trade=0.01))
        # entry=100, stop=99 → risk/share = $1 → max 1000 shares → 100% weight
        sig = make_signal(weight=0.10, entry=100.0, stop=99.0)
        decision = rm.validate(sig, clean_portfolio(), 99.0, 100.0)
        # Stop-loss constraint: 0.01 * 100 / 1 = 1.0 → not binding
        # So approved_weight should stay at target 0.10
        assert decision.approved_weight >= 0.10 - 1e-9
        assert decision.action == RiskAction.ALLOW

    def test_zero_weight_signal_produces_allow_with_zero_weight(self):
        rm = RiskManager(RiskConfig())
        sig = make_signal(weight=0.0)
        decision = rm.validate(sig, clean_portfolio(), 95.0, 100.0)
        assert decision.approved_weight == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ #
# c-2. Rapid-fire blocked                                             #
# ------------------------------------------------------------------ #

class TestRapidFireBlocked:

    def test_daily_trade_limit_blocks_after_n_trades(self):
        rm = RiskManager(RiskConfig(max_daily_trades=5))
        portfolio = clean_portfolio(trade_count=5)
        decision = rm.validate(make_signal(), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK

    def test_at_limit_minus_one_still_allowed(self):
        rm = RiskManager(RiskConfig(max_daily_trades=5))
        portfolio = clean_portfolio(trade_count=4)
        decision = rm.validate(make_signal(), portfolio, 95.0, 100.0)
        assert decision.action != RiskAction.BLOCK

    def test_halted_system_blocks_all_symbols(self):
        rm = RiskManager(RiskConfig())
        rm._halted = True
        rm._halt_reason = "peak drawdown exceeded"
        for sym in ("SPY", "QQQ", "AAPL"):
            decision = rm.validate(make_signal(sym), clean_portfolio(), 95.0, 100.0)
            assert decision.action == RiskAction.BLOCK, f"{sym} should be blocked"

    def test_concurrent_limit_blocks_new_symbol(self):
        rm = RiskManager(RiskConfig(max_concurrent=2))
        existing = {"AAPL": 30_000.0, "MSFT": 30_000.0}
        portfolio = clean_portfolio(positions=existing, gross_exposure=60_000)
        decision = rm.validate(make_signal("NVDA"), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK

    def test_existing_symbol_not_blocked_by_concurrent_limit(self):
        """Adding to an existing position doesn't count as a new position."""
        rm = RiskManager(RiskConfig(max_concurrent=2))
        existing = {"AAPL": 30_000.0, "MSFT": 30_000.0}
        portfolio = clean_portfolio(positions=existing, gross_exposure=60_000)
        # AAPL already held — increasing the position should not hit concurrent limit
        decision = rm.validate(make_signal("AAPL"), portfolio, 95.0, 100.0)
        assert decision.action != RiskAction.BLOCK


# ------------------------------------------------------------------ #
# c-3. No-stop handling                                               #
# ------------------------------------------------------------------ #

class TestNoStopSignal:

    def test_none_stop_loss_is_accepted(self):
        """stop_loss_price=None means no hard stop — signal still processed."""
        rm = RiskManager(RiskConfig())
        sig = make_signal(weight=0.10)
        # Pass None as stop_loss_price — should not raise or block
        decision = rm.validate(sig, clean_portfolio(), stop_loss_price=None, current_price=100.0)
        assert decision.action in (RiskAction.ALLOW, RiskAction.REDUCE)

    def test_stop_above_entry_is_ignored_by_risk_sizing(self):
        """stop_loss_price >= current_price is nonsensical — skip risk sizing."""
        rm = RiskManager(RiskConfig(max_risk_per_trade=0.01))
        sig = make_signal(weight=0.10)
        # Stop above entry — risk_per_share <= 0, skip sizing check
        decision = rm.validate(sig, clean_portfolio(), stop_loss_price=110.0, current_price=100.0)
        assert decision.approved_weight >= 0.10 - 1e-9


# ------------------------------------------------------------------ #
# c-4. Compound drawdown scenarios                                    #
# ------------------------------------------------------------------ #

class TestDrawdownScenarios:

    def test_daily_and_weekly_both_reduce(self):
        """When daily DD is above reduce threshold but below halt, size is reduced."""
        rm = RiskManager(RiskConfig(daily_dd_reduce=0.02, weekly_dd_reduce=0.05))
        # equity=97_500 → daily_dd=2.5%, which is > reduce (2%) but < halt (3%)
        portfolio = clean_portfolio(
            equity=97_500,
            daily_start=100_000,
            weekly_start=100_000,
        )
        decision = rm.validate(make_signal(weight=0.15), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.REDUCE
        assert decision.approved_weight < 0.15

    def test_daily_halt_takes_priority_over_reduce(self):
        rm = RiskManager(RiskConfig(daily_dd_reduce=0.02, daily_dd_halt=0.03))
        # 3.5% daily drawdown → halt
        portfolio = clean_portfolio(equity=96_500, daily_start=100_000)
        decision = rm.validate(make_signal(), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK
        assert rm.is_halted()

    def test_update_snapshot_triggers_peak_halt(self):
        rm = RiskManager(RiskConfig(max_dd_from_peak=0.10))
        portfolio = clean_portfolio(equity=88_000, peak=100_000)
        rm.update_snapshot(portfolio)
        assert rm.is_halted()
        decision = rm.validate(make_signal(), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK

    def test_gross_exposure_fully_deployed_blocks_new(self):
        rm = RiskManager(RiskConfig(max_exposure=0.80))
        portfolio = clean_portfolio(equity=100_000, gross_exposure=82_000)
        decision = rm.validate(make_signal(weight=0.10), portfolio, 95.0, 100.0)
        assert decision.action == RiskAction.BLOCK
