"""
test_integration.py — End-to-end pipeline integration tests.

Tests the full data → features → HMM → strategy → risk → orders chain
using synthetic bars and mocked Alpaca. No real network calls.

Test (a) from Phase 9 spec: end-to-end dry run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import Signal, StrategyOrchestrator
from core.risk_manager import (
    RiskConfig, RiskManager, RiskAction, PortfolioSnapshot,
)
from core.signal_generator import SignalGenerator, ApprovedSignal
from data.feature_engineering import FeatureEngineer


# ------------------------------------------------------------------ #
# Shared fixtures                                                      #
# ------------------------------------------------------------------ #

def make_bars(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with two-regime vol pattern."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    cycle = (np.arange(n) // 60) % 2
    vol   = np.where(cycle == 0, 0.007, 0.022)
    drift = np.where(cycle == 0, 0.0004, -0.0003)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol)))
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.006,
        "low":    close * 0.994,
        "close":  close,
        "volume": rng.integers(500_000, 3_000_000, size=n),
    }, index=dates)


_FAST_CONFIG = HMMConfig(
    n_candidates=[2, 3],
    n_init=3,
    min_train_bars=200,
    stability_bars=2,
    flicker_window=10,
    flicker_threshold=3,
    min_confidence=0.55,
)

_STRATEGY_CFG = {
    "low_vol_allocation": 0.95,
    "mid_vol_allocation_trend": 0.95,
    "mid_vol_allocation_no_trend": 0.60,
    "high_vol_allocation": 0.60,
    "low_vol_leverage": 1.25,
    "rebalance_threshold": 0.10,
    "uncertainty_size_mult": 0.50,
    "min_confidence": 0.55,
}

_CLEAN_PORTFOLIO = PortfolioSnapshot(
    equity=100_000.0,
    cash=100_000.0,
    positions={},
    peak_equity=100_000.0,
    daily_start_equity=100_000.0,
    weekly_start_equity=100_000.0,
    open_trade_count_today=0,
    gross_exposure=0.0,
)


# ------------------------------------------------------------------ #
# Module-scoped fitted engine (trained once for the whole module)     #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def fitted_engine_and_features():
    bars = make_bars(800)
    features = FeatureEngineer.build_hmm_features(bars)
    engine = HMMEngine(_FAST_CONFIG)
    engine.fit(features.iloc[:400])
    return engine, features, bars


# ------------------------------------------------------------------ #
# a. End-to-end dry run                                               #
# ------------------------------------------------------------------ #

class TestEndToEndDryRun:
    """Full pipeline: bars → features → HMM → orchestrator → risk → orders."""

    def test_pipeline_produces_signals(self, fitted_engine_and_features):
        engine, features, bars = fitted_engine_and_features
        enriched = FeatureEngineer.enrich_bars(bars.copy())

        # Predict regime on full feature history
        regime = engine.predict(features)
        assert regime is not None
        assert regime.label != ""
        assert 0.0 < regime.probability <= 1.0

    def test_all_signals_are_long_not_short(self, fitted_engine_and_features):
        engine, features, bars = fitted_engine_and_features
        enriched = FeatureEngineer.enrich_bars(bars.copy())
        regime = engine.predict(features)

        orchestrator = StrategyOrchestrator(_STRATEGY_CFG, engine.get_regime_infos())
        signals = orchestrator.generate_signals(
            ["SPY"], {"SPY": enriched}, regime,
            is_flickering=False, current_weights={"SPY": 0.0},
        )
        for sig in signals:
            assert sig.direction in ("LONG", "FLAT"), (
                f"Strategy produced {sig.direction} — should never be SHORT"
            )

    def test_signal_position_size_in_valid_range(self, fitted_engine_and_features):
        engine, features, bars = fitted_engine_and_features
        enriched = FeatureEngineer.enrich_bars(bars.copy())
        regime = engine.predict(features)

        orchestrator = StrategyOrchestrator(_STRATEGY_CFG, engine.get_regime_infos())
        signals = orchestrator.generate_signals(
            ["SPY"], {"SPY": enriched}, regime,
            is_flickering=False, current_weights={"SPY": 0.0},
        )
        for sig in signals:
            assert 0.0 <= sig.position_size_pct <= 1.0, (
                f"position_size_pct={sig.position_size_pct} out of [0,1]"
            )
            assert sig.leverage >= 1.0

    def test_risk_gate_approves_clean_signals(self, fitted_engine_and_features):
        engine, features, bars = fitted_engine_and_features
        enriched = FeatureEngineer.enrich_bars(bars.copy())
        regime = engine.predict(features)

        orchestrator = StrategyOrchestrator(_STRATEGY_CFG, engine.get_regime_infos())
        signals = orchestrator.generate_signals(
            ["SPY"], {"SPY": enriched}, regime,
            is_flickering=False, current_weights={"SPY": 0.0},
        )
        rm = RiskManager(RiskConfig())
        portfolio = _CLEAN_PORTFOLIO

        for sig in signals:
            decision = rm.validate(
                sig, portfolio,
                stop_loss_price=sig.stop_loss,
                current_price=sig.entry_price,
            )
            # With a clean portfolio (no drawdown, no positions), should not BLOCK
            assert decision.action != RiskAction.BLOCK, (
                f"Clean portfolio signal was blocked: {decision.reason}"
            )
            assert 0.0 <= decision.approved_weight <= 1.0

    def test_signal_generator_full_integration(self, fitted_engine_and_features):
        engine, features, bars = fitted_engine_and_features
        enriched = FeatureEngineer.enrich_bars(bars.copy())

        orchestrator = StrategyOrchestrator(_STRATEGY_CFG, engine.get_regime_infos())
        rm = RiskManager(RiskConfig())

        sg = SignalGenerator(engine, orchestrator, rm)
        approved = sg.process_bar(
            symbols=["SPY"],
            bars_by_symbol={"SPY": enriched},
            features=features,
            portfolio=_CLEAN_PORTFOLIO,
            current_weights={"SPY": 0.0},
        )

        assert isinstance(approved, list)
        for ap in approved:
            assert isinstance(ap, ApprovedSignal)
            assert ap.signal.symbol == "SPY"
            assert ap.decision.approved_weight >= 0.0

    def test_approved_signals_carry_regime_state(self, fitted_engine_and_features):
        engine, features, bars = fitted_engine_and_features
        enriched = FeatureEngineer.enrich_bars(bars.copy())

        orchestrator = StrategyOrchestrator(_STRATEGY_CFG, engine.get_regime_infos())
        rm = RiskManager(RiskConfig())
        sg = SignalGenerator(engine, orchestrator, rm)

        approved = sg.process_bar(
            ["SPY"], {"SPY": enriched}, features, _CLEAN_PORTFOLIO,
        )
        regime = sg.get_current_regime()
        assert regime is not None

        for ap in approved:
            # regime_state on each approved signal should match current regime
            assert ap.regime_state.label == regime.label

    def test_empty_features_returns_no_signals(self, fitted_engine_and_features):
        engine, features, bars = fitted_engine_and_features
        orchestrator = StrategyOrchestrator(_STRATEGY_CFG, engine.get_regime_infos())
        rm = RiskManager(RiskConfig())
        sg = SignalGenerator(engine, orchestrator, rm)

        result = sg.process_bar(
            ["SPY"], {}, pd.DataFrame(), _CLEAN_PORTFOLIO,
        )
        assert result == []


# ------------------------------------------------------------------ #
# b. Look-ahead bias: backtest identical with different end dates      #
# ------------------------------------------------------------------ #

class TestBacktestLookAhead:
    """
    Run two backtests on the same symbol with different end dates.
    The equity curve for the shared OOS period must be numerically
    identical — adding future bars must not change past results.
    """

    @pytest.fixture(scope="class")
    def short_and_long_results(self):
        from backtest.backtester import BacktestConfig, WalkForwardBacktester

        bars = make_bars(800)
        bars_short = {"SPY": bars.iloc[:700].copy()}
        bars_long  = {"SPY": bars.copy()}           # 100 extra bars at the end

        bt_cfg = BacktestConfig(
            train_window=200,
            test_window=100,
            step_size=100,
            initial_capital=100_000,
        )
        bt = WalkForwardBacktester(_FAST_CONFIG, bt_cfg, _STRATEGY_CFG)

        result_short = bt.run(bars_short)
        result_long  = bt.run(bars_long)
        return result_short, result_long

    def test_shared_equity_is_identical(self, short_and_long_results):
        res_short, res_long = short_and_long_results
        if res_short.full_equity.empty or res_long.full_equity.empty:
            pytest.skip("Backtest produced no equity — check data length.")

        # Dates in short result must all appear in long result
        shared_dates = res_short.full_equity.index.intersection(res_long.full_equity.index)
        assert len(shared_dates) > 0, "No shared OOS dates between short and long runs."

        eq_short = res_short.full_equity.loc[shared_dates]
        eq_long  = res_long.full_equity.loc[shared_dates]

        np.testing.assert_allclose(
            eq_short.values, eq_long.values,
            rtol=1e-9,
            err_msg="Equity curve changed for shared period when end date extended — look-ahead bias!",
        )

    def test_regime_history_identical_for_shared_period(self, short_and_long_results):
        res_short, res_long = short_and_long_results
        if res_short.full_regime.empty or res_long.full_regime.empty:
            pytest.skip("No regime history.")

        shared = res_short.full_regime.index.intersection(res_long.full_regime.index)
        if len(shared) == 0:
            pytest.skip("No shared regime dates.")

        assert list(res_short.full_regime.loc[shared]) == list(res_long.full_regime.loc[shared]), (
            "Regime labels differ for the shared period — possible look-ahead in regime detection."
        )


# ------------------------------------------------------------------ #
# Ordered execution / fill-delay invariant                            #
# ------------------------------------------------------------------ #

class TestBacktestFillDelay:
    """Verify signal→fill delay: signals from bar t execute at bar t+1."""

    def test_rebalance_flag_is_one_bar_behind_signal(self):
        from backtest.backtester import BacktestConfig, WalkForwardBacktester

        bars = make_bars(700)
        bt_cfg = BacktestConfig(
            train_window=200, test_window=100, step_size=100,
            initial_capital=100_000,
        )
        bt = WalkForwardBacktester(_FAST_CONFIG, bt_cfg, _STRATEGY_CFG)
        result = bt.run({"SPY": bars})

        if not result.folds:
            pytest.skip("No folds produced.")

        fold = result.folds[0]
        recs = fold.bar_records
        if not recs:
            pytest.skip("Empty bar records.")

        # At bar 0 there is nothing pending — rebalanced must be False
        assert recs[0].rebalanced is False, (
            "First OOS bar should not be rebalanced (no pending signal yet)."
        )
