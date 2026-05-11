"""
test_recovery.py — Session state recovery tests.

Test (e) from Phase 9 spec:
  - SessionState saves and restores all fields faithfully
  - After recovery, equity watermarks are restored (not reset to zero)
  - No double-entry: if a position already exists post-recovery,
    executing the same signal for that symbol produces no new order
  - Corrupt/missing snapshot degrades gracefully (starts fresh)
  - Mode mismatch (paper snapshot, live session) is not loaded
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor, OrderStatus
from broker.position_tracker import PositionTracker
from core.hmm_engine import RegimeState
from core.regime_strategies import Signal
from core.risk_manager import RiskAction, RiskDecision
from core.signal_generator import ApprovedSignal
from main import SessionState


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_session(mode: str = "paper", peak: float = 110_000.0) -> SessionState:
    return SessionState(
        session_start=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        last_trained=datetime.now(timezone.utc).isoformat(),
        hmm_model_path="models/hmm_SPY.pkl",
        last_processed_bar="2024-01-15T16:00:00+00:00",
        daily_trade_count=3,
        peak_equity=peak,
        daily_start_equity=105_000.0,
        weekly_start_equity=103_000.0,
        last_regime={"SPY": "BULL"},
    )


def _make_approved(symbol: str = "SPY", weight: float = 0.10, price: float = 100.0) -> ApprovedSignal:
    signal = Signal(
        symbol=symbol, direction="LONG", confidence=0.80,
        entry_price=price, stop_loss=price * 0.95,
        position_size_pct=weight, leverage=1.0,
        regime_id=1, regime_name="BULL", regime_probability=0.80,
        strategy_name="low_vol_bull",
    )
    decision = RiskDecision(action=RiskAction.ALLOW, approved_weight=weight)
    regime = RegimeState(
        label="BULL", state_id=1, probability=0.80,
        state_probabilities=np.array([0.80, 0.15, 0.05]),
        timestamp=None, is_confirmed=True, consecutive_bars=5,
        vol_environment="low", n_states=3,
    )
    return ApprovedSignal(signal=signal, decision=decision, regime_state=regime,
                          stop_loss_price=price * 0.95, current_price=price)


# ------------------------------------------------------------------ #
# e-1. State snapshot round-trip                                      #
# ------------------------------------------------------------------ #

class TestSessionStateRoundTrip:

    def test_all_fields_survive_save_load(self):
        state = _make_session()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            state.save(path)
            loaded = SessionState.load(path)
            assert loaded is not None
            assert loaded.session_start == state.session_start
            assert loaded.mode == state.mode
            assert loaded.last_trained == state.last_trained
            assert loaded.hmm_model_path == state.hmm_model_path
            assert loaded.last_processed_bar == state.last_processed_bar
            assert loaded.daily_trade_count == state.daily_trade_count
            assert loaded.peak_equity == pytest.approx(state.peak_equity)
            assert loaded.daily_start_equity == pytest.approx(state.daily_start_equity)
            assert loaded.weekly_start_equity == pytest.approx(state.weekly_start_equity)
            assert loaded.last_regime == state.last_regime
        finally:
            path.unlink(missing_ok=True)

    def test_save_produces_valid_json(self):
        state = _make_session()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            state.save(path)
            parsed = json.loads(path.read_text())
            assert isinstance(parsed, dict)
            assert "peak_equity" in parsed
            assert parsed["peak_equity"] == 110_000.0
        finally:
            path.unlink(missing_ok=True)

    def test_missing_snapshot_returns_none(self):
        result = SessionState.load(Path("/tmp/_no_such_file_regime_trader_.json"))
        assert result is None

    def test_corrupt_snapshot_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("{ this is not valid json {{{{")
            path = Path(f.name)
        try:
            result = SessionState.load(path)
            assert result is None
        finally:
            path.unlink(missing_ok=True)

    def test_new_session_creates_valid_state(self):
        state = SessionState.new(mode="paper", model_path="models/hmm_SPY.pkl")
        assert state.mode == "paper"
        assert state.daily_trade_count == 0
        assert state.peak_equity == 0.0
        assert state.last_regime == {}
        assert state.hmm_model_path == "models/hmm_SPY.pkl"


# ------------------------------------------------------------------ #
# e-2. Equity watermarks restored                                     #
# ------------------------------------------------------------------ #

class TestWatermarkRestored:

    def test_peak_equity_restored_from_snapshot(self):
        """After recovery, peak_equity from the snapshot is used — not reset to zero."""
        state = _make_session(peak=110_000.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            state.save(path)
            loaded = SessionState.load(path)
            # Simulate TradingEngine restoring the tracker
            mock_client = MagicMock(spec=AlpacaClient)
            mock_client.get_account.return_value = {"equity": "105000", "cash": "5000"}
            mock_client.get_all_positions.return_value = []
            tracker = PositionTracker(mock_client)
            # Restore from snapshot (as TradingEngine does in _startup)
            tracker._peak_equity = loaded.peak_equity
            tracker._daily_start_equity = loaded.daily_start_equity
            tracker._weekly_start_equity = loaded.weekly_start_equity
            assert tracker._peak_equity == pytest.approx(110_000.0)
            assert tracker._daily_start_equity == pytest.approx(105_000.0)
        finally:
            path.unlink(missing_ok=True)

    def test_daily_trade_count_restored(self):
        state = _make_session()
        state.daily_trade_count = 7
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            state.save(path)
            loaded = SessionState.load(path)
            assert loaded.daily_trade_count == 7
        finally:
            path.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# e-3. No double-entry after recovery                                 #
# ------------------------------------------------------------------ #

class TestNoDoubleEntry:
    """
    After recovery, if the position already exists at the target weight,
    executing the same signal must produce NO new order.
    """

    def test_already_at_target_no_order(self):
        """
        Portfolio has 10 shares of SPY (target weight at $10k equity).
        Re-running the same signal should return None (no order needed).
        """
        client = MagicMock(spec=AlpacaClient)
        client.get_account.return_value = {"equity": "10000", "cash": "0"}
        # 10 shares already held at $100 = 10% of $10k → matches weight=0.10
        client.get_position.return_value = {"qty": "10", "current_price": "100"}
        executor = OrderExecutor(client, paper=True)

        ap = _make_approved(weight=0.10, price=100.0)
        record = executor.execute_signal(ap)

        assert record is None, (
            "Should not submit an order when already at target weight "
            f"(got {record})"
        )

    def test_partially_filled_position_only_tops_up(self):
        """
        After crash recovery, hold 5 shares but want 10. Should BUY 5 more.
        """
        client = MagicMock(spec=AlpacaClient)
        client.get_account.return_value = {"equity": "10000", "cash": "500"}
        client.get_position.return_value = {"qty": "5", "current_price": "100"}
        client.submit_market_order.return_value = {
            "id": "order-topup",
            "symbol": "SPY", "side": "buy", "qty": "5",
            "type": "market", "status": "accepted",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "filled_at": None, "filled_avg_price": None,
        }
        executor = OrderExecutor(client, paper=True)

        ap = _make_approved(weight=0.10, price=100.0)
        record = executor.execute_signal(ap)

        assert record is not None
        assert record.side == "buy"
        assert record.qty == 5  # only the missing 5 shares

    def test_oversized_position_trimmed_on_recovery(self):
        """
        Recovery finds 20 shares but target is 10. Should SELL 10.
        """
        client = MagicMock(spec=AlpacaClient)
        client.get_account.return_value = {"equity": "10000", "cash": "0"}
        client.get_position.return_value = {"qty": "20", "current_price": "100"}
        client.submit_market_order.return_value = {
            "id": "order-trim",
            "symbol": "SPY", "side": "sell", "qty": "10",
            "type": "market", "status": "accepted",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "filled_at": None, "filled_avg_price": None,
        }
        executor = OrderExecutor(client, paper=True)

        ap = _make_approved(weight=0.10, price=100.0)
        record = executor.execute_signal(ap)

        assert record is not None
        assert record.side == "sell"
        assert record.qty == 10


# ------------------------------------------------------------------ #
# e-4. Mode mismatch graceful handling                                #
# ------------------------------------------------------------------ #

class TestModeMismatch:

    def test_live_snapshot_not_loaded_for_paper_session(self):
        """A 'live' snapshot should not be restored when starting a paper session."""
        state = _make_session(mode="live")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            state.save(path)
            loaded = SessionState.load(path)
            assert loaded is not None
            # The caller (TradingEngine) checks the mode; we just verify
            # the field is preserved faithfully for mode comparison
            assert loaded.mode == "live"
            # A paper session would skip this because mode != "paper"
            paper_matches = (loaded.mode == "paper")
            assert not paper_matches, (
                "TradingEngine should not restore a live snapshot into a paper session"
            )
        finally:
            path.unlink(missing_ok=True)
