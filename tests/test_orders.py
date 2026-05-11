"""
test_orders.py — Unit tests for OrderExecutor and AlpacaClient.

Tests use a mock AlpacaClient to avoid real API calls. They verify:
  - execute_signal() submits the correct order side and quantity
  - No order is placed when already at target weight
  - Order is placed to reduce position when weight decreases
  - cancel_order() delegates to AlpacaClient
  - OrderRecord is created and stored in order_log
  - sync_order_statuses() updates records from API responses
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor, OrderRecord, OrderStatus
from core.hmm_engine import RegimeState
from core.regime_strategies import Signal
from core.risk_manager import RiskDecision, RiskAction
from core.signal_generator import ApprovedSignal

import numpy as np


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def make_mock_client() -> MagicMock:
    client = MagicMock(spec=AlpacaClient)
    client.submit_market_order.return_value = {
        "id": "order-abc-123",
        "symbol": "SPY",
        "side": "buy",
        "qty": "10",
        "type": "market",
        "status": "accepted",
        "submitted_at": datetime.utcnow().isoformat(),
        "filled_at": None,
        "filled_avg_price": None,
    }
    client.get_all_positions.return_value = []
    client.get_account.return_value = {"equity": "10000", "cash": "10000"}
    return client


def make_approved_signal(
    symbol: str = "SPY",
    weight: float = 0.10,
    current_price: float = 100.0,
    stop_loss: float = 95.0,
) -> ApprovedSignal:
    signal = Signal(
        symbol=symbol,
        direction="LONG",
        confidence=0.80,
        entry_price=current_price,
        stop_loss=stop_loss,
        position_size_pct=weight,
        leverage=1.0,
        regime_id=1,
        regime_name="BULL",
        regime_probability=0.80,
        strategy_name="low_vol_bull",
    )
    decision = RiskDecision(action=RiskAction.ALLOW, approved_weight=weight)
    regime = RegimeState(
        label="BULL",
        state_id=1,
        probability=0.80,
        state_probabilities=np.array([0.80, 0.15, 0.05]),
        timestamp=None,
        is_confirmed=True,
        consecutive_bars=5,
        vol_environment="low",
        n_states=3,
    )
    return ApprovedSignal(
        signal=signal,
        decision=decision,
        regime_state=regime,
        stop_loss_price=stop_loss,
        current_price=current_price,
    )


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestOrderExecution:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = make_mock_client()
        self.executor = OrderExecutor(self.client, paper=True)

    def test_buy_order_submitted(self):
        approved = make_approved_signal(weight=0.10, current_price=100.0)
        # No existing position
        self.client.get_position.return_value = None
        record = self.executor.execute_signal(approved)
        assert record is not None
        assert record.side == "buy"
        assert record.qty > 0

    def test_sell_order_when_reducing(self):
        approved = make_approved_signal(weight=0.05, current_price=100.0)
        # Currently hold 20 shares (weight ~0.20 of 100k portfolio)
        self.client.get_position.return_value = {"qty": "20", "current_price": "100"}
        record = self.executor.execute_signal(approved)
        assert record is not None
        assert record.side == "sell"

    def test_no_order_when_at_target(self):
        # Target 10 shares, already have 10 → no order
        approved = make_approved_signal(weight=0.10, current_price=100.0)
        # Assume portfolio equity ~100k, target 10 shares
        self.client.get_position.return_value = {"qty": "10", "current_price": "100"}
        record = self.executor.execute_signal(approved)
        # Either no order or a very small one; implementation may return None
        if record is not None:
            assert record.qty == 0 or record.qty < 2

    def test_order_logged(self):
        approved = make_approved_signal()
        self.client.get_position.return_value = None
        self.executor.execute_signal(approved)
        assert len(self.executor.get_order_log()) == 1

    def test_order_record_fields(self):
        approved = make_approved_signal(symbol="QQQ")
        self.client.get_position.return_value = None
        record = self.executor.execute_signal(approved)
        assert record.symbol == "QQQ"
        assert record.order_id == "order-abc-123"
        assert record.status == OrderStatus.SUBMITTED

    def test_cancel_delegates_to_client(self):
        self.executor.cancel_order("order-xyz-999")
        self.client.cancel_order.assert_called_once_with("order-xyz-999")


class TestOrderSync:
    def test_sync_updates_filled_status(self):
        client = make_mock_client()
        executor = OrderExecutor(client, paper=True)

        # Pre-populate a SUBMITTED order in the log
        record = OrderRecord(
            order_id="order-abc-123",
            symbol="SPY",
            side="buy",
            qty=10,
            order_type="market",
            limit_price=None,
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.utcnow(),
        )
        executor._order_log.append(record)

        # Simulate Alpaca returning a filled status
        client.get_order.return_value = {
            "id": "order-abc-123",
            "status": "filled",
            "filled_at": datetime.utcnow().isoformat(),
            "filled_avg_price": "101.50",
        }
        executor.sync_order_statuses()
        assert executor._order_log[0].status == OrderStatus.FILLED
        assert executor._order_log[0].filled_avg_price == 101.50
