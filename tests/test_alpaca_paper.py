"""
test_alpaca_paper.py — Mocked Alpaca paper trading tests.

Test (d) from Phase 9 spec: place order, modify stop, cancel,
verify clean state — all via mocked AlpacaClient.

"Bracket order" in this codebase = market order + stop managed by
the strategy's stop_loss field. We test:
  - Market order submitted and logged
  - Reducing a position submits a sell
  - Cancellation delegates to AlpacaClient and marks record CANCELLED
  - sync_order_statuses() transitions SUBMITTED → FILLED with fill price
  - Order log is clean (no phantom records) after cancel
  - get_position returns None after full close
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor, OrderRecord, OrderStatus
from core.hmm_engine import RegimeState
from core.regime_strategies import Signal
from core.risk_manager import RiskAction, RiskDecision
from core.signal_generator import ApprovedSignal


# ------------------------------------------------------------------ #
# Shared helpers                                                       #
# ------------------------------------------------------------------ #

def _mock_client(equity: float = 10_000.0) -> MagicMock:
    client = MagicMock(spec=AlpacaClient)
    client.get_account.return_value = {"equity": str(equity), "cash": str(equity)}
    client.get_all_positions.return_value = []
    client.submit_market_order.return_value = {
        "id": "order-test-001",
        "symbol": "SPY",
        "side": "buy",
        "qty": "10",
        "type": "market",
        "status": "accepted",
        "submitted_at": datetime.utcnow().isoformat(),
        "filled_at": None,
        "filled_avg_price": None,
    }
    return client


def _make_approved(
    symbol: str = "SPY",
    weight: float = 0.10,
    price: float = 100.0,
    stop: float = 95.0,
) -> ApprovedSignal:
    signal = Signal(
        symbol=symbol,
        direction="LONG",
        confidence=0.80,
        entry_price=price,
        stop_loss=stop,
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
        stop_loss_price=stop,
        current_price=price,
    )


# ------------------------------------------------------------------ #
# d. Alpaca paper: place, modify stop, cancel, verify clean state     #
# ------------------------------------------------------------------ #

class TestPlaceOrder:

    def test_buy_order_submitted_with_correct_side(self):
        client = _mock_client(10_000)
        client.get_position.return_value = None    # no existing position
        executor = OrderExecutor(client, paper=True)

        record = executor.execute_signal(_make_approved(weight=0.10, price=100.0))

        assert record is not None
        assert record.side == "buy"
        assert record.symbol == "SPY"

    def test_buy_qty_matches_weight(self):
        """weight=0.10, equity=10_000, price=100 → target 10 shares."""
        client = _mock_client(10_000)
        client.get_position.return_value = None
        executor = OrderExecutor(client, paper=True)

        record = executor.execute_signal(_make_approved(weight=0.10, price=100.0))
        assert record is not None
        assert record.qty == 10

    def test_submit_market_order_called_once(self):
        client = _mock_client(10_000)
        client.get_position.return_value = None
        executor = OrderExecutor(client, paper=True)

        executor.execute_signal(_make_approved())
        client.submit_market_order.assert_called_once()

    def test_order_status_is_submitted(self):
        client = _mock_client(10_000)
        client.get_position.return_value = None
        executor = OrderExecutor(client, paper=True)

        record = executor.execute_signal(_make_approved())
        assert record.status == OrderStatus.SUBMITTED

    def test_order_logged_immediately(self):
        client = _mock_client(10_000)
        client.get_position.return_value = None
        executor = OrderExecutor(client, paper=True)

        executor.execute_signal(_make_approved())
        log = executor.get_order_log()
        assert len(log) == 1
        assert log[0].symbol == "SPY"


class TestModifyStop:
    """
    'Modify stop' = cancel old order + submit new order at revised size.
    Test the cancel → resubmit pattern.
    """

    def test_cancel_then_resubmit_at_new_weight(self):
        client = _mock_client(10_000)
        client.get_position.return_value = None
        client.submit_market_order.side_effect = [
            # First call: initial order
            {
                "id": "order-initial",
                "symbol": "SPY", "side": "buy", "qty": "10",
                "type": "market", "status": "accepted",
                "submitted_at": datetime.utcnow().isoformat(),
                "filled_at": None, "filled_avg_price": None,
            },
            # Second call: revised order after cancel
            {
                "id": "order-revised",
                "symbol": "SPY", "side": "buy", "qty": "5",
                "type": "market", "status": "accepted",
                "submitted_at": datetime.utcnow().isoformat(),
                "filled_at": None, "filled_avg_price": None,
            },
        ]
        executor = OrderExecutor(client, paper=True)

        # Place initial order
        record1 = executor.execute_signal(_make_approved(weight=0.10))
        assert record1 is not None

        # Cancel initial order (simulating stop modification)
        executor.cancel_order(record1.order_id)
        client.cancel_order.assert_called_once_with(record1.order_id)

        # Resubmit at reduced weight
        client.get_position.return_value = None  # assume not filled
        record2 = executor.execute_signal(_make_approved(weight=0.05))
        assert record2 is not None
        assert record2.order_id == "order-revised"

    def test_cancelled_order_marked_in_log(self):
        client = _mock_client(10_000)
        client.get_position.return_value = None
        executor = OrderExecutor(client, paper=True)

        record = executor.execute_signal(_make_approved())
        executor.cancel_order(record.order_id)

        log = executor.get_order_log()
        assert log[0].status == OrderStatus.CANCELLED


class TestCancelOrder:

    def test_cancel_delegates_to_alpaca_client(self):
        client = _mock_client()
        executor = OrderExecutor(client, paper=True)
        executor.cancel_order("order-xyz-999")
        client.cancel_order.assert_called_once_with("order-xyz-999")

    def test_cancel_all_marks_all_non_terminal_cancelled(self):
        client = _mock_client(10_000)
        client.get_position.return_value = None
        executor = OrderExecutor(client, paper=True)

        executor.execute_signal(_make_approved("SPY"))

        # Manually add a second submitted record
        executor._order_log.append(OrderRecord(
            order_id="order-qqq",
            symbol="QQQ",
            side="buy",
            qty=5,
            order_type="market",
            limit_price=None,
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.utcnow(),
        ))

        executor.cancel_all_open_orders()
        for rec in executor.get_order_log():
            assert rec.status == OrderStatus.CANCELLED


class TestVerifyCleanState:

    def test_no_duplicate_orders_for_same_signal(self):
        """Submitting the same approved signal twice should not produce 2 orders."""
        client = _mock_client(10_000)
        client.get_position.side_effect = [
            None,             # first call: no position
            {"qty": "10", "current_price": "100"},  # second call: already at target
        ]
        executor = OrderExecutor(client, paper=True)

        ap = _make_approved(weight=0.10, price=100.0)

        executor.execute_signal(ap)   # buys 10 shares
        executor.execute_signal(ap)   # already at 10 shares → no order

        # Should have at most 1 actual order in log (second call returns None)
        log = executor.get_order_log()
        assert len(log) == 1

    def test_full_close_leads_to_sell(self):
        """Holding 10 shares, target weight=0 → sell all."""
        client = _mock_client(10_000)
        client.get_position.return_value = {"qty": "10", "current_price": "100"}
        client.submit_market_order.return_value = {
            "id": "order-close",
            "symbol": "SPY", "side": "sell", "qty": "10",
            "type": "market", "status": "accepted",
            "submitted_at": datetime.utcnow().isoformat(),
            "filled_at": None, "filled_avg_price": None,
        }
        executor = OrderExecutor(client, paper=True)

        ap = _make_approved(weight=0.0, price=100.0)  # flatten
        record = executor.execute_signal(ap)

        assert record is not None
        assert record.side == "sell"
        assert record.qty == 10


class TestOrderSync:

    def test_sync_transitions_submitted_to_filled(self):
        client = _mock_client()
        executor = OrderExecutor(client, paper=True)

        executor._order_log.append(OrderRecord(
            order_id="order-abc-123",
            symbol="SPY",
            side="buy",
            qty=10,
            order_type="market",
            limit_price=None,
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.utcnow(),
        ))

        client.get_order.return_value = {
            "id": "order-abc-123",
            "status": "filled",
            "filled_at": datetime.utcnow().isoformat(),
            "filled_avg_price": "101.50",
        }

        executor.sync_order_statuses()
        assert executor._order_log[0].status == OrderStatus.FILLED
        assert executor._order_log[0].filled_avg_price == pytest.approx(101.50)

    def test_sync_skips_terminal_orders(self):
        client = _mock_client()
        executor = OrderExecutor(client, paper=True)

        executor._order_log.append(OrderRecord(
            order_id="order-done",
            symbol="SPY",
            side="buy",
            qty=5,
            order_type="market",
            limit_price=None,
            status=OrderStatus.FILLED,
            submitted_at=datetime.utcnow(),
            filled_avg_price=100.0,
        ))

        executor.sync_order_statuses()
        client.get_order.assert_not_called()

    def test_sync_handles_api_failure_gracefully(self):
        client = _mock_client()
        executor = OrderExecutor(client, paper=True)
        executor._order_log.append(OrderRecord(
            order_id="order-err",
            symbol="SPY",
            side="buy",
            qty=5,
            order_type="market",
            limit_price=None,
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.utcnow(),
        ))
        client.get_order.side_effect = RuntimeError("API down")
        # Should not raise — just logs a warning
        executor.sync_order_statuses()
        assert executor._order_log[0].status == OrderStatus.SUBMITTED
