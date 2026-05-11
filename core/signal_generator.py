"""
signal_generator.py — Combines HMM regime with strategy signals and risk gate.

SignalGenerator is the single integration point that:
  1. Receives fresh bar data for all symbols
  2. Calls HMMEngine.predict() to get the current regime (forward algo)
  3. Calls StrategyOrchestrator.generate_signals() for all symbols
  4. Passes each signal through RiskManager.validate()
  5. Returns approved (or risk-reduced) signals ready for order execution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_strategies import Signal, StrategyOrchestrator
from core.risk_manager import RiskAction, RiskDecision, RiskManager, PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ApprovedSignal:
    """A signal that has cleared (or been reduced by) the risk gate."""

    signal: Signal
    decision: RiskDecision
    regime_state: RegimeState
    stop_loss_price: Optional[float]
    current_price: float


class SignalGenerator:
    """
    Orchestrates HMM + strategy + risk into a list of approved signals.

    Call process_bar() once per bar with updated market data.
    """

    def __init__(
        self,
        hmm_engine: HMMEngine,
        orchestrator: StrategyOrchestrator,
        risk_manager: RiskManager,
    ) -> None:
        self.hmm_engine = hmm_engine
        self.orchestrator = orchestrator
        self.risk_manager = risk_manager
        self._last_regime: Optional[RegimeState] = None

    def process_bar(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        features: pd.DataFrame,
        portfolio: PortfolioSnapshot,
        current_weights: dict[str, float] | None = None,
    ) -> list[ApprovedSignal]:
        """
        Process a single bar and return risk-approved signals.

        Parameters
        ----------
        symbols:
            Tickers to evaluate.
        bars_by_symbol:
            {symbol: OHLCV+indicator DataFrame} (causal window up to this bar).
        features:
            Full feature history fed to HMMEngine (output of build_hmm_features).
        portfolio:
            Current portfolio snapshot for risk checks.
        current_weights:
            {symbol: current_weight} for rebalancing threshold.

        Returns
        -------
        List of ApprovedSignal. BLOCK decisions are filtered out.
        ALLOW and REDUCE decisions are returned so the caller can submit.
        """
        if features.empty:
            logger.warning("process_bar: empty feature DataFrame — skipping.")
            return []

        # 1. Regime detection via forward algorithm
        regime_state = self.hmm_engine.predict(features)
        self._last_regime = regime_state

        logger.debug(
            "Regime: %s (p=%.2f, confirmed=%s, flickering=%s)",
            regime_state.label,
            regime_state.probability,
            regime_state.is_confirmed,
            self.hmm_engine.is_flickering(),
        )

        # 2. Strategy signals for all symbols
        signals = self.orchestrator.generate_signals(
            symbols,
            bars_by_symbol,
            regime_state,
            is_flickering=self.hmm_engine.is_flickering(),
            current_weights=current_weights or {},
        )

        if not signals:
            return []

        # 3. Risk validation
        approved: list[ApprovedSignal] = []
        for sig in signals:
            current_price = sig.entry_price
            stop_loss_price = sig.stop_loss if sig.stop_loss > 0 else None

            decision = self.risk_manager.validate(
                sig, portfolio, stop_loss_price, current_price
            )

            if decision.action == RiskAction.BLOCK:
                logger.info(
                    "Signal BLOCKED for %s (regime=%s): %s",
                    sig.symbol, regime_state.label, decision.reason,
                )
                continue

            if decision.action == RiskAction.REDUCE:
                logger.info(
                    "Signal REDUCED for %s: %.1f%% → %.1f%% (%s)",
                    sig.symbol,
                    sig.target_weight * 100,
                    decision.approved_weight * 100,
                    decision.reason,
                )

            approved.append(ApprovedSignal(
                signal=sig,
                decision=decision,
                regime_state=regime_state,
                stop_loss_price=stop_loss_price,
                current_price=current_price,
            ))

        return approved

    def get_current_regime(self) -> Optional[RegimeState]:
        """Return the most recently computed regime state, or None."""
        return self._last_regime
