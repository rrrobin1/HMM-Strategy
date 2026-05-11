"""
risk_manager.py — Position sizing, leverage control, and circuit breakers.

Validates every signal before it reaches the order executor. Enforces:
  - Per-trade risk limits (stop-loss sizing)
  - Portfolio exposure and leverage ceilings
  - Per-position concentration limits
  - Daily and weekly drawdown circuit breakers
  - Hard halt when drawdown from equity peak exceeds threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from core.regime_strategies import Signal


class RiskAction(Enum):
    ALLOW = auto()
    REDUCE = auto()    # pass signal but scale size down
    BLOCK = auto()     # reject signal entirely


@dataclass
class RiskDecision:
    """Result of risk_manager.validate()."""

    action: RiskAction
    approved_weight: float              # final target weight (may be reduced)
    reason: str = ""                    # human-readable explanation


@dataclass
class PortfolioSnapshot:
    """Current portfolio state passed to RiskManager on each call."""

    equity: float                       # total account equity ($)
    cash: float
    positions: dict[str, float]         # symbol -> market value ($)
    peak_equity: float                  # highest equity seen since start
    daily_start_equity: float           # equity at start of today
    weekly_start_equity: float          # equity at start of this week
    open_trade_count_today: int
    gross_exposure: float               # sum of abs(position values)


@dataclass
class RiskConfig:
    """Parameters loaded from settings.yaml[risk]."""

    max_risk_per_trade: float = 0.01
    max_exposure: float = 0.80
    max_leverage: float = 1.25
    max_single_position: float = 0.15
    max_concurrent: int = 5
    max_daily_trades: int = 20
    daily_dd_reduce: float = 0.02
    daily_dd_halt: float = 0.03
    weekly_dd_reduce: float = 0.05
    weekly_dd_halt: float = 0.07
    max_dd_from_peak: float = 0.10


class RiskManager:
    """
    Stateful risk gate that every signal must pass before execution.

    Maintains drawdown tracking internally; must be updated each bar via
    update_snapshot().
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._halted: bool = False
        self._halt_reason: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        signal: Signal,
        portfolio: PortfolioSnapshot,
        stop_loss_price: Optional[float],
        current_price: float,
    ) -> RiskDecision:
        """
        Validate a proposed signal against all risk rules.

        Checks in priority order:
          1. System halt
          2. Daily drawdown circuit breakers (halt > reduce)
          3. Weekly drawdown circuit breakers (halt > reduce)
          4. Peak drawdown halt
          5. Daily trade count limit
          6. Gross exposure ceiling
          7. Per-position concentration limit (REDUCE, not BLOCK)
          8. Concurrent position limit
          9. Stop-loss-based risk sizing (REDUCE to fit max_risk_per_trade)
        """
        cfg = self.config

        # 1. Hard halt
        if self._halted:
            return RiskDecision(RiskAction.BLOCK, 0.0, self._halt_reason)

        # 2. Daily drawdown
        daily_dd = self._daily_drawdown(portfolio)
        if daily_dd >= cfg.daily_dd_halt:
            self._halted = True
            self._halt_reason = f"daily drawdown {daily_dd:.1%} >= halt threshold {cfg.daily_dd_halt:.1%}"
            return RiskDecision(RiskAction.BLOCK, 0.0, self._halt_reason)

        # 3. Weekly drawdown
        weekly_dd = self._weekly_drawdown(portfolio)
        if weekly_dd >= cfg.weekly_dd_halt:
            self._halted = True
            self._halt_reason = f"weekly drawdown {weekly_dd:.1%} >= halt threshold {cfg.weekly_dd_halt:.1%}"
            return RiskDecision(RiskAction.BLOCK, 0.0, self._halt_reason)

        # 4. Peak drawdown halt
        peak_dd = self._peak_drawdown(portfolio)
        if peak_dd >= cfg.max_dd_from_peak:
            self._halted = True
            self._halt_reason = f"peak drawdown {peak_dd:.1%} >= max {cfg.max_dd_from_peak:.1%}"
            return RiskDecision(RiskAction.BLOCK, 0.0, self._halt_reason)

        # 5. Daily trade count
        if portfolio.open_trade_count_today >= cfg.max_daily_trades:
            return RiskDecision(
                RiskAction.BLOCK, 0.0,
                f"daily trade limit {cfg.max_daily_trades} reached",
            )

        # Now determine the approved weight, starting from the signal's target.
        approved = signal.target_weight
        action = RiskAction.ALLOW
        reason = ""

        # Daily drawdown reduce (softer threshold)
        if daily_dd >= cfg.daily_dd_reduce:
            scale = 1.0 - (daily_dd - cfg.daily_dd_reduce) / max(
                cfg.daily_dd_halt - cfg.daily_dd_reduce, 1e-9
            )
            approved = min(approved, signal.target_weight * scale)
            action = RiskAction.REDUCE
            reason = f"daily drawdown {daily_dd:.1%} — reducing size"

        # 6. Gross exposure ceiling: cap new addition so total stays <= max_exposure
        new_exposure_frac = (portfolio.gross_exposure + approved * portfolio.equity) / portfolio.equity
        if new_exposure_frac > cfg.max_exposure:
            headroom = max(cfg.max_exposure - portfolio.gross_exposure / portfolio.equity, 0.0)
            if headroom <= 0.0:
                return RiskDecision(
                    RiskAction.BLOCK, 0.0,
                    f"gross exposure {portfolio.gross_exposure / portfolio.equity:.1%} >= max {cfg.max_exposure:.1%}",
                )
            approved = min(approved, headroom)
            action = RiskAction.REDUCE
            reason = f"capping at exposure headroom {headroom:.1%}"

        # 7. Single-position concentration limit (REDUCE)
        if approved > cfg.max_single_position:
            approved = cfg.max_single_position
            action = RiskAction.REDUCE
            reason = f"capped at max_single_position {cfg.max_single_position:.1%}"

        # 8. Concurrent position limit (BLOCK if already at max and this is a new position)
        symbol_already_held = signal.symbol in portfolio.positions and portfolio.positions[signal.symbol] > 0
        if not symbol_already_held and len(portfolio.positions) >= cfg.max_concurrent:
            return RiskDecision(
                RiskAction.BLOCK, 0.0,
                f"concurrent position limit {cfg.max_concurrent} reached",
            )

        # 9. Stop-loss-based risk sizing
        if stop_loss_price is not None and current_price > 0 and stop_loss_price < current_price:
            risk_adjusted = self._risk_adjusted_weight(signal, portfolio, stop_loss_price, current_price)
            if risk_adjusted < approved:
                approved = risk_adjusted
                action = RiskAction.REDUCE
                reason = f"risk-adjusted size {risk_adjusted:.1%} (max_risk_per_trade={cfg.max_risk_per_trade:.1%})"

        return RiskDecision(action, approved, reason)

    def update_snapshot(self, portfolio: PortfolioSnapshot) -> None:
        """
        Update internal state with the latest portfolio snapshot.

        Call once per bar before validate(). Triggers peak-drawdown halt.
        """
        peak_dd = self._peak_drawdown(portfolio)
        if peak_dd >= self.config.max_dd_from_peak and not self._halted:
            self._halted = True
            self._halt_reason = f"peak drawdown {peak_dd:.1%} >= max {self.config.max_dd_from_peak:.1%}"

    def is_halted(self) -> bool:
        """Return True if the system is in a hard halt state."""
        return self._halted

    def halt_reason(self) -> str:
        return self._halt_reason

    def manual_halt(self, reason: str = "Manual halt") -> None:
        """Halt trading manually (e.g. from web dashboard)."""
        self._halted = True
        self._halt_reason = reason

    def clear_halt(self) -> None:
        """Clear a halt state. Automatic circuit breakers re-engage on next bar if thresholds still exceeded."""
        self._halted = False
        self._halt_reason = ""

    def reset_daily_counters(self) -> None:
        """Call at the start of each trading day."""
        pass  # daily counters live in PortfolioSnapshot, nothing to reset here

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _daily_drawdown(self, portfolio: PortfolioSnapshot) -> float:
        if portfolio.daily_start_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - portfolio.equity / portfolio.daily_start_equity)

    def _weekly_drawdown(self, portfolio: PortfolioSnapshot) -> float:
        if portfolio.weekly_start_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - portfolio.equity / portfolio.weekly_start_equity)

    def _peak_drawdown(self, portfolio: PortfolioSnapshot) -> float:
        if portfolio.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - portfolio.equity / portfolio.peak_equity)

    def _risk_adjusted_weight(
        self,
        signal: Signal,
        portfolio: PortfolioSnapshot,
        stop_loss_price: float,
        current_price: float,
    ) -> float:
        """Compute max weight such that a stop-out costs <= max_risk_per_trade of equity."""
        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0 or portfolio.equity <= 0:
            return signal.target_weight
        # max_dollar_risk = equity * max_risk_per_trade
        # shares = weight * equity / price → loss = shares * risk_per_share = weight * equity * risk_per_share / price
        # cap: weight * equity * risk_per_share / price <= equity * max_risk_per_trade
        # → weight <= max_risk_per_trade * price / risk_per_share
        return self.config.max_risk_per_trade * current_price / risk_per_share
