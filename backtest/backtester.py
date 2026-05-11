"""
backtester.py — Walk-forward allocation backtester.

DESIGN: This is an ALLOCATION-BASED backtester. It does not track individual
trade entries/exits. It sets a target portfolio allocation per bar based on
the detected volatility regime and rebalances when the allocation shifts by
more than the threshold.

Walk-forward windows:
  In-Sample  (IS)  : 252 bars (1 year)   — HMM training
  Out-of-Sample (OOS): 126 bars (6 months) — evaluation
  Step               : 126 bars           — each fold steps forward 6 months

Allocation math (per symbol):
  per_sym_weight = signal.position_size_pct * leverage / n_symbols
  target_shares  = int(equity * per_sym_weight / price)
  delta_shares   = target_shares - current_shares
  cash          -= delta_shares * fill_price

Fill delay: signal generated on bar t → executed on bar t+1 close.
Slippage: 0.05% applied to fill price (sign-adjusted: buys worse, sells worse).
No stop-losses in backtester (stops are for live trading only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import Signal, StrategyOrchestrator
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

@dataclass
class BacktestConfig:
    """Parameters loaded from settings.yaml[backtest]."""

    slippage_pct: float = 0.0005
    initial_capital: float = 100_000.0
    train_window: int = 252             # IS bars
    test_window: int = 126              # OOS bars per fold
    step_size: int = 126               # bars between fold starts
    risk_free_rate: float = 0.045
    rebalance_threshold: float = 0.10  # min weight change to trigger rebalance
    min_confidence: float = 0.55
    commission: float = 0.0            # per-share commission (Alpaca = $0)


# ------------------------------------------------------------------ #
# Result dataclasses                                                   #
# ------------------------------------------------------------------ #

@dataclass
class BarRecord:
    """Portfolio state at a single bar."""

    date: pd.Timestamp
    equity: float
    cash: float
    regime_label: str
    regime_id: int
    vol_environment: str
    regime_prob: float
    target_allocation: float           # requested by strategy (0 if no signal)
    actual_allocation: float           # (sum of position values) / equity
    rebalanced: bool                   # True if pending signals were executed this bar


@dataclass
class TradeRecord:
    """A single rebalance execution (allocation change for one symbol)."""

    date: pd.Timestamp
    symbol: str
    delta_shares: int
    fill_price: float
    slippage_cost: float
    from_weight: float
    to_weight: float
    regime_label: str
    pnl: float = 0.0                  # realized P&L when reducing a position


@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    equity_curve: pd.Series            # DatetimeIndex → dollar equity
    regime_history: pd.Series          # DatetimeIndex → vol_environment string
    bar_records: list[BarRecord]
    trade_records: list[TradeRecord]
    n_states_selected: int


@dataclass
class BacktestResult:
    """Aggregated results across all walk-forward folds."""

    folds: list[FoldResult]
    full_equity: pd.Series             # continuous, chain-linked equity curve
    full_regime: pd.Series             # vol_environment at each bar
    all_trades: pd.DataFrame           # concatenated trade log
    config: Optional[BacktestConfig] = None


# ------------------------------------------------------------------ #
# Walk-forward backtester                                              #
# ------------------------------------------------------------------ #

class WalkForwardBacktester:
    """
    Executes a walk-forward allocation backtest.

    Usage
    -----
    bt = WalkForwardBacktester(hmm_config, backtest_config, strategy_config)
    result = bt.run(bars_by_symbol, output_dir=Path("results/"))
    """

    def __init__(
        self,
        hmm_config: HMMConfig,
        backtest_config: BacktestConfig,
        strategy_config: dict,
    ) -> None:
        self.hmm_config = hmm_config
        self.bt_config = backtest_config
        self.strategy_config = strategy_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
    ) -> BacktestResult:
        """
        Execute the full walk-forward backtest across all folds.

        Parameters
        ----------
        bars_by_symbol:
            {symbol: OHLCV DataFrame} with DatetimeIndex, lowercase columns.
            All DataFrames should cover the same date range.
        output_dir:
            If provided, writes equity_curve.csv, trade_log.csv,
            regime_history.csv.

        Returns
        -------
        BacktestResult with chain-linked equity curve and trade log.
        """
        primary = list(bars_by_symbol.keys())[0]
        index = bars_by_symbol[primary].index
        folds_config = self._generate_folds(index)

        if not folds_config:
            raise ValueError(
                f"Not enough data for walk-forward. Need "
                f"{self.bt_config.train_window + self.bt_config.test_window} bars, "
                f"got {len(index)}."
            )

        logger.info(
            "Walk-forward: %d symbols, %d folds, IS=%d OOS=%d step=%d",
            len(bars_by_symbol), len(folds_config),
            self.bt_config.train_window, self.bt_config.test_window,
            self.bt_config.step_size,
        )

        # Pre-enrich bars once (adds ema_50, atr, rsi, adx, bb_width).
        enriched = {
            sym: FeatureEngineer.enrich_bars(bars.copy())
            for sym, bars in bars_by_symbol.items()
        }

        # Pre-compute features for the primary symbol over the full window.
        all_features = FeatureEngineer.build_hmm_features(bars_by_symbol[primary])

        fold_results: list[FoldResult] = []
        running_capital = self.bt_config.initial_capital

        for i, (is_start, is_end, oos_start, oos_end) in enumerate(folds_config):
            logger.info(
                "Fold %d/%d  IS: %s→%s  OOS: %s→%s  capital: $%.0f",
                i + 1, len(folds_config),
                is_start.date(), is_end.date(),
                oos_start.date(), oos_end.date(),
                running_capital,
            )
            try:
                fold = self._run_fold(
                    i, bars_by_symbol, enriched, all_features,
                    is_start, is_end, oos_start, oos_end,
                    initial_capital=running_capital,
                )
            except Exception as exc:
                logger.error("Fold %d failed: %s — skipping.", i + 1, exc)
                continue

            fold_results.append(fold)
            if len(fold.equity_curve) > 0:
                running_capital = fold.equity_curve.iloc[-1]
                oos_ret = running_capital / fold.equity_curve.iloc[0] - 1
                logger.info("Fold %d OOS return: %+.2f%%", i + 1, oos_ret * 100)

        result = self._aggregate_folds(fold_results)

        if output_dir:
            self._write_output(result, Path(output_dir))

        return result

    # ------------------------------------------------------------------
    # Single-fold simulation
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold_index: int,
        bars_by_symbol: dict[str, pd.DataFrame],
        enriched: dict[str, pd.DataFrame],
        all_features: pd.DataFrame,
        is_start: pd.Timestamp,
        is_end: pd.Timestamp,
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
        initial_capital: float,
    ) -> FoldResult:
        primary = list(bars_by_symbol.keys())[0]

        # IS features — must satisfy min_train_bars.
        train_feat = all_features.loc[
            (all_features.index >= is_start) & (all_features.index <= is_end)
        ]
        if len(train_feat) < self.hmm_config.min_train_bars:
            raise ValueError(
                f"Fold {fold_index}: only {len(train_feat)} training features "
                f"(need {self.hmm_config.min_train_bars})."
            )

        # Fit HMM on IS features.
        engine = HMMEngine(self.hmm_config)
        engine.fit(train_feat)

        # Build strategy orchestrator from fitted regime infos.
        orchestrator = StrategyOrchestrator(
            self.strategy_config, engine.get_regime_infos()
        )

        # Run forward algorithm on IS+OOS together so stability tracking
        # carries through the training period into the test window.
        full_feat = all_features.loc[
            (all_features.index >= is_start) & (all_features.index <= oos_end)
        ]
        all_states = engine.predict_sequence(full_feat)
        state_by_date: dict[pd.Timestamp, object] = {
            full_feat.index[i]: all_states[i]
            for i in range(len(all_states))
        }

        # OOS dates from primary symbol.
        oos_dates = [
            d for d in bars_by_symbol[primary].index
            if oos_start <= d <= oos_end
        ]

        bar_records, trade_records = self._simulate_oos(
            engine, orchestrator,
            bars_by_symbol, enriched,
            state_by_date, oos_dates,
            initial_capital,
        )

        equity_curve = pd.Series(
            {r.date: r.equity for r in bar_records}, name="equity"
        )
        regime_history = pd.Series(
            {r.date: r.vol_environment for r in bar_records}, name="vol_env"
        )

        return FoldResult(
            fold_index=fold_index,
            train_start=is_start,
            train_end=is_end,
            test_start=oos_start,
            test_end=oos_end,
            equity_curve=equity_curve,
            regime_history=regime_history,
            bar_records=bar_records,
            trade_records=trade_records,
            n_states_selected=engine._n_states,
        )

    def _simulate_oos(
        self,
        engine: HMMEngine,
        orchestrator: StrategyOrchestrator,
        bars_by_symbol: dict[str, pd.DataFrame],
        enriched: dict[str, pd.DataFrame],
        state_by_date: dict,
        oos_dates: list[pd.Timestamp],
        initial_capital: float,
    ) -> tuple[list[BarRecord], list[TradeRecord]]:
        """
        Walk through OOS bars, maintaining portfolio state.

        Fill delay: signals from bar t → executed on bar t+1.
        """
        symbols = list(bars_by_symbol.keys())
        n_syms = len(symbols)

        # Portfolio state
        cash = float(initial_capital)
        shares: dict[str, int] = {s: 0 for s in symbols}
        avg_cost: dict[str, float] = {s: 0.0 for s in symbols}
        current_weights: dict[str, float] = {s: 0.0 for s in symbols}

        pending: Optional[dict[str, int]] = None   # (symbol → target_shares) queued

        bar_records: list[BarRecord] = []
        trade_records: list[TradeRecord] = []

        for bar_i, date in enumerate(oos_dates):
            # Collect prices for this bar.
            prices: dict[str, float] = {}
            for sym in symbols:
                df = bars_by_symbol[sym]
                if date in df.index:
                    prices[sym] = float(df.loc[date, "close"])

            if not prices:
                continue

            equity = cash + sum(shares[s] * prices.get(s, 0.0) for s in symbols)

            # ---- Execute pending rebalance (fill delay) ----
            rebalanced_this_bar = False
            if pending is not None:
                prev_date = oos_dates[bar_i - 1] if bar_i > 0 else date
                prev_regime = state_by_date.get(prev_date)
                regime_label = prev_regime.label if prev_regime else "UNKNOWN"

                for sym, tgt_sh in pending.items():
                    if sym not in prices:
                        continue
                    price = prices[sym]
                    old_sh = shares[sym]
                    delta = tgt_sh - old_sh
                    if delta == 0:
                        continue

                    side = 1 if delta > 0 else -1
                    fill_price = price * (1.0 + side * self.bt_config.slippage_pct)
                    slippage_cost = abs(delta) * price * self.bt_config.slippage_pct
                    commission_cost = abs(delta) * self.bt_config.commission

                    # P&L on sells
                    pnl = 0.0
                    if delta < 0 and avg_cost[sym] > 0:
                        pnl = (-delta) * (fill_price - avg_cost[sym])

                    # Update avg cost on buys
                    if delta > 0 and tgt_sh > 0:
                        prev_val = avg_cost[sym] * old_sh
                        new_val = fill_price * delta
                        avg_cost[sym] = (prev_val + new_val) / tgt_sh
                    elif tgt_sh == 0:
                        avg_cost[sym] = 0.0

                    old_wt = (old_sh * price) / max(equity, 1.0)
                    new_wt = (tgt_sh * price) / max(equity, 1.0)

                    cash -= delta * fill_price + commission_cost
                    shares[sym] = tgt_sh
                    rebalanced_this_bar = True

                    trade_records.append(TradeRecord(
                        date=date,
                        symbol=sym,
                        delta_shares=delta,
                        fill_price=fill_price,
                        slippage_cost=slippage_cost,
                        from_weight=old_wt,
                        to_weight=new_wt,
                        regime_label=regime_label,
                        pnl=pnl,
                    ))

                pending = None
                # Re-mark equity after trades
                equity = cash + sum(shares[s] * prices.get(s, 0.0) for s in symbols)

            # ---- Get regime state for this bar ----
            regime_state = state_by_date.get(date)
            target_alloc = 0.0

            if regime_state is not None:
                # Bars up to this date for strategy indicators
                bars_to_date: dict[str, pd.DataFrame] = {
                    sym: enriched[sym].loc[:date]
                    for sym in symbols
                    if sym in enriched and not enriched[sym].loc[:date].empty
                }

                # Current weights after mark-to-market
                current_weights = {
                    sym: (shares[sym] * prices.get(sym, 0.0)) / max(equity, 1.0)
                    for sym in symbols
                }

                # Generate signals
                signals = orchestrator.generate_signals(
                    symbols,
                    bars_to_date,
                    regime_state,
                    is_flickering=engine.is_flickering(),
                    current_weights=current_weights,
                )

                if signals:
                    # Build target shares for all symbols (equal weight split)
                    new_targets: dict[str, int] = {}
                    first_sig = signals[0]
                    target_alloc = first_sig.position_size_pct * first_sig.leverage

                    for sig in signals:
                        per_sym_weight = sig.position_size_pct * sig.leverage / n_syms
                        notional = equity * per_sym_weight
                        p = prices.get(sig.symbol, 0.0)
                        if p > 0:
                            new_targets[sig.symbol] = max(0, int(notional / p))

                    # Also build target=0 for symbols with no signal (flat)
                    for sym in symbols:
                        if sym not in new_targets and shares[sym] > 0:
                            new_targets[sym] = 0

                    if new_targets:
                        pending = new_targets

            # ---- Record bar state ----
            pos_value = sum(shares[s] * prices.get(s, 0.0) for s in symbols)
            actual_alloc = pos_value / max(equity, 1.0)

            bar_records.append(BarRecord(
                date=date,
                equity=equity,
                cash=cash,
                regime_label=regime_state.label if regime_state else "UNKNOWN",
                regime_id=regime_state.state_id if regime_state else -1,
                vol_environment=regime_state.vol_environment if regime_state else "unknown",
                regime_prob=regime_state.probability if regime_state else 0.0,
                target_allocation=target_alloc,
                actual_allocation=actual_alloc,
                rebalanced=rebalanced_this_bar,
            ))

        return bar_records, trade_records

    # ------------------------------------------------------------------
    # Fold management
    # ------------------------------------------------------------------

    def _generate_folds(
        self, index: pd.DatetimeIndex
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (is_start, is_end, oos_start, oos_end) tuples."""
        folds = []
        n = len(index)
        is_w = self.bt_config.train_window
        oos_w = self.bt_config.test_window
        step = self.bt_config.step_size

        start = 0
        while start + is_w + oos_w <= n:
            is_end_idx = start + is_w - 1
            oos_end_idx = min(start + is_w + oos_w - 1, n - 1)
            folds.append((
                index[start],
                index[is_end_idx],
                index[start + is_w],
                index[oos_end_idx],
            ))
            start += step

        return folds

    def _aggregate_folds(self, folds: list[FoldResult]) -> BacktestResult:
        """
        Chain-link fold equity curves into a single continuous curve.

        Each fold starts from where the previous fold ended. Daily returns
        within each fold are preserved exactly.
        """
        if not folds:
            return BacktestResult([], pd.Series(), pd.Series(), pd.DataFrame(), self.bt_config)

        eq_parts: list[pd.Series] = []
        reg_parts: list[pd.Series] = []
        trade_dfs: list[pd.DataFrame] = []

        for fold in folds:
            eq = fold.equity_curve
            if len(eq) > 0:
                eq_parts.append(eq)
            if len(fold.regime_history) > 0:
                reg_parts.append(fold.regime_history)
            if fold.trade_records:
                trade_dfs.append(
                    pd.DataFrame([vars(t) for t in fold.trade_records])
                )

        full_equity = pd.concat(eq_parts) if eq_parts else pd.Series(dtype=float)
        full_regime = pd.concat(reg_parts) if reg_parts else pd.Series(dtype=str)
        all_trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

        return BacktestResult(
            folds=folds,
            full_equity=full_equity,
            full_regime=full_regime,
            all_trades=all_trades,
            config=self.bt_config,
        )

    def _apply_slippage(self, price: float, side: str) -> float:
        sign = 1.0 if side == "buy" else -1.0
        return price * (1.0 + sign * self.bt_config.slippage_pct)

    def _write_output(self, result: BacktestResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not result.full_equity.empty:
            result.full_equity.to_csv(output_dir / "equity_curve.csv", header=True)
        if not result.full_regime.empty:
            result.full_regime.to_csv(output_dir / "regime_history.csv", header=True)
        if not result.all_trades.empty:
            result.all_trades.to_csv(output_dir / "trade_log.csv", index=False)
        logger.info("Backtest results written to %s/", output_dir)
