"""
performance.py — Performance metrics and Rich-formatted reporting.

Computes standard quantitative metrics from an equity curve:
  Core     : total return, CAGR, Sharpe, Sortino, Calmar, max drawdown
  Trade    : win rate, avg win/loss, profit factor, total trades
  Regime   : per vol-environment breakdown (% time, contribution, Sharpe)
  Confidence: metrics bucketed by regime probability
  Benchmark : vs buy-and-hold and 200-SMA trend-following

Outputs Rich tables to terminal and writes CSV files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeMetrics:
    """Performance metrics for a single vol-environment."""

    vol_env: str
    pct_time: float
    total_return: float
    annualized_return: float
    sharpe: float
    win_rate: float
    avg_trade_pnl: float
    n_trades: int


@dataclass
class ConfidenceBucket:
    """Performance metrics grouped by regime-probability bucket."""

    label: str
    n_trades: int
    win_rate: float
    sharpe: float
    avg_pnl: float


@dataclass
class BenchmarkResult:
    name: str
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float


@dataclass
class WorstCase:
    worst_day: float
    worst_week: float
    worst_month: float
    max_consecutive_losses: int
    max_days_underwater: int


@dataclass
class PerformanceReport:
    """All computed metrics for a backtest run."""

    # Core
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int          # trading days
    volatility: float                   # annualized

    # Trade stats
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    avg_holding_days: float

    # Worst case
    worst_case: WorstCase

    # Per-regime breakdown
    regime_metrics: list[RegimeMetrics] = field(default_factory=list)

    # Confidence buckets
    confidence_buckets: list[ConfidenceBucket] = field(default_factory=list)

    # Benchmarks
    benchmarks: list[BenchmarkResult] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Analyzer                                                             #
# ------------------------------------------------------------------ #

class PerformanceAnalyzer:
    """Computes PerformanceReport from equity curve, trades, and regime history."""

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self.risk_free_rate = risk_free_rate
        self._rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        regime_history: Optional[pd.Series] = None,
        bar_regime_prob: Optional[pd.Series] = None,
        benchmark_prices: Optional[dict[str, pd.Series]] = None,
    ) -> PerformanceReport:
        """
        Compute all performance metrics.

        Parameters
        ----------
        equity_curve:
            Daily portfolio equity, DatetimeIndex.
        trades:
            Trade log from backtester (columns: date, symbol, delta_shares,
            fill_price, pnl, from_weight, to_weight, regime_label).
        regime_history:
            Optional series: DatetimeIndex → vol_environment string.
        bar_regime_prob:
            Optional series: DatetimeIndex → float (regime probability per bar).
        benchmark_prices:
            Optional {name: price_series} for benchmark comparison.
        """
        eq = equity_curve.dropna().sort_index()
        if len(eq) < 2:
            raise ValueError("Need at least 2 equity data points.")

        daily_rets = eq.pct_change().dropna()

        max_dd, max_dd_dur = self._max_drawdown(eq)

        report = PerformanceReport(
            total_return=float(eq.iloc[-1] / eq.iloc[0] - 1),
            cagr=self._cagr(eq),
            sharpe=self._sharpe(daily_rets),
            sortino=self._sortino(daily_rets),
            calmar=self._cagr(eq) / max(abs(max_dd), 1e-10),
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_dur,
            volatility=float(daily_rets.std() * math.sqrt(252)),
            **self._trade_stats(trades),
            worst_case=self._worst_case(daily_rets),
        )

        if regime_history is not None and not regime_history.empty:
            report.regime_metrics = self._regime_breakdown(eq, daily_rets, regime_history, trades)

        if bar_regime_prob is not None and not trades.empty:
            report.confidence_buckets = self._confidence_buckets(trades, bar_regime_prob)

        if benchmark_prices:
            report.benchmarks = self._compute_benchmarks(eq, benchmark_prices, daily_rets)

        return report

    def print_report(self, report: PerformanceReport) -> None:
        """Print formatted tables to terminal using Rich."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
            console = Console()
        except ImportError:
            self._print_plain(report)
            return

        console.print("\n[bold cyan]═══ REGIME-TRADER BACKTEST RESULTS ═══[/bold cyan]\n")

        # ---- Core metrics ----
        t = Table(title="Core Metrics", box=box.SIMPLE_HEAD, show_header=True)
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right")
        t.add_row("Total Return", f"{report.total_return:.2%}")
        t.add_row("CAGR", f"{report.cagr:.2%}")
        t.add_row("Annualized Vol", f"{report.volatility:.2%}")
        t.add_row("Sharpe Ratio", f"{report.sharpe:.3f}")
        t.add_row("Sortino Ratio", f"{report.sortino:.3f}")
        t.add_row("Calmar Ratio", f"{report.calmar:.3f}")
        t.add_row("Max Drawdown", f"{report.max_drawdown:.2%}")
        t.add_row("Max DD Duration", f"{report.max_drawdown_duration} days")
        console.print(t)

        # ---- Trade stats ----
        t2 = Table(title="Trade Statistics", box=box.SIMPLE_HEAD)
        t2.add_column("Metric", style="bold")
        t2.add_column("Value", justify="right")
        t2.add_row("Total Trades", str(report.total_trades))
        t2.add_row("Win Rate", f"{report.win_rate:.1%}")
        t2.add_row("Avg Win", f"{report.avg_win:.2%}")
        t2.add_row("Avg Loss", f"{report.avg_loss:.2%}")
        t2.add_row("Profit Factor", f"{report.profit_factor:.2f}")
        t2.add_row("Avg Holding (days)", f"{report.avg_holding_days:.1f}")
        console.print(t2)

        # ---- Worst case ----
        t3 = Table(title="Worst-Case Scenarios", box=box.SIMPLE_HEAD)
        t3.add_column("Scenario", style="bold")
        t3.add_column("Value", justify="right")
        wc = report.worst_case
        t3.add_row("Worst Day", f"{wc.worst_day:.2%}")
        t3.add_row("Worst Week", f"{wc.worst_week:.2%}")
        t3.add_row("Worst Month", f"{wc.worst_month:.2%}")
        t3.add_row("Max Consecutive Losses", str(wc.max_consecutive_losses))
        t3.add_row("Max Days Underwater", str(wc.max_days_underwater))
        console.print(t3)

        # ---- Regime breakdown ----
        if report.regime_metrics:
            t4 = Table(title="Regime Breakdown", box=box.SIMPLE_HEAD)
            for col in ["Vol Env", "% Time", "Ann. Return", "Sharpe", "Win Rate", "Trades"]:
                t4.add_column(col, justify="right" if col != "Vol Env" else "left")
            color_map = {"low": "green", "mid": "yellow", "high": "red", "unknown": "white"}
            for rm in sorted(report.regime_metrics, key=lambda r: r.vol_env):
                c = color_map.get(rm.vol_env, "white")
                t4.add_row(
                    f"[{c}]{rm.vol_env}[/{c}]",
                    f"{rm.pct_time:.1%}",
                    f"{rm.annualized_return:.2%}",
                    f"{rm.sharpe:.3f}",
                    f"{rm.win_rate:.1%}",
                    str(rm.n_trades),
                )
            console.print(t4)

        # ---- Confidence buckets ----
        if report.confidence_buckets:
            t5 = Table(title="Performance by Confidence Bucket", box=box.SIMPLE_HEAD)
            for col in ["Confidence", "Trades", "Win Rate", "Sharpe", "Avg P&L"]:
                t5.add_column(col, justify="right" if col != "Confidence" else "left")
            for cb in report.confidence_buckets:
                t5.add_row(
                    cb.label, str(cb.n_trades),
                    f"{cb.win_rate:.1%}", f"{cb.sharpe:.3f}", f"{cb.avg_pnl:.4f}",
                )
            console.print(t5)

        # ---- Benchmarks ----
        if report.benchmarks:
            t6 = Table(title="Benchmark Comparison", box=box.SIMPLE_HEAD)
            for col in ["Strategy", "Total Return", "CAGR", "Sharpe", "Max DD"]:
                t6.add_column(col, justify="right" if col != "Strategy" else "left")
            for bm in report.benchmarks:
                t6.add_row(
                    bm.name, f"{bm.total_return:.2%}", f"{bm.cagr:.2%}",
                    f"{bm.sharpe:.3f}", f"{bm.max_drawdown:.2%}",
                )
            console.print(t6)

        console.print()

    def to_csv(
        self,
        report: PerformanceReport,
        equity_curve: pd.Series,
        output_dir: Path,
    ) -> None:
        """Write equity curve and benchmark comparison to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        equity_curve.to_csv(output_dir / "equity_curve.csv", header=["equity"])

        if report.benchmarks:
            rows = [
                {
                    "name": bm.name,
                    "total_return": bm.total_return,
                    "cagr": bm.cagr,
                    "sharpe": bm.sharpe,
                    "max_drawdown": bm.max_drawdown,
                }
                for bm in report.benchmarks
            ]
            pd.DataFrame(rows).to_csv(output_dir / "benchmark_comparison.csv", index=False)

        if report.regime_metrics:
            rows = [vars(rm) for rm in report.regime_metrics]
            pd.DataFrame(rows).to_csv(output_dir / "regime_breakdown.csv", index=False)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _sharpe(self, daily_rets: pd.Series) -> float:
        excess = daily_rets - self._rf_daily
        std = daily_rets.std()
        if std < 1e-10:
            return 0.0
        return float(excess.mean() / std * math.sqrt(252))

    def _sortino(self, daily_rets: pd.Series) -> float:
        excess = daily_rets - self._rf_daily
        neg = daily_rets[daily_rets < 0]
        downside_std = neg.std()
        if downside_std < 1e-10:
            return float(excess.mean() * math.sqrt(252) * 10)  # very high → cap
        return float(excess.mean() / downside_std * math.sqrt(252))

    def _cagr(self, equity: pd.Series) -> float:
        n_years = len(equity) / 252
        if n_years < 1e-6:
            return 0.0
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)

    def _max_drawdown(self, equity: pd.Series) -> tuple[float, int]:
        """Return (max_drawdown_fraction, duration_in_trading_days)."""
        running_max = equity.cummax()
        dd = equity / running_max - 1.0
        max_dd = float(dd.min())

        # Duration: longest consecutive underwater period
        underwater = dd < 0
        max_dur = 0
        cur_dur = 0
        for uw in underwater:
            if uw:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0

        return max_dd, max_dur

    def _trade_stats(self, trades: pd.DataFrame) -> dict:
        """Return dict matching PerformanceReport field names."""
        if trades is None or len(trades) == 0:
            return dict(
                win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                profit_factor=0.0, total_trades=0, avg_holding_days=0.0,
            )

        pnls = trades["pnl"].values if "pnl" in trades else np.array([])
        if len(pnls) == 0:
            return dict(
                win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                profit_factor=0.0, total_trades=len(trades), avg_holding_days=0.0,
            )

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        total = len(pnls)

        pf = (wins.sum() / max(abs(losses.sum()), 1e-10)) if len(losses) > 0 else float("inf")

        # Avg holding: approximate from trade pairs per symbol
        avg_hold = 1.0  # default 1 day for rebalancing
        if "date" in trades.columns:
            trades_by_sym = trades.groupby("symbol")["date"].agg(["min", "max"])
            if not trades_by_sym.empty:
                durations = (trades_by_sym["max"] - trades_by_sym["min"]).dt.days
                avg_hold = float(durations.mean()) if len(durations) > 0 else 1.0

        return dict(
            win_rate=float(len(wins) / max(total, 1)),
            avg_win=float(wins.mean()) if len(wins) > 0 else 0.0,
            avg_loss=float(losses.mean()) if len(losses) > 0 else 0.0,
            profit_factor=float(pf),
            total_trades=total,
            avg_holding_days=avg_hold,
        )

    def _worst_case(self, daily_rets: pd.Series) -> WorstCase:
        worst_day = float(daily_rets.min())

        # Worst week (rolling 5 bars)
        weekly = daily_rets.rolling(5).apply(lambda r: (1 + r).prod() - 1, raw=True)
        worst_week = float(weekly.min()) if len(weekly) > 0 else 0.0

        # Worst month (rolling 21 bars)
        monthly = daily_rets.rolling(21).apply(lambda r: (1 + r).prod() - 1, raw=True)
        worst_month = float(monthly.min()) if len(monthly) > 0 else 0.0

        # Max consecutive losses
        max_cons = 0
        cur_cons = 0
        for r in daily_rets:
            if r < 0:
                cur_cons += 1
                max_cons = max(max_cons, cur_cons)
            else:
                cur_cons = 0

        # Max days underwater (from peak)
        eq_proxy = (1 + daily_rets).cumprod()
        running_max = eq_proxy.cummax()
        underwater = eq_proxy < running_max
        max_uw = 0
        cur_uw = 0
        for uw in underwater:
            if uw:
                cur_uw += 1
                max_uw = max(max_uw, cur_uw)
            else:
                cur_uw = 0

        return WorstCase(
            worst_day=worst_day,
            worst_week=worst_week,
            worst_month=worst_month,
            max_consecutive_losses=max_cons,
            max_days_underwater=max_uw,
        )

    def _regime_breakdown(
        self,
        equity: pd.Series,
        daily_rets: pd.Series,
        regime_history: pd.Series,
        trades: pd.DataFrame,
    ) -> list[RegimeMetrics]:
        aligned_regime = regime_history.reindex(daily_rets.index, method="ffill")
        total_bars = len(daily_rets)
        metrics_list = []

        for vol_env in aligned_regime.dropna().unique():
            mask = aligned_regime == vol_env
            env_rets = daily_rets[mask]
            pct_time = mask.sum() / max(total_bars, 1)
            ann_ret = float((1 + env_rets.mean()) ** 252 - 1) if len(env_rets) > 0 else 0.0

            env_sharpe = self._sharpe(env_rets) if len(env_rets) > 1 else 0.0
            wins = env_rets[env_rets > 0]
            win_rate = len(wins) / max(len(env_rets), 1)

            # Trade P&L for this regime
            n_trades = 0
            avg_pnl = 0.0
            if not trades.empty and "regime_label" in trades.columns:
                # Map regime_label (e.g. "BULL") → vol_environment via the history
                if "vol_environment" in trades.columns:
                    env_trades = trades[trades["vol_environment"] == vol_env]
                else:
                    env_trades = trades.iloc[0:0]
                n_trades = len(env_trades)
                avg_pnl = float(env_trades["pnl"].mean()) if n_trades > 0 and "pnl" in env_trades else 0.0

            metrics_list.append(RegimeMetrics(
                vol_env=vol_env,
                pct_time=float(pct_time),
                total_return=float((1 + env_rets).prod() - 1) if len(env_rets) > 0 else 0.0,
                annualized_return=ann_ret,
                sharpe=env_sharpe,
                win_rate=float(win_rate),
                avg_trade_pnl=avg_pnl,
                n_trades=n_trades,
            ))

        return metrics_list

    def _confidence_buckets(
        self,
        trades: pd.DataFrame,
        bar_regime_prob: pd.Series,
    ) -> list[ConfidenceBucket]:
        buckets = [
            ("<50%", 0.0, 0.5),
            ("50–60%", 0.5, 0.6),
            ("60–70%", 0.6, 0.7),
            ("70%+", 0.7, 1.01),
        ]
        result = []
        for label, lo, hi in buckets:
            prob_at_trade = bar_regime_prob.reindex(trades["date"], method="nearest") if "date" in trades else pd.Series()
            mask = (prob_at_trade >= lo) & (prob_at_trade < hi)
            bucket_trades = trades[mask.values] if len(mask) == len(trades) else trades.iloc[0:0]
            n = len(bucket_trades)
            if n == 0:
                result.append(ConfidenceBucket(label=label, n_trades=0, win_rate=0.0, sharpe=0.0, avg_pnl=0.0))
                continue

            pnls = bucket_trades["pnl"].values if "pnl" in bucket_trades else np.zeros(n)
            wins = pnls[pnls > 0]
            pnl_rets = pd.Series(pnls / max(abs(pnls).max(), 1.0))
            sharpe = self._sharpe(pnl_rets) if len(pnl_rets) > 1 else 0.0
            result.append(ConfidenceBucket(
                label=label,
                n_trades=n,
                win_rate=float(len(wins) / n),
                sharpe=sharpe,
                avg_pnl=float(pnls.mean()),
            ))
        return result

    def _compute_benchmarks(
        self,
        strategy_equity: pd.Series,
        benchmark_prices: dict[str, pd.Series],
        strategy_rets: pd.Series,
    ) -> list[BenchmarkResult]:
        results = []
        initial = float(strategy_equity.iloc[0])

        for name, prices in benchmark_prices.items():
            prices = prices.reindex(strategy_equity.index, method="ffill").dropna()
            if len(prices) < 2:
                continue

            if name == "buy_and_hold":
                bm_equity = initial * prices / prices.iloc[0]
            elif name == "sma200":
                sma200 = prices.rolling(200, min_periods=1).mean()
                in_market = prices > sma200
                daily_px_rets = prices.pct_change().fillna(0)
                strat_rets_bm = daily_px_rets * in_market.shift(1).fillna(0)
                bm_equity = initial * (1 + strat_rets_bm).cumprod()
            else:
                bm_equity = initial * prices / prices.iloc[0]

            bm_rets = bm_equity.pct_change().dropna()
            dd, _ = self._max_drawdown(bm_equity)
            results.append(BenchmarkResult(
                name=name,
                total_return=float(bm_equity.iloc[-1] / bm_equity.iloc[0] - 1),
                cagr=self._cagr(bm_equity),
                sharpe=self._sharpe(bm_rets),
                max_drawdown=dd,
            ))

        return results

    def _print_plain(self, report: PerformanceReport) -> None:
        """Fallback plain-text output when Rich is not available."""
        print(f"\n=== BACKTEST RESULTS ===")
        print(f"Total Return : {report.total_return:.2%}")
        print(f"CAGR         : {report.cagr:.2%}")
        print(f"Sharpe       : {report.sharpe:.3f}")
        print(f"Sortino      : {report.sortino:.3f}")
        print(f"Max Drawdown : {report.max_drawdown:.2%}")
        print(f"Win Rate     : {report.win_rate:.1%}")
        print(f"Total Trades : {report.total_trades}")
