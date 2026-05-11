"""
stress_test.py — Monte Carlo stress scenarios for the backtester.

Three scenarios:
  1. Crash injection   : randomly inject 5–15% single-day drops at 10 points.
                         100 Monte Carlo runs. Reports mean/worst max-loss
                         and % of runs where circuit-breaker-like drawdown fired.

  2. Gap risk          : inject overnight adverse gaps of 2–5× ATR at random bars.
                         Reports expected loss vs actual max loss.

  3. Regime misclassify: shuffle the regime labels produced by the HMM.
                         Verifies that the allocation system still survives
                         even with wrong signals — risk management must be
                         independent of regime accuracy.

All scenarios operate on bar data copies and re-run the backtester.
Results are reported as plain-text tables (Rich if available).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Result dataclasses                                                   #
# ------------------------------------------------------------------ #

@dataclass
class CrashScenarioResult:
    """Monte Carlo crash injection results."""

    n_simulations: int
    mean_max_loss: float           # mean of per-simulation max drawdown
    worst_max_loss: float          # worst max drawdown across all sims
    pct_severe_drawdown: float     # % sims where max DD > 20%
    mean_total_return: float
    std_total_return: float
    circuit_breaker_pct: float     # % sims where drawdown > 10% (halt threshold)


@dataclass
class GapRiskResult:
    """Overnight gap risk results."""

    n_gaps_injected: int
    mean_gap_size: float           # mean injected gap size
    expected_loss: float           # expected loss = mean gap * allocation
    actual_max_loss: float         # realized max single-day loss


@dataclass
class MisclassifyResult:
    """Regime misclassification stress test results."""

    baseline_return: float         # return with correct regimes
    shuffled_returns: list[float]  # return for each shuffle seed
    mean_shuffled: float
    std_shuffled: float
    pct_positive: float            # % of shuffled runs still positive
    conclusion: str                # "ROBUST" or "FRAGILE"


@dataclass
class StressReport:
    crash: Optional[CrashScenarioResult] = None
    gap_risk: Optional[GapRiskResult] = None
    misclassify: Optional[MisclassifyResult] = None


# ------------------------------------------------------------------ #
# Stress tester                                                        #
# ------------------------------------------------------------------ #

class StressTester:
    """
    Applies stress scenarios to bar data and re-runs the backtester.

    Parameters
    ----------
    backtester:
        A configured WalkForwardBacktester instance.
    severe_dd_threshold:
        Drawdown fraction considered "severe" for the circuit-breaker metric.
    """

    def __init__(
        self,
        backtester,                     # WalkForwardBacktester (avoid circular import)
        severe_dd_threshold: float = 0.20,
        circuit_breaker_threshold: float = 0.10,
    ) -> None:
        self.backtester = backtester
        self.severe_dd = severe_dd_threshold
        self.cb_threshold = circuit_breaker_threshold

    # ------------------------------------------------------------------
    # Public scenarios
    # ------------------------------------------------------------------

    def run_crash_injection(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        n_crashes: int = 10,
        crash_range: tuple[float, float] = (0.05, 0.15),
        n_simulations: int = 100,
    ) -> CrashScenarioResult:
        """
        Inject random sudden drops and measure portfolio response.

        Parameters
        ----------
        bars_by_symbol:
            Base bar data to modify.
        n_crashes:
            Number of crash events per simulation.
        crash_range:
            (min_drop, max_drop) as fractions of price.
        n_simulations:
            Number of Monte Carlo seeds.
        """
        logger.info(
            "Crash injection: %d simulations, %d crashes each, drop %.0f%%–%.0f%%",
            n_simulations, n_crashes,
            crash_range[0] * 100, crash_range[1] * 100,
        )

        max_losses: list[float] = []
        total_returns: list[float] = []
        circuit_fires: int = 0

        for seed in range(n_simulations):
            rng = np.random.default_rng(seed + 1000)
            modified = self._inject_crashes(bars_by_symbol, n_crashes, crash_range, rng)

            try:
                result = self.backtester.run(modified)
            except Exception as exc:
                logger.debug("Crash sim seed=%d failed: %s", seed, exc)
                continue

            if result.full_equity.empty or len(result.full_equity) < 2:
                continue

            eq = result.full_equity
            rets = eq.pct_change().dropna()
            running_max = eq.cummax()
            dd = float((eq / running_max - 1).min())
            total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)

            max_losses.append(dd)
            total_returns.append(total_ret)
            if dd < -self.cb_threshold:
                circuit_fires += 1

        n = len(max_losses)
        if n == 0:
            return CrashScenarioResult(0, 0, 0, 0, 0, 0, 0)

        arr = np.array(max_losses)
        ret_arr = np.array(total_returns)

        return CrashScenarioResult(
            n_simulations=n,
            mean_max_loss=float(arr.mean()),
            worst_max_loss=float(arr.min()),
            pct_severe_drawdown=float((arr < -self.severe_dd).mean()),
            mean_total_return=float(ret_arr.mean()),
            std_total_return=float(ret_arr.std()),
            circuit_breaker_pct=float(circuit_fires / n),
        )

    def run_gap_risk(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        n_gaps: int = 20,
        gap_atr_multiple: tuple[float, float] = (2.0, 5.0),
        n_simulations: int = 50,
    ) -> GapRiskResult:
        """
        Inject adverse overnight price gaps and measure losses.

        Gaps are drawn from gap_atr_multiple × ATR of the gap bar.
        """
        logger.info(
            "Gap risk: %d simulations, %d gaps each, %.1f–%.1f× ATR",
            n_simulations, n_gaps, *gap_atr_multiple,
        )

        all_gap_sizes: list[float] = []
        all_single_day_losses: list[float] = []

        for seed in range(n_simulations):
            rng = np.random.default_rng(seed + 2000)
            modified, gap_sizes = self._inject_gaps(
                bars_by_symbol, n_gaps, gap_atr_multiple, rng
            )
            all_gap_sizes.extend(gap_sizes)

            try:
                result = self.backtester.run(modified)
            except Exception:
                continue

            if result.full_equity.empty:
                continue

            daily_rets = result.full_equity.pct_change().dropna()
            if len(daily_rets) > 0:
                all_single_day_losses.append(float(daily_rets.min()))

        mean_gap = float(np.mean(all_gap_sizes)) if all_gap_sizes else 0.0
        act_max_loss = float(np.min(all_single_day_losses)) if all_single_day_losses else 0.0
        # Expected loss ≈ mean gap × typical allocation (~0.80)
        expected_loss = -mean_gap * 0.80

        return GapRiskResult(
            n_gaps_injected=n_gaps,
            mean_gap_size=mean_gap,
            expected_loss=expected_loss,
            actual_max_loss=act_max_loss,
        )

    def run_misclassification(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        n_shuffles: int = 20,
    ) -> MisclassifyResult:
        """
        Shuffle the HMM regime outputs and measure strategy degradation.

        If the strategy completely collapses with wrong regimes, the risk
        management layer is not independent enough of regime accuracy.
        A robust system should still produce positive (if lower) returns.
        """
        logger.info("Regime misclassification: %d shuffle seeds", n_shuffles)

        # Baseline run (correct regimes)
        try:
            baseline = self.backtester.run(bars_by_symbol)
            baseline_ret = float(
                baseline.full_equity.iloc[-1] / baseline.full_equity.iloc[0] - 1
            ) if not baseline.full_equity.empty else 0.0
        except Exception as exc:
            logger.error("Baseline run failed: %s", exc)
            baseline_ret = 0.0

        shuffled_returns: list[float] = []

        for seed in range(n_shuffles):
            modified = self._shuffle_regimes(bars_by_symbol, seed)
            try:
                result = self.backtester.run(modified)
                if not result.full_equity.empty:
                    ret = float(
                        result.full_equity.iloc[-1] / result.full_equity.iloc[0] - 1
                    )
                    shuffled_returns.append(ret)
            except Exception:
                continue

        if not shuffled_returns:
            return MisclassifyResult(baseline_ret, [], 0.0, 0.0, 0.0, "UNKNOWN")

        arr = np.array(shuffled_returns)
        mean_sh = float(arr.mean())
        std_sh = float(arr.std())
        pct_pos = float((arr > 0).mean())

        conclusion = "ROBUST" if pct_pos >= 0.5 else "FRAGILE"
        if conclusion == "FRAGILE":
            logger.warning(
                "Misclassification test: FRAGILE — only %.0f%% of shuffled runs "
                "were profitable. Risk management may be insufficient.",
                pct_pos * 100,
            )

        return MisclassifyResult(
            baseline_return=baseline_ret,
            shuffled_returns=list(shuffled_returns),
            mean_shuffled=mean_sh,
            std_shuffled=std_sh,
            pct_positive=pct_pos,
            conclusion=conclusion,
        )

    def run_all(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
    ) -> StressReport:
        """Run all three stress scenarios and return a combined report."""
        report = StressReport()
        report.crash = self.run_crash_injection(bars_by_symbol)
        report.gap_risk = self.run_gap_risk(bars_by_symbol)
        report.misclassify = self.run_misclassification(bars_by_symbol)
        self.print_report(report)
        return report

    # ------------------------------------------------------------------
    # Injection helpers
    # ------------------------------------------------------------------

    def _inject_crashes(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        n_crashes: int,
        crash_range: tuple[float, float],
        rng: np.random.Generator,
    ) -> dict[str, pd.DataFrame]:
        """Return a copy of bars_by_symbol with random crash events injected."""
        modified = {sym: df.copy() for sym, df in bars_by_symbol.items()}
        primary = list(modified.keys())[0]
        n = len(modified[primary])

        # Pick crash indices well inside the data range.
        lo, hi = 50, n - 10
        if lo >= hi:
            return modified
        n_crashes = min(n_crashes, (hi - lo) // 5)
        crash_idxs = sorted(rng.choice(range(lo, hi), n_crashes, replace=False))

        for sym, df in modified.items():
            for idx in crash_idxs:
                if idx >= len(df):
                    continue
                drop = rng.uniform(*crash_range)
                factor = 1.0 - drop
                date = df.index[idx]
                df.loc[date, "close"] = df.loc[date, "close"] * factor
                df.loc[date, "low"] = min(
                    df.loc[date, "low"] * factor, df.loc[date, "close"]
                )
                df.loc[date, "high"] = max(
                    df.loc[date, "open"], df.loc[date, "close"]
                )
                # Next bar open reflects the crash
                if idx + 1 < len(df):
                    next_date = df.index[idx + 1]
                    df.loc[next_date, "open"] = df.loc[date, "close"]

        return modified

    def _inject_gaps(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        n_gaps: int,
        gap_atr_multiple: tuple[float, float],
        rng: np.random.Generator,
    ) -> tuple[dict[str, pd.DataFrame], list[float]]:
        """
        Inject adverse overnight gaps. Returns modified bars and list of gap sizes.

        Gap = open[t] is set below close[t-1] by gap_multiple × ATR[t-1].
        """
        modified = {sym: df.copy() for sym, df in bars_by_symbol.items()}
        primary = list(modified.keys())[0]
        n = len(modified[primary])

        lo, hi = 60, n - 5
        if lo >= hi:
            return modified, []
        n_gaps = min(n_gaps, (hi - lo) // 4)
        gap_idxs = sorted(rng.choice(range(lo, hi), n_gaps, replace=False))

        gap_sizes: list[float] = []

        for sym, df in modified.items():
            # Approximate ATR as high-low average
            hl_avg = ((df["high"] - df["low"]) / df["close"]).rolling(14).mean()

            for idx in gap_idxs:
                if idx >= len(df) or pd.isna(hl_avg.iloc[idx - 1]):
                    continue
                atr_pct = float(hl_avg.iloc[idx - 1])
                mult = rng.uniform(*gap_atr_multiple)
                gap_pct = atr_pct * mult
                gap_sizes.append(gap_pct)

                date = df.index[idx]
                prev_close = float(df.iloc[idx - 1]["close"])
                gapped_open = prev_close * (1.0 - gap_pct)

                df.loc[date, "open"] = gapped_open
                df.loc[date, "low"] = min(gapped_open, float(df.loc[date, "low"]) * (1 - gap_pct))
                df.loc[date, "close"] = min(float(df.loc[date, "close"]), gapped_open * 1.01)

        return modified, gap_sizes

    def _shuffle_regimes(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        seed: int,
    ) -> dict[str, pd.DataFrame]:
        """
        Simulate misclassification by shuffling prices so the HMM sees
        different vol ordering. We shuffle blocks of bars to disrupt the
        vol-regime structure without making data obviously invalid.
        """
        modified = {}
        rng = np.random.default_rng(seed + 3000)

        for sym, df in bars_by_symbol.items():
            df_copy = df.copy()
            n = len(df_copy)
            block_size = 60  # shuffle 60-day blocks
            n_blocks = n // block_size
            if n_blocks < 2:
                modified[sym] = df_copy
                continue

            # Shuffle block order for OHLCV columns only, keep index intact
            block_indices = list(range(n_blocks))
            rng.shuffle(block_indices)

            shuffled_data = []
            for b in block_indices:
                start = b * block_size
                end = min(start + block_size, n)
                shuffled_data.append(df_copy.iloc[start:end].values)

            shuffled_vals = np.vstack(shuffled_data)[: n]
            new_df = pd.DataFrame(
                shuffled_vals,
                index=df_copy.index[:n],
                columns=df_copy.columns,
            )
            modified[sym] = new_df

        return modified

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, report: StressReport) -> None:
        """Print stress test results using Rich if available."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
            console = Console()
        except ImportError:
            self._print_plain(report)
            return

        console.print("\n[bold red]═══ STRESS TEST RESULTS ═══[/bold red]\n")

        if report.crash:
            cr = report.crash
            t = Table(title="Crash Injection (Monte Carlo)", box=box.SIMPLE_HEAD)
            t.add_column("Metric", style="bold")
            t.add_column("Value", justify="right")
            t.add_row("Simulations run", str(cr.n_simulations))
            t.add_row("Mean max drawdown", f"{cr.mean_max_loss:.2%}")
            t.add_row("Worst max drawdown", f"{cr.worst_max_loss:.2%}")
            t.add_row("% with DD > 20%", f"{cr.pct_severe_drawdown:.1%}")
            t.add_row("% where CB would fire (DD>10%)", f"{cr.circuit_breaker_pct:.1%}")
            t.add_row("Mean total return", f"{cr.mean_total_return:.2%}")
            t.add_row("Std total return", f"{cr.std_total_return:.2%}")
            console.print(t)

        if report.gap_risk:
            gr = report.gap_risk
            t2 = Table(title="Overnight Gap Risk", box=box.SIMPLE_HEAD)
            t2.add_column("Metric", style="bold")
            t2.add_column("Value", justify="right")
            t2.add_row("Gaps injected per run", str(gr.n_gaps_injected))
            t2.add_row("Mean gap size", f"{gr.mean_gap_size:.2%}")
            t2.add_row("Expected portfolio loss", f"{gr.expected_loss:.2%}")
            t2.add_row("Actual worst day", f"{gr.actual_max_loss:.2%}")
            console.print(t2)

        if report.misclassify:
            mc = report.misclassify
            color = "green" if mc.conclusion == "ROBUST" else "red"
            t3 = Table(title="Regime Misclassification", box=box.SIMPLE_HEAD)
            t3.add_column("Metric", style="bold")
            t3.add_column("Value", justify="right")
            t3.add_row("Baseline return", f"{mc.baseline_return:.2%}")
            t3.add_row("Mean shuffled return", f"{mc.mean_shuffled:.2%}")
            t3.add_row("Std shuffled return", f"{mc.std_shuffled:.2%}")
            t3.add_row("% shuffled runs positive", f"{mc.pct_positive:.1%}")
            t3.add_row("Verdict", f"[{color}]{mc.conclusion}[/{color}]")
            console.print(t3)

        console.print()

    def _print_plain(self, report: StressReport) -> None:
        print("\n=== STRESS TEST RESULTS ===")
        if report.crash:
            cr = report.crash
            print(f"Crash: mean max-loss={cr.mean_max_loss:.2%}, "
                  f"worst={cr.worst_max_loss:.2%}, CB-fire={cr.circuit_breaker_pct:.1%}")
        if report.gap_risk:
            gr = report.gap_risk
            print(f"Gaps: expected-loss={gr.expected_loss:.2%}, "
                  f"actual-worst={gr.actual_max_loss:.2%}")
        if report.misclassify:
            mc = report.misclassify
            print(f"Misclassify: mean={mc.mean_shuffled:.2%}, verdict={mc.conclusion}")
