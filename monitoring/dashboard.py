"""
monitoring/dashboard.py — Terminal live dashboard using Rich.

Layout (refreshes every N seconds):

┌─ REGIME ──────────────────────────────────────────────────────┐
│ BULL (72%) │ Stability: 14 bars │ Flicker: 1/20              │
├─ PORTFOLIO ────────────────────────────────────────────────────┤
│ Equity: $105,230  │ Daily: +$340 (+0.32%)                     │
│ Allocation: 95%   │ Leverage: 1.25×                           │
├─ POSITIONS ────────────────────────────────────────────────────┤
│ SPY  │ LONG  │ $520.30 │ +1.2% │ Stop: $508 │ 3h ago         │
├─ RECENT SIGNALS ───────────────────────────────────────────────┤
│ 14:30 │ SPY │ Rebalance 60%→95% │ Low vol                    │
├─ RISK STATUS ──────────────────────────────────────────────────┤
│ Daily DD: 0.3% / 3% ✓    │ From Peak: 1.2% / 10% ✓          │
├─ SYSTEM ───────────────────────────────────────────────────────┤
│ Data: ✓  │ API: ✓ 23ms  │ HMM: 2d ago  │ PAPER             │
└────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from core.hmm_engine import RegimeState
    from broker.position_tracker import PositionTracker
    from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

_VOL_COLOR = {"low": "green", "mid": "yellow", "high": "red"}
_DIR_COLOR = {"LONG": "green", "FLAT": "dim white"}


class Dashboard:
    """
    Renders a live terminal dashboard that refreshes every N seconds.

    Usage
    -----
    dash = Dashboard(refresh_seconds=5)
    dash.start()
    # … call dash.update(…) each bar …
    dash.stop()
    """

    def __init__(self, refresh_seconds: int = 5, paper: bool = True) -> None:
        self.refresh_seconds = refresh_seconds
        self.paper = paper
        self._console = Console()
        self._live: Optional[Live] = None

        # Latest data cache — written by update(), read by _build_layout()
        self._regime_states: dict[str, "RegimeState"] = {}
        self._position_tracker: Optional["PositionTracker"] = None
        self._risk_manager: Optional["RiskManager"] = None
        self._recent_alerts: list[str] = []
        self._recent_signals: list[str] = []
        self._system_status: dict = {}
        self._started_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enter the Rich Live context and begin rendering."""
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=1.0 / max(self.refresh_seconds, 1),
            screen=False,
        )
        self._live.start()
        logger.debug("Dashboard started.")

    def stop(self) -> None:
        """Exit the Rich Live context cleanly."""
        if self._live:
            self._live.stop()
            self._live = None
        logger.debug("Dashboard stopped.")

    def update(
        self,
        regime_states: dict[str, "RegimeState"],
        position_tracker: "PositionTracker",
        risk_manager: "RiskManager",
        recent_alerts: list[str],
        recent_signals: Optional[list[str]] = None,
        system_status: Optional[dict] = None,
    ) -> None:
        """
        Push fresh data into the dashboard; Rich re-renders automatically.

        Called once per bar from the main loop.
        """
        self._regime_states = regime_states
        self._position_tracker = position_tracker
        self._risk_manager = risk_manager
        self._recent_alerts = recent_alerts[-10:]
        self._recent_signals = (recent_signals or [])[-8:]
        self._system_status = system_status or {}

        if self._live:
            self._live.update(self._build_layout())

    # ------------------------------------------------------------------
    # Layout builder
    # ------------------------------------------------------------------

    def _build_layout(self) -> Layout:
        sections = []

        sections.append(Panel(
            self._build_regime_table(),
            title="[bold]REGIME[/bold]",
            border_style="blue",
        ))
        sections.append(Panel(
            self._build_portfolio_table(),
            title="[bold]PORTFOLIO[/bold]",
            border_style="blue",
        ))
        sections.append(Panel(
            self._build_position_table(),
            title="[bold]POSITIONS[/bold]",
            border_style="blue",
        ))
        sections.append(Panel(
            self._build_signals_table(),
            title="[bold]RECENT SIGNALS[/bold]",
            border_style="blue",
        ))
        sections.append(Panel(
            self._build_risk_table(),
            title="[bold]RISK STATUS[/bold]",
            border_style="blue",
        ))
        sections.append(Panel(
            self._build_system_row(),
            title="[bold]SYSTEM[/bold]",
            border_style="blue",
        ))

        from rich.columns import Columns
        from rich.console import Group
        return Group(*sections)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_regime_table(self) -> Table:
        t = Table.grid(padding=(0, 2))
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)

        if not self._regime_states:
            t.add_row("[dim]No regime data yet[/dim]", "", "")
            return t

        for sym, rs in self._regime_states.items():
            vol = getattr(rs, "vol_environment", "mid")
            color = _VOL_COLOR.get(vol, "white")
            label = Text(f"{sym}: {rs.label} ({rs.probability:.0%})", style=f"bold {color}")
            stability = Text(
                f"Stability: {getattr(rs, 'consecutive_bars', 0)} bars",
                style="cyan",
            )
            flicker_rate = ""
            # flicker info not on RegimeState directly — shown if available in system_status
            t.add_row(label, stability, flicker_rate)

        return t

    def _build_portfolio_table(self) -> Table:
        t = Table.grid(padding=(0, 2))
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)

        if self._position_tracker is None:
            t.add_row("[dim]No portfolio data[/dim]", "")
            return t

        snap = self._position_tracker.get_snapshot()
        equity = snap.equity
        daily_start = snap.daily_start_equity
        daily_pnl = equity - daily_start if daily_start > 0 else 0.0
        daily_ret = daily_pnl / daily_start if daily_start > 0 else 0.0

        pnl_color = "green" if daily_pnl >= 0 else "red"
        sign = "+" if daily_pnl >= 0 else ""

        gross = snap.gross_exposure
        alloc_pct = gross / equity * 100 if equity > 0 else 0.0

        t.add_row(
            f"Equity: [bold]${equity:,.2f}[/bold]",
            f"Daily: [{pnl_color}]{sign}${daily_pnl:,.2f} ({sign}{daily_ret:.2%})[/{pnl_color}]",
        )
        t.add_row(
            f"Allocation: [bold]{alloc_pct:.0f}%[/bold]",
            f"Trades today: {snap.open_trade_count_today}",
        )
        return t

    def _build_position_table(self) -> Table:
        t = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        t.add_column("Symbol", style="bold", no_wrap=True)
        t.add_column("Dir", no_wrap=True)
        t.add_column("Price", justify="right", no_wrap=True)
        t.add_column("P&L", justify="right", no_wrap=True)
        t.add_column("Mkt Value", justify="right", no_wrap=True)

        if self._position_tracker is None:
            t.add_row("[dim]–[/dim]", "", "", "", "")
            return t

        positions = self._position_tracker.get_positions()
        if not positions:
            t.add_row("[dim]No open positions[/dim]", "", "", "", "")
            return t

        for sym, rec in sorted(positions.items()):
            pnl_color = "green" if rec.unrealized_pnl >= 0 else "red"
            sign = "+" if rec.unrealized_pnl >= 0 else ""
            t.add_row(
                sym,
                Text("LONG" if rec.qty > 0 else "SHORT", style=_DIR_COLOR.get("LONG", "green")),
                f"${rec.current_price:,.2f}",
                Text(f"{sign}{rec.unrealized_pnl_pct:.2%}", style=pnl_color),
                f"${rec.market_value:,.0f}",
            )
        return t

    def _build_signals_table(self) -> Table:
        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column(no_wrap=True)

        if not self._recent_signals:
            t.add_row("[dim]No signals yet[/dim]")
        else:
            for sig in self._recent_signals[-8:]:
                t.add_row(sig)
        return t

    def _build_risk_table(self) -> Table:
        t = Table.grid(padding=(0, 2))
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)

        if self._position_tracker is None or self._risk_manager is None:
            t.add_row("[dim]No risk data[/dim]", "")
            return t

        snap = self._position_tracker.get_snapshot()
        cfg = self._risk_manager.config

        # Daily drawdown
        daily_dd = (
            max(0.0, 1.0 - snap.equity / snap.daily_start_equity)
            if snap.daily_start_equity > 0 else 0.0
        )
        peak_dd = (
            max(0.0, 1.0 - snap.equity / snap.peak_equity)
            if snap.peak_equity > 0 else 0.0
        )
        weekly_dd = (
            max(0.0, 1.0 - snap.equity / snap.weekly_start_equity)
            if snap.weekly_start_equity > 0 else 0.0
        )

        def _dd_text(val: float, warn: float, halt: float) -> Text:
            tick = "✓" if val < warn else ("⚠" if val < halt else "✗")
            color = "green" if val < warn else ("yellow" if val < halt else "red")
            return Text(f"{val:.1%} / {halt:.0%} {tick}", style=color)

        halted = self._risk_manager.is_halted()
        halt_text = Text(
            f"HALTED: {self._risk_manager.halt_reason()}", style="bold red"
        ) if halted else Text("Running ✓", style="green")

        t.add_row(
            Text("Daily DD: ") + _dd_text(daily_dd, cfg.daily_dd_reduce, cfg.daily_dd_halt),
            Text("From Peak: ") + _dd_text(peak_dd, cfg.max_dd_from_peak * 0.5, cfg.max_dd_from_peak),
        )
        t.add_row(
            Text("Weekly DD: ") + _dd_text(weekly_dd, cfg.weekly_dd_reduce, cfg.weekly_dd_halt),
            halt_text,
        )
        return t

    def _build_system_row(self) -> Table:
        t = Table.grid(padding=(0, 2))
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)

        ss = self._system_status
        mode = "[bold magenta]PAPER[/bold magenta]" if self.paper else "[bold red]LIVE[/bold red]"

        data_ok = ss.get("data_feed_ok", True)
        api_ok = ss.get("api_ok", True)
        api_ms = ss.get("api_latency_ms", 0)
        hmm_age = ss.get("hmm_age_days", 0)
        uptime_secs = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        uptime = _fmt_uptime(uptime_secs)

        data_txt = Text("Data: ✓", style="green") if data_ok else Text("Data: ✗", style="red")
        api_txt = (
            Text(f"API: ✓ {api_ms:.0f}ms", style="green")
            if api_ok else Text("API: ✗", style="red")
        )
        hmm_txt = Text(f"HMM: {hmm_age}d ago", style="cyan" if hmm_age < 7 else "yellow")
        uptime_txt = Text(f"Up: {uptime}", style="dim")

        t.add_row(data_txt, api_txt, hmm_txt, uptime_txt, Text(mode))

        # Recent alerts (last 3) below the system row
        if self._recent_alerts:
            from rich.console import Group as RGroup
            lines = "\n".join(f"[dim]{a}[/dim]" for a in self._recent_alerts[-3:])
            # Appended as a second row spanning all columns
            t.add_row(Text(lines), "", "", "", "")

        return t


# ------------------------------------------------------------------ #
# Utility                                                              #
# ------------------------------------------------------------------ #

def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"
