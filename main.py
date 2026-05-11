"""
main.py — Entry point for regime-trader.

Usage
-----
# Walk-forward backtest:
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
python main.py backtest --symbols SPY --start 2019-01-01 --compare
python main.py backtest --symbols SPY --start 2019-01-01 --stress-test

# Train HMM and exit (no trading):
python main.py paper --train-only
python main.py live  --train-only

# Paper-trading live loop:
python main.py paper [--dry-run]

# Live trading (real capital — use only after 30+ days paper results):
python main.py live [--dry-run]

# Show dashboard for a running instance (reads state_snapshot.json):
python main.py dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal as signal_module
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SETTINGS_PATH = Path("config/settings.yaml")
STATE_SNAPSHOT_PATH = Path("state_snapshot.json")
WEB_STATE_PATH = Path("web_state.json")
CONTROL_PATH = Path("control.json")
MODEL_DIR = Path("models")
MODEL_MAX_AGE_DAYS = 7
HISTORY_BARS = 600          # bars to load for HMM feature warmup


# ------------------------------------------------------------------ #
# Config loading                                                       #
# ------------------------------------------------------------------ #

def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        logger.warning("settings.yaml not found — using defaults.")
        return {}
    with open(SETTINGS_PATH) as f:
        return yaml.safe_load(f) or {}


# ------------------------------------------------------------------ #
# Retry helper                                                         #
# ------------------------------------------------------------------ #

def retry(fn, max_attempts: int = 3, backoff_base: float = 2.0, label: str = ""):
    """Call fn(), retrying up to max_attempts times with exponential backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = backoff_base ** attempt
                logger.warning(
                    "%s attempt %d/%d failed: %s. Retrying in %.1fs …",
                    label or fn.__name__, attempt + 1, max_attempts, exc, wait,
                )
                time.sleep(wait)
    raise last_exc


# ------------------------------------------------------------------ #
# Session state (persisted to state_snapshot.json)                    #
# ------------------------------------------------------------------ #

@dataclass
class SessionState:
    session_start: str
    mode: str                           # "paper" | "live"
    last_trained: str                   # ISO datetime string
    hmm_model_path: str
    last_processed_bar: Optional[str]   # ISO datetime of last processed bar
    daily_trade_count: int
    peak_equity: float
    daily_start_equity: float
    weekly_start_equity: float
    last_regime: dict                   # {symbol: label}

    def save(self, path: Path = STATE_SNAPSHOT_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.debug("State snapshot saved → %s", path)

    @classmethod
    def load(cls, path: Path = STATE_SNAPSHOT_PATH) -> Optional["SessionState"]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(**data)
        except Exception as exc:
            logger.warning("Could not load state snapshot: %s", exc)
            return None

    @classmethod
    def new(cls, mode: str, model_path: str) -> "SessionState":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            session_start=now,
            mode=mode,
            last_trained=now,
            hmm_model_path=model_path,
            last_processed_bar=None,
            daily_trade_count=0,
            peak_equity=0.0,
            daily_start_equity=0.0,
            weekly_start_equity=0.0,
            last_regime={},
        )


# ------------------------------------------------------------------ #
# Bar feed (WebSocket + polling fallback)                              #
# ------------------------------------------------------------------ #

class BarFeed:
    """
    Provides a thread-safe bar queue populated via Alpaca WebSocket.

    Falls back to polling when WebSocket is unavailable (e.g. outside
    market hours or in development).
    """

    def __init__(self, symbols: list[str], api_key: str, secret_key: str,
                 paper: bool = True) -> None:
        self.symbols = symbols
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the WebSocket listener in a background thread."""
        self._thread = threading.Thread(
            target=self._run_stream, daemon=True, name="bar-feed"
        )
        self._thread.start()
        logger.info("Bar feed started for %s", self.symbols)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def get(self, timeout: float = 60.0) -> Optional[dict]:
        """Block until a bar arrives (or timeout). Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run_stream(self) -> None:
        try:
            self._stream_alpaca()
        except Exception as exc:
            logger.warning(
                "WebSocket stream failed: %s — falling back to polling.", exc
            )
            self._poll_fallback()

    def _stream_alpaca(self) -> None:
        """Use alpaca-trade-api async streaming for minute bars."""
        import alpaca_trade_api as tradeapi  # type: ignore
        base_url = (
            "https://paper-api.alpaca.markets"
            if self.paper
            else "https://api.alpaca.markets"
        )
        conn = tradeapi.Stream(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=base_url,
            data_feed="iex",
        )

        @conn.on(r"^AM\." + "|^AM\\.".join(self.symbols) + "$")
        async def on_bar(conn, channel, bar):
            self._queue.put({
                "symbol": bar.symbol,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "timestamp": str(bar.start),
            })

        channels = [f"AM.{sym}" for sym in self.symbols]
        conn.run(channels)

    def _poll_fallback(self) -> None:
        """Poll Alpaca REST API for the latest bar every 60 seconds."""
        import alpaca_trade_api as tradeapi  # type: ignore
        base_url = (
            "https://paper-api.alpaca.markets"
            if self.paper
            else "https://api.alpaca.markets"
        )
        api = tradeapi.REST(self.api_key, self._secret_key_for_poll(), base_url=base_url)
        last_ts: dict[str, str] = {}

        while not self._stop_event.is_set():
            for sym in self.symbols:
                try:
                    bar = api.get_latest_bar(sym)
                    ts = str(bar.t)
                    if last_ts.get(sym) != ts:
                        last_ts[sym] = ts
                        self._queue.put({
                            "symbol": sym,
                            "open": float(bar.o),
                            "high": float(bar.h),
                            "low": float(bar.l),
                            "close": float(bar.c),
                            "volume": float(bar.v),
                            "timestamp": ts,
                        })
                except Exception as exc:
                    logger.debug("Poll error for %s: %s", sym, exc)
            self._stop_event.wait(60.0)

    def _secret_key_for_poll(self) -> str:
        return os.environ.get("ALPACA_SECRET_KEY", "")


# ------------------------------------------------------------------ #
# Main trading engine                                                  #
# ------------------------------------------------------------------ #

class TradingEngine:
    """
    Full orchestration: startup → main loop → shutdown.

    Instantiate once and call run(). SIGINT/SIGTERM trigger clean shutdown.
    """

    def __init__(self, cfg: dict, paper: bool, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.paper = paper
        self.dry_run = dry_run

        broker_cfg = cfg.get("broker", {})
        hmm_cfg_raw = cfg.get("hmm", {})
        bt_cfg_raw = cfg.get("backtest", {})
        strategy_cfg = cfg.get("strategy", {})
        strategy_cfg.update(hmm_cfg_raw)

        self.symbols: list[str] = broker_cfg.get("symbols", ["SPY"])
        self.primary = self.symbols[0]
        self.timeframe: str = broker_cfg.get("timeframe", "1Day")

        from core.hmm_engine import HMMConfig
        from backtest.backtester import BacktestConfig

        self.hmm_config = HMMConfig(
            n_candidates=hmm_cfg_raw.get("n_candidates", [3, 4, 5, 6, 7]),
            n_init=hmm_cfg_raw.get("n_init", 10),
            covariance_type=hmm_cfg_raw.get("covariance_type", "full"),
            min_train_bars=hmm_cfg_raw.get("min_train_bars", 504),
            stability_bars=hmm_cfg_raw.get("stability_bars", 3),
            flicker_window=hmm_cfg_raw.get("flicker_window", 20),
            flicker_threshold=hmm_cfg_raw.get("flicker_threshold", 4),
            min_confidence=hmm_cfg_raw.get("min_confidence", 0.55),
        )
        self.strategy_cfg = strategy_cfg
        self.risk_cfg_raw = cfg.get("risk", {})
        self.monitoring_cfg = cfg.get("monitoring", {})

        self.model_path = MODEL_DIR / f"hmm_{self.primary}.pkl"

        self._shutdown_event = threading.Event()
        self._session_state: Optional[SessionState] = None

        # Components — initialized in _startup()
        self._client = None
        self._position_tracker = None
        self._risk_manager = None
        self._hmm_engine = None
        self._orchestrator = None
        self._signal_generator = None
        self._bar_feed: Optional[BarFeed] = None
        self._data_fetcher = None

        # Monitoring — initialized in _startup()
        from monitoring.alerts import AlertManager
        from monitoring.dashboard import Dashboard
        from monitoring.logger import get_main_logger, get_trade_logger, get_regime_logger
        refresh = self.monitoring_cfg.get("dashboard_refresh_seconds", 5)
        rate_limit = self.monitoring_cfg.get("alert_rate_limit_minutes", 15)
        self._alert_manager = AlertManager(rate_limit_minutes=rate_limit)
        self._dashboard = Dashboard(refresh_seconds=refresh, paper=paper)
        self._main_log = get_main_logger()
        self._trade_log = get_trade_logger()
        self._regime_log = get_regime_logger()

        # Rolling bar buffer per symbol
        self._bar_buffers: dict[str, object] = {}

        # Regime tracking for change detection
        self._last_regime_label: str = ""
        self._last_retrain_date: Optional[date] = None
        self._recent_signals: list[str] = []    # last 8 signals for dashboard

        # Alert log (shown on dashboard)
        self._alerts: list[str] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        signal_module.signal(signal_module.SIGINT, self._handle_signal)
        signal_module.signal(signal_module.SIGTERM, self._handle_signal)

        try:
            self._startup()
            self._main_loop()
        except Exception as exc:
            logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
            self._alert(f"SYSTEM_ERROR: {exc}")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _startup(self) -> None:
        load_dotenv()
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        # 1. Connect to Alpaca and verify account
        from broker.alpaca_client import AlpacaClient
        self._client = AlpacaClient(paper=self.paper)
        logger.info("Connecting to Alpaca (%s)…", "paper" if self.paper else "LIVE")
        retry(self._client.connect, label="Alpaca.connect")

        acct = retry(self._client.get_account, label="get_account")
        logger.info(
            "Account verified. Equity: $%.2f  Cash: $%.2f",
            float(acct["equity"]), float(acct["cash"]),
        )

        # 2. Check market hours
        clock = retry(self._client.get_clock, label="get_clock")
        if not clock["is_open"]:
            logger.warning(
                "Market is closed. Next open: %s", clock.get("next_open", "?")
            )
            # Continue anyway — we may be in a test/dry-run context

        # 3. Initialize position tracker and sync
        from broker.position_tracker import PositionTracker
        self._position_tracker = PositionTracker(self._client)
        retry(self._position_tracker.sync, label="PositionTracker.sync")

        # 4. Initialize risk manager
        from core.risk_manager import RiskConfig, RiskManager
        risk_cfg = RiskConfig(
            max_risk_per_trade=self.risk_cfg_raw.get("max_risk_per_trade", 0.01),
            max_exposure=self.risk_cfg_raw.get("max_exposure", 0.80),
            max_leverage=self.risk_cfg_raw.get("max_leverage", 1.25),
            max_single_position=self.risk_cfg_raw.get("max_single_position", 0.15),
            max_concurrent=self.risk_cfg_raw.get("max_concurrent", 5),
            max_daily_trades=self.risk_cfg_raw.get("max_daily_trades", 20),
            daily_dd_reduce=self.risk_cfg_raw.get("daily_dd_reduce", 0.02),
            daily_dd_halt=self.risk_cfg_raw.get("daily_dd_halt", 0.03),
            weekly_dd_reduce=self.risk_cfg_raw.get("weekly_dd_reduce", 0.05),
            weekly_dd_halt=self.risk_cfg_raw.get("weekly_dd_halt", 0.07),
            max_dd_from_peak=self.risk_cfg_raw.get("max_dd_from_peak", 0.10),
        )
        self._risk_manager = RiskManager(risk_cfg)
        portfolio = self._position_tracker.get_snapshot()
        self._risk_manager.update_snapshot(portfolio)

        # 5. Load historical bars for HMM training / warmup
        from data.market_data import MarketDataFetcher
        self._data_fetcher = MarketDataFetcher(self._client, use_cache=True)
        logger.info("Loading historical bars for %d symbols…", len(self.symbols))
        bars_by_symbol = self._load_history()
        self._bar_buffers = {sym: bars.copy() for sym, bars in bars_by_symbol.items()}

        # 6. Load or train HMM
        self._hmm_engine = self._load_or_train_hmm(bars_by_symbol)

        # 7. Build strategy orchestrator
        from core.regime_strategies import StrategyOrchestrator
        self._orchestrator = StrategyOrchestrator(
            self.strategy_cfg, self._hmm_engine.get_regime_infos()
        )

        # 8. Build signal generator
        from core.signal_generator import SignalGenerator
        self._signal_generator = SignalGenerator(
            self._hmm_engine, self._orchestrator, self._risk_manager
        )

        # 9. Check for session recovery
        prev_state = SessionState.load()
        if prev_state and prev_state.mode == ("paper" if self.paper else "live"):
            logger.info(
                "Recovering from previous session (started %s)", prev_state.session_start
            )
            self._session_state = prev_state
            # Restore peak/daily/weekly equity from snapshot
            pt = self._position_tracker
            if prev_state.peak_equity > 0:
                pt._peak_equity = prev_state.peak_equity
            if prev_state.daily_start_equity > 0:
                pt._daily_start_equity = prev_state.daily_start_equity
            if prev_state.weekly_start_equity > 0:
                pt._weekly_start_equity = prev_state.weekly_start_equity
        else:
            self._session_state = SessionState.new(
                mode="paper" if self.paper else "live",
                model_path=str(self.model_path),
            )

        # 10. Start bar feed
        self._bar_feed = BarFeed(
            symbols=self.symbols,
            api_key=api_key,
            secret_key=secret_key,
            paper=self.paper,
        )
        self._bar_feed.start()

        mode = "PAPER" if self.paper else "LIVE"
        dry = " [DRY-RUN]" if self.dry_run else ""
        logger.info("=" * 60)
        logger.info("System online — %s mode%s", mode, dry)
        self._main_log.info("System online", extra={"mode": mode, "dry_run": self.dry_run})
        logger.info("Symbols: %s", " ".join(self.symbols))
        logger.info("Timeframe: %s  Model: %s", self.timeframe, self.model_path.name)
        regime = self._signal_generator.get_current_regime()
        if regime:
            logger.info(
                "Current regime: %s (p=%.2f, %s)",
                regime.label, regime.probability, regime.vol_environment
            )
        logger.info("=" * 60)

        # Start Rich live dashboard (graceful fallback if no TTY)
        try:
            self._dashboard.start()
        except Exception as exc:
            logger.debug("Dashboard not started: %s", exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        logger.info("Entering main loop. Ctrl-C to stop.")
        _consecutive_timeouts = 0

        while not self._shutdown_event.is_set():
            bar = self._bar_feed.get(timeout=120.0)

            if bar is None:
                _consecutive_timeouts += 1
                self._housekeeping()
                if _consecutive_timeouts >= 3:
                    from monitoring.alerts import AlertType
                    self._alert_manager.fire(
                        AlertType.DATA_FEED_DOWN,
                        title="Data feed timeout",
                        body=f"No bar received in {_consecutive_timeouts * 120}s",
                    )
                continue

            _consecutive_timeouts = 0
            if self._shutdown_event.is_set():
                break

            symbol = bar.get("symbol", self.primary)
            try:
                self._process_bar(bar, symbol)
            except Exception as exc:
                logger.error("Error processing bar for %s: %s", symbol, exc)
                from monitoring.alerts import AlertType
                self._alert_manager.fire(
                    AlertType.SYSTEM_ERROR,
                    title=f"Bar processing error: {symbol}",
                    body=str(exc),
                    force=True,
                )

    def _process_bar(self, bar: dict, symbol: str) -> None:
        """Process one incoming bar through the full pipeline."""
        from data.feature_engineering import FeatureEngineer

        # 0. Apply any pending control commands from the web dashboard
        self._apply_controls()

        # 1. Append new bar to rolling buffer
        if symbol in self._bar_buffers:
            new_row = {
                "open": bar["open"], "high": bar["high"],
                "low": bar["low"],   "close": bar["close"],
                "volume": bar["volume"],
            }
            import pandas as pd
            ts = pd.to_datetime(bar["timestamp"])
            buf = self._bar_buffers[symbol]
            if ts not in buf.index:
                import numpy as np
                new_df = pd.DataFrame([new_row], index=[ts])
                self._bar_buffers[symbol] = pd.concat([buf, new_df]).tail(HISTORY_BARS)

        # Only process the pipeline when ALL primary-symbol bars arrive
        if symbol != self.primary:
            return

        primary_bars = self._bar_buffers[self.primary]

        # 2. Enrich and build features (causal — uses only data up to this bar)
        try:
            enriched_primary = FeatureEngineer.enrich_bars(primary_bars.copy())
            features = FeatureEngineer.build_hmm_features(primary_bars)
            if features.empty:
                logger.debug("No features yet — waiting for more history.")
                return
        except Exception as exc:
            logger.error("Feature engineering failed: %s — holding regime.", exc)
            return

        # Enriched bars for all symbols (causal window)
        enriched_by_sym = {}
        for sym in self.symbols:
            if sym in self._bar_buffers:
                try:
                    enriched_by_sym[sym] = FeatureEngineer.enrich_bars(
                        self._bar_buffers[sym].copy()
                    )
                except Exception:
                    pass

        if not enriched_by_sym:
            return

        # 3. Get current portfolio snapshot
        try:
            retry(self._position_tracker.sync, label="PositionTracker.sync")
        except Exception as exc:
            logger.warning("Portfolio sync failed: %s — using cached snapshot.", exc)
        portfolio = self._position_tracker.get_snapshot()

        # Update risk manager with fresh snapshot
        self._risk_manager.update_snapshot(portfolio)

        # Circuit breaker check — skip signal generation if halted
        if self._risk_manager.is_halted():
            logger.warning(
                "Risk manager HALTED: %s — no new orders.", self._risk_manager.halt_reason()
            )
            self._update_state(bar)
            return

        # Current weights
        equity = portfolio.equity
        current_weights = {
            sym: portfolio.positions.get(sym, 0.0) / max(equity, 1.0)
            for sym in self.symbols
        }

        # 4–6. HMM predict + orchestrator + risk gate
        approved_signals = self._signal_generator.process_bar(
            symbols=self.symbols,
            bars_by_symbol=enriched_by_sym,
            features=features,
            portfolio=portfolio,
            current_weights=current_weights,
        )

        regime = self._signal_generator.get_current_regime()
        if regime:
            new_label = regime.label
            if new_label != self._last_regime_label:
                logger.info(
                    "REGIME CHANGE: %s → %s (p=%.2f, vol=%s)",
                    self._last_regime_label or "–", new_label,
                    regime.probability, regime.vol_environment,
                )
                from monitoring.alerts import AlertType
                from monitoring.logger import log_regime_change
                self._alert_manager.fire(
                    AlertType.REGIME_CHANGE,
                    title="Regime change",
                    body=f"{self._last_regime_label or '–'} → {new_label} (p={regime.probability:.2f})",
                )
                log_regime_change(
                    self._regime_log,
                    symbol=self.primary,
                    old_regime=self._last_regime_label or "–",
                    new_regime=new_label,
                    confidence=regime.probability,
                    vol_environment=regime.vol_environment,
                    equity=portfolio.equity,
                )
                self._last_regime_label = new_label

        if self._risk_manager.is_halted():
            from monitoring.alerts import AlertType
            self._alert_manager.fire(
                AlertType.CIRCUIT_BREAKER,
                title="Circuit breaker triggered",
                body=self._risk_manager.halt_reason(),
                force=True,
            )

        # 7. Submit orders
        if not self.dry_run:
            self._execute_signals(approved_signals)
        else:
            for ap in approved_signals:
                sig_line = (
                    f"{datetime.now().strftime('%H:%M')} | {ap.signal.symbol} | "
                    f"{ap.signal.direction} {ap.decision.approved_weight:.0%} | "
                    f"{getattr(regime, 'vol_environment', '?')}"
                )
                self._recent_signals.append(sig_line)
                logger.info("[DRY-RUN] %s", sig_line)

        # 8. Structured bar-summary log
        from monitoring.logger import log_bar_summary
        daily_start = self._position_tracker._daily_start_equity
        daily_pnl = portfolio.equity - daily_start if daily_start > 0 else 0.0
        log_bar_summary(
            self._main_log,
            regime=regime.label if regime else "UNKNOWN",
            probability=regime.probability if regime else 0.0,
            equity=portfolio.equity,
            positions={sym: v for sym, v in portfolio.positions.items()},
            daily_pnl=daily_pnl,
        )

        # 9. Dashboard refresh
        try:
            self._dashboard.update(
                regime_states={self.primary: regime} if regime else {},
                position_tracker=self._position_tracker,
                risk_manager=self._risk_manager,
                recent_alerts=self._alert_manager.get_recent(),
                recent_signals=self._recent_signals[-8:],
                system_status={
                    "data_feed_ok": True,
                    "api_ok": True,
                    "api_latency_ms": 0,
                    "hmm_age_days": (
                        (datetime.now() - datetime.fromtimestamp(
                            self.model_path.stat().st_mtime
                        )).days
                        if self.model_path.exists() else 0
                    ),
                },
            )
        except Exception as exc:
            logger.debug("Dashboard update failed: %s", exc)

        # 10. Update state snapshot
        self._update_state(bar, regime)

        # 11. Write web dashboard state
        self._write_web_state(bar, regime)

        # 12. Weekly retraining
        self._maybe_retrain(self._bar_buffers)

    # ------------------------------------------------------------------
    # Web state / control channel
    # ------------------------------------------------------------------

    def _write_web_state(self, bar: Optional[dict] = None, regime=None) -> None:
        """Serialize current engine state to web_state.json for the web dashboard."""
        try:
            portfolio_data = None
            risk_data: dict = {}
            positions: list = []
            config_data: dict = {}

            if self._position_tracker is not None:
                snap = self._position_tracker.get_snapshot()
                portfolio_data = {
                    "equity": snap.equity,
                    "cash": snap.cash,
                    "daily_start": snap.daily_start_equity,
                    "weekly_start": snap.weekly_start_equity,
                    "gross_exposure": snap.gross_exposure,
                    "open_trade_count_today": snap.open_trade_count_today,
                }

                if self._risk_manager is not None:
                    cfg = self._risk_manager.config
                    daily_dd = max(0.0, 1.0 - snap.equity / snap.daily_start_equity) if snap.daily_start_equity > 0 else 0.0
                    weekly_dd = max(0.0, 1.0 - snap.equity / snap.weekly_start_equity) if snap.weekly_start_equity > 0 else 0.0
                    peak_dd = max(0.0, 1.0 - snap.equity / snap.peak_equity) if snap.peak_equity > 0 else 0.0
                    risk_data = {
                        "halted": self._risk_manager.is_halted(),
                        "halt_reason": self._risk_manager.halt_reason() if self._risk_manager.is_halted() else None,
                        "daily_dd": daily_dd,
                        "weekly_dd": weekly_dd,
                        "peak_dd": peak_dd,
                        "daily_dd_halt": cfg.daily_dd_halt,
                        "daily_dd_reduce": cfg.daily_dd_reduce,
                        "weekly_dd_halt": cfg.weekly_dd_halt,
                        "weekly_dd_reduce": cfg.weekly_dd_reduce,
                        "max_dd_from_peak": cfg.max_dd_from_peak,
                    }
                    config_data = {
                        "daily_dd_reduce": cfg.daily_dd_reduce,
                        "daily_dd_halt": cfg.daily_dd_halt,
                        "weekly_dd_reduce": cfg.weekly_dd_reduce,
                        "weekly_dd_halt": cfg.weekly_dd_halt,
                        "max_dd_from_peak": cfg.max_dd_from_peak,
                        "max_exposure": cfg.max_exposure,
                        "max_leverage": cfg.max_leverage,
                        "max_risk_per_trade": cfg.max_risk_per_trade,
                        "max_single_position": cfg.max_single_position,
                        "max_concurrent": cfg.max_concurrent,
                        "max_daily_trades": cfg.max_daily_trades,
                    }

                for sym, rec in self._position_tracker.get_positions().items():
                    positions.append({
                        "symbol": sym,
                        "qty": rec.qty,
                        "market_value": rec.market_value,
                        "unrealized_pnl": rec.unrealized_pnl,
                        "unrealized_pnl_pct": rec.unrealized_pnl_pct,
                    })

            regime_data = None
            if regime is not None:
                regime_data = {
                    "label": regime.label,
                    "probability": regime.probability,
                    "vol_environment": getattr(regime, "vol_environment", "mid"),
                    "consecutive_bars": getattr(regime, "consecutive_bars", 0),
                    "is_confirmed": getattr(regime, "is_confirmed", False),
                }

            hmm_age = 0
            if self.model_path.exists():
                hmm_age = (datetime.now() - datetime.fromtimestamp(self.model_path.stat().st_mtime)).days

            recent_orders: list = []
            if self._client is not None:
                try:
                    recent_orders = self._client.list_recent_orders(limit=20)
                except Exception:
                    pass

            state = {
                "regime": regime_data,
                "portfolio": portfolio_data,
                "risk": risk_data,
                "positions": positions,
                "recent_orders": recent_orders,
                "recent_signals": self._recent_signals[-12:],
                "recent_alerts": self._alert_manager.get_recent() if self._alert_manager else [],
                "system": {
                    "mode": "paper" if self.paper else "live",
                    "hmm_model": self.model_path.name,
                    "hmm_age_days": hmm_age,
                    "last_bar": bar.get("timestamp") if bar else None,
                    "dry_run": self.dry_run,
                    "session_start": self._session_state.session_start if self._session_state else None,
                },
                "config": config_data,
            }
            WEB_STATE_PATH.write_text(json.dumps(state))
        except Exception as exc:
            logger.debug("Failed to write web state: %s", exc)

    def _apply_controls(self) -> None:
        """Read and execute commands from control.json (written by the web dashboard)."""
        if not CONTROL_PATH.exists():
            return
        try:
            raw = CONTROL_PATH.read_text()
            CONTROL_PATH.unlink()
            commands = json.loads(raw)
        except Exception as exc:
            logger.error("Failed to read control.json: %s", exc)
            return

        if isinstance(commands, dict):
            commands = [commands]

        for cmd in commands:
            action = cmd.get("action", "")
            try:
                if action == "halt":
                    self._risk_manager.manual_halt("Manual halt via web dashboard")
                    logger.info("Web dashboard: manual halt activated.")
                elif action == "resume":
                    self._risk_manager.clear_halt()
                    logger.info("Web dashboard: halt cleared, trading resumed.")
                elif action == "retrain":
                    logger.info("Web dashboard: force retrain requested.")
                    self._hmm_engine = self._train_hmm(self._bar_buffers)
                    from core.regime_strategies import StrategyOrchestrator
                    from core.signal_generator import SignalGenerator
                    self._orchestrator = StrategyOrchestrator(
                        self.strategy_cfg, self._hmm_engine.get_regime_infos()
                    )
                    self._signal_generator = SignalGenerator(
                        self._hmm_engine, self._orchestrator, self._risk_manager
                    )
                    logger.info("Web dashboard: retrain complete.")
                elif action == "set_dry_run":
                    self.dry_run = bool(cmd.get("value", False))
                    logger.info("Web dashboard: dry_run set to %s.", self.dry_run)
                elif action == "set_risk":
                    params = cmd.get("params", {})
                    cfg = self._risk_manager.config
                    for k, v in params.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, float(v))
                            logger.info("Web dashboard: risk.%s = %.4f", k, float(v))
                elif action == "close_position":
                    symbol = cmd.get("symbol")
                    if symbol and self._client:
                        self._client.close_position(symbol)
                        logger.info("Web dashboard: closed position %s.", symbol)
                else:
                    logger.warning("Unknown control action: %s", action)
            except Exception as exc:
                logger.error("Control action '%s' failed: %s", action, exc)

    def _execute_signals(self, approved_signals) -> None:
        from broker.order_executor import OrderExecutor
        from monitoring.alerts import AlertType
        from monitoring.logger import log_order_event
        executor = OrderExecutor(self._client, paper=self.paper)

        for ap in approved_signals:
            try:
                record = retry(
                    lambda: executor.execute_signal(ap),
                    label=f"execute_signal({ap.signal.symbol})",
                )
                if record is not None:
                    self._position_tracker.record_trade(ap.signal.symbol)
                    logger.info(
                        "Order submitted: %s %s %d shares (regime=%s)",
                        record.side.upper(), record.symbol, record.qty,
                        ap.regime_state.label,
                    )
                    log_order_event(
                        self._trade_log, "SUBMITTED",
                        order_id=record.order_id,
                        symbol=record.symbol,
                        side=record.side,
                        qty=record.qty,
                        price=ap.current_price,
                        regime=ap.regime_state.label,
                    )
                    sig_line = (
                        f"{datetime.now().strftime('%H:%M')} | {record.symbol} | "
                        f"{record.side.upper()} {record.qty}sh | "
                        f"{ap.regime_state.vol_environment}"
                    )
                    self._recent_signals.append(sig_line)
            except Exception as exc:
                logger.error(
                    "Order execution failed for %s: %s", ap.signal.symbol, exc
                )
                self._alert_manager.fire(
                    AlertType.ORDER_FAILURE,
                    title=f"Order failed: {ap.signal.symbol}",
                    body=str(exc),
                )

    # ------------------------------------------------------------------
    # HMM management
    # ------------------------------------------------------------------

    def _load_history(self) -> dict:
        """Load HISTORY_BARS of daily bars from Alpaca (or cache)."""
        from datetime import timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        # Load enough bars to cover min_train_bars + feature warmup
        start_date = (
            datetime.now() - timedelta(days=int(HISTORY_BARS * 1.5))
        ).strftime("%Y-%m-%d")

        bars_by_symbol = {}
        for sym in self.symbols:
            try:
                df = retry(
                    lambda s=sym: self._data_fetcher.get_bars(
                        s, self.timeframe, start_date, end_date
                    ),
                    label=f"get_bars({sym})",
                )
                if not df.empty:
                    bars_by_symbol[sym] = df.tail(HISTORY_BARS)
                    logger.info(
                        "Loaded %d bars for %s (%s → %s)",
                        len(bars_by_symbol[sym]), sym,
                        df.index[0].date(), df.index[-1].date(),
                    )
            except Exception as exc:
                logger.error("Could not load history for %s: %s", sym, exc)

        if not bars_by_symbol:
            raise RuntimeError("No historical data loaded. Check Alpaca credentials.")
        return bars_by_symbol

    def _load_or_train_hmm(self, bars_by_symbol: dict) -> object:
        from core.hmm_engine import HMMEngine
        from data.feature_engineering import FeatureEngineer

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Check if saved model is fresh enough
        if self.model_path.exists():
            age_days = (
                datetime.now() - datetime.fromtimestamp(self.model_path.stat().st_mtime)
            ).days
            if age_days <= MODEL_MAX_AGE_DAYS:
                try:
                    engine = HMMEngine.load(self.model_path)
                    logger.info(
                        "Loaded HMM model from %s (age: %d days, n_states: %d)",
                        self.model_path.name, age_days, engine._n_states,
                    )
                    # Run predict_sequence so stability state is initialized
                    features = FeatureEngineer.build_hmm_features(
                        bars_by_symbol[self.primary]
                    )
                    if not features.empty:
                        engine.predict_sequence(features)
                    return engine
                except Exception as exc:
                    logger.warning("Failed to load model: %s — retraining.", exc)

        return self._train_hmm(bars_by_symbol)

    def _train_hmm(self, bars_by_symbol: dict) -> object:
        from core.hmm_engine import HMMEngine
        from data.feature_engineering import FeatureEngineer

        logger.info("Training HMM on %s (%s bars)…", self.primary,
                    len(bars_by_symbol.get(self.primary, [])))
        features = FeatureEngineer.build_hmm_features(bars_by_symbol[self.primary])

        if len(features) < self.hmm_config.min_train_bars:
            raise RuntimeError(
                f"Need at least {self.hmm_config.min_train_bars} feature rows to train HMM, "
                f"got {len(features)}."
            )

        engine = HMMEngine(self.hmm_config)
        engine.fit(features)
        engine.save(self.model_path)
        self._last_retrain_date = date.today()

        if self._session_state:
            self._session_state.last_trained = datetime.now(timezone.utc).isoformat()

        logger.info(
            "HMM trained: %d states, model saved → %s",
            engine._n_states, self.model_path,
        )
        return engine

    def _maybe_retrain(self, bars_by_symbol: dict) -> None:
        """Retrain HMM weekly."""
        today = date.today()
        if self._last_retrain_date is None:
            self._last_retrain_date = today
            return
        if (today - self._last_retrain_date).days >= 7:
            logger.info("Weekly retraining triggered.")
            try:
                self._hmm_engine = self._train_hmm(bars_by_symbol)
                from core.regime_strategies import StrategyOrchestrator
                from core.signal_generator import SignalGenerator
                from monitoring.alerts import AlertType
                self._orchestrator = StrategyOrchestrator(
                    self.strategy_cfg, self._hmm_engine.get_regime_infos()
                )
                self._signal_generator = SignalGenerator(
                    self._hmm_engine, self._orchestrator, self._risk_manager
                )
                self._alert_manager.fire(
                    AlertType.HMM_RETRAINED,
                    title="HMM retrained",
                    body=f"{self._hmm_engine._n_states} states, model={self.model_path.name}",
                )
            except Exception as exc:
                logger.error("Weekly retrain failed: %s — keeping current model.", exc)

    # ------------------------------------------------------------------
    # Housekeeping (called on bar-feed timeout)
    # ------------------------------------------------------------------

    def _housekeeping(self) -> None:
        """Periodic tasks: daily counter reset, sync portfolio."""
        self._apply_controls()
        self._write_web_state()
        now = datetime.now()
        # Reset daily counters at 9:30 AM ET on weekdays
        if now.weekday() < 5 and now.hour == 9 and now.minute == 30:
            self._position_tracker.reset_daily_counters()
        # Reset weekly counters on Monday
        if now.weekday() == 0 and now.hour == 9 and now.minute == 30:
            self._position_tracker.reset_weekly_counters()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _update_state(self, bar: dict, regime=None) -> None:
        if self._session_state is None:
            return
        self._session_state.last_processed_bar = bar.get("timestamp")
        if regime:
            self._session_state.last_regime[bar.get("symbol", self.primary)] = regime.label
        pt = self._position_tracker
        self._session_state.peak_equity = pt._peak_equity
        self._session_state.daily_start_equity = pt._daily_start_equity
        self._session_state.weekly_start_equity = pt._weekly_start_equity
        self._session_state.daily_trade_count = pt._daily_trade_count
        self._session_state.save()

    def _alert(self, msg: str) -> None:
        """Legacy helper — kept for data-feed / system-error callsites."""
        from monitoring.alerts import AlertType
        self._alert_manager.fire(AlertType.SYSTEM_ERROR, title="System alert", body=msg)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Received signal %d — shutting down.", signum)
        self._shutdown_event.set()

    def _shutdown(self) -> None:
        logger.info("Shutting down …")

        # Stop dashboard before printing anything to stdout
        try:
            self._dashboard.stop()
        except Exception:
            pass

        if self._bar_feed:
            self._bar_feed.stop()

        if self._session_state:
            self._session_state.save()
            logger.info("State saved → %s", STATE_SNAPSHOT_PATH)

        self._main_log.info("System shutdown")

        # Print session summary
        self._print_summary()

        logger.info("Shutdown complete. Positions remain open (stops in place).")

    def _print_summary(self) -> None:
        if self._session_state is None:
            return
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print(f"  Mode:          {'PAPER' if self.paper else 'LIVE'}")
        print(f"  Started:       {self._session_state.session_start}")
        print(f"  Stopped:       {datetime.now(timezone.utc).isoformat()}")
        if self._position_tracker:
            snapshot = self._position_tracker.get_snapshot()
            print(f"  Final equity:  ${snapshot.equity:,.2f}")
            if snapshot.daily_start_equity > 0:
                daily_ret = (snapshot.equity / snapshot.daily_start_equity - 1) * 100
                print(f"  Daily return:  {daily_ret:+.2f}%")
            print(f"  Trades today:  {snapshot.open_trade_count_today}")
            print(f"  Open positions: {len(snapshot.positions)}")
        regime = (
            self._signal_generator.get_current_regime()
            if self._signal_generator else None
        )
        if regime:
            print(f"  Last regime:   {regime.label} (p={regime.probability:.2f})")
        print("=" * 60)


# ------------------------------------------------------------------ #
# Train-only helper                                                    #
# ------------------------------------------------------------------ #

def run_train_only(cfg: dict, paper: bool) -> None:
    """Load data from Alpaca, train HMM, save model, and exit."""
    load_dotenv()
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set.", file=sys.stderr)
        sys.exit(1)

    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketDataFetcher
    from core.hmm_engine import HMMConfig, HMMEngine
    from data.feature_engineering import FeatureEngineer
    from datetime import timedelta

    broker_cfg = cfg.get("broker", {})
    hmm_cfg_raw = cfg.get("hmm", {})
    symbols: list[str] = broker_cfg.get("symbols", ["SPY"])
    primary = symbols[0]
    timeframe = broker_cfg.get("timeframe", "1Day")

    hmm_config = HMMConfig(
        n_candidates=hmm_cfg_raw.get("n_candidates", [3, 4, 5, 6, 7]),
        n_init=hmm_cfg_raw.get("n_init", 10),
        covariance_type=hmm_cfg_raw.get("covariance_type", "full"),
        min_train_bars=hmm_cfg_raw.get("min_train_bars", 504),
    )

    client = AlpacaClient(paper=paper)
    client.connect()
    fetcher = MarketDataFetcher(client, use_cache=True)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=900)).strftime("%Y-%m-%d")

    df = fetcher.get_bars(primary, timeframe, start_date, end_date)
    if df.empty:
        print(f"ERROR: No data for {primary}.", file=sys.stderr)
        sys.exit(1)

    features = FeatureEngineer.build_hmm_features(df)
    logger.info("Training HMM on %d feature rows…", len(features))

    engine = HMMEngine(hmm_config)
    engine.fit(features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"hmm_{primary}.pkl"
    engine.save(model_path)
    logger.info("HMM trained: %d states → %s", engine._n_states, model_path)


# ------------------------------------------------------------------ #
# Dashboard command (reads state_snapshot.json)                        #
# ------------------------------------------------------------------ #

def run_dashboard() -> None:
    """Read state_snapshot.json and show a static summary dashboard."""
    state = SessionState.load()
    if state is None:
        print("No state_snapshot.json found. Is a trading session running?")
        return

    print("\n" + "=" * 60)
    print("REGIME-TRADER  STATUS")
    print(f"  Session started: {state.session_start}")
    print(f"  Mode:            {state.mode.upper()}")
    print(f"  Last bar:        {state.last_processed_bar or 'none'}")
    print(f"  Last trained:    {state.last_trained}")
    print(f"  Peak equity:     ${state.peak_equity:,.2f}")
    print(f"  Daily start:     ${state.daily_start_equity:,.2f}")
    print(f"  Trades today:    {state.daily_trade_count}")
    if state.last_regime:
        print("  Last regimes:")
        for sym, label in state.last_regime.items():
            print(f"    {sym}: {label}")
    print("=" * 60)


# ------------------------------------------------------------------ #
# Backtest helpers (unchanged from Phase 4)                            #
# ------------------------------------------------------------------ #

def fetch_bars(symbols: list[str], start: str, end: str | None) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        print(
            "ERROR: yfinance is not installed.\n"
            "  pip install yfinance",
            file=sys.stderr,
        )
        sys.exit(1)

    bars_by_symbol = {}
    for sym in symbols:
        logger.info("Downloading %s from %s to %s …", sym, start, end or "today")
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                logger.warning("No data returned for %s — skipping.", sym)
                continue
            # yfinance ≥0.2 returns a MultiIndex (metric, ticker); flatten to metric only
            if hasattr(df.columns, "levels"):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            bars_by_symbol[sym] = df
            logger.info(
                "  %s: %d bars (%s → %s)", sym, len(df),
                df.index[0].date(), df.index[-1].date()
            )
        except Exception as exc:
            logger.error("Failed to download %s: %s", sym, exc)

    if not bars_by_symbol:
        print("ERROR: No data downloaded. Check symbols and date range.", file=sys.stderr)
        sys.exit(1)

    return bars_by_symbol


def run_backtest(args: argparse.Namespace) -> None:
    from backtest.backtester import BacktestConfig, WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer
    from backtest.stress_test import StressTester
    from core.hmm_engine import HMMConfig

    cfg = load_settings()
    hmm_cfg_raw = cfg.get("hmm", {})
    bt_cfg_raw = cfg.get("backtest", {})
    strategy_cfg = cfg.get("strategy", {})
    strategy_cfg.update(hmm_cfg_raw)

    hmm_config = HMMConfig(
        n_candidates=hmm_cfg_raw.get("n_candidates", [3, 4, 5, 6, 7]),
        n_init=hmm_cfg_raw.get("n_init", 10),
        covariance_type=hmm_cfg_raw.get("covariance_type", "full"),
        min_train_bars=hmm_cfg_raw.get("min_train_bars", 504),
        stability_bars=hmm_cfg_raw.get("stability_bars", 3),
        flicker_window=hmm_cfg_raw.get("flicker_window", 20),
        flicker_threshold=hmm_cfg_raw.get("flicker_threshold", 4),
        min_confidence=hmm_cfg_raw.get("min_confidence", 0.55),
    )
    bt_config = BacktestConfig(
        slippage_pct=bt_cfg_raw.get("slippage_pct", 0.0005),
        initial_capital=bt_cfg_raw.get("initial_capital", 100_000),
        train_window=bt_cfg_raw.get("train_window", 252),
        test_window=bt_cfg_raw.get("test_window", 126),
        step_size=bt_cfg_raw.get("step_size", 126),
        risk_free_rate=bt_cfg_raw.get("risk_free_rate", 0.045),
        rebalance_threshold=strategy_cfg.get("rebalance_threshold", 0.10),
        min_confidence=hmm_cfg_raw.get("min_confidence", 0.55),
    )

    bars_by_symbol = fetch_bars(args.symbols, args.start, args.end)

    backtester = WalkForwardBacktester(hmm_config, bt_config, strategy_cfg)
    logger.info("Starting walk-forward backtest…")
    result = backtester.run(bars_by_symbol, output_dir=args.output)

    if result.full_equity.empty:
        print("Backtest produced no equity data.", file=sys.stderr)
        sys.exit(1)

    analyzer = PerformanceAnalyzer(risk_free_rate=bt_config.risk_free_rate)
    benchmark_prices = None
    if args.compare:
        primary = args.symbols[0]
        benchmark_prices = {
            "buy_and_hold": bars_by_symbol[primary]["close"],
            "sma200": bars_by_symbol[primary]["close"],
        }

    report = analyzer.analyze(
        equity_curve=result.full_equity,
        trades=result.all_trades,
        regime_history=result.full_regime,
        benchmark_prices=benchmark_prices,
    )
    analyzer.print_report(report)
    analyzer.to_csv(report, result.full_equity, args.output)

    if args.stress_test:
        logger.info("Running stress tests (%d sims)…", args.n_stress_sims)
        tester = StressTester(backtester)
        tester.run_crash_injection(bars_by_symbol, n_simulations=args.n_stress_sims)
        tester.run_gap_risk(bars_by_symbol)
        tester.run_misclassification(bars_by_symbol)

    logger.info("Done. Results written to %s/", args.output)


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="regime-trader",
        description="HMM-based market regime detection and allocation system.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- backtest ----
    bt = sub.add_parser("backtest", help="Run walk-forward backtest (yfinance data)")
    bt.add_argument("--symbols", nargs="+", default=["SPY"])
    bt.add_argument("--start", default="2019-01-01")
    bt.add_argument("--end", default=None)
    bt.add_argument("--output", type=Path, default=Path("results/"))
    bt.add_argument("--compare", action="store_true",
                    help="Add buy-and-hold / 200-SMA benchmark comparisons")
    bt.add_argument("--stress-test", action="store_true",
                    help="Run Monte Carlo stress tests after backtest")
    bt.add_argument("--n-stress-sims", type=int, default=100)

    # ---- paper ----
    paper = sub.add_parser("paper", help="Paper-trading live loop (no real capital)")
    paper.add_argument("--dry-run", action="store_true",
                       help="Full pipeline but no orders submitted")
    paper.add_argument("--train-only", action="store_true",
                       help="Train HMM on Alpaca data and exit without trading")

    # ---- live ----
    live_p = sub.add_parser("live", help="Live trading with real capital")
    live_p.add_argument("--dry-run", action="store_true",
                        help="Full pipeline but no orders submitted")
    live_p.add_argument("--train-only", action="store_true",
                        help="Train HMM on Alpaca data and exit without trading")

    # ---- dashboard ----
    sub.add_parser("dashboard", help="Show status dashboard for a running instance")

    # ---- webdashboard ----
    wd = sub.add_parser("webdashboard", help="Start browser-based dashboard with controls")
    wd.add_argument("--port", type=int, default=8080)

    return parser.parse_args()


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()

    if args.command == "backtest":
        run_backtest(args)

    elif args.command == "dashboard":
        run_dashboard()

    elif args.command == "webdashboard":
        from monitoring.web_dashboard import run as run_web
        run_web(port=getattr(args, "port", 8080))

    elif args.command in ("paper", "live"):
        cfg = load_settings()
        paper = args.command == "paper"
        train_only = getattr(args, "train_only", False)
        dry_run = getattr(args, "dry_run", False)

        if train_only:
            run_train_only(cfg, paper=paper)
        else:
            engine = TradingEngine(cfg, paper=paper, dry_run=dry_run)
            engine.run()

    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
