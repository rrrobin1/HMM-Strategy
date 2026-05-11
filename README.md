# regime-trader

HMM-based market regime detection and adaptive allocation system for US equities, using Alpaca as broker.

> **Philosophy:** Risk management is more important than signal generation. The system is designed to survive drawdowns first, then compound gains.

---

## Architecture

```
Market Data (Alpaca REST/WebSocket)
        │
        ▼
Feature Engineering
  (returns, vol, momentum, vol_z, drawdown)
        │
        ▼
HMM Engine (hmmlearn GaussianHMM, BIC selection)
  ┌─────┴──────┐
  │  Vol-rank  │   sort states by expected_vol → LOW / MID / HIGH
  └─────┬──────┘
        │
        ▼
Strategy Orchestrator
  ├── LowVolBull     → 95% allocated, 1.25× leverage
  ├── MidVolCautious → 60–95% depending on trend
  └── HighVolDefensive → 60%, wait for stability
        │
        ▼
Risk Manager (9-check pipeline)
  halt → daily DD → weekly DD → peak DD → trade count
  → exposure ceiling → single-position cap → concurrent cap
  → stop-loss sizing
        │
        ▼
Order Executor (Alpaca market orders, delta-based sizing)
        │
        ▼
Position Tracker + Session State (crash recovery)
        │
        ▼
Monitoring: Rich dashboard, JSON logs, email/webhook alerts
```

**Key design decisions:**

- **Forward algorithm, not Viterbi** — causal inference only; no future bars leak into today's regime.
- **BIC model selection** — number of HMM states (2–5) chosen automatically each training run.
- **Always LONG, never SHORT** — high-vol response is allocation reduction, not reversal.
- **Rebalance threshold** — only trade when |target − current weight| > 10% to limit turnover.
- **Fill delay** — signal at bar *t* executes at bar *t+1* open, matching live execution reality.

---

## Quick Start

```bash
# 1. Clone and create virtual environment
git clone <repo> regime-trader && cd regime-trader
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Alpaca credentials
cp .env.example .env
# Edit .env: set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

# 4. Review config
vi config/settings.yaml          # symbols, timeframe, risk params

# 5. Run backtest to validate strategy on historical data
python main.py backtest

# 6. Start paper trading
python main.py paper
```

---

## CLI Reference

| Command | Description |
|---|---|
| `python main.py backtest` | Walk-forward backtest on configured symbols |
| `python main.py paper` | Paper trading loop (Alpaca paper endpoint) |
| `python main.py live` | Live trading (real money — paper first!) |
| `python main.py dashboard` | Attach Rich dashboard to a running session |
| `python main.py paper --dry-run` | Run pipeline but submit no orders |
| `python main.py paper --train-only` | Train HMM and exit without trading |
| `python main.py live --dry-run` | Live data, no order submission |

---

## Configuration Guide

`config/settings.yaml` (annotated):

```yaml
# Symbols to trade (primary symbol drives HMM training)
symbols: ["SPY"]

# Bar timeframe fed to HMM features
timeframe: "1Day"

# HMM configuration
hmm:
  n_candidates: [2, 3, 4]      # state counts evaluated by BIC
  n_init: 10                   # EM restarts per candidate
  min_train_bars: 252          # minimum history to train
  stability_bars: 3            # consecutive bars to confirm regime
  flicker_window: 20           # lookback for flicker detection
  flicker_threshold: 4         # max regime changes in window
  min_confidence: 0.55         # p threshold; below → uncertainty mode

# Strategy allocation weights
strategy:
  low_vol_allocation: 0.95
  low_vol_leverage: 1.25
  mid_vol_allocation_trend: 0.95
  mid_vol_allocation_no_trend: 0.60
  high_vol_allocation: 0.60
  rebalance_threshold: 0.10    # minimum weight change to trigger order
  uncertainty_size_mult: 0.50  # scale factor when flickering or p < min_confidence

# Risk circuit breakers
risk:
  max_daily_trades: 10         # halt new trades after N per day
  max_concurrent: 5            # max simultaneous open positions
  max_single_position: 0.40    # largest single position (weight)
  max_exposure: 1.00           # gross exposure ceiling
  max_risk_per_trade: 0.01     # max equity at risk per trade (stop-loss sizing)
  daily_dd_reduce: 0.02        # daily drawdown → reduce sizes
  daily_dd_halt: 0.03          # daily drawdown → halt all trading
  weekly_dd_reduce: 0.04       # weekly drawdown → reduce sizes
  weekly_dd_halt: 0.06         # weekly drawdown → halt all trading
  max_dd_from_peak: 0.10       # peak drawdown → permanent halt (session)
```

---

## Project Layout

```
regime-trader/
├── main.py                    # TradingEngine, BarFeed, SessionState, CLI
├── config/
│   └── settings.yaml          # all tunable parameters
├── core/
│   ├── hmm_engine.py          # HMM training, predict, regime stability
│   ├── regime_strategies.py   # Signal dataclass, strategy classes, orchestrator
│   ├── risk_manager.py        # RiskManager, RiskConfig, circuit breakers
│   └── signal_generator.py    # SignalGenerator.process_bar()
├── broker/
│   ├── alpaca_client.py       # Alpaca REST wrapper
│   ├── order_executor.py      # OrderExecutor, OrderRecord, sync
│   └── position_tracker.py    # PositionTracker, PortfolioSnapshot
├── data/
│   ├── feature_engineering.py # FeatureEngineer — HMM features + enriched bars
│   └── market_data.py         # MarketDataFetcher with Parquet cache
├── backtest/
│   └── backtester.py          # WalkForwardBacktester, BacktestResult, metrics
├── monitoring/
│   ├── logger.py              # JSONFormatter, rotating log files
│   ├── dashboard.py           # Rich Live 6-panel terminal dashboard
│   └── alerts.py              # AlertManager, 9 alert types, email/webhook
├── tests/
│   ├── test_hmm.py
│   ├── test_strategies.py
│   ├── test_risk.py
│   ├── test_orders.py
│   ├── test_integration.py    # end-to-end pipeline + look-ahead bias
│   ├── test_risk_stress.py    # extreme signals, rapid-fire, drawdown compound
│   ├── test_alpaca_paper.py   # mocked Alpaca paper trading
│   └── test_recovery.py       # session state save/load, crash recovery
├── logs/                      # JSON log files (git-ignored)
├── models/                    # saved HMM .pkl files (git-ignored)
└── data/cache/                # Parquet bar cache (git-ignored)
```

---

## Running the Tests

```bash
# Full suite
pytest

# Fast unit tests only (no HMM training)
pytest tests/test_risk.py tests/test_orders.py tests/test_recovery.py

# Integration + stress (trains HMM — takes ~30 s)
pytest tests/test_integration.py tests/test_risk_stress.py

# Verbose with coverage
pytest --tb=short -q
```

---

## FAQ

**Why forward algorithm instead of Viterbi?**
Viterbi finds the globally optimal state sequence using all bars including future ones. The forward algorithm computes `p(state | data up to t)` for each bar causally — identical to how the system will behave live. Using Viterbi in backtesting would introduce look-ahead bias.

**How does the system pick the number of HMM states?**
`HMMEngine` trains candidate models with 2–5 states (configurable), each with multiple random restarts, and selects the model with the lowest Bayesian Information Criterion (BIC). BIC penalises model complexity, preventing overfitting to noise.

**Why are some trades rejected?**
`RiskManager.validate()` runs 9 ordered checks. Common reasons: daily drawdown exceeded the halt threshold; gross exposure ceiling reached; daily trade count limit hit; stop-loss sizing constrains the position below 1 share. Check `logs/main.log` for the `reason` field on each blocked signal.

**How do I switch from paper to live trading?**
Change `ALPACA_BASE_URL` in `.env` from the paper endpoint to `https://api.alpaca.markets`. Run `python main.py live`. The session state is mode-tagged — a paper snapshot will not be restored into a live session (mode mismatch check).

**How often does the HMM retrain?**
Every 7 calendar days (configurable). The `TradingEngine._maybe_retrain()` method compares `last_trained` in `SessionState` to the current date and fires `HMM_RETRAINED` alert on completion.

**What happens on a crash?**
`SessionState` is saved to `session_state.json` after every processed bar. On restart, `TradingEngine._startup()` loads the snapshot, restores equity watermarks into `PositionTracker`, and the order executor reconciles positions via `get_position()` delta logic — it will only submit orders for the shares that are missing or excess relative to target weights.

**The system halted and won't trade — what do I do?**
A halt (`rm.is_halted()`) means a drawdown circuit breaker fired (daily, weekly, or peak). It persists for the session. Options: (1) let the session end and restart tomorrow (daily counters reset); (2) inspect `rm._halt_reason` in the logs; (3) if the halt was erroneous, restart the process after fixing the underlying condition.

---

## Disclaimer

This software is for **educational purposes only**. It does not constitute financial advice. Past backtested performance does not guarantee future results. Markets are non-stationary — a regime model trained on historical data may fail without warning. **Always paper trade for at least 30 days before considering live deployment.** Never risk capital you cannot afford to lose.
