"""
monitoring/web_dashboard.py — Browser-based dashboard with live controls.

Reads web_state.json (written by TradingEngine on every bar) and serves a
live-updating HTML page at http://localhost:8080.  Control commands (halt,
resume, retrain, set_risk, close_position, toggle dry-run) are written to
control.json which TradingEngine reads at the start of every bar.

Run via:  python3 main.py webdashboard [--port 8080]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WEB_STATE_PATH = Path("web_state.json")
CONTROL_PATH = Path("control.json")

_shared_client = None


def _get_alpaca_client():
    """Return a cached AlpacaClient, creating one if needed."""
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    import os, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    paper = True
    settings_path = Path("config/settings.yaml")
    if settings_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(settings_path.read_text()) or {}
            paper = cfg.get("broker", {}).get("paper_trading", True)
        except Exception:
            pass
    from broker.alpaca_client import AlpacaClient
    _shared_client = AlpacaClient(paper=paper)
    _shared_client.connect()
    return _shared_client

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Robin Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #f1f5f9;
  --surface: #ffffff;
  --surface2: #f8fafc;
  --border: #e2e8f0;
  --border2: #cbd5e1;
  --text: #1e293b;
  --muted: #94a3b8;
  --green: #16a34a; --green-dim: #dcfce7; --green-border: #86efac;
  --yellow: #d97706; --yellow-dim: #fef9c3; --yellow-border: #fde68a;
  --red: #dc2626;   --red-dim: #fee2e2;   --red-border: #fca5a5;
  --blue: #2563eb;  --blue-dim: #dbeafe;  --blue-border: #93c5fd;
  --shadow: 0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
}
body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; font-size: 13px; line-height: 1.5; min-height: 100vh; }

/* ── HEADER ─────────────────────────────────────────── */
header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 13px 24px; display: flex; align-items: center; justify-content: space-between; box-shadow: var(--shadow); }
header h1 { font-size: 15px; font-weight: 700; letter-spacing: -0.01em; color: var(--text); }
.badge { padding: 3px 11px; border-radius: 20px; font-size: 11px; font-weight: 600; letter-spacing: 0.03em; }
.badge-paper { background: var(--blue-dim); color: #1d4ed8; }
.badge-live  { background: var(--red-dim);  color: #b91c1c; }
.badge-offline { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }
.status-row { display: flex; align-items: center; gap: 12px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
.dot.offline { background: var(--muted); animation: none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.last-update { color: var(--muted); font-size: 11px; }

/* ── CONTROLS BAR ────────────────────────────────────── */
.controls-bar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 10px 24px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ctrl-group { display: flex; gap: 6px; align-items: center; }
.ctrl-sep { width: 1px; height: 24px; background: var(--border2); margin: 0 6px; }
.ctrl-btn { padding: 6px 14px; border-radius: 7px; border: 1px solid; font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s; white-space: nowrap; }
.ctrl-btn:hover { filter: brightness(0.93); }
.ctrl-btn:active { transform: scale(0.97); }
.ctrl-btn.danger  { background: var(--red-dim);    color: #b91c1c; border-color: var(--red-border); }
.ctrl-btn.success { background: var(--green-dim);  color: #15803d; border-color: var(--green-border); }
.ctrl-btn.warning { background: var(--yellow-dim); color: #b45309; border-color: var(--yellow-border); }
.ctrl-btn.info    { background: var(--blue-dim);   color: #1d4ed8; border-color: var(--blue-border); }
.ctrl-btn.active  { filter: brightness(0.88); }
.ctrl-btn:disabled { opacity: 0.35; cursor: not-allowed; }
.ctrl-status { margin-left: auto; font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 20px; }
.ctrl-status.ok    { color: #15803d; background: var(--green-dim); }
.ctrl-status.error { color: #b91c1c; background: var(--red-dim); }

/* ── PAGE BODY + SIDEBAR LAYOUT ─────────────────────── */
.page-body { display: flex; padding: 18px 24px; gap: 16px; align-items: flex-start; }
main { flex: 1; display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; min-width: 0; }
.sidebar { width: 380px; flex-shrink: 0; display: flex; flex-direction: column; gap: 14px; position: sticky; top: 14px; }
@media(max-width:1280px){ .sidebar { width: 320px; } }
@media(max-width:1060px){ .page-body { flex-direction: column; } .sidebar { width: 100%; position: static; } main { grid-template-columns: 1fr 1fr; } }
@media(max-width:600px){ main { grid-template-columns: 1fr; } }

/* ── CHART ───────────────────────────────────────────── */
#chart-container { width: 100%; height: 310px; }
.chart-sym-input { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 3px 8px; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; color: var(--text); width: 90px; text-transform: uppercase; outline: none; }
.chart-sym-input:focus { border-color: var(--blue-border); box-shadow: 0 0 0 2px #dbeafe; }

/* ── QUICK ORDER ─────────────────────────────────────── */
.order-inputs { display: flex; gap: 8px; margin-bottom: 12px; }
.order-input { padding: 8px 10px; border: 1px solid var(--border); border-radius: 7px; font-family: 'Inter', sans-serif; font-size: 13px; color: var(--text); background: var(--surface2); outline: none; width: 100%; }
.order-input:focus { border-color: var(--blue-border); box-shadow: 0 0 0 2px #dbeafe; }
.order-btns { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.order-btn { padding: 11px 8px; border-radius: 7px; border: 1px solid; font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.15s; letter-spacing: 0.01em; }
.order-btn:hover { filter: brightness(0.93); }
.order-btn:active { transform: scale(0.97); }
.buy-market  { background: #dcfce7; color: #15803d; border-color: #86efac; }
.buy-ask     { background: #d1fae5; color: #065f46; border-color: #6ee7b7; }
.sell-market { background: #fee2e2; color: #b91c1c; border-color: #fca5a5; }
.sell-ask    { background: #fce7f3; color: #9d174d; border-color: #f9a8d4; }
.order-feedback { margin-top: 10px; padding: 8px 12px; border-radius: 7px; font-size: 12px; font-weight: 500; }
.order-feedback.ok    { background: var(--green-dim); color: #15803d; }
.order-feedback.error { background: var(--red-dim);   color: #b91c1c; }

/* ── CARDS ───────────────────────────────────────────── */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; box-shadow: var(--shadow); }
.card-full { grid-column: 1 / -1; }
.card-header { padding: 10px 16px; border-bottom: 1px solid var(--border); font-size: 11px; font-weight: 600; letter-spacing: 0.07em; color: var(--muted); text-transform: uppercase; display: flex; justify-content: space-between; align-items: center; background: var(--surface2); }
.card-body { padding: 14px 16px; }

/* ── STAT ROWS ───────────────────────────────────────── */
.stat-row { display: flex; justify-content: space-between; align-items: baseline; padding: 5px 0; border-bottom: 1px solid var(--surface2); }
.stat-row:last-child { border-bottom: none; }
.stat-label { color: var(--muted); font-size: 12px; font-weight: 500; }
.stat-value { font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 13px; }

/* ── COLORS ──────────────────────────────────────────── */
.green  { color: var(--green); }
.yellow { color: var(--yellow); }
.red    { color: var(--red); }
.blue   { color: var(--blue); }
.dim    { color: var(--muted); }

/* ── REGIME ──────────────────────────────────────────── */
.regime-label { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 3px; }
.regime-prob  { font-size: 12px; color: var(--muted); margin-bottom: 10px; }

/* ── RISK BARS ───────────────────────────────────────── */
.risk-bar-wrap { display: flex; align-items: center; gap: 10px; padding: 6px 0; }
.risk-bar-bg   { flex: 1; height: 6px; background: var(--border); border-radius: 3px; }
.risk-bar-fill { height: 6px; border-radius: 3px; transition: width 0.5s; }
.risk-label    { min-width: 76px; color: var(--muted); font-size: 11px; font-weight: 500; }
.risk-value    { min-width: 68px; text-align: right; font-size: 12px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.halted-banner { background: var(--red-dim); border: 1px solid var(--red-border); color: #b91c1c; padding: 9px 13px; border-radius: 8px; margin-bottom: 10px; font-weight: 600; }

/* ── TABLES ──────────────────────────────────────────── */
table { width: 100%; border-collapse: collapse; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; padding: 7px 10px; border-bottom: 1px solid var(--border); background: var(--surface2); }
td { padding: 8px 10px; border-bottom: 1px solid var(--surface2); font-size: 13px; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #fafbfc; }

/* ── POSITION CLOSE BUTTON ───────────────────────────── */
.close-btn { background: var(--red-dim); color: #b91c1c; border: 1px solid var(--red-border); border-radius: 5px; padding: 3px 9px; font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600; cursor: pointer; }
.close-btn:hover { background: var(--red-border); }

/* ── SIGNAL / ALERT LINES ────────────────────────────── */
.signal-line { padding: 6px 0; border-bottom: 1px solid var(--surface2); display: flex; gap: 12px; }
.signal-line:last-child { border-bottom: none; }
.signal-time { color: var(--muted); min-width: 44px; font-size: 12px; }

/* ── CONFIG PANEL ────────────────────────────────────── */
.config-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 40px; margin-bottom: 16px; }
@media(max-width:700px){ .config-grid { grid-template-columns: 1fr; } }
.cfg-item { display: flex; align-items: center; gap: 12px; padding: 7px 0; border-bottom: 1px solid var(--surface2); }
.cfg-item:last-child { border-bottom: none; }
.cfg-label { min-width: 148px; color: var(--muted); font-size: 12px; font-weight: 500; }
input[type=range] { flex: 1; appearance: none; height: 4px; background: var(--border); border-radius: 2px; cursor: pointer; }
input[type=range]::-webkit-slider-thumb { appearance: none; width: 15px; height: 15px; border-radius: 50%; background: var(--blue); cursor: pointer; border: 2px solid white; box-shadow: 0 1px 4px rgba(0,0,0,0.2); }
.cfg-val { min-width: 48px; text-align: right; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text); }
.apply-btn { width: 100%; padding: 10px; background: var(--blue); color: #fff; border: none; border-radius: 8px; font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
.apply-btn:hover { background: #1d4ed8; }

/* ── DAILY HALT STATE ────────────────────────────────── */
.order-card-inner { position: relative; }
.order-card-inner.halted .order-inputs,
.order-card-inner.halted .order-btns { filter: blur(4px); pointer-events: none; user-select: none; opacity: 0.25; transition: all 0.4s; }
.halt-overlay { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 12px 16px; background: rgba(255,255,255,0.92); backdrop-filter: blur(4px); border-radius: 8px; z-index: 5; }
.halt-icon { font-size: 36px; margin-bottom: 8px; }
.halt-msg { font-size: 13px; font-weight: 700; color: var(--text); line-height: 1.55; margin-bottom: 6px; }
.halt-sub { font-size: 11px; color: var(--muted); font-weight: 500; }
.config-halted { opacity: 0.45; pointer-events: none; }
.config-halted input[type=range] { cursor: not-allowed; }
.halt-banner-cfg { background: #fef9c3; border: 1px solid #fde68a; color: #92400e; border-radius: 7px; padding: 9px 14px; margin-bottom: 12px; font-size: 12px; font-weight: 600; display: none; }

/* ── EMPTY / OFFLINE ─────────────────────────────────── */
.empty-state { color: var(--muted); text-align: center; padding: 20px 0; font-size: 12px; }
.offline-overlay { text-align: center; padding: 28px; color: var(--muted); }
.offline-overlay .icon { font-size: 28px; margin-bottom: 8px; }

/* ── MODAL ───────────────────────────────────────────── */
#modal-overlay { display: none; position: fixed; inset: 0; background: rgba(15,23,42,0.35); backdrop-filter: blur(3px); z-index: 1000; align-items: center; justify-content: center; }
#modal-box { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 28px 32px; max-width: 420px; width: 90%; box-shadow: var(--shadow-md); }
#modal-msg { font-size: 14px; line-height: 1.6; margin-bottom: 20px; color: var(--text); }
.modal-btns { display: flex; gap: 10px; justify-content: flex-end; }
.modal-btns button { padding: 8px 20px; border-radius: 7px; font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 600; cursor: pointer; border: 1px solid; }
#modal-confirm { background: var(--red-dim); color: #b91c1c; border-color: var(--red-border); }
#modal-cancel  { background: var(--surface2); color: var(--muted); border-color: var(--border2); }
</style>
</head>
<body>

<!-- ── HEADER ──────────────────────────────────────────── -->
<header>
  <h1>Robin Dashboard</h1>
  <div class="status-row">
    <div class="dot" id="dot"></div>
    <span class="last-update" id="last-update">Connecting…</span>
    <span class="badge badge-offline" id="mode-badge">—</span>
  </div>
</header>

<!-- ── CONTROLS BAR ────────────────────────────────────── -->
<div class="controls-bar">
  <div class="ctrl-group">
    <button id="btn-halt"   class="ctrl-btn danger"  onclick="confirmAndSend('halt',   'Halt all trading? No new orders will be placed until you resume.')">⛔ HALT</button>
    <button id="btn-resume" class="ctrl-btn success" onclick="sendControl('resume')">▶ RESUME</button>
  </div>
  <div class="ctrl-sep"></div>
  <div class="ctrl-group">
    <button id="btn-dry-run" class="ctrl-btn warning" onclick="toggleDryRun()">DRY RUN: OFF</button>
  </div>
  <div class="ctrl-sep"></div>
  <div class="ctrl-group">
    <button id="btn-retrain" class="ctrl-btn info" onclick="confirmAndSend('retrain', 'Force HMM retrain now? This runs in the engine process and may take a few minutes.')">↺ FORCE RETRAIN</button>
  </div>
  <div class="ctrl-status" id="ctrl-status"></div>
</div>

<!-- ── PAGE BODY ─────────────────────────────────────────── -->
<div class="page-body">
<main>

  <!-- Regime -->
  <div class="card">
    <div class="card-header">Regime</div>
    <div class="card-body" id="regime-body"><div class="offline-overlay"><div class="icon">📡</div>Waiting…</div></div>
  </div>

  <!-- Portfolio -->
  <div class="card">
    <div class="card-header">Portfolio</div>
    <div class="card-body" id="portfolio-body"><div class="offline-overlay"><div class="icon">📊</div>Waiting…</div></div>
  </div>

  <!-- Risk -->
  <div class="card">
    <div class="card-header">Risk</div>
    <div class="card-body" id="risk-body"><div class="offline-overlay"><div class="icon">🛡️</div>Waiting…</div></div>
  </div>

  <!-- Positions (full width) -->
  <div class="card card-full">
    <div class="card-header">Positions</div>
    <div class="card-body" id="positions-body"><div class="offline-overlay"><div class="icon">📋</div>Waiting…</div></div>
  </div>

  <!-- Recent Orders (full width) -->
  <div class="card card-full">
    <div class="card-header">Recent Orders <span id="orders-source" style="font-weight:400;text-transform:none;letter-spacing:0"></span></div>
    <div class="card-body" id="orders-body"><div class="offline-overlay"><div class="icon">📋</div>Waiting…</div></div>
  </div>

  <!-- Recent Signals (2-col span) -->
  <div class="card" style="grid-column: span 2;">
    <div class="card-header">Recent Signals</div>
    <div class="card-body" id="signals-body"><div class="offline-overlay"><div class="icon">📈</div>Waiting…</div></div>
  </div>

  <!-- System -->
  <div class="card">
    <div class="card-header">System</div>
    <div class="card-body" id="system-body"><div class="offline-overlay"><div class="icon">⚙️</div>Waiting…</div></div>
  </div>

  <!-- Alerts (full width) -->
  <div class="card card-full">
    <div class="card-header">Recent Alerts</div>
    <div class="card-body" id="alerts-body"><div class="empty-state">No alerts</div></div>
  </div>

  <!-- Risk Config (full width) -->
  <div class="card card-full" id="config-card">
    <div class="card-header">
      Risk Configuration
      <span style="color:var(--muted);font-weight:400;font-size:11px">Changes take effect on next bar processed by the engine</span>
    </div>
    <div class="card-body" id="config-card-body">
      <div class="halt-banner-cfg" id="cfg-halt-banner">⚠ Configuration locked — daily limit active. Changes resume tomorrow.</div>
      <div class="config-grid" id="cfg-grid">

        <!-- Left column: DD thresholds -->
        <div>
          <div class="cfg-item">
            <span class="cfg-label">Daily DD Reduce</span>
            <input type="range" id="cfg-daily_dd_reduce" min="0.005" max="0.05" step="0.001" oninput="showCfgVal(this,'%')">
            <span class="cfg-val" id="cfg-daily_dd_reduce-val">—</span>
          </div>
          <div class="cfg-item">
            <span class="cfg-label">Daily DD Halt</span>
            <input type="range" id="cfg-daily_dd_halt" min="0.01" max="0.10" step="0.001" oninput="showCfgVal(this,'%')">
            <span class="cfg-val" id="cfg-daily_dd_halt-val">—</span>
          </div>
          <div class="cfg-item">
            <span class="cfg-label">Weekly DD Reduce</span>
            <input type="range" id="cfg-weekly_dd_reduce" min="0.01" max="0.10" step="0.001" oninput="showCfgVal(this,'%')">
            <span class="cfg-val" id="cfg-weekly_dd_reduce-val">—</span>
          </div>
          <div class="cfg-item">
            <span class="cfg-label">Weekly DD Halt</span>
            <input type="range" id="cfg-weekly_dd_halt" min="0.02" max="0.15" step="0.001" oninput="showCfgVal(this,'%')">
            <span class="cfg-val" id="cfg-weekly_dd_halt-val">—</span>
          </div>
          <div class="cfg-item">
            <span class="cfg-label">Peak DD Halt</span>
            <input type="range" id="cfg-max_dd_from_peak" min="0.05" max="0.30" step="0.005" oninput="showCfgVal(this,'%')">
            <span class="cfg-val" id="cfg-max_dd_from_peak-val">—</span>
          </div>
        </div>

        <!-- Right column: limits -->
        <div>
          <div class="cfg-item">
            <span class="cfg-label">Max Exposure</span>
            <input type="range" id="cfg-max_exposure" min="0.20" max="1.50" step="0.01" oninput="showCfgVal(this,'%')">
            <span class="cfg-val" id="cfg-max_exposure-val">—</span>
          </div>
          <div class="cfg-item">
            <span class="cfg-label">Max Leverage</span>
            <input type="range" id="cfg-max_leverage" min="1.00" max="3.00" step="0.05" oninput="showCfgVal(this,'x')">
            <span class="cfg-val" id="cfg-max_leverage-val">—</span>
          </div>
          <div class="cfg-item">
            <span class="cfg-label">Max Risk/Trade</span>
            <input type="range" id="cfg-max_risk_per_trade" min="0.001" max="0.05" step="0.001" oninput="showCfgVal(this,'%')">
            <span class="cfg-val" id="cfg-max_risk_per_trade-val">—</span>
          </div>
          <div class="cfg-item">
            <span class="cfg-label">Max Single Position</span>
            <input type="range" id="cfg-max_single_position" min="0.02" max="0.50" step="0.01" oninput="showCfgVal(this,'%')">
            <span class="cfg-val" id="cfg-max_single_position-val">—</span>
          </div>
          <div class="cfg-item">
            <span class="cfg-label">Max Concurrent</span>
            <input type="range" id="cfg-max_concurrent" min="1" max="20" step="1" oninput="showCfgVal(this,'')">
            <span class="cfg-val" id="cfg-max_concurrent-val">—</span>
          </div>
        </div>

      </div>
      <button class="apply-btn" onclick="applyConfig()">Apply Risk Config →</button>
    </div>
  </div>

</main>

<!-- ── SIDEBAR ───────────────────────────────────────────── -->
<aside class="sidebar">

  <!-- Chart -->
  <div class="card">
    <div class="card-header">
      <span id="chart-label">SPY · 1m</span>
      <input class="chart-sym-input" id="chart-sym-input" value="SPY" placeholder="Symbol"
             onkeydown="if(event.key==='Enter')loadChart(this.value)" title="Press Enter to load">
    </div>
    <div id="chart-container"></div>
    <div id="chart-msg" class="empty-state" style="display:none;padding:40px 0"></div>
  </div>

  <!-- Quick Order -->
  <div class="card">
    <div class="card-header">Quick Order</div>
    <div class="card-body">
      <div class="order-card-inner" id="order-card-inner">
        <div class="order-inputs">
          <input class="order-input" id="order-symbol" value="SPY" placeholder="Symbol" style="flex:1;text-transform:uppercase">
          <input class="order-input" id="order-qty" type="number" value="1" min="1" style="width:72px;text-align:center">
        </div>
        <div class="order-btns">
          <button class="order-btn buy-market"  onclick="placeOrder('buy','market')">▲ Buy Market</button>
          <button class="order-btn buy-ask"     onclick="placeOrder('buy','ask')">▲ Buy Ask</button>
          <button class="order-btn sell-market" onclick="placeOrder('sell','market')">▼ Sell Market</button>
          <button class="order-btn sell-ask"    onclick="placeOrder('sell','ask')">▼ Sell Ask</button>
        </div>
        <!-- Halt overlay (shown when daily limit hit) -->
        <div class="halt-overlay" id="halt-overlay" style="display:none">
          <div class="halt-icon" id="halt-icon">🛑</div>
          <div class="halt-msg" id="halt-msg">Daily limit reached.</div>
          <div class="halt-sub" id="halt-sub">Trading is paused for today.</div>
        </div>
      </div>
      <div id="order-feedback" style="margin-top:10px"></div>
    </div>
  </div>

</aside>
</div><!-- end .page-body -->

<!-- ── MODAL ─────────────────────────────────────────────── -->
<div id="modal-overlay">
  <div id="modal-box">
    <div id="modal-msg"></div>
    <div class="modal-btns">
      <button id="modal-cancel" onclick="hideModal()">Cancel</button>
      <button id="modal-confirm">Confirm</button>
    </div>
  </div>
</div>

<script>
// ── State ────────────────────────────────────────────────────
const VOL_COLOR = { low: 'green', mid: 'yellow', high: 'red' };
let _dryRun = false;
let _cfgInitialized = false;

// ── Halt messages ─────────────────────────────────────────────
const HALT_MSGS = [
  ["Today's limit is your edge, not a failure.", "The discipline you show now is the discipline that compounds."],
  ["The best trade right now is no trade.", "Protect your capital — it's the only tool that matters."],
  ["Every great trader has a hard stop on their day.", "You kept your rule. That's the real win."],
  ["Markets will be here tomorrow. Make sure you are too.", "Step away, reset, return stronger."],
  ["Stopping here took more discipline than any trade you placed.", "That discipline is what separates pros from the rest."],
  ["Your future self is grateful you walked away.", "Rest is not weakness — it's part of the strategy."],
  ["You followed your system today. Systems beat impulses every time.", "Tomorrow is a fresh slate."],
  ["The market doesn't care about your P&L. Your risk rules do.", "Honor them and they'll protect you."],
  ["One bad afternoon can erase a great week.", "You just prevented that. Well done."],
  ["Close the screen. Go outside. Come back fresh.", "That's not retreating — that's professional trading."],
];
let _haltMsgIdx = Math.floor(Math.random() * HALT_MSGS.length);
let _lastHalted = false;

function updateHaltUI(risk) {
  const halted = !!(risk && (risk.halted || (risk.daily_dd >= risk.daily_dd_halt && risk.daily_dd_halt > 0)));

  const inner   = document.getElementById('order-card-inner');
  const overlay = document.getElementById('halt-overlay');
  const cfgBody = document.getElementById('config-card-body');
  const cfgBanner = document.getElementById('cfg-halt-banner');

  if (halted) {
    // Rotate message each time halt is freshly detected
    if (!_lastHalted) { _haltMsgIdx = Math.floor(Math.random() * HALT_MSGS.length); }
    const [main, sub] = HALT_MSGS[_haltMsgIdx];
    inner.classList.add('halted');
    overlay.style.display = '';
    document.getElementById('halt-msg').textContent  = main;
    document.getElementById('halt-sub').textContent  = sub;
    // Lock config
    cfgBody.classList.add('config-halted');
    cfgBanner.style.display = '';
    cfgBody.querySelectorAll('input[type=range]').forEach(el => el.disabled = true);
    document.querySelector('.apply-btn').disabled = true;
  } else {
    inner.classList.remove('halted');
    overlay.style.display = 'none';
    cfgBody.classList.remove('config-halted');
    cfgBanner.style.display = 'none';
    cfgBody.querySelectorAll('input[type=range]').forEach(el => el.disabled = false);
    document.querySelector('.apply-btn').disabled = false;
  }
  _lastHalted = halted;
}

// ── Formatting ───────────────────────────────────────────────
function fmt(n, d=2) { return n==null ? '—' : Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}); }
function pct(n) { return n==null ? '—' : (n>=0?'+':'')+fmt(n*100)+'%'; }
function money(n) { return n==null ? '—' : '$'+fmt(n); }
function cc(n, inv=false) { if(n==null) return ''; return (n>0) !== inv ? 'green' : n===0 ? '' : 'red'; }

// ── Render functions ─────────────────────────────────────────
function renderRegime(r) {
  if (!r) return '<div class="empty-state">No regime detected</div>';
  const c = VOL_COLOR[r.vol_environment] || 'dim';
  return `
    <div class="regime-label ${c}">${r.label}</div>
    <div class="regime-prob">Confidence: <b>${fmt(r.probability*100,1)}%</b></div>
    <div style="margin-top:10px">
      <div class="stat-row"><span class="stat-label">Vol environment</span><span class="stat-value ${c}">${(r.vol_environment||'').toUpperCase()}</span></div>
      <div class="stat-row"><span class="stat-label">Consecutive bars</span><span class="stat-value">${r.consecutive_bars??'—'}</span></div>
      <div class="stat-row"><span class="stat-label">Confirmed</span><span class="stat-value ${r.is_confirmed?'green':'yellow'}">${r.is_confirmed?'✓ Yes':'⏳ No'}</span></div>
    </div>`;
}

function renderPortfolio(p) {
  if (!p) return '<div class="empty-state">No portfolio data</div>';
  const dailyPct  = p.daily_start  > 0 ? (p.equity/p.daily_start  - 1) : null;
  const weeklyPct = p.weekly_start > 0 ? (p.equity/p.weekly_start - 1) : null;
  const expPct    = p.equity > 0 ? p.gross_exposure/p.equity : 0;
  return `
    <div class="stat-row"><span class="stat-label">Equity</span><span class="stat-value">${money(p.equity)}</span></div>
    <div class="stat-row"><span class="stat-label">Cash</span><span class="stat-value">${money(p.cash)}</span></div>
    <div class="stat-row"><span class="stat-label">Daily P&amp;L</span><span class="stat-value ${cc(dailyPct)}">${pct(dailyPct)}</span></div>
    <div class="stat-row"><span class="stat-label">Weekly P&amp;L</span><span class="stat-value ${cc(weeklyPct)}">${pct(weeklyPct)}</span></div>
    <div class="stat-row"><span class="stat-label">Gross exposure</span><span class="stat-value">${pct(expPct)}</span></div>
    <div class="stat-row"><span class="stat-label">Trades today</span><span class="stat-value">${p.open_trade_count_today??0}</span></div>`;
}

function riskBar(label, value, reduce, halt, max) {
  const p = Math.min((value/max)*100, 100);
  const color = value>=halt ? 'red' : value>=reduce ? 'yellow' : 'green';
  const icon  = value>=halt ? '✗'  : value>=reduce ? '⚠'     : '✓';
  return `<div class="risk-bar-wrap">
    <span class="risk-label">${label}</span>
    <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${p}%;background:var(--${color})"></div></div>
    <span class="risk-value ${color}">${icon} ${fmt(value*100,2)}%</span>
  </div>`;
}

function renderRisk(r) {
  if (!r) return '<div class="empty-state">No risk data</div>';
  let h = '';
  if (r.halted) h += `<div class="halted-banner">🚨 HALTED — ${r.halt_reason||'circuit breaker'}</div>`;
  h += riskBar('Daily DD',  r.daily_dd,  r.daily_dd_reduce||0.02, r.daily_dd_halt||0.03,    0.06);
  h += riskBar('Weekly DD', r.weekly_dd, r.weekly_dd_reduce||0.05, r.weekly_dd_halt||0.07,  0.12);
  h += riskBar('Peak DD',   r.peak_dd,   (r.max_dd_from_peak||0.10)*0.6, r.max_dd_from_peak||0.10, 0.20);
  return h;
}

function renderPositions(positions) {
  if (!positions||!positions.length) return '<div class="empty-state">No open positions</div>';
  let h = '<table><thead><tr><th>Symbol</th><th>Qty</th><th>Mkt Value</th><th>Unreal P&amp;L</th><th>Return</th><th></th></tr></thead><tbody>';
  for (const p of positions) {
    const c = cc(p.unrealized_pnl_pct);
    h += `<tr>
      <td><b>${p.symbol}</b></td>
      <td>${p.qty}</td>
      <td>${money(p.market_value)}</td>
      <td class="${c}">${money(p.unrealized_pnl)}</td>
      <td class="${c}">${pct(p.unrealized_pnl_pct)}</td>
      <td><button class="close-btn" onclick="confirmClose('${p.symbol}')">✕ Close</button></td>
    </tr>`;
  }
  return h + '</tbody></table>';
}

function renderSignals(signals) {
  if (!signals||!signals.length) return '<div class="empty-state">No signals yet</div>';
  return signals.slice(-12).reverse().map(s => {
    const parts = s.split('|').map(x=>x.trim());
    return `<div class="signal-line">
      <span class="signal-time">${parts[0]||''}</span>
      <span><b>${parts[1]||''}</b></span>
      <span class="green">${parts[2]||''}</span>
      <span class="dim">${parts[3]||''}</span>
    </div>`;
  }).join('');
}

function renderSystem(s) {
  if (!s) return '<div class="empty-state">No system data</div>';
  const start = s.session_start ? new Date(s.session_start) : null;
  const uptimeSec = start ? Math.floor((Date.now()-start.getTime())/1000) : 0;
  const h = Math.floor(uptimeSec/3600), m = Math.floor((uptimeSec%3600)/60);
  const uptime = h>0 ? `${h}h ${m}m` : `${m}m`;
  return `
    <div class="stat-row"><span class="stat-label">Mode</span><span class="stat-value ${s.mode==='live'?'red':'blue'}">${(s.mode||'').toUpperCase()}</span></div>
    <div class="stat-row"><span class="stat-label">HMM model</span><span class="stat-value">${s.hmm_model||'—'}</span></div>
    <div class="stat-row"><span class="stat-label">HMM age</span><span class="stat-value ${(s.hmm_age_days||0)<7?'green':'yellow'}">${s.hmm_age_days??'—'}d</span></div>
    <div class="stat-row"><span class="stat-label">Last bar</span><span class="stat-value dim">${s.last_bar?new Date(s.last_bar).toLocaleTimeString():'—'}</span></div>
    <div class="stat-row"><span class="stat-label">Uptime</span><span class="stat-value">${uptime}</span></div>
    <div class="stat-row"><span class="stat-label">Dry run</span><span class="stat-value ${s.dry_run?'yellow':'green'}">${s.dry_run?'Yes':'No'}</span></div>`;
}

function renderAlerts(alerts) {
  if (!alerts||!alerts.length) return '<div class="empty-state">No recent alerts</div>';
  return alerts.slice(-10).reverse().map(a=>`<div class="signal-line"><span class="red">${a}</span></div>`).join('');
}

function renderOrders(orders) {
  if (!orders||!orders.length) return '<div class="empty-state">No recent orders</div>';
  const STATUS_COLOR = {
    filled: 'green', partially_filled: 'yellow',
    canceled: 'dim', cancelled: 'dim', expired: 'dim', replaced: 'dim',
    new: 'blue', accepted: 'blue', pending_new: 'blue',
    pending_cancel: 'yellow', rejected: 'red',
  };
  let h = '<table><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Type</th><th>Status</th><th>Filled @</th></tr></thead><tbody>';
  for (const o of orders.slice(0, 20)) {
    const color = STATUS_COLOR[o.status] || 'dim';
    const ts = o.submitted_at ? new Date(o.submitted_at).toLocaleTimeString() : '—';
    const filled = o.filled_avg_price ? '$'+fmt(parseFloat(o.filled_avg_price)) : '—';
    const sideColor = o.side==='buy' ? 'green' : 'red';
    h += `<tr>
      <td class="dim">${ts}</td>
      <td><b>${o.symbol}</b></td>
      <td class="${sideColor}">${(o.side||'').toUpperCase()}</td>
      <td>${o.qty}</td>
      <td class="dim">${o.type||'—'}</td>
      <td class="${color}">${o.status||'—'}</td>
      <td class="dim">${filled}</td>
    </tr>`;
  }
  return h + '</tbody></table>';
}

// ── Config panel ─────────────────────────────────────────────
const CFG_FIELDS_PCT = ['daily_dd_reduce','daily_dd_halt','weekly_dd_reduce','weekly_dd_halt','max_dd_from_peak','max_exposure','max_risk_per_trade','max_single_position'];

function showCfgVal(input, unit) {
  const val = parseFloat(input.value);
  const display = unit === '%'  ? fmt(val*100,1)+'%'
                : unit === 'x'  ? fmt(val,2)+'×'
                :                 String(Math.round(val));
  document.getElementById(input.id+'-val').textContent = display;
}

function loadConfig(config) {
  if (!config || _cfgInitialized) return;
  _cfgInitialized = true;
  const percentFields = ['daily_dd_reduce','daily_dd_halt','weekly_dd_reduce','weekly_dd_halt','max_dd_from_peak','max_exposure','max_risk_per_trade','max_single_position'];
  for (const f of percentFields) {
    const el = document.getElementById('cfg-'+f);
    if (el && config[f] != null) { el.value = config[f]; showCfgVal(el, '%'); }
  }
  // leverage
  const lev = document.getElementById('cfg-max_leverage');
  if (lev && config.max_leverage != null) { lev.value = config.max_leverage; showCfgVal(lev, 'x'); }
  // max_concurrent (integer)
  const mc = document.getElementById('cfg-max_concurrent');
  if (mc && config.max_concurrent != null) { mc.value = config.max_concurrent; showCfgVal(mc, ''); }
}

function applyConfig() {
  const params = {};
  const percentFields = ['daily_dd_reduce','daily_dd_halt','weekly_dd_reduce','weekly_dd_halt','max_dd_from_peak','max_exposure','max_risk_per_trade','max_single_position'];
  for (const f of percentFields) {
    const el = document.getElementById('cfg-'+f);
    if (el) params[f] = parseFloat(el.value);
  }
  const lev = document.getElementById('cfg-max_leverage');
  if (lev) params['max_leverage'] = parseFloat(lev.value);
  const mc = document.getElementById('cfg-max_concurrent');
  if (mc) params['max_concurrent'] = parseInt(mc.value);
  sendControl('set_risk', {params});
}

// ── Control buttons ──────────────────────────────────────────
function updateControlButtons(d) {
  _dryRun = d.system?.dry_run || false;
  const halted = d.risk?.halted || false;

  document.getElementById('btn-halt').disabled   = halted;
  document.getElementById('btn-resume').disabled = !halted;

  const drBtn = document.getElementById('btn-dry-run');
  drBtn.textContent = _dryRun ? 'DRY RUN: ON' : 'DRY RUN: OFF';
  drBtn.classList.toggle('active', _dryRun);
}

function toggleDryRun() { sendControl('set_dry_run', {value: !_dryRun}); }

function confirmClose(symbol) {
  showModal(`Close position in ${symbol}? A market sell order will be submitted immediately.`,
    () => sendControl('close_position', {symbol}));
}

// ── Control API ──────────────────────────────────────────────
async function sendControl(action, extra = {}) {
  try {
    const res = await fetch('/api/control', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({action, ...extra}),
    });
    const data = await res.json();
    showCtrlStatus(data.ok ? `✓ ${action} queued` : `✗ ${data.error||'error'}`, !data.ok);
  } catch {
    showCtrlStatus('✗ Connection error', true);
  }
}

function confirmAndSend(action, msg) {
  showModal(msg, () => sendControl(action));
}

function showCtrlStatus(msg, isErr=false) {
  const el = document.getElementById('ctrl-status');
  el.textContent = msg;
  el.className = 'ctrl-status ' + (isErr ? 'error' : 'ok');
  setTimeout(() => { el.textContent=''; el.className='ctrl-status'; }, 4500);
}

// ── Modal ────────────────────────────────────────────────────
function showModal(msg, onConfirm) {
  document.getElementById('modal-msg').textContent = msg;
  document.getElementById('modal-confirm').onclick = () => { hideModal(); onConfirm(); };
  document.getElementById('modal-overlay').style.display = 'flex';
}
function hideModal() { document.getElementById('modal-overlay').style.display = 'none'; }
document.getElementById('modal-overlay').addEventListener('click', e => { if(e.target===e.currentTarget) hideModal(); });

// ── Polling ──────────────────────────────────────────────────
async function refresh() {
  try {
    const res = await fetch('/api/state');
    if (!res.ok) throw new Error();
    const d = await res.json();

    document.getElementById('dot').className = 'dot';
    document.getElementById('last-update').textContent = 'Updated ' + new Date().toLocaleTimeString();
    const mode = (d.system?.mode||'').toLowerCase();
    const badge = document.getElementById('mode-badge');
    badge.textContent = mode.toUpperCase()||'—';
    badge.className = 'badge badge-'+(mode==='live'?'live':mode==='paper'?'paper':'offline');

    document.getElementById('regime-body').innerHTML    = renderRegime(d.regime);
    document.getElementById('portfolio-body').innerHTML = renderPortfolio(d.portfolio);
    document.getElementById('risk-body').innerHTML      = renderRisk(d.risk);
    document.getElementById('positions-body').innerHTML = renderPositions(d.positions);
    document.getElementById('orders-body').innerHTML    = renderOrders(d.recent_orders);
    document.getElementById('signals-body').innerHTML   = renderSignals(d.recent_signals);
    document.getElementById('system-body').innerHTML    = renderSystem(d.system);
    document.getElementById('alerts-body').innerHTML    = renderAlerts(d.recent_alerts);

    const src = document.getElementById('orders-source');
    if (src) src.textContent = d._source === 'alpaca_direct' ? '(live from Alpaca)' : '(engine)';

    updateControlButtons(d);
    updateHaltUI(d.risk);
    loadConfig(d.config);
  } catch {
    document.getElementById('dot').className = 'dot offline';
    document.getElementById('last-update').textContent = 'Engine offline';
  }
}

refresh();
setInterval(refresh, 3000);
</script>

<!-- TradingView Lightweight Charts -->
<script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<script>
// ── Chart ────────────────────────────────────────────────────
let _chart = null, _candles = null, _chartSym = 'SPY';

function initChart() {
  const el = document.getElementById('chart-container');
  _chart = LightweightCharts.createChart(el, {
    layout: { background: { color: '#ffffff' }, textColor: '#475569' },
    grid: { vertLines: { color: '#f8fafc' }, horzLines: { color: '#f8fafc' } },
    rightPriceScale: { borderColor: '#e2e8f0' },
    timeScale: { borderColor: '#e2e8f0', timeVisible: true, secondsVisible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    width: el.clientWidth,
    height: 310,
  });
  _candles = _chart.addCandlestickSeries({
    upColor: '#16a34a', downColor: '#dc2626',
    borderUpColor: '#16a34a', borderDownColor: '#dc2626',
    wickUpColor: '#16a34a', wickDownColor: '#dc2626',
  });
  new ResizeObserver(() => _chart && _chart.applyOptions({ width: el.clientWidth })).observe(el);
  loadChart(_chartSym);
  setInterval(() => loadChart(_chartSym), 60000);
}

async function loadChart(symbol) {
  symbol = symbol.trim().toUpperCase();
  if (!symbol) return;
  _chartSym = symbol;
  document.getElementById('chart-label').textContent = symbol + ' · 1m';
  document.getElementById('chart-sym-input').value = symbol;
  try {
    const res = await fetch('/api/chart?symbol=' + encodeURIComponent(symbol) + '&limit=200');
    const d = await res.json();
    if (d.ok && d.data.length > 0) {
      document.getElementById('chart-container').style.display = '';
      document.getElementById('chart-msg').style.display = 'none';
      _candles.setData(d.data);
      _chart.timeScale().fitContent();
    } else {
      showChartMsg(d.error || 'No data returned for ' + symbol);
    }
  } catch(e) { showChartMsg('Could not load chart data'); }
}

function showChartMsg(msg) {
  document.getElementById('chart-container').style.display = 'none';
  const el = document.getElementById('chart-msg');
  el.style.display = '';
  el.textContent = '⚠ ' + msg;
}

// init chart after DOM + LW Charts are both ready
document.addEventListener('DOMContentLoaded', () => {
  if (window.LightweightCharts) initChart();
  else document.querySelector('script[src*="lightweight"]').addEventListener('load', initChart);
});

// ── Quick Order ──────────────────────────────────────────────
async function placeOrder(side, type) {
  const symbol = (document.getElementById('order-symbol').value || '').trim().toUpperCase();
  const qty    = parseInt(document.getElementById('order-qty').value) || 1;
  if (!symbol) return;
  const fb = document.getElementById('order-feedback');
  fb.innerHTML = '<div class="order-feedback" style="background:#f1f5f9;color:#64748b">Sending…</div>';
  try {
    const res = await fetch('/api/order', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, qty, side, order_type: type }),
    });
    const d = await res.json();
    if (d.ok) {
      const o = d.order;
      fb.innerHTML = `<div class="order-feedback ok">✓ ${o.status} — ${side.toUpperCase()} ${qty} ${symbol} @ ${type}</div>`;
    } else {
      fb.innerHTML = `<div class="order-feedback error">✗ ${d.error}</div>`;
    }
  } catch(e) {
    fb.innerHTML = `<div class="order-feedback error">✗ Connection error</div>`;
  }
  setTimeout(() => { fb.innerHTML = ''; }, 7000);
}
</script>
</body>
</html>
"""


_ENGINE_STALE_SECS = 90  # treat web_state.json as stale after this many seconds


def _alpaca_direct_state() -> dict:
    """
    Fetch live account + position data straight from Alpaca.
    Called when the trading engine is not running (web_state.json missing or stale).
    Merges in static context from state_snapshot.json and settings.yaml where available.
    """
    import os, time, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key    = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return {}

    settings_path = Path("config/settings.yaml")
    cfg: dict = {}
    paper = True
    if settings_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(settings_path.read_text()) or {}
            paper = cfg.get("broker", {}).get("paper_trading", True)
        except Exception:
            pass

    try:
        client = _get_alpaca_client()

        acct     = client.get_account()
        equity   = float(acct["equity"])
        cash     = float(acct["cash"])

        positions_raw = client.get_all_positions()
        positions = []
        gross_exposure = 0.0
        for p in positions_raw:
            qty = int(float(p.get("qty", 0)))
            if qty == 0:
                continue
            mkt_val = float(p.get("market_value", 0))
            gross_exposure += abs(mkt_val)
            positions.append({
                "symbol":            p["symbol"],
                "qty":               qty,
                "market_value":      mkt_val,
                "unrealized_pnl":    float(p.get("unrealized_pl",   0)),
                "unrealized_pnl_pct": float(p.get("unrealized_plpc", 0)),
            })

        # Pull daily/weekly baselines from state_snapshot.json if available
        daily_start  = 0.0
        weekly_start = 0.0
        last_regime: dict = {}
        last_trained = ""
        snapshot_path = Path("state_snapshot.json")
        if snapshot_path.exists():
            try:
                snap = json.loads(snapshot_path.read_text())
                daily_start  = snap.get("daily_start_equity",  0.0) or 0.0
                weekly_start = snap.get("weekly_start_equity", 0.0) or 0.0
                last_regime  = snap.get("last_regime", {})
                last_trained = snap.get("last_trained", "")
            except Exception:
                pass

        # Config from settings.yaml
        config: dict = {}
        if settings_path.exists():
            try:
                risk_cfg = cfg.get("risk", {})
                config = {
                    "daily_dd_reduce":    risk_cfg.get("daily_dd_reduce",    0.02),
                    "daily_dd_halt":      risk_cfg.get("daily_dd_halt",      0.03),
                    "weekly_dd_reduce":   risk_cfg.get("weekly_dd_reduce",   0.05),
                    "weekly_dd_halt":     risk_cfg.get("weekly_dd_halt",     0.07),
                    "max_dd_from_peak":   risk_cfg.get("max_dd_from_peak",   0.10),
                    "max_exposure":       risk_cfg.get("max_exposure",        0.80),
                    "max_leverage":       risk_cfg.get("max_leverage",        1.25),
                    "max_risk_per_trade": risk_cfg.get("max_risk_per_trade",  0.01),
                    "max_single_position":risk_cfg.get("max_single_position", 0.15),
                    "max_concurrent":     risk_cfg.get("max_concurrent",       5),
                    "max_daily_trades":   risk_cfg.get("max_daily_trades",    20),
                }
            except Exception:
                pass

        # Best-effort drawdown from snapshot baselines
        daily_dd  = max(0.0, 1.0 - equity / daily_start)  if daily_start  > 0 else 0.0
        weekly_dd = max(0.0, 1.0 - equity / weekly_start) if weekly_start > 0 else 0.0

        # Last known regime from snapshot
        regime_data = None
        if last_regime:
            first_sym, first_label = next(iter(last_regime.items()))
            regime_data = {
                "label": first_label, "probability": 0.0,
                "vol_environment": "mid", "consecutive_bars": 0, "is_confirmed": False,
            }

        # HMM model age
        hmm_age = 0
        model_name = "—"
        model_path = Path("models")
        if settings_path.exists():
            try:
                primary = cfg.get("broker", {}).get("symbols", ["SPY"])[0]
                mp = model_path / f"hmm_{primary}.pkl"
                if mp.exists():
                    hmm_age = int((time.time() - mp.stat().st_mtime) / 86400)
                    model_name = mp.name
            except Exception:
                pass

        recent_orders: list = []
        try:
            recent_orders = client.list_recent_orders(limit=20)
        except Exception:
            pass

        return {
            "regime":    regime_data,
            "portfolio": {
                "equity":               equity,
                "cash":                 cash,
                "daily_start":          daily_start  or equity,
                "weekly_start":         weekly_start or equity,
                "gross_exposure":       gross_exposure,
                "open_trade_count_today": 0,
            },
            "risk": {
                "halted":          False,
                "halt_reason":     None,
                "daily_dd":        daily_dd,
                "weekly_dd":       weekly_dd,
                "peak_dd":         0.0,
                "daily_dd_halt":   config.get("daily_dd_halt",   0.03),
                "daily_dd_reduce": config.get("daily_dd_reduce", 0.02),
                "weekly_dd_halt":  config.get("weekly_dd_halt",  0.07),
                "weekly_dd_reduce":config.get("weekly_dd_reduce",0.05),
                "max_dd_from_peak":config.get("max_dd_from_peak",0.10),
            },
            "positions":      positions,
            "recent_orders":  recent_orders,
            "recent_signals": [],
            "recent_alerts":  [],
            "system": {
                "mode":         "paper" if paper else "live",
                "hmm_model":    model_name,
                "hmm_age_days": hmm_age,
                "last_bar":     None,
                "dry_run":      False,
                "session_start": None,
            },
            "config": config,
            "_source": "alpaca_direct",
        }

    except Exception as exc:
        logger.warning("Alpaca direct fetch failed: %s", exc)
        return {}


def build_fastapi_app() -> Any:
    import time as _time

    try:
        from fastapi import FastAPI, Body
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError:
        raise ImportError("fastapi is required: pip install fastapi uvicorn")

    app = FastAPI(title="Regime Trader Dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML

    @app.get("/api/state", response_class=JSONResponse)
    async def state():
        # Use engine-written state if it exists and is fresh
        if WEB_STATE_PATH.exists():
            age = _time.time() - WEB_STATE_PATH.stat().st_mtime
            if age < _ENGINE_STALE_SECS:
                try:
                    return JSONResponse(json.loads(WEB_STATE_PATH.read_text()))
                except Exception:
                    pass

        # Engine not running — fetch live from Alpaca
        live = _alpaca_direct_state()
        if live:
            return JSONResponse(live)
        return JSONResponse({"error": "No session running and Alpaca credentials not found"}, status_code=503)

    @app.get("/api/chart", response_class=JSONResponse)
    async def chart_data(symbol: str = "SPY", limit: int = 200):
        try:
            from datetime import datetime, timedelta
            client = _get_alpaca_client()
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            end   = datetime.now().strftime("%Y-%m-%d")
            bars  = client.get_bars(symbol.upper(), "1Min", start, end, limit=limit)
            if bars.empty:
                return JSONResponse({
                    "ok": False,
                    "error": f"No 1-min data for {symbol.upper()} — IEX feed only covers US stocks (e.g. SPY, QQQ, AAPL)"
                })
            result = [
                {
                    "time":  int(ts.timestamp()),
                    "open":  float(row["open"]),
                    "high":  float(row["high"]),
                    "low":   float(row["low"]),
                    "close": float(row["close"]),
                }
                for ts, row in bars.iterrows()
            ]
            return JSONResponse({"ok": True, "data": result, "symbol": symbol.upper()})
        except Exception as exc:
            logger.warning("Chart data failed for %s: %s", symbol, exc)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    @app.post("/api/order", response_class=JSONResponse)
    async def place_order(cmd: dict = Body(...)):
        try:
            client = _get_alpaca_client()
            symbol     = cmd.get("symbol", "").upper()
            qty        = int(cmd.get("qty", 1))
            side       = cmd.get("side", "buy").lower()
            order_type = cmd.get("order_type", "market")
            if not symbol:
                return JSONResponse({"ok": False, "error": "Symbol required"}, status_code=400)
            if order_type == "market":
                result = client.submit_market_order(symbol, qty, side)
            elif order_type == "ask":
                quote = client.get_latest_quote(symbol)
                price = float(quote["ask"])
                result = client.submit_limit_order(symbol, qty, side, price)
            else:
                return JSONResponse({"ok": False, "error": f"Unknown order type: {order_type}"}, status_code=400)
            logger.info("Quick order: %s %s %d %s → %s", order_type, side, qty, symbol, result.get("status"))
            return JSONResponse({"ok": True, "order": result})
        except Exception as exc:
            logger.error("Quick order failed: %s", exc)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    @app.post("/api/control", response_class=JSONResponse)
    async def control(cmd: dict = Body(...)):
        valid_actions = {"halt", "resume", "retrain", "set_dry_run", "set_risk", "close_position"}
        action = cmd.get("action", "")
        if action not in valid_actions:
            return JSONResponse({"ok": False, "error": f"Unknown action: {action}"}, status_code=400)

        try:
            pending: list = []
            if CONTROL_PATH.exists():
                pending = json.loads(CONTROL_PATH.read_text())
            pending.append(cmd)
            CONTROL_PATH.write_text(json.dumps(pending))
            logger.info("Control command queued: %s", action)
            return JSONResponse({"ok": True, "action": action})
        except Exception as exc:
            logger.error("Failed to write control.json: %s", exc)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    return app


def run(port: int = 8080) -> None:
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required: pip install uvicorn")

    app = build_fastapi_app()
    logger.info("Web dashboard starting at http://localhost:%d", port)
    print(f"\n  Regime Trader dashboard → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
