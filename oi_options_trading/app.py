"""OI-driven options trading — strategy suite (FastAPI).

For now the suite ships the NIFTY index chart visualization. Strategies will
plug in over time (OI-skew screen, max-pain rebalancer, etc.).

Data source: the continuous 1-min NIFTY master parquet maintained by
fetch_nifty_master.py. Resampling to 1m/5m/15m/1h/1d happens here.

Run:
    python -m uvicorn oi_options_trading.app:app --host 0.0.0.0 --port 8705

Mounted behind nginx at /strategy/ (set APP_BASE=/strategy via env).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse


ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
MASTER     = ROOT / "nifty_1m_master.parquet"
OI_INTRA   = ROOT / "_atm_oi_intraday.parquet"
APP_BASE   = os.environ.get("APP_BASE", "").rstrip("/")


TF_RULE = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "60min", "4h": "240min", "1d": "1D",
}


@lru_cache(maxsize=1)
def _load_master_1m() -> pd.DataFrame:
    if not MASTER.exists():
        return pd.DataFrame()
    df = pd.read_parquet(MASTER)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "volume" not in df.columns:
        df["volume"] = 0
    return (df.sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last")
              .reset_index(drop=True))


@lru_cache(maxsize=8)
def _resampled(tf: str) -> pd.DataFrame:
    df = _load_master_1m()
    if df.empty:
        return df
    if tf == "1m":
        return df
    rule = TF_RULE.get(tf)
    if rule is None:
        raise HTTPException(400, f"unknown tf: {tf}")
    if tf == "1d":
        d = df.copy()
        d["day"] = d["timestamp"].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
        out = (d.groupby("day", as_index=False)
                 .agg(open=("open", "first"), high=("high", "max"),
                      low=("low", "min"),     close=("close", "last"),
                      volume=("volume", "sum")))
        return out.rename(columns={"day": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
    out = (df.set_index("timestamp")
             .resample(rule, closed="left", label="left")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"})
             .dropna(subset=["open", "close"])
             .reset_index())
    return out


def to_unix(ts: pd.Series) -> np.ndarray:
    return (ts.view("int64") // 1_000_000_000).to_numpy()


app = FastAPI(title="Strategy Suite — OI Options")


@app.get("/api/nifty")
def nifty(tf: str = Query("1d")):
    bars = _resampled(tf)
    if bars.empty:
        return JSONResponse({"bars": [], "n": 0, "tf": tf})
    t = to_unix(bars["timestamp"])
    out = [{
        "time" : int(t[i]),
        "open" : float(bars["open"].iloc[i]),
        "high" : float(bars["high"].iloc[i]),
        "low"  : float(bars["low"].iloc[i]),
        "close": float(bars["close"].iloc[i]),
    } for i in range(len(bars))]
    return JSONResponse({"bars": out, "n": len(out), "tf": tf,
                         "first": out[0]["time"], "last": out[-1]["time"]})


@lru_cache(maxsize=1)
def _load_oi_intraday() -> pd.DataFrame:
    if not OI_INTRA.exists():
        return pd.DataFrame()
    df = pd.read_parquet(OI_INTRA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=8)
def _oi_resampled(tf: str) -> pd.DataFrame:
    """Resample the per-tick ATM±1 OI series to the requested TF using 'last'
    aggregation. For TFs coarser than the native ~3-min cadence the line is
    smooth; for 1m the line steps (same value held for 3 consecutive 1m bars)."""
    df = _load_oi_intraday()
    if df.empty:
        return df
    if tf == "1d":
        d = df.copy()
        # Pin the daily timestamp to 09:15 IST so the OI 1d series shares
        # an x-axis bucket with the candle 1d series (which also uses 09:15).
        # The aggregation still picks each day's LAST tick (EOD value).
        d["day"] = d["timestamp"].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
        out = (d.groupby("day", as_index=False)
                 .agg(ce_oi=("ce_oi", "last"),
                      pe_oi=("pe_oi", "last"),
                      atm=("atm", "last"),
                      spot=("spot", "last")))
        return out.rename(columns={"day": "timestamp"})
    rule = TF_RULE.get(tf)
    if rule is None:
        raise HTTPException(400, f"unknown tf: {tf}")
    out = (df.set_index("timestamp")
             .resample(rule, closed="left", label="left")
             .agg({"ce_oi": "last", "pe_oi": "last",
                   "atm":   "last", "spot":  "last"})
             .dropna(subset=["ce_oi", "pe_oi"])
             .reset_index())
    return out


@app.get("/api/nifty_oi")
def nifty_oi(tf: str = Query("1d")):
    """ATM±1 combined OI series (CE + PE) at the requested TF.
    Returns two time-series ready for lightweight-charts line series."""
    df = _oi_resampled(tf)
    if df.empty:
        return JSONResponse({"ce_oi": [], "pe_oi": [], "n": 0, "tf": tf})
    t = to_unix(df["timestamp"])
    ce = [{"time": int(t[i]), "value": float(df["ce_oi"].iloc[i])}
          for i in range(len(df))]
    pe = [{"time": int(t[i]), "value": float(df["pe_oi"].iloc[i])}
          for i in range(len(df))]
    return JSONResponse({"ce_oi": ce, "pe_oi": pe, "n": len(df), "tf": tf})


@lru_cache(maxsize=1)
def _strategy_signals_cached():
    """Run Steps 1+2 once per process; OI aggregate changes only when the
    daily 5 PM cron rebuilds it, so a process-level cache is fine."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from strategies.multi_strike_oi_crossover import compute_signals
    return compute_signals()


@app.get("/api/strategy/multi_strike_oi_crossover")
def strategy_signals():
    """All days that fired a Step-2 crossover signal — BULLISH / BEARISH +
    surrounding OI snapshot, ready to overlay as chart markers."""
    sigs, stats = _strategy_signals_cached()
    return JSONResponse({"signals": sigs, "stats": stats, "n": len(sigs)})


@app.get("/api/range")
def date_range():
    df = _load_master_1m()
    if df.empty:
        return {"first": None, "last": None, "rows": 0}
    return {
        "first": df["timestamp"].iloc[0].isoformat(),
        "last":  df["timestamp"].iloc[-1].isoformat(),
        "rows":  int(len(df)),
        "days":  int(df["timestamp"].dt.date.nunique()),
    }


# ─────────────────────────────────── UI ─────────────────────────────────
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Suite — NIFTY</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg:#0b0d11; --panel:#10131a; --panel-2:#161a23; --border:#202533;
    --border-hi:#2b3142; --text:#e7eaf1; --muted:#7b8294; --accent:#60a5fa;
    --green:#26a69a; --red:#ef5350;
  }
  * { box-sizing: border-box; }
  html, body { height:100%; margin:0; background:var(--bg); color:var(--text);
               font-family:'JetBrains Mono', 'SF Mono', Consolas, monospace; font-size:13px; }
  .bar {
    display:flex; align-items:center; gap:14px; padding:10px 18px;
    background:var(--panel); border-bottom:1px solid var(--border); flex-wrap:wrap;
  }
  .brand { display:flex; align-items:baseline; gap:8px; margin-right:18px; }
  .brand .logo { color:var(--accent); font-size:18px; }
  .brand .title { font-weight:700; letter-spacing:0.5px; }
  .brand .sub { color:var(--muted); font-size:11px; }
  .bar label { color:var(--muted); font-size:11px; letter-spacing:0.5px; text-transform:uppercase; }
  .tf-grp { display:inline-flex; background:var(--panel-2); border:1px solid var(--border-hi);
            border-radius:6px; overflow:hidden; }
  .tf-grp button {
    background:transparent; color:var(--text); border:none; padding:6px 14px; cursor:pointer;
    font:inherit; font-family:inherit; border-right:1px solid var(--border-hi);
  }
  .tf-grp button:last-child { border-right:none; }
  .tf-grp button.on { background:var(--accent); color:#0a0c10; font-weight:600; }
  .tf-grp button:hover:not(.on) { background:var(--panel); }
  .btn {
    background:var(--panel-2); color:var(--text); border:1px solid var(--border-hi);
    border-radius:6px; padding:6px 14px; cursor:pointer; font:inherit; font-family:inherit;
  }
  .btn:hover { background:var(--panel); }
  .spot {
    display:flex; align-items:baseline; gap:8px; padding:5px 12px;
    background:var(--panel-2); border:1px solid var(--border-hi); border-radius:6px;
  }
  .spot .l { color:var(--muted); font-size:11px; }
  .spot strong { color:var(--accent); font-size:15px; font-weight:700; }
  .meta {
    margin-left:auto; color:var(--muted); font-size:11px; display:flex; gap:14px;
  }
  .meta b { color:var(--text); font-weight:600; }
  /* Two-pane layout when OI sub-pane is visible. The pane's flex-basis is
     driven by JS (initial 35%, drag between 15% and 50%) — the divider
     gives a draggable handle to resize. */
  .charts { display:flex; flex-direction:column; width:100%; height:calc(100vh - 56px); }
  #chart   { flex: 1 1 auto; min-height: 0; width:100%; position:relative; }
  #pane-resizer {
    flex: 0 0 6px; width: 100%; background: var(--border);
    cursor: row-resize; transition: background 0.15s;
    border-top: 1px solid var(--border-hi);
    border-bottom: 1px solid var(--border-hi);
  }
  #pane-resizer:hover, #pane-resizer.dragging { background: var(--accent, #60a5fa); }
  #oi-pane { flex: 0 0 35%; min-height: 0; width:100%; position:relative; }
  #oi-pane.hidden, #pane-resizer.hidden, #oi-label.hidden { display: none !important; }
  #oi-label {
    position:absolute; left:14px; top:6px; z-index:5;
    color: var(--muted); font-size:11px; letter-spacing:0.6px;
    background: rgba(11,13,17,0.85); padding:3px 8px; border-radius:4px;
    border: 1px solid var(--border-hi); pointer-events:none;
  }
  #oi-label b { color: var(--text); font-weight:600; }
  #oi-label .ce { color: #26a69a; }
  #oi-label .pe { color: #ef5350; }
</style>
</head>
<body>

<div class="bar">
  <div class="brand">
    <span class="logo">◆</span>
    <span class="title">STRATEGY SUITE</span>
    <span class="sub">NIFTY 50 · index</span>
  </div>

  <label>Timeframe</label>
  <span class="tf-grp" id="tf-grp">
    <button data-tf="1m">1m</button>
    <button data-tf="3m">3m</button>
    <button data-tf="5m">5m</button>
    <button data-tf="15m">15m</button>
    <button data-tf="1h">1h</button>
    <button data-tf="1d" class="on">1d</button>
  </span>

  <button class="btn" id="fit-all">FIT ALL</button>
  <button class="btn" id="oi-toggle" title="Show/hide the ATM±1 combined OI sub-pane">ATM·OI</button>
  <button class="btn" id="strat-toggle" title="Show Step-2 crossover signals (BULLISH ▲ / BEARISH ▼) on the candle chart">STRATEGY</button>
  <span id="strat-stat" style="color:var(--muted); font-size:11px; display:none;">
    <b id="ss-bull" style="color:#26a69a;">—</b> bullish ·
    <b id="ss-bear" style="color:#ef5350;">—</b> bearish ·
    <b id="ss-no">—</b> no-cross
  </span>

  <div class="spot">
    <span class="l">LAST</span>
    <strong id="last-val">—</strong>
  </div>

  <div class="meta">
    <span><b id="bar-count">—</b> bars</span>
    <span><b id="day-count">—</b> days</span>
    <span><b id="range">—</b></span>
  </div>
</div>

<div class="charts" id="charts-wrap">
  <div id="chart"></div>
  <div id="pane-resizer" class="hidden" title="Drag to resize the OI pane (15% – 50%)"></div>
  <div id="oi-pane" class="hidden">
    <div id="oi-label" class="hidden">
      ATM±1 OI &nbsp;·&nbsp; <span class="ce">CE <b id="ce-val">—</b></span>
      &nbsp;·&nbsp; <span class="pe">PE <b id="pe-val">—</b></span>
      &nbsp;·&nbsp; ATM <b id="atm-val">—</b>
    </div>
  </div>
</div>

<script>
const APP_BASE = "__APP_BASE__";
const apiUrl = (p) => (APP_BASE || "") + p;

const state = {
  tf: "1d",
  chart: null,
  candle: null,
  bars: [],
  // OI sub-pane
  oiChart: null,
  ceLine: null, peLine: null,
  ceBars: [], peBars: [],
  showOi: false,
  // Per-tf cache so toggling/switching is instant after first load
  oiByTf: {},
  // Strategy markers
  showStrategy: false,
  signals: [],     // list of {day, signal, signal_time, day_open_time, ...}
};

const $ = (id) => document.getElementById(id);

function fmtPrice(v) { return v == null ? "—" : v.toFixed(2); }
function fmtDateUnix(t) {
  const d = new Date(t * 1000);
  // The bars carry IST timestamps encoded as UTC seconds at the same wall-clock,
  // so calling getUTCxxx gives back the original IST date.
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth()+1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function setupChart() {
  const baseOpts = {
    layout: { background: { color: "#0b0d11" }, textColor: "#e7eaf1" },
    grid: {
      vertLines: { color: "#1a1f2b", visible: true },
      horzLines: { color: "#1a1f2b" },
    },
    crosshair: { mode: 0 },
    rightPriceScale: { borderColor: "#2b3142" },
  };
  state.chart = LightweightCharts.createChart($("chart"), Object.assign({}, baseOpts, {
    timeScale: {
      borderColor: "#2b3142",
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 12,
      barSpacing: 8,
      minBarSpacing: 1.5,
      tickMarkFormatter: (t, tickType) => {
        const d = new Date(t * 1000);
        const yr = d.getUTCFullYear();
        const mo = d.toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" });
        const day = d.getUTCDate();
        if (state.tf === "1d") {
          // For 1d ticks, show "5 Jan '22"
          return `${day} ${mo} '${String(yr).slice(2)}`;
        }
        // For intraday, show "Jan 5 14:30"
        const hh = String(d.getUTCHours()).padStart(2, "0");
        const mm = String(d.getUTCMinutes()).padStart(2, "0");
        return `${mo} ${day} ${hh}:${mm}`;
      },
    },
  }));
  state.candle = state.chart.addCandlestickSeries({
    upColor: "#26a69a", downColor: "#ef5350",
    borderUpColor: "#26a69a", borderDownColor: "#ef5350",
    wickUpColor: "#26a69a", wickDownColor: "#ef5350",
  });

  // OI sub-pane chart — two line series (CE teal / PE red), x-axis hidden
  // (synced with the main chart's axis via subscribeVisibleLogicalRangeChange)
  state.oiChart = LightweightCharts.createChart($("oi-pane"), Object.assign({}, baseOpts, {
    timeScale: { borderColor: "#2b3142", timeVisible: false, secondsVisible: false,
                 visible: false },
    rightPriceScale: { borderColor: "#2b3142" },
  }));
  // Compact OI formatter: 12,299,040 -> "12.3M", 850000 -> "850K".
  // Without this the OI right-axis labels are 8-10 chars wide vs. ~6 for
  // the candle prices, so the two charts' plot areas don't end at the
  // same X coordinate and time positions visually drift.
  const oiFmt = {
    type: 'custom',
    formatter: (v) => {
      if (v >= 1e7) return (v/1e7).toFixed(1) + 'Cr';
      if (v >= 1e5) return (v/1e5).toFixed(1) + 'L';
      if (v >= 1e3) return (v/1e3).toFixed(0) + 'K';
      return String(v|0);
    },
  };
  state.ceLine = state.oiChart.addLineSeries({
    color: "#26a69a", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: oiFmt,
  });
  state.peLine = state.oiChart.addLineSeries({
    color: "#ef5350", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: oiFmt,
  });
  // Lock both charts to the same right-gutter width so a given time t
  // lands at exactly the same X coordinate on both panes.
  const GUTTER = 64;
  state.chart.priceScale('right').applyOptions({ minimumWidth: GUTTER });
  state.oiChart.priceScale('right').applyOptions({ minimumWidth: GUTTER });
  // Two-way TIME-axis sync. We can't use logical range because the two
  // series carry different bar counts (1591 daily candles vs 1569 daily OI
  // points across 2020-2026). subscribeVisibleTimeRangeChange + setVisibleRange
  // keeps both panes locked to the same wall-clock window regardless.
  let syncing = false;
  const sync = (src, dst) => {
    src.timeScale().subscribeVisibleTimeRangeChange((r) => {
      if (syncing || !r || r.from == null || r.to == null) return;
      syncing = true;
      try { dst.timeScale().setVisibleRange({ from: r.from, to: r.to }); } catch (_) {}
      syncing = false;
    });
  };
  sync(state.chart,   state.oiChart);
  sync(state.oiChart, state.chart);

  // ── Crosshair sync ─────────────────────────────────────────────
  // Hovering EITHER pane draws the dashed crosshair on BOTH at the same
  // time-cursor. The horizontal line on the "other" chart sits at the
  // value of that other chart's series at the hovered time, so each pane
  // shows its own scale-appropriate reading.
  let xhSyncing = false;
  state.chart.subscribeCrosshairMove((p) => {
    // Update top-bar readout
    if (p && p.seriesData) {
      const c = p.seriesData.get(state.candle);
      if (c && c.close != null) $("last-val").textContent = fmtPrice(c.close);
    }
    if (xhSyncing || !state.oiChart || !state.ceLine) return;
    xhSyncing = true;
    try {
      if (!p || p.time == null) {
        state.oiChart.clearCrosshairPosition();
      } else {
        const ce = findAtTime(state.ceBars, p.time);
        if (ce != null) {
          state.oiChart.setCrosshairPosition(ce.value, p.time, state.ceLine);
        }
      }
    } catch (_) {}
    xhSyncing = false;
  });
  state.oiChart.subscribeCrosshairMove((p) => {
    // OI readouts in the sub-pane label
    if (p && p.seriesData) {
      const ce = p.seriesData.get(state.ceLine);
      const pe = p.seriesData.get(state.peLine);
      if (ce) $("ce-val").textContent = (ce.value/1e5).toFixed(2) + " L";
      if (pe) $("pe-val").textContent = (pe.value/1e5).toFixed(2) + " L";
    }
    if (xhSyncing || !state.chart || !state.candle) return;
    xhSyncing = true;
    try {
      if (!p || p.time == null) {
        state.chart.clearCrosshairPosition();
      } else {
        const c = findAtTime(state.bars, p.time);
        if (c != null) {
          state.chart.setCrosshairPosition(c.close, p.time, state.candle);
        }
      }
    } catch (_) {}
    xhSyncing = false;
  });
}

// Convert one of the signal objects into a lightweight-charts marker, anchored
// to the candle time appropriate for the active TF (day-open at 1d, exact
// signal minute on intraday TFs).
//
// Display:
//   BULLISH  →  green ▲ below bar  · text "BUY <spot>"
//   BEARISH  →  red   ▼ above bar  · text "SELL <spot>"
// `size: 2` ~doubles the arrow + text vs default; lightweight-charts renders
// the text in a heavier weight at that size so it reads bold.
function signalToMarker(s) {
  const isDaily = state.tf === '1d';
  const isoStr = isDaily ? s.day_open_time : s.signal_time;
  const t = Math.floor(new Date(isoStr + (isoStr.endsWith('Z') ? '' : 'Z')).getTime() / 1000);
  const spot = Math.round(s.signal_spot).toLocaleString();
  if (s.signal === 'BULLISH') {
    return { time: t, position: 'belowBar', color: '#22c55e',
             shape: 'arrowUp',  text: 'BUY ' + spot, size: 2 };
  }
  return   { time: t, position: 'aboveBar', color: '#ef4444',
             shape: 'arrowDown', text: 'SELL ' + spot, size: 2 };
}

function applyStrategyMarkers() {
  if (!state.candle) return;
  if (!state.showStrategy || !state.signals.length) {
    state.candle.setMarkers([]);
    return;
  }
  // Markers must match a bar time exactly. setMarkers expects sorted ascending.
  const ms = state.signals.map(signalToMarker)
                          .sort((a, b) => a.time - b.time);
  state.candle.setMarkers(ms);
}

async function loadStrategySignals() {
  if (state.signals.length) {
    applyStrategyMarkers();
    return;
  }
  try {
    const r = await fetch(apiUrl('/api/strategy/multi_strike_oi_crossover'));
    const j = await r.json();
    state.signals = j.signals || [];
    const st = j.stats || {};
    $('ss-bull').textContent = (st.bull || 0).toLocaleString();
    $('ss-bear').textContent = (st.bear || 0).toLocaleString();
    $('ss-no').textContent   = (st.no_signal || 0).toLocaleString();
    applyStrategyMarkers();
  } catch (_) {}
}

function toggleStrategy() {
  state.showStrategy = !state.showStrategy;
  $('strat-stat').style.display = state.showStrategy ? 'inline' : 'none';
  const btn = $('strat-toggle');
  if (state.showStrategy) {
    btn.style.background = 'linear-gradient(135deg, #a78bfa 0%, #7c5fd6 100%)';
    btn.style.color = '#0a0c10';
    btn.style.borderColor = 'transparent';
    btn.style.fontWeight = '700';
    loadStrategySignals();
  } else {
    btn.style.background = ''; btn.style.color = '';
    btn.style.borderColor = ''; btn.style.fontWeight = '';
    state.candle.setMarkers([]);
  }
}

// Binary search for the bar at-or-before `time` in a sorted-by-time array.
function findAtTime(bars, time) {
  if (!bars || !bars.length) return null;
  let lo = 0, hi = bars.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (bars[mid].time <= time) lo = mid + 1;
    else hi = mid - 1;
  }
  return hi >= 0 ? bars[hi] : null;
}

async function loadTF(tf) {
  state.tf = tf;
  $("tf-grp").querySelectorAll("button").forEach(b => {
    b.classList.toggle("on", b.dataset.tf === tf);
  });
  const r = await fetch(apiUrl(`/api/nifty?tf=${tf}`));
  const j = await r.json();
  state.bars = j.bars || [];
  state.candle.setData(state.bars);
  if (state.bars.length) {
    const lastBar = state.bars[state.bars.length - 1];
    $("last-val").textContent = fmtPrice(lastBar.close);
    $("bar-count").textContent = state.bars.length.toLocaleString();
    const first = fmtDateUnix(state.bars[0].time);
    const last  = fmtDateUnix(lastBar.time);
    $("range").textContent = `${first} → ${last}`;
  }
  state.chart.timeScale().fitContent();
  // Re-load OI at this TF if the sub-pane is open
  if (state.showOi) await loadOI(tf);
  // Re-anchor strategy markers (timestamps differ by TF)
  if (state.showStrategy) applyStrategyMarkers();
}

async function loadOI(tf) {
  if (state.oiByTf[tf]) {
    state.ceBars = state.oiByTf[tf].ce;
    state.peBars = state.oiByTf[tf].pe;
    state.ceLine.setData(state.ceBars);
    state.peLine.setData(state.peBars);
    syncOiToIndex();
    return;
  }
  try {
    const r = await fetch(apiUrl(`/api/nifty_oi?tf=${tf}`));
    const j = await r.json();
    state.ceBars = j.ce_oi || [];
    state.peBars = j.pe_oi || [];
    state.oiByTf[tf] = { ce: state.ceBars, pe: state.peBars };
    state.ceLine.setData(state.ceBars);
    state.peLine.setData(state.peBars);
    if (state.ceBars.length) {
      const last = state.ceBars[state.ceBars.length - 1];
      $("ce-val").textContent = (last.value/1e5).toFixed(2) + " L";
    }
    if (state.peBars.length) {
      const last = state.peBars[state.peBars.length - 1];
      $("pe-val").textContent = (last.value/1e5).toFixed(2) + " L";
    }
    syncOiToIndex();
  } catch (e) {
    state.ceBars = []; state.peBars = [];
  }
}

// Force the OI pane to match the index chart's current visible time range.
// Called after setData (when the OI chart auto-fits to its own data, which
// is wider than the index range and breaks the visual lockstep).
function syncOiToIndex() {
  if (!state.chart || !state.oiChart) return;
  try {
    const r = state.chart.timeScale().getVisibleRange();
    if (r && r.from != null && r.to != null) {
      state.oiChart.timeScale().setVisibleRange({ from: r.from, to: r.to });
    }
  } catch (_) {}
}

function toggleOi() {
  state.showOi = !state.showOi;
  $("oi-pane").classList.toggle("hidden",  !state.showOi);
  $("pane-resizer").classList.toggle("hidden", !state.showOi);
  $("oi-label").classList.toggle("hidden", !state.showOi);
  $("oi-toggle").classList.toggle("on", state.showOi);
  // Style the toggle for an active state similar to TF buttons
  const btn = $("oi-toggle");
  if (state.showOi) {
    btn.style.background = "linear-gradient(135deg, #26a69a 0%, #1d8a80 100%)";
    btn.style.color = "#0a0c10";
    btn.style.borderColor = "transparent";
    btn.style.fontWeight = "700";
  } else {
    btn.style.background = "";
    btn.style.color = "";
    btn.style.borderColor = "";
    btn.style.fontWeight = "";
  }
  if (state.showOi) {
    loadOI(state.tf);
    setTimeout(() => {
      state.oiChart.applyOptions({ autoSize: true });
      state.chart.applyOptions({ autoSize: true });
    }, 30);
  }
}

async function loadMeta() {
  try {
    const r = await fetch(apiUrl(`/api/range`));
    const j = await r.json();
    if (j.days) $("day-count").textContent = j.days.toLocaleString();
  } catch (_) {}
}

function init() {
  setupChart();
  $("tf-grp").querySelectorAll("button").forEach(b => {
    b.addEventListener("click", () => loadTF(b.dataset.tf));
  });
  $("fit-all").addEventListener("click", () => {
    state.chart.timeScale().fitContent();
    if (state.oiChart) state.oiChart.timeScale().fitContent();
  });
  $("oi-toggle").addEventListener("click", toggleOi);
  $("strat-toggle").addEventListener("click", toggleStrategy);

  // ── OI pane drag-to-resize ───────────────────────────────────────
  // The pane occupies 15%–50% of the .charts container; user drags the
  // 6px handle to size between. Clamped both ways so neither pane can
  // collapse and the OI pane can't exceed half the viewport.
  (function attachResizer() {
    const handle = $("pane-resizer");
    const wrap   = $("charts-wrap");
    const pane   = $("oi-pane");
    const MIN_PCT = 15, MAX_PCT = 50;
    let dragging = false;
    handle.addEventListener("mousedown", (e) => {
      dragging = true;
      handle.classList.add("dragging");
      document.body.style.cursor = "row-resize";
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const rect = wrap.getBoundingClientRect();
      const fromBottom = rect.bottom - e.clientY;
      let pct = (fromBottom / rect.height) * 100;
      pct = Math.max(MIN_PCT, Math.min(MAX_PCT, pct));
      pane.style.flexBasis = pct.toFixed(2) + "%";
      // Both charts need to re-measure when the pane resizes
      if (state.chart)   state.chart.applyOptions({ autoSize: true });
      if (state.oiChart) state.oiChart.applyOptions({ autoSize: true });
    });
    window.addEventListener("mouseup", () => {
      if (!dragging) return;
      dragging = false;
      handle.classList.remove("dragging");
      document.body.style.cursor = "";
    });
  })();
  loadTF("1d");
  loadMeta();
  window.addEventListener("resize", () => {
    state.chart.applyOptions({ autoSize: true });
    if (state.oiChart) state.oiChart.applyOptions({ autoSize: true });
  });
}

document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML.replace("__APP_BASE__", APP_BASE)
