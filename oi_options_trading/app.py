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
        d["day"] = d["timestamp"].dt.normalize() + pd.Timedelta(hours=15, minutes=30)
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
  /* Two-pane layout when OI sub-pane is visible */
  .charts { display:flex; flex-direction:column; width:100%; height:calc(100vh - 56px); }
  #chart   { flex: 1 1 auto; min-height: 0; width:100%; position:relative; }
  #oi-pane { flex: 0 0 200px; min-height: 160px; width:100%; position:relative;
             border-top: 1px solid var(--border); }
  #oi-pane.hidden, #oi-label.hidden { display: none !important; }
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

<div class="charts">
  <div id="chart"></div>
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
  // Show "last bar" value in the top bar as user scrubs the crosshair
  state.chart.subscribeCrosshairMove((p) => {
    if (!p || !p.seriesData) return;
    const v = p.seriesData.get(state.candle);
    if (v && v.close != null) $("last-val").textContent = fmtPrice(v.close);
  });

  // OI sub-pane chart — two line series (CE teal / PE red), x-axis hidden
  // (synced with the main chart's axis via subscribeVisibleLogicalRangeChange)
  state.oiChart = LightweightCharts.createChart($("oi-pane"), Object.assign({}, baseOpts, {
    timeScale: { borderColor: "#2b3142", timeVisible: false, secondsVisible: false,
                 visible: false },
    rightPriceScale: { borderColor: "#2b3142" },
  }));
  state.ceLine = state.oiChart.addLineSeries({
    color: "#26a69a", lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
  });
  state.peLine = state.oiChart.addLineSeries({
    color: "#ef5350", lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
  });
  // Two-way time-axis sync so panning/zooming either chart moves both
  let syncing = false;
  const sync = (src, dst) => {
    src.timeScale().subscribeVisibleLogicalRangeChange((r) => {
      if (syncing || !r) return;
      syncing = true;
      try { dst.timeScale().setVisibleLogicalRange(r); } catch (_) {}
      syncing = false;
    });
  };
  sync(state.chart,   state.oiChart);
  sync(state.oiChart, state.chart);
  // Update the OI readout label from the OI chart's crosshair
  state.oiChart.subscribeCrosshairMove((p) => {
    if (!p || !p.seriesData) return;
    const ce = p.seriesData.get(state.ceLine);
    const pe = p.seriesData.get(state.peLine);
    if (ce) $("ce-val").textContent = (ce.value/1e5).toFixed(2) + " L";
    if (pe) $("pe-val").textContent = (pe.value/1e5).toFixed(2) + " L";
    // Also try to populate from the candle chart's data if hovered there
    const c = p.seriesData.get(state.candle);
    if (c && c.close != null) $("last-val").textContent = fmtPrice(c.close);
  });
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
}

async function loadOI(tf) {
  if (state.oiByTf[tf]) {
    state.ceBars = state.oiByTf[tf].ce;
    state.peBars = state.oiByTf[tf].pe;
    state.ceLine.setData(state.ceBars);
    state.peLine.setData(state.peBars);
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
  } catch (e) {
    state.ceBars = []; state.peBars = [];
  }
}

function toggleOi() {
  state.showOi = !state.showOi;
  $("oi-pane").classList.toggle("hidden",  !state.showOi);
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
