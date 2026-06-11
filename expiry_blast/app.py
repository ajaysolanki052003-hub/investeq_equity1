"""Expiry Blast — dashboard (FastAPI).

Replay any historical expiry day through the entry engine and watch the four
conditions (A proximity / B call unwind / C put writing / D price confirm)
evaluate tick by tick against the call wall, with the audit trail and the
fired entry signal (if any).

Run:
    python -m uvicorn expiry_blast.app:app --host 0.0.0.0 --port 8706

Mounted behind nginx at /blast/ (set APP_BASE=/blast via env).
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .config import BlastConfig
from .data import HistoricalFeed, expiry_days
from .engine import SignalEngine

APP_BASE = os.environ.get("APP_BASE", "").rstrip("/")
OUT_DIR = Path(__file__).parent / "output"
SIGNALS_CSV = OUT_DIR / "signals.csv"

app = FastAPI(title="Expiry Blast — Short Covering entry engine")

CFG = BlastConfig()
ENGINE = SignalEngine(CFG)

# Two studied configurations, switchable in the UI. "default" is the
# quality set (engine defaults); "volume" trades ~5.6/yr at +0.25R
# (TP+80/SL-40 bracket) — see README §Relaxation sweep.
PRESETS: dict[str, dict] = {
    "default": {},
    "volume": {"spot_proximity_pct": 0.30, "call_oi_unwind_pct": 10.0,
               "put_oi_buildup_pct": 5.0, "put_strikes_required": 1,
               "oi_lookback_min": 30},
}
PRESET_SIGNALS = {
    "default": SIGNALS_CSV,
    "volume": OUT_DIR / "signals_volume.csv",
}


@lru_cache(maxsize=1)
def _expiry_days() -> list[str]:
    return [d.isoformat() for d in expiry_days()]


def _signals_df(preset: str) -> pd.DataFrame:
    p = PRESET_SIGNALS.get(preset, SIGNALS_CSV)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _signal_days(preset: str) -> set[str]:
    df = _signals_df(preset)
    if df.empty or "timestamp" not in df.columns:
        return set()
    return set(pd.to_datetime(df["timestamp"]).dt.date.astype(str))


@app.get("/api/config")
def config(preset: str = "default"):
    cfg = (BlastConfig.from_dict({**CFG.to_dict(), **PRESETS[preset]})
           if PRESETS.get(preset) else CFG)
    return JSONResponse({**cfg.to_dict(), "preset": preset,
                         "presets": list(PRESETS)})


@app.get("/api/days")
def days(preset: str = "default"):
    sig = _signal_days(preset)
    return JSONResponse({"days": [{"day": d, "signal": d in sig}
                                  for d in _expiry_days()]})


@app.get("/api/signals")
def signals(preset: str = "default"):
    df = _signals_df(preset)
    if df.empty:
        return JSONResponse({"signals": [], "note": "run the backtest first: "
                             "python -m expiry_blast.backtest"})
    df = df.where(pd.notna(df), None)
    return JSONResponse({"signals": df.to_dict(orient="records"),
                         "preset": preset})


@app.get("/api/trade_chart")
def trade_chart(day: str, strike: int, entry_ts: str):
    """1-min premium bars of the bought CE for the whole expiry day —
    the trade's life from entry to EOD, chart-ready."""
    from .data import DATA_ROOT
    d = date.fromisoformat(day)
    p = DATA_ROOT / "options" / f"{d:%Y%m%d}.parquet"
    if not p.exists():
        raise HTTPException(404, f"no options data for {day}")
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    leg = df[(df["strike"] == strike) & (df["option_type"] == "CE")
             & (pd.to_datetime(df["expiry"]).dt.date == d)
             ].sort_values("timestamp")
    if leg.empty:
        raise HTTPException(404, f"no bars for {strike}CE on {day}")
    bars = [{"time": int(r["timestamp"].value // 1_000_000_000),
             "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"])}
            for _, r in leg.iterrows()]
    ent_t = pd.Timestamp(entry_ts)
    nxt = leg[leg["timestamp"] >= ent_t]
    entry_premium = float(nxt["open"].iloc[0]) if len(nxt) else None
    return JSONResponse({
        "bars": bars, "strike": strike, "day": day,
        "entry_time": int(ent_t.value // 1_000_000_000),
        "entry_premium": entry_premium,
        "tp": entry_premium * 1.8 if entry_premium else None,
        "sl": entry_premium * 0.6 if entry_premium else None})


@lru_cache(maxsize=32)
def _replay(day_iso: str, overrides: tuple = ()) -> dict:
    d = date.fromisoformat(day_iso)
    cfg = (BlastConfig.from_dict({**CFG.to_dict(), **dict(overrides)})
           if overrides else CFG)
    try:
        feed = HistoricalFeed(d)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    res = SignalEngine(cfg).run_day(feed)

    bars = feed._candles(cfg.candle_tf_min)
    candles = [{"time": int(r["timestamp"].value // 1_000_000_000),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"])}
               for r in bars.to_dict(orient="records")]

    # Wall step-line over evaluation time, for overlay on the candle chart.
    wall = [{"time": int(pd.Timestamp(e["ts"]).value // 1_000_000_000),
             "value": e["wall_strike"]}
            for e in res.audit if e.get("wall_strike")]

    return {"day": day_iso, "expiry": res.expiry,
            "is_expiry_day": res.skipped_reason is None,
            "skipped_reason": res.skipped_reason,
            "candles": candles, "wall": wall,
            "audit": res.audit,
            "signals": res.signals,
            "config": cfg.to_dict()}


@app.get("/api/day/{day_iso}")
def day(day_iso: str,
        preset: str = "default",
        put_strikes: int | None = None,
        buildup: float | None = None,
        unwind: float | None = None,
        proximity: float | None = None,
        lookback: int | None = None):
    over = dict(PRESETS.get(preset) or {})
    over.update({k: v for k, v in {
        "put_strikes_required": put_strikes,
        "put_oi_buildup_pct": buildup,
        "call_oi_unwind_pct": unwind,
        "spot_proximity_pct": proximity,
        "oi_lookback_min": lookback,
    }.items() if v is not None})
    return JSONResponse(_replay(day_iso, tuple(sorted(over.items()))))


# ─── UI ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Expiry Blast — Short Covering Entry Engine</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root {
  --bg:#0b0e13; --panel:#11161f; --line:#1d2533; --text:#d7dde8;
  --muted:#67718a; --green:#26a69a; --red:#ef5350; --amber:#facc15;
  --blue:#60a5fa; --violet:#a78bfa;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text);
       font:13px/1.45 "Segoe UI", system-ui, sans-serif; }
header { display:flex; align-items:center; gap:14px; padding:10px 16px;
         border-bottom:1px solid var(--line); }
header h1 { font-size:15px; font-weight:600; letter-spacing:.4px; }
header .sub { color:var(--muted); font-size:12px; }
select, button {
  background:var(--panel); color:var(--text); border:1px solid var(--line);
  border-radius:6px; padding:5px 9px; font:inherit; cursor:pointer;
}
.layout { display:grid; grid-template-columns: 270px 1fr; gap:10px;
          padding:10px 16px; height:calc(100vh - 49px); }
.side { display:flex; flex-direction:column; gap:10px; overflow:hidden; }
.panel { background:var(--panel); border:1px solid var(--line);
         border-radius:8px; padding:10px; }
.panel h2 { font-size:11px; text-transform:uppercase; letter-spacing:1px;
            color:var(--muted); margin-bottom:8px; }
#daylist { overflow-y:auto; flex:1; }
#daylist .d { padding:4px 8px; border-radius:5px; cursor:pointer;
              display:flex; justify-content:space-between; }
#daylist .d:hover { background:#171e2b; }
#daylist .d.sel { background:#1b2435; }
#daylist .d .dot { color:var(--green); }
#cfg { font-size:11.5px; color:var(--muted); }
#cfg b { color:var(--text); font-weight:600; }
.main { display:flex; flex-direction:column; gap:10px; overflow:hidden; }
#banner { display:none; padding:9px 12px; border-radius:8px; font-weight:600; }
#banner.sig { display:block; background:#10271f; border:1px solid var(--green);
              color:var(--green); }
#banner.nosig { display:block; background:#16181d; border:1px solid var(--line);
                color:var(--muted); font-weight:400; }
#chart { flex:1.4; min-height:240px; }
#audit-wrap { flex:1; overflow-y:auto; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th, td { padding:3px 8px; text-align:right; white-space:nowrap; }
th { position:sticky; top:0; background:var(--panel); color:var(--muted);
     font-weight:500; text-transform:uppercase; font-size:10.5px;
     letter-spacing:.6px; border-bottom:1px solid var(--line); }
td:first-child, th:first-child { text-align:left; }
tr.ev:hover { background:#171e2b; cursor:pointer; }
tr.sigrow { background:#10271f; }
.ok { color:var(--green); font-weight:700; }
.no { color:#3d4458; }
.skip { color:var(--amber); }
tr.detail td { text-align:left; background:#0d1118; color:var(--muted);
               font:11px/1.5 Consolas, monospace; white-space:pre-wrap; }
</style>
</head>
<body>
<header>
  <h1>EXPIRY BLAST</h1>
  <span class="sub">expiry-day short covering · entry engine (phase 1)</span>
  <select id="day-sel"></select>
  <span class="sub">preset:</span>
  <select id="preset" title="DEFAULT: quality set — A 0.15% · B 10% · C 10% (1of2) · LB 15m → 14 trades, 43% bracket-win. VOLUME: A 0.30% · B 10% · C 5% (1of2) · LB 30m → 36 trades, 42% win, ~5.6/yr.">
    <option value="default">DEFAULT · 14 trades</option>
    <option value="volume">VOLUME · 36 trades</option>
  </select>
  <span class="sub" id="stat"></span>
</header>

<div class="layout">
  <div class="side">
    <div class="panel" style="flex:1.2; display:flex; flex-direction:column; overflow:hidden;">
      <h2>Trades <span id="ntrades"></span></h2>
      <div id="tradelist" style="overflow-y:auto; flex:1;"></div>
    </div>
    <div class="panel" style="flex:1; display:flex; flex-direction:column; overflow:hidden;">
      <h2>Expiry days <span id="ndays"></span></h2>
      <div id="daylist"></div>
    </div>
    <div class="panel">
      <h2>Config</h2>
      <div id="cfg"></div>
    </div>
  </div>

  <div class="main">
    <div id="banner"></div>
    <div id="chart" class="panel"></div>
    <div id="prem-wrap" class="panel" style="display:none; flex:1; min-height:180px; position:relative;">
      <span id="prem-title" style="position:absolute; top:6px; left:10px; z-index:5;
            font-size:11px; color:var(--muted); letter-spacing:.6px;"></span>
      <button id="prem-close" style="position:absolute; top:4px; right:8px; z-index:5;
            background:none; border:none; color:var(--muted); cursor:pointer;">✕</button>
      <div id="prem-chart" style="position:absolute; inset:0; padding-top:22px;"></div>
    </div>
    <div id="audit-wrap" class="panel">
      <table id="audit">
        <thead><tr>
          <th>time</th><th>spot</th><th>wall</th>
          <th>A prox</th><th>B CE unwind</th><th>C PE build</th><th>D close></th>
          <th></th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const APP_BASE = "__APP_BASE__";
const api = (p) => fetch((APP_BASE || "") + p).then(r => {
  if (!r.ok) throw new Error(r.status); return r.json(); });

let chart, candleSeries, wallSeries;

function initChart() {
  const el = document.getElementById("chart");
  chart = LightweightCharts.createChart(el, {
    layout: { background: { color: "#11161f" }, textColor: "#67718a" },
    grid: { vertLines: { color: "#161c28" }, horzLines: { color: "#161c28" } },
    timeScale: { timeVisible: true, secondsVisible: false },
    rightPriceScale: { borderColor: "#1d2533" },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: "#26a69a", downColor: "#ef5350", borderVisible: false,
    wickUpColor: "#26a69a", wickDownColor: "#ef5350" });
  wallSeries = chart.addLineSeries({
    color: "#facc15", lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed,
    priceLineVisible: false, lastValueVisible: true, title: "call wall" });
  new ResizeObserver(() => chart.applyOptions(
    { width: el.clientWidth, height: el.clientHeight })).observe(el);
}

function fmtPct(v) { return v === null || v === undefined ? "·" : v.toFixed(1) + "%"; }
function cell(cond, text) {
  if (!cond) return `<td class="skip">skip</td>`;
  return `<td class="${cond.pass ? "ok" : "no"}">${text}</td>`;
}

let curDay = null;
const curPreset = () => document.getElementById("preset").value;

async function loadDay(day) {
  curDay = day;
  document.getElementById("stat").textContent = "loading " + day + " ...";
  const d = await api("/api/day/" + day + "?preset=" + curPreset());
  document.getElementById("stat").textContent =
    d.expiry ? "expiry " + d.expiry : "";

  candleSeries.setData(d.candles);
  wallSeries.setData(d.wall);

  // Entry marker
  const markers = [];
  for (const s of d.signals) {
    markers.push({ time: Math.floor(new Date(s.timestamp.replace(" ","T")
        + "+05:30").getTime() / 1000),
      position: "belowBar", color: "#26a69a", shape: "arrowUp",
      text: "BUY " + s.entry_strike + "CE @ " + (s.entry_premium ?? "?") });
  }
  candleSeries.setMarkers(markers);
  chart.timeScale().fitContent();

  // Banner
  const b = document.getElementById("banner");
  if (d.signals.length) {
    const s = d.signals[0];
    b.className = "sig";
    b.textContent = `ENTRY ${s.timestamp} — BUY 1 lot ${s.entry_strike} CE @ `
      + `${s.entry_premium ?? "n/a"} | wall ${s.max_call_oi_strike} `
      + `(CE ${s.call_oi_pct_change}%) | PE build `
      + Object.entries(s.put_oi_pct_changes).map(([k,v]) => `${k}: ${v}%`).join(", ")
      + ` | 5m close ${s.candle_close}`;
  } else {
    b.className = "nosig";
    b.textContent = d.skipped_reason
      ? "day skipped — " + d.skipped_reason
      : "no entry signal fired on " + day
        + " — all four conditions never aligned (see audit below)";
  }

  // Audit table
  const tb = document.querySelector("#audit tbody");
  tb.innerHTML = "";
  for (const e of d.audit) {
    const tr = document.createElement("tr");
    tr.className = "ev" + ("signal" in e ? " sigrow" : "");
    const t = e.ts.slice(11, 16);
    if (e.skipped) {
      tr.innerHTML = `<td>${t}</td><td colspan="6" class="skip">${e.skipped}</td><td></td>`;
    } else {
      const c = e.conditions;
      const cPct = c.C_put_writing.strikes.map(s => fmtPct(s.pct_change)).join(" / ");
      tr.innerHTML =
        `<td>${t}</td><td>${e.spot}</td><td>${e.wall_strike}</td>`
        + cell(c.A_proximity, fmtPct(c.A_proximity.dist_pct))
        + cell(c.B_call_unwind, fmtPct(c.B_call_unwind.pct_change))
        + cell(c.C_put_writing, cPct)
        + cell(c.D_price_confirm, c.D_price_confirm.close ?? "·")
        + `<td>${"signal" in e ? "★" : ""}</td>`;
    }
    tr.onclick = () => toggleDetail(tr, e);
    tb.appendChild(tr);
  }
}

function toggleDetail(tr, e) {
  if (tr.nextSibling && tr.nextSibling.classList?.contains("detail")) {
    tr.nextSibling.remove(); return;
  }
  const d = document.createElement("tr");
  d.className = "detail";
  d.innerHTML = `<td colspan="8">${JSON.stringify(e, null, 1)}</td>`;
  tr.after(d);
}

// ── Trade premium chart (entry → EOD) ────────────────────────────────
let premChart = null, premSeries = null, premLines = [];

function showTradeChart(sig) {
  const wrap = document.getElementById("prem-wrap");
  wrap.style.display = "block";
  if (!premChart) {
    const el = document.getElementById("prem-chart");
    premChart = LightweightCharts.createChart(el, {
      layout: { background: { color: "#11161f" }, textColor: "#67718a" },
      grid: { vertLines: { color: "#161c28" }, horzLines: { color: "#161c28" } },
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: "#1d2533" },
    });
    premSeries = premChart.addCandlestickSeries({
      upColor: "#26a69a", downColor: "#ef5350", borderVisible: false,
      wickUpColor: "#26a69a", wickDownColor: "#ef5350" });
    new ResizeObserver(() => premChart.applyOptions(
      { width: el.clientWidth, height: el.clientHeight - 22 })).observe(el);
  }
  const day = sig.timestamp.slice(0, 10);
  api(`/api/trade_chart?day=${day}&strike=${sig.entry_strike}`
      + `&entry_ts=${encodeURIComponent(sig.timestamp)}`)
    .then(d => {
      premSeries.setData(d.bars);
      for (const ln of premLines) premSeries.removePriceLine(ln);
      premLines = [];
      const mk = (price, color, title) => premLines.push(
        premSeries.createPriceLine({ price, color, lineWidth: 1,
          lineStyle: LightweightCharts.LineStyle.Dashed, title }));
      if (d.entry_premium) {
        mk(d.entry_premium, "#60a5fa", "entry " + d.entry_premium.toFixed(1));
        mk(d.tp, "#26a69a", "TP +80%");
        mk(d.sl, "#ef5350", "SL -40%");
      }
      premSeries.setMarkers([{ time: d.entry_time, position: "belowBar",
        color: "#26a69a", shape: "arrowUp", text: "BUY" }]);
      premChart.timeScale().fitContent();
      document.getElementById("prem-title").textContent =
        `${day} · ${sig.entry_strike} CE premium (1m) — entry to EOD`;
    })
    .catch(err => {
      document.getElementById("prem-title").textContent =
        "premium data unavailable: " + err.message;
      premSeries && premSeries.setData([]);
    });
}

// ── Preset-aware loaders ─────────────────────────────────────────────
let dayPick = null;

async function loadPreset() {
  const preset = curPreset();
  const cfg = await api("/api/config?preset=" + preset);
  document.getElementById("cfg").innerHTML = [
    ["instrument", cfg.instrument],
    ["window", cfg.window_start + " – " + cfg.window_end + " IST"],
    ["proximity", cfg.spot_proximity_pct + " %"],
    ["CE unwind ≥", cfg.call_oi_unwind_pct + " %"],
    ["PE buildup ≥", cfg.put_oi_buildup_pct + " % ("
       + cfg.put_strikes_required + " of 2)"],
    ["OI lookback", cfg.oi_lookback_min + " min"],
    ["candle TF", cfg.candle_tf_min + " min"],
    ["max entries/day", cfg.max_signals_per_day],
  ].map(([k, v]) => `${k} <b style="float:right">${v}</b>`).join("<br>");

  const [{ days }, { signals }] = await Promise.all([
    api("/api/days?preset=" + preset),
    api("/api/signals?preset=" + preset)]);

  // Day list + selector
  document.getElementById("ndays").textContent = "(" + days.length + ")";
  const list = document.getElementById("daylist");
  const sel = document.getElementById("day-sel");
  list.innerHTML = ""; sel.innerHTML = "";
  for (const d of [...days].reverse()) {
    const row = document.createElement("div");
    row.className = "d"; row.dataset.day = d.day;
    row.innerHTML = `<span>${d.day}</span>` +
      (d.signal ? `<span class="dot">●</span>` : "");
    row.onclick = () => dayPick(d.day);
    list.appendChild(row);
    sel.add(new Option(d.day + (d.signal ? " ●" : ""), d.day));
  }

  // Trade list — newest first, click = open that day + premium chart
  const tl = document.getElementById("tradelist");
  document.getElementById("ntrades").textContent = "(" + signals.length + ")";
  tl.innerHTML = "";
  for (const s of [...signals].reverse()) {
    const day = s.timestamp.slice(0, 10);
    const e = s.entry_premium, eod = s.premium_eod, peak = s.premium_max_after;
    const ret = (e && eod != null) ? ((eod / e - 1) * 100) : null;
    const col = ret == null ? "var(--muted)"
              : ret >= 0 ? "var(--green)" : "var(--red)";
    const row = document.createElement("div");
    row.className = "d";
    row.innerHTML =
      `<span>${day} <span style="color:var(--muted)">${s.timestamp.slice(11,16)}</span>`
      + ` ${s.entry_strike}CE @ ${e ? e.toFixed(1) : "?"}</span>`
      + `<span style="color:${col}">${ret == null ? "·" : (ret>=0?"+":"") + ret.toFixed(0) + "%"}</span>`;
    row.title = `wall ${s.max_call_oi_strike} · CE ${s.call_oi_pct_change}% · `
      + `peak ${peak ? peak.toFixed(1) : "?"} · EOD ${eod != null ? eod.toFixed(2) : "?"}`;
    row.onclick = () => { dayPick(day); showTradeChart(s); };
    tl.appendChild(row);
  }

  sel.onchange = () => dayPick(sel.value);
  dayPick = (day) => {
    sel.value = day;
    for (const r of list.children)
      r.classList.toggle("sel", r.dataset.day === day);
    loadDay(day).catch(err => {
      document.getElementById("stat").textContent = "error: " + err.message; });
  };
  if (curDay && days.some(d => d.day === curDay)) dayPick(curDay);
  else if (days.length) dayPick(days[days.length - 1].day);
}

async function boot() {
  initChart();
  document.getElementById("preset").onchange = loadPreset;
  document.getElementById("prem-close").onclick = () =>
    document.getElementById("prem-wrap").style.display = "none";
  await loadPreset();
}
boot();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML.replace("__APP_BASE__", APP_BASE)
