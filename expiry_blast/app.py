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
SIGNALS_CSV = Path(__file__).parent / "output" / "signals.csv"

app = FastAPI(title="Expiry Blast — Short Covering entry engine")

CFG = BlastConfig()
ENGINE = SignalEngine(CFG)


@lru_cache(maxsize=1)
def _expiry_days() -> list[str]:
    return [d.isoformat() for d in expiry_days()]


def _signal_days() -> set[str]:
    if not SIGNALS_CSV.exists():
        return set()
    try:
        df = pd.read_csv(SIGNALS_CSV)
        if df.empty or "timestamp" not in df.columns:
            return set()
        return set(pd.to_datetime(df["timestamp"]).dt.date.astype(str))
    except Exception:
        return set()


@app.get("/api/config")
def config():
    return JSONResponse(CFG.to_dict())


@app.get("/api/days")
def days():
    sig = _signal_days()
    return JSONResponse({"days": [{"day": d, "signal": d in sig}
                                  for d in _expiry_days()]})


@app.get("/api/signals")
def signals():
    if not SIGNALS_CSV.exists():
        return JSONResponse({"signals": [], "note": "run the backtest first: "
                             "python -m expiry_blast.backtest"})
    df = pd.read_csv(SIGNALS_CSV)
    df = df.where(pd.notna(df), None)
    return JSONResponse({"signals": df.to_dict(orient="records")})


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
        put_strikes: int | None = None,
        buildup: float | None = None,
        unwind: float | None = None,
        proximity: float | None = None,
        lookback: int | None = None):
    over = {k: v for k, v in {
        "put_strikes_required": put_strikes,
        "put_oi_buildup_pct": buildup,
        "call_oi_unwind_pct": unwind,
        "spot_proximity_pct": proximity,
        "oi_lookback_min": lookback,
    }.items() if v is not None}
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
  <span class="sub">C strikes:</span>
  <select id="cstrikes" title="how many of the 2 below-wall PE strikes must pass condition C">
    <option value="1">1 of 2 (default)</option>
    <option value="2">2 of 2 (strict spec)</option>
  </select>
  <span class="sub" id="stat"></span>
</header>

<div class="layout">
  <div class="side">
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
async function loadDay(day) {
  curDay = day;
  document.getElementById("stat").textContent = "loading " + day + " ...";
  const ps = document.getElementById("cstrikes").value;
  const d = await api("/api/day/" + day + "?put_strikes=" + ps);
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

async function boot() {
  initChart();
  const cfg = await api("/api/config");
  document.getElementById("cfg").innerHTML = [
    ["instrument", cfg.instrument],
    ["window", cfg.window_start + " – " + cfg.window_end + " IST"],
    ["proximity", cfg.spot_proximity_pct + " %"],
    ["CE unwind ≥", cfg.call_oi_unwind_pct + " %"],
    ["PE buildup ≥", cfg.put_oi_buildup_pct + " %"],
    ["OI lookback", cfg.oi_lookback_min + " min"],
    ["candle TF", cfg.candle_tf_min + " min"],
    ["max entries/day", cfg.max_signals_per_day],
    ["stale OI cutoff", cfg.stale_oi_max_min + " min"],
  ].map(([k, v]) => `${k} <b style="float:right">${v}</b>`).join("<br>");

  const { days } = await api("/api/days");
  document.getElementById("ndays").textContent = "(" + days.length + ")";
  const list = document.getElementById("daylist");
  const sel = document.getElementById("day-sel");
  for (const d of [...days].reverse()) {
    const row = document.createElement("div");
    row.className = "d"; row.dataset.day = d.day;
    row.innerHTML = `<span>${d.day}</span>` +
      (d.signal ? `<span class="dot">●</span>` : "");
    row.onclick = () => pick(d.day);
    list.appendChild(row);
    sel.add(new Option(d.day + (d.signal ? " ●" : ""), d.day));
  }
  sel.onchange = () => pick(sel.value);
  document.getElementById("cstrikes").onchange =
    () => { if (curDay) loadDay(curDay); };
  function pick(day) {
    sel.value = day;
    for (const r of list.children)
      r.classList.toggle("sel", r.dataset.day === day);
    loadDay(day).catch(err => {
      document.getElementById("stat").textContent = "error: " + err.message; });
  }
  if (days.length) pick(days[days.length - 1].day);
}
boot();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML.replace("__APP_BASE__", APP_BASE)
