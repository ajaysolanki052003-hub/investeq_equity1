"""Indicator Screener — filter the NSE equity universe by threshold values of
the most-used technical indicators.

Reuses the daily/hourly OHLCV the Daily Scanner already maintains
(ema_scanner/data/{1d,1h}/*.csv, kept current by the candles timers). For each
symbol we compute a snapshot of 10 classic indicators at the latest bar and let
the user AND/OR threshold filters together to get a list of matching stocks.

Run:
    python -m uvicorn stock_screener.app:app --host 0.0.0.0 --port 8707
Mounted behind nginx at /screener/ (set APP_BASE=/screener via env).
"""

from __future__ import annotations

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

APP_BASE = os.environ.get("APP_BASE", "").rstrip("/")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ema_scanner" / "data"
TAIL     = 320           # bars used per symbol (enough for the longest window)

# ── Indicator catalogue (key, label, group, range hint, default op + value) ──
INDICATORS = [
    {"key": "rsi",       "label": "RSI (14)",            "group": "Momentum",   "op": "<",  "value": 30,  "min": 0,   "max": 100,  "hint": "<30 oversold · >70 overbought"},
    {"key": "macd_hist", "label": "MACD Histogram",      "group": "Momentum",   "op": ">",  "value": 0,   "min": None,"max": None, "hint": ">0 bullish momentum"},
    {"key": "stoch_k",   "label": "Stochastic %K (14)",  "group": "Momentum",   "op": "<",  "value": 20,  "min": 0,   "max": 100,  "hint": "<20 oversold · >80 overbought"},
    {"key": "cci",       "label": "CCI (20)",            "group": "Momentum",   "op": "<",  "value": -100,"min": None,"max": None, "hint": "<-100 oversold · >100 overbought"},
    {"key": "roc",       "label": "ROC (10) %",          "group": "Momentum",   "op": ">",  "value": 0,   "min": None,"max": None, "hint": "% change over 10 bars"},
    {"key": "sma50_pct", "label": "Price vs SMA-50 (%)", "group": "Trend",      "op": ">",  "value": 0,   "min": None,"max": None, "hint": ">0 price above SMA-50"},
    {"key": "ema20_pct", "label": "Price vs EMA-20 (%)", "group": "Trend",      "op": ">",  "value": 0,   "min": None,"max": None, "hint": ">0 price above EMA-20"},
    {"key": "adx",       "label": "ADX (14)",            "group": "Trend",      "op": ">",  "value": 25,  "min": 0,   "max": 100,  "hint": ">25 trending · >40 strong"},
    {"key": "bb_pctb",   "label": "Bollinger %B",        "group": "Volatility", "op": "<",  "value": 0,   "min": None,"max": None, "hint": "<0 below lower · >1 above upper band"},
    {"key": "atr_pct",   "label": "ATR (%)",             "group": "Volatility", "op": ">",  "value": 3,   "min": 0,   "max": None, "hint": "ATR as % of price"},
]
IND_KEYS = [i["key"] for i in INDICATORS]

PRESETS = [
    {"name": "Oversold bounce",  "logic": "AND", "filters": [{"key": "rsi", "op": "<", "value": 35}, {"key": "stoch_k", "op": "<", "value": 25}]},
    {"name": "Strong uptrend",   "logic": "AND", "filters": [{"key": "adx", "op": ">", "value": 25}, {"key": "sma50_pct", "op": ">", "value": 0}, {"key": "ema20_pct", "op": ">", "value": 0}]},
    {"name": "Momentum breakout","logic": "AND", "filters": [{"key": "macd_hist", "op": ">", "value": 0}, {"key": "roc", "op": ">", "value": 5}, {"key": "rsi", "op": ">", "value": 55}]},
    {"name": "Below lower band", "logic": "AND", "filters": [{"key": "bb_pctb", "op": "<", "value": 0.05}]},
    {"name": "High volatility",  "logic": "AND", "filters": [{"key": "atr_pct", "op": ">", "value": 4}]},
]


# ─────────────────────────── indicator maths (pure pandas) ──────────────────
def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _compute_one(df: pd.DataFrame) -> dict | None:
    if len(df) < 60:
        return None
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # RSI(14)
    d = c.diff()
    rsi = 100 - 100 / (1 + _wilder(d.clip(lower=0), 14) / _wilder(-d.clip(upper=0), 14).replace(0, np.nan))

    # MACD(12,26,9)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()

    # Stochastic %K(14) smoothed-3
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()

    # CCI(20)
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # ROC(10)
    roc = (c / c.shift(10) - 1) * 100

    # Trend distances
    sma50 = c.rolling(50).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    sma50_pct = (c / sma50 - 1) * 100
    ema20_pct = (c / ema20 - 1) * 100

    # ADX(14)
    up, dn = h.diff(), -l.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14)
    plus_di  = 100 * _wilder(pd.Series(plus_dm, index=c.index), 14) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=c.index), 14) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder(dx, 14)

    # Bollinger %B(20,2)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    bb_pctb = (c - lower) / (upper - lower).replace(0, np.nan)

    # ATR %
    atr_pct = atr / c * 100

    def last(s):
        s = s.dropna()
        return float(s.iloc[-1]) if len(s) else None

    return {
        "close":     last(c),
        "volume":    last(v),
        "rsi":       last(rsi),
        "macd_hist": last(macd_hist),
        "stoch_k":   last(stoch_k),
        "cci":       last(cci),
        "roc":       last(roc),
        "sma50_pct": last(sma50_pct),
        "ema20_pct": last(ema20_pct),
        "adx":       last(adx),
        "bb_pctb":   last(bb_pctb),
        "atr_pct":   last(atr_pct),
    }


# ─────────────────────────── snapshot cache (mtime-keyed) ───────────────────
_CACHE: dict[str, tuple[float, pd.DataFrame, str]] = {}


def _snapshot(tf: str) -> tuple[pd.DataFrame, str]:
    d = DATA_DIR / tf
    files = sorted(glob.glob(str(d / "*.csv")))
    if not files:
        return pd.DataFrame(), ""
    sig = max(os.path.getmtime(f) for f in files)
    cached = _CACHE.get(tf)
    if cached and cached[0] == sig:
        return cached[1], cached[2]

    rows, asof = {}, ""
    for f in files:
        sym = os.path.basename(f).replace("_historical.csv", "").replace(".csv", "")
        try:
            df = pd.read_csv(f).tail(TAIL)
            if "datetime" in df.columns and len(df):
                asof = max(asof, str(df["datetime"].iloc[-1])[:10])
            vals = _compute_one(df)
        except Exception:
            vals = None
        if vals and vals["close"]:
            rows[sym] = vals

    snap = pd.DataFrame.from_dict(rows, orient="index")
    snap.index.name = "symbol"
    _CACHE[tf] = (sig, snap, asof)
    return snap, asof


# ─────────────────────────────────── API ───────────────────────────────────
app = FastAPI(title="Indicator Screener")


class Filter(BaseModel):
    key: str
    op: str
    value: float


class ScreenReq(BaseModel):
    timeframe: str = "1d"
    logic: str = "AND"
    filters: list[Filter] = []
    sort_by: str | None = None
    sort_dir: str = "desc"


@app.get("/api/meta")
def meta():
    return {"indicators": INDICATORS, "presets": PRESETS,
            "timeframes": ["1d", "1h"]}


@app.post("/api/screen")
def screen(req: ScreenReq):
    tf = req.timeframe if req.timeframe in ("1d", "1h") else "1d"
    snap, asof = _snapshot(tf)
    if snap.empty:
        return JSONResponse({"count": 0, "universe": 0, "asof": "", "rows": [], "cols": []})

    mask = None
    ops = {">": np.greater, ">=": np.greater_equal, "<": np.less,
           "<=": np.less_equal}
    used_keys = []
    for f in req.filters:
        if f.key not in snap.columns or f.op not in ops:
            continue
        used_keys.append(f.key)
        col = snap[f.key]
        m = ops[f.op](col, f.value) & col.notna()
        mask = m if mask is None else (mask & m if req.logic == "AND" else mask | m)

    res = snap if mask is None else snap[mask]

    sort_by = req.sort_by if (req.sort_by in snap.columns) else (used_keys[0] if used_keys else "close")
    res = res.sort_values(sort_by, ascending=(req.sort_dir == "asc"))

    show = ["close"] + [k for k in IND_KEYS if k in (used_keys or IND_KEYS)]
    rows = []
    for sym, r in res.iterrows():
        row = {"symbol": sym}
        for k in show:
            val = r.get(k)
            row[k] = None if (val is None or pd.isna(val)) else round(float(val), 2)
        rows.append(row)
    return {"count": len(rows), "universe": len(snap), "asof": asof,
            "cols": show, "rows": rows}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML.replace("__APP_BASE__", APP_BASE)


# ──────────────────────────────────── UI ───────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Indicator Screener</title>
<style>
  :root{
    --bg:#0a0e17; --panel:#111726; --panel2:#0d1320; --line:#1e2940;
    --txt:#e6edf6; --mut:#8a98b2; --accent:#34d399; --accent2:#22d3ee;
    --pos:#34d399; --neg:#f87171; --chip:#172033;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#16243d 0%,var(--bg) 55%);
       color:var(--txt);font:14px/1.5 'Inter',system-ui,Segoe UI,Roboto,sans-serif;min-height:100vh}
  a{color:inherit}
  .wrap{max-width:1240px;margin:0 auto;padding:28px 22px 60px}
  header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px}
  .title{font-size:26px;font-weight:800;letter-spacing:.2px;display:flex;align-items:center;gap:12px}
  .title .dot{width:11px;height:11px;border-radius:50%;background:var(--accent);box-shadow:0 0 16px 2px var(--accent)}
  .sub{color:var(--mut);font-size:13px;margin-top:4px}
  .asof{color:var(--mut);font-size:12.5px;text-align:right}
  .asof b{color:var(--txt)}
  .grid{display:grid;grid-template-columns:380px 1fr;gap:20px}
  @media(max-width:920px){.grid{grid-template-columns:1fr}}
  .card{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
        border:1px solid var(--line);border-radius:16px;padding:18px}
  .card h3{margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--mut)}
  .seg{display:inline-flex;background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:3px}
  .seg button{background:none;border:0;color:var(--mut);padding:6px 14px;border-radius:8px;cursor:pointer;font-weight:600;font-size:13px}
  .seg button.on{background:var(--accent);color:#04130d}
  .seg.cyan button.on{background:var(--accent2);color:#042027}
  .row{display:flex;gap:8px;align-items:center;margin-bottom:8px}
  select,input{background:var(--panel2);border:1px solid var(--line);color:var(--txt);
        border-radius:9px;padding:8px 9px;font-size:13px;outline:none}
  select:focus,input:focus{border-color:var(--accent)}
  .fk{flex:1;min-width:0}
  .fo{width:62px;text-align:center}
  .fv{width:84px}
  .rm{background:none;border:0;color:var(--mut);cursor:pointer;font-size:17px;padding:4px 6px;border-radius:7px}
  .rm:hover{color:var(--neg);background:#2a1620}
  .hint{color:var(--mut);font-size:11.5px;margin:-2px 0 12px 2px;min-height:14px}
  .addbtn{width:100%;background:var(--chip);border:1px dashed var(--line);color:var(--mut);
        padding:9px;border-radius:10px;cursor:pointer;font-weight:600;margin-top:4px}
  .addbtn:hover{border-color:var(--accent);color:var(--accent)}
  .run{width:100%;margin-top:14px;background:linear-gradient(90deg,var(--accent),var(--accent2));
        color:#04130d;font-weight:800;border:0;border-radius:11px;padding:12px;cursor:pointer;font-size:14px;letter-spacing:.02em}
  .run:active{transform:translateY(1px)}
  .presets{display:flex;flex-wrap:wrap;gap:7px}
  .chip{background:var(--chip);border:1px solid var(--line);color:var(--txt);border-radius:999px;
        padding:6px 12px;font-size:12.5px;cursor:pointer}
  .chip:hover{border-color:var(--accent);color:var(--accent)}
  .resbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;flex-wrap:wrap;gap:10px}
  .count{font-size:15px}
  .count b{font-size:22px;color:var(--accent);font-weight:800}
  .dl{background:var(--chip);border:1px solid var(--line);color:var(--txt);border-radius:9px;padding:7px 13px;cursor:pointer;font-size:12.5px}
  .dl:hover{border-color:var(--accent2);color:var(--accent2)}
  .tablewrap{overflow:auto;border:1px solid var(--line);border-radius:12px;max-height:70vh}
  table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
  th,td{padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
  th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--panel)}
  th{position:sticky;top:0;background:var(--panel2);color:var(--mut);font-size:11.5px;text-transform:uppercase;
     letter-spacing:.07em;cursor:pointer;user-select:none;z-index:2}
  th:first-child{z-index:3}
  th.sorted{color:var(--accent)}
  tbody tr:hover{background:#0f1830}
  td.sym{font-weight:700}
  td.pos{color:var(--pos)} td.neg{color:var(--neg)}
  .empty{padding:48px;text-align:center;color:var(--mut)}
  .spin{display:inline-block;width:15px;height:15px;border:2px solid var(--line);border-top-color:var(--accent);
        border-radius:50%;animation:spin .7s linear infinite;vertical-align:-3px;margin-right:8px}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div class="title"><span class="dot"></span>Indicator Screener</div>
      <div class="sub">Filter the NSE universe by threshold values of the 10 most-used technical indicators.</div>
    </div>
    <div class="asof" id="asof"></div>
  </header>

  <div class="grid">
    <!-- LEFT: filter builder -->
    <div class="card">
      <h3>Timeframe</h3>
      <div class="seg cyan" id="tf">
        <button data-tf="1d" class="on">Daily</button>
        <button data-tf="1h">Hourly</button>
      </div>

      <h3 style="margin-top:18px">Combine with</h3>
      <div class="seg" id="logic">
        <button data-logic="AND" class="on">ALL · AND</button>
        <button data-logic="OR">ANY · OR</button>
      </div>

      <h3 style="margin-top:18px">Filters</h3>
      <div id="filters"></div>
      <button class="addbtn" id="add">+ Add indicator filter</button>
      <button class="run" id="run">Run Screen</button>

      <h3 style="margin-top:20px">Quick presets</h3>
      <div class="presets" id="presets"></div>
    </div>

    <!-- RIGHT: results -->
    <div class="card">
      <div class="resbar">
        <div class="count" id="count">Ready — build a filter and run.</div>
        <button class="dl" id="dl" style="display:none">⬇ Download CSV</button>
      </div>
      <div class="tablewrap"><div id="res"><div class="empty">No results yet.</div></div></div>
    </div>
  </div>
</div>

<script>
const APP_BASE="__APP_BASE__";
const api=(p,o)=>fetch((APP_BASE||"")+p,o).then(r=>r.json());
let META=null, STATE={tf:"1d",logic:"AND"}, LAST=null, SORT={by:null,dir:"desc"};

const labelOf=k=>(META.indicators.find(i=>i.key===k)||{}).label||k;
const hintOf =k=>(META.indicators.find(i=>i.key===k)||{}).hint||"";

function indOptions(sel){
  let groups={};
  META.indicators.forEach(i=>{(groups[i.group]=groups[i.group]||[]).push(i)});
  let h="";
  for(const g in groups){
    h+=`<optgroup label="${g}">`;
    groups[g].forEach(i=>h+=`<option value="${i.key}" ${i.key===sel?"selected":""}>${i.label}</option>`);
    h+="</optgroup>";
  }
  return h;
}
function opOptions(sel){
  return ["<",">","<=",">="].map(o=>`<option ${o===sel?"selected":""}>${o}</option>`).join("");
}

function addFilter(f){
  f=f||{key:"rsi",op:"<",value:30};
  const def=META.indicators.find(i=>i.key===f.key)||{};
  const wrap=document.getElementById("filters");
  const div=document.createElement("div");
  div.className="frow";
  div.innerHTML=`
    <div class="row">
      <select class="fk">${indOptions(f.key)}</select>
      <select class="fo">${opOptions(f.op)}</select>
      <input class="fv" type="number" step="any" value="${f.value}"/>
      <button class="rm" title="remove">×</button>
    </div>
    <div class="hint">${def.hint||""}</div>`;
  wrap.appendChild(div);
  const kSel=div.querySelector(".fk"), hint=div.querySelector(".hint");
  kSel.onchange=()=>hint.textContent=hintOf(kSel.value);
  div.querySelector(".rm").onclick=()=>div.remove();
}

function readFilters(){
  return [...document.querySelectorAll(".frow")].map(d=>({
    key:d.querySelector(".fk").value,
    op:d.querySelector(".fo").value,
    value:parseFloat(d.querySelector(".fv").value)
  })).filter(f=>!isNaN(f.value));
}

function setFilters(list){
  document.getElementById("filters").innerHTML="";
  list.forEach(addFilter);
}

async function run(){
  const filters=readFilters();
  document.getElementById("count").innerHTML='<span class="spin"></span>Screening…';
  const body={timeframe:STATE.tf,logic:STATE.logic,filters,
              sort_by:SORT.by,sort_dir:SORT.dir};
  const r=await api("/api/screen",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  LAST=r; render(r);
}

function render(r){
  document.getElementById("asof").innerHTML=`Universe <b>${r.universe}</b> · ${STATE.tf.toUpperCase()} · as of <b>${r.asof||"—"}</b>`;
  document.getElementById("count").innerHTML=`<b>${r.count}</b> stocks match`;
  document.getElementById("dl").style.display=r.count?"block":"none";
  if(!r.count){document.getElementById("res").innerHTML='<div class="empty">No stocks match these filters. Loosen a threshold or switch AND→OR.</div>';return;}
  const cols=r.cols;
  const head=`<tr><th data-c="symbol">Symbol</th>${cols.map(c=>`<th data-c="${c}" class="${SORT.by===c?'sorted':''}">${c==='close'?'Close':labelOf(c)}</th>`).join("")}</tr>`;
  const fmt=(c,v)=>{
    if(v===null||v===undefined)return '<td>—</td>';
    let cls="";
    if(["macd_hist","roc","sma50_pct","ema20_pct"].includes(c))cls=v>0?"pos":(v<0?"neg":"");
    return `<td class="${cls}">${v}</td>`;
  };
  const rows=r.rows.map(o=>`<tr><td class="sym">${o.symbol}</td>${cols.map(c=>fmt(c,o[c])).join("")}</tr>`).join("");
  document.getElementById("res").innerHTML=`<table><thead>${head}</thead><tbody>${rows}</tbody></table>`;
  document.querySelectorAll("th").forEach(th=>th.onclick=()=>{
    const c=th.dataset.c;
    SORT.dir=(SORT.by===c&&SORT.dir==="desc")?"asc":"desc"; SORT.by=c; run();
  });
}

function downloadCSV(){
  if(!LAST||!LAST.rows.length)return;
  const cols=["symbol",...LAST.cols];
  const lines=[cols.join(",")].concat(LAST.rows.map(o=>cols.map(c=>o[c]).join(",")));
  const blob=new Blob([lines.join("\n")],{type:"text/csv"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);a.download=`screen_${STATE.tf}_${LAST.asof||""}.csv`;a.click();
}

function bindSeg(id,key){
  document.querySelectorAll(`#${id} button`).forEach(b=>b.onclick=()=>{
    document.querySelectorAll(`#${id} button`).forEach(x=>x.classList.remove("on"));
    b.classList.add("on"); STATE[key]=b.dataset[key];
  });
}

(async()=>{
  META=await api("/api/meta");
  bindSeg("tf","tf"); bindSeg("logic","logic");
  document.getElementById("add").onclick=()=>addFilter();
  document.getElementById("run").onclick=run;
  document.getElementById("dl").onclick=downloadCSV;
  const pc=document.getElementById("presets");
  META.presets.forEach(p=>{
    const c=document.createElement("button");c.className="chip";c.textContent=p.name;
    c.onclick=()=>{
      document.querySelectorAll('#logic button').forEach(x=>x.classList.toggle("on",x.dataset.logic===p.logic));
      STATE.logic=p.logic; setFilters(p.filters); run();
    };
    pc.appendChild(c);
  });
  addFilter();           // start with one filter row
  run();
})();
</script>
</body>
</html>"""
