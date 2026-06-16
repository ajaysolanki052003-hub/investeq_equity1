"""Resistance Marker (/resistance) — interactive swing-high resistance on the
VISIBLE window of a stock chart.

Load any NIFTY-500 stock, scroll/zoom to any window, press "Apply Resistance":
swing-high pivot zones are computed from ONLY the candles currently on screen
and drawn as horizontal level segments confined to that window (not the whole
history). Additive — mark several windows; "Clear" wipes them.

Data: the per-stock candle CSVs the screener/scanner already maintain
(ema_scanner/data/{1d,1h}/<SYM>_historical.csv). No broker calls — pure reads.

Run:
    APP_BASE=/resistance uvicorn stock_resistance.app:app --host 127.0.0.1 --port 8709
"""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ema_scanner" / "data"
APP_BASE = os.environ.get("APP_BASE", "")

app = FastAPI(title="Resistance Marker")

_cache: dict[tuple, tuple] = {}   # (sym, tf) -> (mtime, candles)


def _symbols() -> list[str]:
    d = DATA_DIR / "1d"
    files = glob.glob(str(d / "*_historical.csv"))
    return sorted(os.path.basename(f).replace("_historical.csv", "") for f in files)


def _candles(sym: str, tf: str) -> list[dict]:
    if tf not in ("1d", "1h"):
        raise HTTPException(400, "tf must be 1d or 1h")
    path = DATA_DIR / tf / f"{sym}_historical.csv"
    if not path.exists():
        raise HTTPException(404, f"no data for {sym} ({tf})")
    mt = path.stat().st_mtime
    hit = _cache.get((sym, tf))
    if hit and hit[0] == mt:
        return hit[1]
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close"])
    df = df.dropna().sort_values("timestamp")
    out = [{"time": int(t), "open": float(o), "high": float(h),
            "low": float(l), "close": float(c)}
           for t, o, h, l, c in zip(df["timestamp"], df["open"], df["high"],
                                    df["low"], df["close"])]
    _cache[(sym, tf)] = (mt, out)
    return out


@app.get("/api/meta")
def meta():
    syms = _symbols()
    default = "RELIANCE" if "RELIANCE" in syms else (syms[0] if syms else "")
    return {"symbols": syms, "timeframes": ["1d", "1h"], "default": default}


@app.get("/api/candles")
def candles(symbol: str = Query(...), tf: str = Query("1d")):
    sym = symbol.strip().upper()
    return JSONResponse({"symbol": sym, "tf": tf, "candles": _candles(sym, tf)})


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(PAGE.replace("__APP_BASE__", APP_BASE))


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resistance Marker</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0b0d11;--border:#222838;--text:#e7eaf1;--muted:#8a93a6;--accent:#f59e0b;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,Segoe UI,Roboto,sans-serif}
#bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:linear-gradient(180deg,#0d1018,#10131a);border-bottom:1px solid var(--border)}
#bar label{color:var(--muted);font-size:11px;margin-right:3px}
select,input,button{background:#0d1117;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 9px;font:inherit}
button{cursor:pointer}
button.on{background:linear-gradient(135deg,#f59e0b,#b45309);color:#1a1206;border-color:transparent;font-weight:700}
#apply{background:linear-gradient(135deg,#f59e0b,#b45309);color:#1a1206;border-color:transparent;font-weight:700}
.seg button{border-radius:0;margin:0}
.seg button:first-child{border-radius:6px 0 0 6px}
.seg button:last-child{border-radius:0 6px 6px 0}
.tag{color:var(--accent);font-weight:700;letter-spacing:.4px;margin-right:4px}
#title{font-weight:700;font-size:13px;color:var(--muted);margin-left:auto}
.charts{height:calc(100vh - 56px)}
#chart{height:100%}
.hint{color:var(--muted);font-size:11px}
</style></head><body>

<div id="bar">
  <span class="tag">RESISTANCE MARKER</span>
  <span><label>Stock</label><select id="sym" style="width:160px"></select></span>
  <span class="seg" id="tf"><button data-tf="1d" class="on">Daily</button><button data-tf="1h">Hourly</button></span>
  <span><label>Pivot</label><select id="k"><option selected>8</option><option>13</option><option>21</option></select></span>
  <span><label>Min touches</label><select id="touch"><option selected>1</option><option>2</option><option>3</option></select></span>
  <button id="apply">⬛ Apply Resistance</button>
  <button id="clear">Clear</button>
  <span class="hint">scroll/zoom to a window, then Apply — marks only the visible screen</span>
  <span id="title"></span>
</div>

<div class="charts"><div id="chart"></div></div>

<script>
const APP_BASE="__APP_BASE__";
const api=function(p){return (APP_BASE||"")+p;};
const $=function(id){return document.getElementById(id);};

const S={candles:[], tf:"1d", marks:[]};

const chart=LightweightCharts.createChart($("chart"),{autoSize:true,
  layout:{background:{color:"#0b0d11"},textColor:"#cfd6e4"},
  grid:{vertLines:{color:"#161b27"},horzLines:{color:"#161b27"}},
  timeScale:{borderColor:"#222838",rightOffset:6,timeVisible:false,secondsVisible:false},
  rightPriceScale:{borderColor:"#222838"},crosshair:{mode:0}});
const candle=chart.addCandlestickSeries({upColor:"#26a69a",downColor:"#ef5350",
  borderUpColor:"#26a69a",borderDownColor:"#ef5350",wickUpColor:"#26a69a",wickDownColor:"#ef5350"});

// ── load candles for a symbol/tf ─────────────────────────────────────
async function load(){
  const sym=$("sym").value.trim().toUpperCase(); if(!sym)return;
  clearMarks();
  $("title").textContent="loading "+sym+"…";
  let j; try{j=await(await fetch(api("/api/candles?symbol="+sym+"&tf="+S.tf))).json();}
  catch(e){$("title").textContent="not found";return;}
  if(j.error||!j.candles||!j.candles.length){$("title").textContent=sym+": no data";candle.setData([]);return;}
  S.candles=j.candles;
  candle.setData(S.candles);
  chart.timeScale().fitContent();
  $("title").textContent=sym+" · "+S.tf+" · "+S.candles.length+" bars";
}

// ── resistance on the VISIBLE window only ────────────────────────────
function applyResistance(){
  if(!S.candles.length)return;
  const r=chart.timeScale().getVisibleRange();
  if(!r){return;}
  const vis=S.candles.filter(function(c){return c.time>=r.from && c.time<=r.to;});
  if(vis.length<5){$("title").textContent="zoom out a little — too few bars on screen";return;}
  const k=+$("k").value, minTouch=+$("touch").value;
  // 1) swing-high pivots: high[i] is the max within +/- k bars
  const piv=[];
  for(let i=k;i<vis.length-k;i++){
    let isHigh=true;
    for(let j=i-k;j<=i+k;j++){ if(vis[j].high>vis[i].high){isHigh=false;break;} }
    if(isHigh) piv.push(vis[i].high);
  }
  if(!piv.length){$("title").textContent="no swing highs in this window";return;}
  // 2) cluster nearby pivots into zones (within 0.6% of the running average)
  piv.sort(function(a,b){return a-b;});
  const TOL=0.006, zones=[];
  for(let i=0;i<piv.length;i++){
    const z=zones[zones.length-1];
    if(z && Math.abs(piv[i]-z.avg)<=z.avg*TOL){ z.sum+=piv[i]; z.n++; z.avg=z.sum/z.n; }
    else zones.push({sum:piv[i], n:1, avg:piv[i]});
  }
  // 3) keep zones with >= minTouch pivots; draw a segment across the visible window
  const t0=vis[0].time, t1=vis[vis.length-1].time;
  const kept=zones.filter(function(z){return z.n>=minTouch;}).sort(function(a,b){return b.n-a.n;});
  let drawn=0;
  kept.forEach(function(z){
    const strong=z.n>=3, mid=z.n===2;
    const seg=chart.addLineSeries({
      color: strong?"#f59e0b":(mid?"#fbbf24":"#fcd34d"),
      lineWidth: strong?3:(mid?2:1), lineStyle:0,
      priceLineVisible:false, lastValueVisible:true, crosshairMarkerVisible:false,
      title: z.n+"×",
    });
    seg.setData([{time:t0,value:z.avg},{time:t1,value:z.avg}]);
    S.marks.push(seg); drawn++;
  });
  $("title").textContent=drawn+" resistance level(s) on this window  ("+vis.length+" bars)";
}

function clearMarks(){ S.marks.forEach(function(s){try{chart.removeSeries(s)}catch(e){}}); S.marks=[];
  $("apply").classList.remove("on"); $("apply").textContent="⬛ Apply Resistance"; }

// ── wire-up ──────────────────────────────────────────────────────────
[].slice.call($("tf").children).forEach(function(b){b.onclick=function(){
  [].slice.call($("tf").children).forEach(function(x){x.classList.remove("on")});
  b.classList.add("on"); S.tf=b.dataset.tf; load();};});
$("sym").addEventListener("change",load);
// Toggle: 1st click draws resistance for the visible window, 2nd click removes.
$("apply").onclick=function(){
  if(S.marks.length){ clearMarks(); $("title").textContent="resistance removed"; return; }
  applyResistance();
  if(S.marks.length){ $("apply").classList.add("on"); $("apply").textContent="✕ Remove Resistance"; }
};
$("clear").onclick=function(){clearMarks(); $("title").textContent="cleared";};

// ── boot ─────────────────────────────────────────────────────────────
(async function(){
  const m=await(await fetch(api("/api/meta"))).json();
  $("sym").innerHTML=(m.symbols||[]).map(function(s){
    return "<option"+(s===m.default?" selected":"")+">"+s+"</option>";}).join("");
  await load();
})();
</script>
</body></html>"""
