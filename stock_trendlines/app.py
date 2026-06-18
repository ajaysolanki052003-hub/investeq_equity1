"""Trendline Plotter (/trend) — draw trendlines on any stock chart.

Two ways to mark a trend on the VISIBLE window:
  • Auto      — swing-high / swing-low pivots on screen are connected into upper
                (resistance) and lower (support) trend boundaries via the convex
                hull, so the lines hug the tops/bottoms and never cut through a bar.
  • Draw      — click two points on the chart and a straight trendline is drawn
                between them. Additive; draw as many as you like. "Clear" wipes all.

Data: the per-stock candle CSVs the screener/scanner already maintain
(ema_scanner/data/{1d,1h}/<SYM>_historical.csv). No broker calls — pure reads.

Run:
    APP_BASE=/trend uvicorn stock_trendlines.app:app --host 127.0.0.1 --port 8710
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

app = FastAPI(title="Trendline Plotter")

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


def _earliest_year(sym: str) -> int:
    """Start year of a symbol's daily CSV (cheap: reads the first data row)."""
    path = DATA_DIR / "1d" / f"{sym}_historical.csv"
    try:
        with open(path) as f:
            f.readline()                       # header
            ts = int(f.readline().split(",")[0])
        return time.gmtime(ts).tm_year
    except Exception:
        return 9999


@app.get("/api/meta")
def meta():
    syms = _symbols()
    # Default to a long-history, recognizable name so the chart opens on a deep
    # window. The marquee F&O names (RELIANCE, HDFC, INFY...) are capped at 2024
    # locally, so prefer a liquid large-cap that actually has 2020+ history.
    default = ""
    for pref in ("SBIN", "RELIANCE"):
        if pref in syms and _earliest_year(pref) <= 2020:
            default = pref
            break
    if not default:
        deep = [s for s in syms if _earliest_year(s) <= 2020]
        default = (deep or syms or [""])[0]
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
<title>Trendline Plotter</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0b0d11;--border:#222838;--text:#e7eaf1;--muted:#8a93a6;--accent:#38bdf8;}
*{box-sizing:border-box}
body{margin:0;height:100vh;display:flex;flex-direction:column;overflow:hidden;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,Segoe UI,Roboto,sans-serif}
#bar{flex:0 0 auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:linear-gradient(180deg,#0d1018,#10131a);border-bottom:1px solid var(--border)}
#bar label{color:var(--muted);font-size:11px;margin-right:3px}
select,input,button{background:#0d1117;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 9px;font:inherit}
button{cursor:pointer}
button.on{background:linear-gradient(135deg,#38bdf8,#0369a1);color:#04121c;border-color:transparent;font-weight:700}
.seg button{border-radius:0;margin:0}
.seg button:first-child{border-radius:6px 0 0 6px}
.seg button:last-child{border-radius:0 6px 6px 0}
.tag{color:var(--accent);font-weight:700;letter-spacing:.4px;margin-right:4px}
#title{font-weight:700;font-size:13px;color:var(--muted);margin-left:auto}
.charts{flex:1 1 auto;min-height:0}
#chart{height:100%}
#chart.drawing{cursor:crosshair}
.hint{color:var(--muted);font-size:11px}
</style></head><body>

<div id="bar">
  <span class="tag">TRENDLINE PLOTTER</span>
  <span><label>Stock</label><select id="sym" style="width:160px"></select></span>
  <span class="seg" id="tf"><button data-tf="1d" class="on">Daily</button><button data-tf="1h">Hourly</button></span>
  <span><label>Pivot</label><select id="k"><option>5</option><option selected>8</option><option>13</option><option>21</option></select></span>
  <span><label>Max lines</label><select id="maxl"><option>6</option><option selected>12</option><option>20</option><option>30</option></select></span>
  <button id="auto">📈 Auto Trendlines</button>
  <button id="draw">✏️ Draw</button>
  <button id="clear">Clear</button>
  <span class="hint">Auto = connect swing pivots on screen · Draw = click two points</span>
  <span id="title"></span>
</div>

<div class="charts"><div id="chart"></div></div>

<script>
const APP_BASE="__APP_BASE__";
const api=function(p){return (APP_BASE||"")+p;};
const $=function(id){return document.getElementById(id);};

const S={candles:[], tf:"1d", lines:[]};
let drawMode=false, pending=null;

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
  clearLines();
  $("title").textContent="loading "+sym+"…";
  let j; try{j=await(await fetch(api("/api/candles?symbol="+sym+"&tf="+S.tf))).json();}
  catch(e){$("title").textContent="not found";return;}
  if(j.error||!j.candles||!j.candles.length){$("title").textContent=sym+": no data";candle.setData([]);return;}
  S.candles=j.candles;
  candle.setData(S.candles);
  chart.timeScale().fitContent();
  $("title").textContent=sym+" · "+S.tf+" · "+S.candles.length+" bars";
}

// ── find ALL valid trendlines among pivots ───────────────────────────
// A line through two pivots is valid (upper) if no pivot pokes above it
// beyond TOL; touches = pivots lying on it within TOL. Keep >= 2 touches.
function findTrendlines(piv, upper){
  const TOL=0.006, out=[];
  for(let a=0;a<piv.length-1;a++){
    for(let b=a+1;b<piv.length;b++){
      const A=piv[a], B=piv[b];
      if(B.i===A.i) continue;
      const slope=(B.value-A.value)/(B.i-A.i);
      let ok=true, touches=0;
      for(let p=0;p<piv.length;p++){
        const P=piv[p];
        const lv=A.value+slope*(P.i-A.i);
        const d=(P.value-lv)/lv;
        if(upper){ if(d>TOL){ok=false;break;} } else { if(d<-TOL){ok=false;break;} }
        if(Math.abs(d)<=TOL) touches++;
      }
      if(ok && touches>=2) out.push({A:A, B:B, slope:slope, touches:touches, span:B.i-A.i});
    }
  }
  return out;
}

// drop near-duplicate lines (similar slope + intercept); strongest first
function dedupeLines(lines, avg){
  lines.sort(function(a,b){ return (b.touches-a.touches) || (b.span-a.span); });
  const out=[];
  lines.forEach(function(L){
    const sR=L.slope/avg, v0=L.A.value-L.slope*L.A.i;
    const dup=out.some(function(K){
      const sRk=K.slope/avg, v0k=K.A.value-K.slope*K.A.i;
      return Math.abs(sR-sRk)<0.0006 && Math.abs(v0-v0k)/avg<0.012;
    });
    if(!dup) out.push(L);
  });
  return out;
}

function drawTrend(vis, L, color){
  const lastI=vis.length-1;
  const v0=L.A.value, vEnd=L.A.value+L.slope*(lastI-L.A.i);
  const ln=chart.addLineSeries({color:color, lineWidth:(L.touches>=3?2:1), lineStyle:0,
    priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false, title:L.touches+"×"});
  ln.setData([{time:vis[L.A.i].time,value:v0},{time:vis[lastI].time,value:vEnd}]);
  S.lines.push(ln);
}

// ── auto trendlines on the VISIBLE window only ───────────────────────
function autoTrendlines(){
  if(!S.candles.length)return;
  const r=chart.timeScale().getVisibleRange(); if(!r)return;
  const vis=S.candles.filter(function(c){return c.time>=r.from && c.time<=r.to;});
  if(vis.length<10){$("title").textContent="zoom out a little — too few bars on screen";return;}
  const k=+$("k").value, maxl=+$("maxl").value;
  const highs=[], lows=[];
  for(let i=k;i<vis.length-k;i++){
    let isH=true,isL=true;
    for(let j=i-k;j<=i+k;j++){
      if(vis[j].high>vis[i].high) isH=false;
      if(vis[j].low <vis[i].low ) isL=false;
    }
    if(isH) highs.push({i:i, value:vis[i].high});
    if(isL) lows.push({i:i, value:vis[i].low});
  }
  const avg=(vis[0].close+vis[vis.length-1].close)/2 || 1;
  const R =dedupeLines(findTrendlines(highs,true ), avg).slice(0,maxl);
  const Sp=dedupeLines(findTrendlines(lows ,false), avg).slice(0,maxl);
  R.forEach (function(L){ drawTrend(vis,L,"#f87171"); });
  Sp.forEach(function(L){ drawTrend(vis,L,"#4ade80"); });
  const n=R.length+Sp.length;
  $("title").textContent = n ? (n+" trendlines · "+R.length+" resistance / "+Sp.length+" support  ("+highs.length+" highs, "+lows.length+" lows)")
                              : "no valid trendlines — try a smaller Pivot or zoom out";
}

// ── manual draw: click two points → straight trendline ───────────────
chart.subscribeClick(function(param){
  if(!drawMode||!param.point) return;
  const t=chart.timeScale().coordinateToTime(param.point.x);
  const v=candle.coordinateToPrice(param.point.y);
  if(t==null||v==null) return;
  if(!pending){ pending={time:t,value:v}; $("title").textContent="click the 2nd point…"; return; }
  let p1=pending, p2={time:t,value:v};
  if(p2.time<p1.time){ const tmp=p1;p1=p2;p2=tmp; }
  if(p2.time===p1.time){ pending=null; $("title").textContent="pick two different points"; return; }
  const ln=chart.addLineSeries({color:"#38bdf8",lineWidth:2,lineStyle:0,
    priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
  ln.setData([p1,p2]); S.lines.push(ln);
  projectForward(p1,p2,"#38bdf8");
  pending=null;
  $("title").textContent=S.lines.length+" series · click two more, or press Draw to stop";
});

function setDraw(on){
  drawMode=on; pending=null;
  $("draw").classList.toggle("on",on);
  $("chart").classList.toggle("drawing",on);
  if(on)$("title").textContent="draw mode — click two points on the chart";
}

// dotted continuation of a line (slope of p1->p2) forward to the latest bar
function projectForward(p1,p2,color){
  if(!S.candles.length) return;
  const tEnd=S.candles[S.candles.length-1].time;
  if(tEnd<=p2.time || p2.time===p1.time) return;
  const slope=(p2.value-p1.value)/(p2.time-p1.time);
  const vEnd=p1.value+slope*(tEnd-p1.time);
  const proj=chart.addLineSeries({color:color,lineWidth:2,lineStyle:1,
    priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
  proj.setData([p2,{time:tEnd,value:vEnd}]); S.lines.push(proj);
}

function clearLines(){ S.lines.forEach(function(s){try{chart.removeSeries(s)}catch(e){}}); S.lines=[]; pending=null; }

// ── wire-up ──────────────────────────────────────────────────────────
[].slice.call($("tf").children).forEach(function(b){b.onclick=function(){
  [].slice.call($("tf").children).forEach(function(x){x.classList.remove("on")});
  b.classList.add("on"); S.tf=b.dataset.tf; load();};});
$("sym").addEventListener("change",load);
$("auto").onclick=autoTrendlines;
$("draw").onclick=function(){ setDraw(!drawMode); };
$("clear").onclick=function(){clearLines(); $("title").textContent="cleared";};
document.addEventListener("keydown",function(e){ if(e.key==="Escape"){ if(pending){pending=null;$("title").textContent="point reset — click again";} else setDraw(false);} });

// ── boot ─────────────────────────────────────────────────────────────
(async function(){
  const m=await(await fetch(api("/api/meta"))).json();
  $("sym").innerHTML=(m.symbols||[]).map(function(s){
    return "<option"+(s===m.default?" selected":"")+">"+s+"</option>";}).join("");
  await load();
})();
</script>
</body></html>"""
