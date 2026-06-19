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


def _resample_30m(bars: list[dict]) -> list[dict]:
    """Build 30m bars from 15m: pair consecutive 15m bars within each trading
    day (day = timestamp // 86400, since times are naive-IST-as-UTC)."""
    out: list[dict] = []
    day = None
    idx = 0
    cur = None
    for b in bars:
        d = b["time"] // 86400
        if d != day:
            day, idx = d, 0
        if idx % 2 == 0:
            cur = {"time": b["time"], "open": b["open"], "high": b["high"],
                   "low": b["low"], "close": b["close"], "volume": b["volume"]}
            out.append(cur)
        else:
            cur["high"] = max(cur["high"], b["high"])
            cur["low"] = min(cur["low"], b["low"])
            cur["close"] = b["close"]
            cur["volume"] += b["volume"]
        idx += 1
    return out


def _candles(sym: str, tf: str) -> list[dict]:
    if tf not in ("1d", "1h", "30m", "15m"):
        raise HTTPException(400, "tf must be 1d, 1h, 30m or 15m")
    src_tf = "15m" if tf == "30m" else tf      # 30m is resampled from 15m
    path = DATA_DIR / src_tf / f"{sym}_historical.csv"
    if not path.exists():
        raise HTTPException(404, f"no data for {sym} ({tf})")
    mt = path.stat().st_mtime
    hit = _cache.get((sym, tf))
    if hit and hit[0] == mt:
        return hit[1]
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    df["volume"] = df["volume"].fillna(0.0)
    base = [{"time": int(t), "open": float(o), "high": float(h),
             "low": float(l), "close": float(c), "volume": float(v)}
            for t, o, h, l, c, v in zip(df["timestamp"], df["open"], df["high"],
                                        df["low"], df["close"], df["volume"])]
    out = _resample_30m(base) if tf == "30m" else base
    _cache[(sym, tf)] = (mt, out)
    return out


@app.get("/api/meta")
def meta():
    syms = _symbols()
    default = "RELIANCE" if "RELIANCE" in syms else (syms[0] if syms else "")
    return {"symbols": syms, "timeframes": ["1d", "1h", "30m", "15m"], "default": default}


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
#logic{position:absolute;right:14px;top:54px;z-index:9;width:430px;max-width:calc(100vw - 28px);
  background:#0d1117;border:1px solid var(--border);border-radius:10px;padding:14px 16px;
  box-shadow:0 12px 40px rgba(0,0,0,.55);display:none;font-size:12px;line-height:1.55}
#logic.on{display:block}
#logic h4{margin:0 0 6px;font-size:12px;letter-spacing:.4px;color:var(--accent)}
#logic ol{margin:4px 0 10px;padding-left:18px}#logic li{margin:2px 0}
#logic .cat{display:flex;gap:8px;align-items:flex-start;margin:5px 0}
#logic .sw{flex:0 0 12px;height:12px;border-radius:3px;margin-top:3px}
#logic .muted{color:var(--muted)}
#logic code{background:#161b27;padding:1px 5px;border-radius:4px;color:#cfd6e4}
</style></head><body>

<div id="bar">
  <span class="tag">RESISTANCE MARKER</span>
  <span><label>Stock</label><select id="sym" style="width:160px"></select></span>
  <span class="seg" id="tf"><button data-tf="1d" class="on">Daily</button><button data-tf="1h">Hourly</button><button data-tf="30m">30m</button><button data-tf="15m">15m</button></span>
  <span><label>Pivot</label><select id="k"><option selected>8</option><option>13</option><option>21</option></select></span>
  <span><label>Min touches</label><select id="touch"><option selected>1</option><option>2</option><option>3</option></select></span>
  <button id="apply">⬛ Apply Resistance</button>
  <button id="setups">🎯 Find Setups</button>
  <button id="vwap">VWAP</button>
  <button id="clear">Clear</button>
  <button id="info">ℹ Logic</button>
  <span class="hint">Apply = current resistance (intact; one false breakout tolerated if rejected ≥3×) · Find Setups = scans the FULL chart for levels tested ≥2× where price holds BELOW resistance, closes HUG a flat VWAP for ≥5 candles, then a GREEN candle lifts off above the VWAP</span>
  <span id="title"></span>
</div>

<div id="logic">
  <h4>HOW FIND SETUPS WORKS</h4>
  <div class="muted">Scans the FULL chart history (the past), per resistance level.</div>
  <b>The setup — hug the flat VWAP, then GREEN lift-off, UNDER resistance:</b>
  <ol>
    <li>Level <b>tested ≥ 2×</b> (price rejected there — the resistance "takes it back"). <i>One <b>false breakout</b> — a poke above that returns below — is tolerated if the level rejected <b>≥ 3×</b>.</i></li>
    <li>Price <b>HOLDS BELOW</b> the resistance up to any sustained break (the lone false breakout aside).</li>
    <li>For <b>≥ 5 candles the closes HUG the VWAP</b> — each close within <code>0.4%</code> of the line (price sits on the VWAP).</li>
    <li>The <b>VWAP is flat / slight-slope</b> across that hug (within <code>±0.4%</code>).</li>
    <li><b>Trigger:</b> the next candle is <b>GREEN</b> (close &gt; open) and <b>closes clearly ABOVE the VWAP</b> — the lift-off off the line — still below the resistance.</li>
  </ol>
  <div class="cat"><span class="sw" style="background:#34d399"></span><span>The <b>bold green line</b> inside each box is the <b>flat VWAP segment</b> threading through the hug and lift-off — drawn over the thin blue VWAP. The <b>▲ triangle</b> marks the green lift-off candle (the entry).</span></div>
  <div class="cat" style="margin-top:6px"><span class="sw" style="background:#ef4444"></span><span>Each setup draws <b>two boxes</b>: the <b>green outer box</b> = the FULL setup (first test → lift-off, incl. the resistance line), and the <b>red dotted inner box</b> = just the <b>VWAP-logic region</b> (the hug → lift-off).</span></div>
  <div class="muted" style="margin-top:9px">
    The label shows the <code>VWAP slope</code> across the hug and how many
    <code>candles</code> hugged the line before the lift-off.
  </div>
</div>

<div class="charts"><div id="chart"></div></div>

<script>
const APP_BASE="__APP_BASE__";
const api=function(p){return (APP_BASE||"")+p;};
const $=function(id){return document.getElementById(id);};

const S={candles:[], tf:"1d", zones:[], setups:[]};  // zones=resistance blocks · setups=scenario boxes
let vwapOn=false, vwapSeries=null;

const chart=LightweightCharts.createChart($("chart"),{autoSize:true,
  layout:{background:{color:"#0b0d11"},textColor:"#cfd6e4"},
  grid:{vertLines:{color:"#161b27"},horzLines:{color:"#161b27"}},
  timeScale:{borderColor:"#222838",rightOffset:6,timeVisible:false,secondsVisible:false},
  rightPriceScale:{borderColor:"#222838"},crosshair:{mode:0}});
const candle=chart.addCandlestickSeries({upColor:"#26a69a",downColor:"#ef5350",
  borderUpColor:"#26a69a",borderDownColor:"#ef5350",wickUpColor:"#26a69a",wickDownColor:"#ef5350"});

// ── overlay canvas: resistance is drawn as shaded BLOCKS, not lines ──
$("chart").style.position="relative";
const ov=document.createElement("canvas");
ov.style.cssText="position:absolute;left:0;top:0;pointer-events:none;z-index:2;";
$("chart").appendChild(ov);

function drawBlocks(){
  const w=$("chart").clientWidth, h=$("chart").clientHeight;
  const dpr=window.devicePixelRatio||1;
  if(ov.width!==w*dpr||ov.height!==h*dpr){ ov.width=w*dpr; ov.height=h*dpr; ov.style.width=w+"px"; ov.style.height=h+"px"; }
  const ctx=ov.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  if(!S.zones.length && !S.setups.length) return;
  const ts=chart.timeScale();
  S.zones.forEach(function(z){
    const y=candle.priceToCoordinate(z.top);   // line at the cluster's top (the ceiling)
    if(y==null) return;
    let x0=ts.timeToCoordinate(z.fromTime);     // where the level first formed
    if(x0==null||x0<0) x0=0;
    // end the line at the LAST test, not the right edge — don't extend to full
    let lastT=z.fromTime;
    if(z.touches){ z.touches.forEach(function(p){ if(p.time>lastT) lastT=p.time; }); }
    let x1=ts.timeToCoordinate(lastT);
    if(x1==null||x1>w) x1=w;
    if(x1<=x0) return;
    const strong=z.n>=3, mid=z.n===2;
    const stroke = strong?"#f59e0b":(mid?"#fbbf24":"#fcd34d");
    ctx.strokeStyle=stroke; ctx.lineWidth=strong?3:(mid?2:1.5);
    ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
    ctx.fillStyle=stroke; ctx.font="600 11px system-ui,Segoe UI,sans-serif";
    ctx.textBaseline="bottom"; ctx.textAlign="left";
    ctx.fillText(z.n+"× touch", x1+5, y-3);
    // multi-touch: mark each contributing peak with a dot at its exact high
    if(z.n>=2 && z.touches){
      z.touches.forEach(function(p){
        const px=ts.timeToCoordinate(p.time), py=candle.priceToCoordinate(p.value);
        if(px==null||py==null) return;
        ctx.beginPath(); ctx.arc(px,py,4.5,0,2*Math.PI);
        ctx.fillStyle=stroke; ctx.fill();
        ctx.lineWidth=1.5; ctx.strokeStyle="#0b0d11"; ctx.stroke();
      });
    }
  });
  // ── scenario boxes: highlight WHERE the bullish setup conditions were met ──
  S.setups.forEach(function(b){
    let x0=ts.timeToCoordinate(b.x0), x1=ts.timeToCoordinate(b.x1);
    const yT=candle.priceToCoordinate(b.yTop), yB=candle.priceToCoordinate(b.yBottom);
    if(x0==null||x1==null||yT==null||yB==null) return;
    if((x0<0&&x1<0) || (x0>w&&x1>w)) return;   // entirely outside the view → don't clutter the edge
    if(x0<0)x0=0; if(x1>w)x1=w;
    const col = b.kind===1 ? "#22d3ee" : (b.kind===3 ? "#34d399" : "#a78bfa");  // ① cyan · ② violet · ③ green
    ctx.save();
    ctx.fillStyle = b.kind===1 ? "rgba(34,211,238,0.10)" : (b.kind===3 ? "rgba(52,211,153,0.10)" : "rgba(167,139,250,0.10)");
    ctx.fillRect(x0, yT, x1-x0, yB-yT);
    ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
    ctx.strokeRect(x0, yT, x1-x0, yB-yT); ctx.setLineDash([]);
    // ── highlight the JUDGED VWAP segment (2nd test → re-approach): a bold,
    //    box-coloured polyline over the plain blue VWAP so you can SEE the
    //    slope/hug the category was decided on ──
    if(b.vwPts && b.vwPts.length>1){
      ctx.beginPath(); let started=false;
      b.vwPts.forEach(function(p){
        const px=ts.timeToCoordinate(p.time), py=candle.priceToCoordinate(p.value);
        if(px==null||py==null) return;
        if(!started){ ctx.moveTo(px,py); started=true; } else ctx.lineTo(px,py);
      });
      ctx.strokeStyle=col; ctx.lineWidth=3; ctx.globalAlpha=0.9; ctx.stroke(); ctx.globalAlpha=1;
    }
    // INNER box — the VWAP-logic region only (hug → lift-off), drawn in amber so it
    // stands out from the green full-setup box
    if(b.inner){
      let ix0=ts.timeToCoordinate(b.inner.x0), ix1=ts.timeToCoordinate(b.inner.x1);
      const iyT=candle.priceToCoordinate(b.inner.yTop), iyB=candle.priceToCoordinate(b.inner.yBottom);
      if(ix0!=null&&ix1!=null&&iyT!=null&&iyB!=null){
        if(ix0<0)ix0=0; if(ix1>w)ix1=w;
        ctx.fillStyle="rgba(239,68,68,0.10)";
        ctx.fillRect(ix0, iyT, ix1-ix0, iyB-iyT);
        ctx.strokeStyle="#ef4444"; ctx.lineWidth=1.4; ctx.setLineDash([2,2]);
        ctx.strokeRect(ix0, iyT, ix1-ix0, iyB-iyT); ctx.setLineDash([]);
        ctx.fillStyle="#ef4444"; ctx.font="600 9px system-ui,Segoe UI,sans-serif";
        ctx.textBaseline="bottom"; ctx.textAlign="left";
        ctx.fillText("VWAP logic", ix0+3, iyB-2);
      }
    }
    // label + the measured stats behind the category
    const stat = "  (VWAP "+(b.slope>=0?"+":"")+b.slope+"% · hug "+b.nCand+" candles → lift-off)";
    ctx.fillStyle=col; ctx.font="700 11px system-ui,Segoe UI,sans-serif";
    ctx.textBaseline="top"; ctx.textAlign="left";
    ctx.fillText(b.label+stat, x0+5, yT+4);
    (b.pts||[]).forEach(function(p){
      const px=ts.timeToCoordinate(p.time), py=candle.priceToCoordinate(p.value);
      if(px==null||py==null) return;
      if(p.star){   // reclaim candle — point of interest, drawn as an up-triangle
        ctx.beginPath(); ctx.moveTo(px,py-10); ctx.lineTo(px-6,py+3); ctx.lineTo(px+6,py+3); ctx.closePath();
        ctx.fillStyle=col; ctx.fill(); ctx.lineWidth=1.2; ctx.strokeStyle="#0b0d11"; ctx.stroke();
      } else {      // a held swing low
        ctx.beginPath(); ctx.arc(px,py,3.5,0,2*Math.PI);
        ctx.fillStyle=col; ctx.fill(); ctx.lineWidth=1.2; ctx.strokeStyle="#0b0d11"; ctx.stroke();
      }
    });
    ctx.restore();
  });
}
(function loop(){ drawBlocks(); requestAnimationFrame(loop); })();

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
  // Daily: fit the full history. Intraday (1h/30m/15m): thousands of bars, so
  // fitContent looks fully compressed — default to the most recent ~200 bars.
  if(S.tf!=="1d"){
    const n=S.candles.length;
    chart.timeScale().setVisibleLogicalRange({from:Math.max(0,n-200), to:n-1+6});
  } else {
    chart.timeScale().fitContent();
  }
  drawVWAP();
  $("title").textContent=sym+" · "+S.tf+" · "+S.candles.length+" bars";
}

// ── VWAP: typical price (H+L+C)/3 weighted by volume. Resets each trading
//    day on Hourly (intraday session VWAP); cumulative anchor on Daily. ──
function computeVWAP(candles, resetDaily){
  const out=[]; let pv=0, vv=0, curDay=null;
  for(let i=0;i<candles.length;i++){
    const c=candles[i], day=Math.floor(c.time/86400);
    if(resetDaily && day!==curDay){ pv=0; vv=0; curDay=day; }
    const tp=(c.high+c.low+c.close)/3, v=c.volume||0;
    pv+=tp*v; vv+=v;
    out.push({time:c.time, value: vv>0 ? pv/vv : tp});
  }
  return out;
}

function drawVWAP(){
  if(vwapSeries){ try{chart.removeSeries(vwapSeries)}catch(e){} vwapSeries=null; }
  if(!vwapOn || !S.candles.length) return;
  vwapSeries=chart.addLineSeries({color:"#a855f7", lineWidth:2, lineStyle:0,
    priceLineVisible:false, lastValueVisible:true, crosshairMarkerVisible:false, title:"VWAP"});
  vwapSeries.setData(computeVWAP(S.candles, S.tf!=="1d"));
}

// VWAP values aligned to a visible slice (uses the same full-history/daily-reset
// series that is drawn, so detection matches what the eye sees on the chart).
function visVwap(vis){
  const full=computeVWAP(S.candles, S.tf!=="1d");
  const off=S.candles.indexOf(vis[0]);
  return vis.map(function(c,i){ const w=full[off+i]; return w?w.value:(c.high+c.low+c.close)/3; });
}

// ── cluster swing-high pivots into resistance zones ────────────────────
// Base collecting tolerance scales with the timeframe (1d 1.6% / 1h 0.8% /
// 30m·15m 0.35%). Two refinements:
//  • ADAPTIVE %: once a level has 2 touches that sit close together, the
//    collecting band shrinks proportionally (down to TOL/4) so a far-away pivot
//    can't widen an already-tight resistance.
//  • MIN GAP: two touches must be ≥12 candles apart in time — a pivot too close
//    to an existing touch is the same swing, so it's ignored (not a new test).
function clusterPivots(piv, tf){
  piv.sort(function(a,b){return a.price-b.price;});
  const TOL = tf==="1d" ? 0.016 : (tf==="1h" ? 0.008 : 0.0035);
  const MINTOL=TOL*0.25, MINBARS=12;
  const zones=[];
  for(let i=0;i<piv.length;i++){
    const z=zones[zones.length-1];
    if(z){
      let effTol=TOL;
      if(z.n>=2){ const sr=(z.top-z.bottom)/z.avg; effTol=Math.min(TOL, Math.max(MINTOL, sr*1.5)); }
      if(Math.abs(piv[i].price-z.avg)<=z.avg*effTol){
        // too close in time to an existing touch → same swing, ignore it
        if(z.pts.some(function(p){return Math.abs(piv[i].i-p.i)<MINBARS;})) continue;
        z.sum+=piv[i].price; z.n++; z.avg=z.sum/z.n;
        z.top=Math.max(z.top,piv[i].price); z.bottom=Math.min(z.bottom,piv[i].price);
        z.firstIdx=Math.min(z.firstIdx,piv[i].i); z.pts.push(piv[i]);
        continue;
      }
    }
    zones.push({sum:piv[i].price, n:1, avg:piv[i].price,
                top:piv[i].price, bottom:piv[i].price, firstIdx:piv[i].i, pts:[piv[i]]});
  }
  return zones;
}

// ── classify how a level was broken, TOLERATING ONE false breakout ──────
// A "false breakout" is a close above the level that later RETURNS below it (a
// bullish candle pokes through then price comes back and keeps rejecting). We
// allow at most one. Returns {falseBreaks, terminal} where `terminal` is the
// index of the first SUSTAINED (real) break — i.e. price closed above and never
// came back — or vis.length if the level was never sustainably broken.
function levelBreaks(vis, top, fromIdx){
  const BREAK=0.0015, up=top*(1+BREAK), n=vis.length;
  let above=false, breakStart=-1, falseBreaks=0;
  for(let j=fromIdx+1;j<n;j++){
    if(!above && vis[j].close>up){ above=true; breakStart=j; }     // poked above the level
    else if(above && vis[j].close<top){ above=false; falseBreaks++; } // came back under → that break was FALSE
  }
  return {falseBreaks:falseBreaks, terminal: above ? breakStart : n};
}

// ── setup detection for ONE resistance level, scanning the PAST ─────────
// Returns 0 or 1 box. A level is active from its first test until a close
// decisively breaks above it; the setup is only sought inside that window.
//
// THE setup (single definition) — hug-the-flat-VWAP, then GREEN lift-off, UNDER resistance:
//   • the level is tested >= 2 times (price rejected there — "resistance takes it back"), then
//   • price HOLDS BELOW the resistance (never closes above the level), and
//   • for >= 5 candles the closes HUG the VWAP (sit on the line) while the VWAP is
//     FLAT/slight-slope — price and VWAP coil together, then
//   • the next candle is a GREEN candle that closes CLEARLY ABOVE the VWAP — the
//     lift-off off the line — still below the resistance. That is the trigger.
function detectScenario(vis, vw, z){
  const out=[], n=vis.length, BREAK=0.0015;
  const HUG=0.004;    // a close within 0.4% of the VWAP = "sitting on the line"
  const FLAT=0.004;   // VWAP slope across the hug must stay within +/-0.4% (flat/slight)
  const MINHUG=5;     // ...for at least 5 candles
  const top=z.top;
  const touches=z.pts.map(function(p){return p.i;}).sort(function(a,b){return a-b;});
  if(touches.length<2) return out;                   // MANDATORY: tested >= 2 times
  const firstTouch=touches[0], secondTouch=touches[1], lastTouch=touches[touches.length-1];
  // Break handling, tolerating ONE false breakout (a poke above that returns below):
  //   • qualifies for tolerance when there's <=1 false breakout AND (none, or >=3
  //     rejections) → the active window extends PAST the lone false breakout to the
  //     first SUSTAINED (real) break, so re-rejections after the poke still count.
  //   • otherwise (>1 poke, or 1 poke with <3 rejections) → fall back to the strict
  //     ceiling at the FIRST close above (old behaviour) — pre-breakout setups kept.
  const bk=levelBreaks(vis, top, firstTouch);
  let brkIdx;
  if(bk.falseBreaks<=1 && (bk.falseBreaks===0 || touches.length>=3)){
    brkIdx=bk.terminal;
  } else {
    brkIdx=n;
    for(let j=firstTouch+1;j<n;j++){ if(vis[j].close>top*(1+BREAK)){ brkIdx=j; break; } }
  }
  // a touch after the (effective) break = role-reversal → drop
  if(lastTouch > brkIdx) return out;

  const a=secondTouch;                               // the rejection (2nd test)
  const cap=Math.min(brkIdx-1, n-1, a+80);           // stay within the active window (below the level)
  const belowRes=function(i){ return vis[i].close < top*(1+BREAK); };

  // scan for: a run of >=5 closes hugging a flat VWAP, immediately followed by a
  // GREEN candle that closes clearly above the VWAP (the lift-off). All below resistance.
  let i=a+1;
  while(i+MINHUG<=cap){
    // extend a hug run while closes sit on the line (within HUG) and stay below resistance
    let h0=i, h1=i-1, j=i;
    while(j<=cap && belowRes(j) && Math.abs(vis[j].close-vw[j])/vw[j] <= HUG){ h1=j; j++; }
    const len=h1-h0+1;
    if(len>=MINHUG){
      const flat = Math.abs(vw[h1]-vw[h0])/vw[h0] <= FLAT;     // VWAP flat across the hug
      const L=j;                                               // candle that ended the hug = lift-off candidate
      if(flat && L<=cap){
        const c=vis[L];
        const greenLift = c.close>c.open && c.close > vw[L]*(1+HUG) && belowRes(L);
        if(greenLift){
          const slopePct=+(((vw[h1]-vw[h0])/vw[h0])*100).toFixed(2);
          // OUTER box: the FULL setup — first test through the lift-off, enclosing the resistance line
          const left=firstTouch, right=L;
          let minLow=Infinity; for(let k=left;k<=right;k++) minLow=Math.min(minLow,vis[k].low);
          // INNER box: just the VWAP-logic region — the hug → lift-off, framed tightly
          let hiH=-Infinity, loL=Infinity;
          for(let k=h0;k<=L;k++){ hiH=Math.max(hiH,vis[k].high); loL=Math.min(loL,vis[k].low); }
          // the flat VWAP segment threading through the hug + lift-off
          const vwPts=[]; for(let k=h0;k<=L;k++) vwPts.push({time:vis[k].time, value:vw[k]});
          out.push({kind:3, x0:vis[left].time, x1:vis[right].time, li:left, ri:right,
                    yTop:top*(1+0.004), yBottom:minLow*(1-0.002),
                    label:"tested ≥2× · hug flat VWAP · green lift-off ↑",
                    slope:slopePct, nCand:len, vwPts:vwPts,
                    inner:{x0:vis[h0].time, x1:vis[L].time, yTop:hiH*(1+0.002), yBottom:loL*(1-0.002)},
                    pts:[{time:vis[L].time, value:vis[L].close, star:true}]});  // lift-off candle (triangle)
          return out;
        }
      }
      i=h1+1;          // hug found but no valid lift-off — search on from the run's end
    } else {
      i++;
    }
  }
  return out;
}

// ── Find Setups: scan the FULL chart, box every lift-off setup, zoom to latest ──
function detectSetups(){
  if(S.setups.length){ clearMarks(); $("title").textContent="setups cleared"; return; }
  if(!S.candles.length) return;
  const vis=S.candles;                       // FULL chart history — not just the visible window
  if(vis.length<10){$("title").textContent="not enough bars on this chart";return;}
  chart.timeScale().fitContent();
  const k=+$("k").value, vw=visVwap(vis), piv=[];
  for(let i=k;i<vis.length-k;i++){
    let hi=true; for(let j=i-k;j<=i+k;j++){ if(vis[j].high>vis[i].high){hi=false;break;} }
    if(hi) piv.push({i:i, price:vis[i].high});
  }
  S.zones=[]; S.setups=[];
  clusterPivots(piv, S.tf).forEach(function(z){
    if(z.n<2) return;
    const found=detectScenario(vis,vw,z);
    if(!found.length) return;
    found.forEach(function(s){S.setups.push(s);});
    S.zones.push({top:z.top, bottom:z.bottom, n:z.n, fromTime:vis[z.firstIdx].time,
                  touches: z.pts.map(function(p){return {time:vis[p.i].time, value:p.price};})});
  });
  if(!S.setups.length){
    $("title").textContent = S.tf==="1d"
      ? "no setups on Daily — this VWAP-hug setup needs an INTRADAY timeframe (15m/30m/1h); Daily VWAP is a slow cumulative line price doesn't hug"
      : "no setups (need ≥2-tested level, closes hugging a flat VWAP below it, then a green lift-off above VWAP)";
    return;
  }
  // zoom to the MOST RECENT setup (older ones stay drawn — scroll left to see them)
  let last=S.setups[0];
  S.setups.forEach(function(s){ if(s.ri>last.ri) last=s; });
  chart.timeScale().setVisibleLogicalRange({from:Math.max(0,last.li-25), to:last.ri+30});
  drawBlocks();
  $("setups").classList.add("on"); $("setups").textContent="✕ Clear Setups";
  $("title").textContent = S.zones.length+" level(s) · "+S.setups.length+" setup(s) — showing the latest; scroll ← for earlier";
}

// ── resistance on the VISIBLE window only ────────────────────────────
function applyResistance(){
  if(!S.candles.length)return;
  const r=chart.timeScale().getVisibleRange();
  if(!r){return;}
  const vis=S.candles.filter(function(c){return c.time>=r.from && c.time<=r.to;});
  if(vis.length<5){$("title").textContent="zoom out a little — too few bars on screen";return;}
  const k=+$("k").value, minTouch=+$("touch").value;
  // 1) swing-high pivots: high[i] is the max within +/- k bars (keep bar index)
  const piv=[];
  for(let i=k;i<vis.length-k;i++){
    let isHigh=true;
    for(let j=i-k;j<=i+k;j++){ if(vis[j].high>vis[i].high){isHigh=false;break;} }
    if(isHigh) piv.push({i:i, price:vis[i].high});
  }
  if(!piv.length){$("title").textContent="no swing highs in this window";return;}
  // 2) cluster nearby pivots into zones (shared recipe: timeframe-scaled tolerance
  //    that tightens for close touches, and a ≥12-candle min gap between touches).
  const zones=clusterPivots(piv, S.tf);
  // 3) CURRENT resistance only: tested >= minTouch AND still intact. A level the
  //    price SUSTAINABLY closed above is gone — but we TOLERATE one false breakout
  //    (a poke above that returns below), provided the level rejected >= 3 times.
  const kept=[];
  zones.forEach(function(z){
    if(z.n<minTouch) return;
    const bk=levelBreaks(vis, z.top, z.firstIdx);
    const intact = bk.terminal>=vis.length && bk.falseBreaks<=1 && (bk.falseBreaks===0 || z.n>=3);
    if(intact) kept.push(z);
  });
  if(!kept.length){$("title").textContent="no intact resistance overhead in this window (price near its high)";return;}
  kept.sort(function(a,b){return b.n-a.n;});
  // store as blocks (band = lowest..highest peak of the cluster), drawn on the overlay
  S.zones=kept.map(function(z){
    return {top:z.top, bottom:z.bottom, n:z.n, fromTime:vis[z.firstIdx].time,
            touches: z.pts.map(function(p){return {time:vis[p.i].time, value:p.price};})};
  });
  drawBlocks();
  $("title").textContent=S.zones.length+" current resistance line(s) · tested & unbroken  ("+vis.length+" bars)";
}

function clearMarks(){ S.zones=[]; S.setups=[]; drawBlocks();
  $("apply").classList.remove("on"); $("apply").textContent="⬛ Apply Resistance";
  $("setups").classList.remove("on"); $("setups").textContent="🎯 Find Setups"; }

// ── wire-up ──────────────────────────────────────────────────────────
[].slice.call($("tf").children).forEach(function(b){b.onclick=function(){
  [].slice.call($("tf").children).forEach(function(x){x.classList.remove("on")});
  b.classList.add("on"); S.tf=b.dataset.tf; load();};});
$("sym").addEventListener("change",load);
// Toggle: 1st click draws resistance for the visible window, 2nd click removes.
$("apply").onclick=function(){
  if(S.zones.length){ clearMarks(); $("title").textContent="resistance removed"; return; }
  applyResistance();
  if(S.zones.length){ $("apply").classList.add("on"); $("apply").textContent="✕ Remove Resistance"; }
};
$("setups").onclick=function(){ detectSetups(); };
$("vwap").onclick=function(){ vwapOn=!vwapOn; $("vwap").classList.toggle("on",vwapOn); drawVWAP();
  $("title").textContent="VWAP "+(vwapOn?("on · "+(S.tf==="1d"?"cumulative anchor":"daily-session reset")):"off"); };
$("clear").onclick=function(){clearMarks(); $("title").textContent="cleared";};
$("info").onclick=function(){ const on=$("logic").classList.toggle("on"); $("info").classList.toggle("on",on); };

// ── boot ─────────────────────────────────────────────────────────────
(async function(){
  const m=await(await fetch(api("/api/meta"))).json();
  $("sym").innerHTML=(m.symbols||[]).map(function(s){
    return "<option"+(s===m.default?" selected":"")+">"+s+"</option>";}).join("");
  await load();
})();
</script>
</body></html>"""
