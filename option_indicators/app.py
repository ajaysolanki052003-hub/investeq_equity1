"""Option Indicator Lab (/optlab) — live + replay indicator behaviour on any
NIFTY option contract.

Pick an expiry + strike + CE/PE and watch the popular indicators (Volume
Profile, MACD, RSI, VWAP, Bollinger, EMA stack, ADX, Stochastic, OBV) evolve
on that contract's premium — LIVE during market hours (auto-polled), or REPLAY
any past session by scrubbing/playing back its 1-minute bars.

Single data path: Groww `/v1/historical/candles` (segment=FNO) serves a
contract's 1-min OHLCV+volume for TODAY up to now AND any day back to 2020 —
so replay needs no separate recorder.

Run:
    APP_BASE=/optlab uvicorn option_indicators.app:app --host 127.0.0.1 --port 8708
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ema_scanner"))
from groww_client import get_access_token, fetch_candles          # noqa: E402
from fetch_oi_intraday import _gsym, _next_weekly_expiries, STRIKE_STEP  # noqa: E402
from option_indicators import indicators as ind                    # noqa: E402

DATA = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
SPOT_MASTER = DATA / "nifty_1m_master.parquet"
APP_BASE = os.environ.get("APP_BASE", "")

app = FastAPI(title="Option Indicator Lab")


# ─────────────────────────────── auth (cached) ──────────────────────────────
_auth = {"token": None, "at": 0.0}
_auth_lock = threading.Lock()


def _token() -> str:
    with _auth_lock:
        if _auth["token"] is None or time.time() - _auth["at"] > 3000:
            _auth["token"] = get_access_token(os.environ["GROWW_TOTP_JWT"],
                                              os.environ["GROWW_TOTP_SECRET"])
            _auth["at"] = time.time()
        return _auth["token"]


def _now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _today_ist() -> date:
    return _now_ist().date()


# ─────────────────────────── contract universe ──────────────────────────────
def _atm() -> int:
    """Latest NIFTY spot snapped to the 50-strike grid (from the 1m master)."""
    try:
        df = pd.read_parquet(SPOT_MASTER, columns=["close"])
        spot = float(df["close"].iloc[-1])
    except Exception:
        spot = 23000.0
    return int(round(spot / STRIKE_STEP) * STRIKE_STEP)


def _expiries(n: int = 6) -> list[str]:
    return [e.isoformat() for e in _next_weekly_expiries(_today_ist(), n=n)]


def _strikes(width: int = 15) -> list[int]:
    a = _atm()
    return list(range(a - width * STRIKE_STEP, a + width * STRIKE_STEP + 1, STRIKE_STEP))


# ───────────────────────────── bar fetching ─────────────────────────────────
# Cache keyed (gsym, date, interval). Today's entry carries a short TTL so the
# live poll picks up new 1-min bars; past days are immutable so cached forever.
_bars: dict[tuple, tuple] = {}
_bars_lock = threading.Lock()
TODAY_TTL = 20.0


def _fetch_bars(gsym: str, d: date, interval: str) -> pd.DataFrame:
    key = (gsym, d.isoformat(), interval)
    is_today = (d == _today_ist())
    with _bars_lock:
        hit = _bars.get(key)
        if hit and (not is_today or time.time() - hit[0] < TODAY_TTL):
            return hit[1]
    s = f"{d.isoformat()} 09:15:00"
    e = f"{d.isoformat()} 15:30:00"
    df = fetch_candles(gsym, interval, s, e, _token(), segment="FNO", delay_s=0)
    if df is None or df.empty:
        out = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    else:
        out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["close"]).reset_index(drop=True)
    with _bars_lock:
        _bars[key] = (time.time(), out)
    return out


# ─────────────────────────────── endpoints ──────────────────────────────────
@app.get("/api/meta")
def meta():
    return {"expiries": _expiries(), "strikes": _strikes(), "atm": _atm(),
            "today": _today_ist().isoformat()}


@app.get("/api/strikes")
def strikes():
    return {"strikes": _strikes(), "atm": _atm()}


@app.get("/api/series")
def series(expiry: str = Query(...), strike: int = Query(...),
           type: str = Query("CE"), date_: str = Query(None, alias="date"),
           interval: str = Query("1m")):
    typ = type.upper()
    if typ not in ("CE", "PE"):
        raise HTTPException(400, "type must be CE or PE")
    try:
        exp = datetime.strptime(expiry, "%Y-%m-%d").date()
        d = (datetime.strptime(date_, "%Y-%m-%d").date() if date_ else _today_ist())
    except ValueError:
        raise HTTPException(400, "bad date/expiry (YYYY-MM-DD)")
    if interval not in ("1m", "3m", "5m", "15m"):
        raise HTTPException(400, "interval must be 1m|3m|5m|15m")

    gsym = _gsym(exp, int(strike), typ)
    df = _fetch_bars(gsym, d, interval)
    if df.empty:
        return JSONResponse({"symbol": gsym, "candles": [], "volume": [],
                             "indicators": {}, "n": 0,
                             "live": d == _today_ist()})

    candles = [{"time": int(t), "open": float(o), "high": float(h),
                "low": float(l), "close": float(c)}
               for t, o, h, l, c in zip(df["timestamp"], df["open"],
                                        df["high"], df["low"], df["close"])]
    return JSONResponse({
        "symbol":  gsym,
        "candles": candles,
        "volume":  ind.volume_bars(df),   # drives the client-side Volume Profile
        "n":       len(df),
        "live":    d == _today_ist(),
        "asof":    _now_ist().strftime("%H:%M:%S"),
    })


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(PAGE.replace("__APP_BASE__", APP_BASE))


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Option Volume Profile</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0b0d11;--panel:#10131a;--border:#222838;--text:#e7eaf1;--muted:#8a93a6;--accent:#22d3ee;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,Segoe UI,Roboto,sans-serif}
#bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:linear-gradient(180deg,#0d1018,#10131a);border-bottom:1px solid var(--border)}
#bar label{color:var(--muted);font-size:11px;margin-right:3px}
select,input,button{background:#0d1117;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 9px;font:inherit}
button{cursor:pointer}
button.on{background:linear-gradient(135deg,#22d3ee,#0891b2);color:#04141a;border-color:transparent;font-weight:700}
.seg button{border-radius:0;margin:0}
.seg button:first-child{border-radius:6px 0 0 6px}
.seg button:last-child{border-radius:0 6px 6px 0}
.tag{color:var(--accent);font-weight:700;letter-spacing:.4px;margin-right:4px}
#title{font-weight:700;font-size:13px;color:var(--muted);margin-left:auto}
.charts{height:calc(100vh - 56px)}
#price{height:100%;position:relative}
#vpcanvas{position:absolute;top:0;left:0;height:100%;pointer-events:none;z-index:5}
#status{color:var(--muted);font-size:11px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;margin-right:5px;vertical-align:middle}
.dot.live{background:#22c55e;box-shadow:0 0 6px #22c55e}
#replay{display:flex;gap:8px;align-items:center}
#scrub{width:220px}
</style></head><body>

<div id="bar">
  <span class="tag">VOLUME PROFILE</span>
  <span><label>Expiry</label><select id="expiry"></select></span>
  <span><label>Strike</label><select id="strike"></select></span>
  <span class="seg" id="cepe"><button data-t="CE" class="on">CE</button><button data-t="PE">PE</button></span>
  <span><label>Date</label><input type="date" id="date"></span>
  <span class="seg" id="ivl">
    <button data-i="1m" class="on">1m</button><button data-i="3m">3m</button>
    <button data-i="5m">5m</button><button data-i="15m">15m</button></span>
  <span><label>Bins</label><select id="bins">
    <option>24</option><option selected>48</option><option>72</option><option>96</option></select></span>
  <button id="live-btn">&#9679; LIVE</button>
  <span id="replay">
    <button id="play">&#9654;</button>
    <input type="range" id="scrub" min="0" max="0" value="0">
    <select id="speed"><option value="500">1x</option><option value="250">2x</option>
      <option value="100">5x</option><option value="40">12x</option></select>
  </span>
  <span id="status"><span class="dot"></span><span id="status-txt">&mdash;</span></span>
  <span id="title"></span>
</div>

<div class="charts"><div id="price"><canvas id="vpcanvas" width="200"></canvas></div></div>

<script>
const APP_BASE="__APP_BASE__";
const api=function(p){return (APP_BASE||"")+p;};
const $=function(id){return document.getElementById(id);};

const S={data:null, idx:0, live:false, playing:false, timer:null, liveTimer:null,
         tf:"1m", type:"CE", today:null};

const price=LightweightCharts.createChart($("price"),{autoSize:true,
  layout:{background:{color:"#0b0d11"},textColor:"#cfd6e4"},
  grid:{vertLines:{color:"#161b27"},horzLines:{color:"#161b27"}},
  timeScale:{timeVisible:true,secondsVisible:false,borderColor:"#222838",rightOffset:6},
  rightPriceScale:{borderColor:"#222838"},crosshair:{mode:0}});
const candle=price.addCandlestickSeries({upColor:"#26a69a",downColor:"#ef5350",
  borderUpColor:"#26a69a",borderDownColor:"#ef5350",wickUpColor:"#26a69a",wickDownColor:"#ef5350"});
let vpLines=[];

price.timeScale().subscribeVisibleTimeRangeChange(function(){drawVP();});

function render(t){ candle.setData(S.data.candles.filter(function(c){return c.time<=t;}));
  drawVP(); requestAnimationFrame(drawVP); }

function drawVP(){
  const cv=$("vpcanvas"),ctx=cv.getContext("2d");
  cv.height=$("price").clientHeight; cv.style.width="200px";
  ctx.clearRect(0,0,cv.width,cv.height);
  vpLines.forEach(function(l){try{candle.removePriceLine(l)}catch(e){}}); vpLines=[];
  if(!S.data||!S.data.candles.length)return;
  const t=S.data.candles[S.idx]?S.data.candles[S.idx].time:1e18;
  const cs=S.data.candles.filter(function(c){return c.time<=t;});
  if(cs.length<2)return;
  const vmap={}; (S.data.volume||[]).forEach(function(v){vmap[v.time]=v.value;});
  let lo=Infinity,hi=-Infinity; cs.forEach(function(c){lo=Math.min(lo,c.low);hi=Math.max(hi,c.high);});
  if(!(hi>lo))return;
  const N=+$("bins").value,bins=new Array(N).fill(0),step=(hi-lo)/N;
  cs.forEach(function(c){const b=Math.min(N-1,Math.floor((c.close-lo)/step));bins[b]+=(vmap[c.time]||0);});
  const max=Math.max.apply(null,bins)||1, pocB=bins.indexOf(max);
  const tot=bins.reduce(function(a,b){return a+b;},0); let lo_i=pocB,hi_i=pocB,acc=bins[pocB];
  while(acc<0.7*tot&&(lo_i>0||hi_i<N-1)){
    const dn=lo_i>0?bins[lo_i-1]:-1, up=hi_i<N-1?bins[hi_i+1]:-1;
    if(up>=dn){hi_i++;acc+=bins[hi_i];}else{lo_i--;acc+=bins[lo_i];}}
  const y=function(p){const c=candle.priceToCoordinate(p);return c==null?null:c;};
  for(let i=0;i<N;i++){
    const p0=lo+i*step,p1=p0+step,y0=y(p1),y1=y(p0); if(y0==null||y1==null)continue;
    const w=(bins[i]/max)*188;
    // POC bin brightest, value-area bins solid cyan, the rest a muted slate —
    // left-anchored solid horizontal volume bars (extend right from x=0).
    ctx.fillStyle = i===pocB ? "#22d3eecc"
                  : (i>=lo_i&&i<=hi_i) ? "#22d3ee88" : "#64748b77";
    ctx.fillRect(0,y0,w,Math.max(1,(y1-y0)-1));
  }
  const poc=lo+(pocB+0.5)*step, vah=lo+(hi_i+1)*step, val=lo+lo_i*step;
  vpLines.push(candle.createPriceLine({price:poc,color:"#22d3ee",lineWidth:2,title:"POC"}));
  vpLines.push(candle.createPriceLine({price:vah,color:"#22d3ee66",lineStyle:2,lineWidth:1,title:"VAH"}));
  vpLines.push(candle.createPriceLine({price:val,color:"#22d3ee66",lineStyle:2,lineWidth:1,title:"VAL"}));
}

async function load(){
  const exp=$("expiry").value, k=$("strike").value, d=$("date").value;
  if(!exp||!k)return;
  setStatus(false,"loading...");
  const u=api("/api/series?expiry="+exp+"&strike="+k+"&type="+S.type+"&date="+d+"&interval="+S.tf);
  let j; try{j=await(await fetch(u)).json();}catch(e){setStatus(false,"fetch error");return;}
  S.data=j; S.idx=(j.candles.length||1)-1;
  $("title").textContent=(j.symbol||"")+" - "+(j.n||0)+" bars";
  $("scrub").max=Math.max(0,(j.candles.length||1)-1); $("scrub").value=S.idx;
  if(!j.candles||!j.candles.length){setStatus(false,"no data for this contract/day");candle.setData([]);return;}
  render(j.candles[S.idx].time);
  price.timeScale().fitContent();
  S.live=j.live; setStatus(j.live,j.asof?("live - "+j.asof):"loaded");
}
function setStatus(live,txt){$("status").querySelector(".dot").className="dot"+(live?" live":"");
  $("status-txt").textContent=txt||"";}

function step(){ if(!S.data)return;
  if(S.idx>=S.data.candles.length-1){pause();return;}
  S.idx++; $("scrub").value=S.idx; render(S.data.candles[S.idx].time); }
function play(){if(!S.data)return; stopLive(); S.playing=true; $("play").textContent="⏸";
  setStatus(false,"replay"); S.timer=setInterval(step,+$("speed").value);}
function pause(){S.playing=false; $("play").textContent="▶"; clearInterval(S.timer);}

function startLive(){stopLive(); if($("date").value!==S.today)return;
  S.liveTimer=setInterval(function(){if(!S.playing)load();},20000);}
function stopLive(){clearInterval(S.liveTimer);}

function seg(id,attr,fn){[].slice.call($(id).children).forEach(function(b){b.onclick=function(){
  [].slice.call($(id).children).forEach(function(x){x.classList.remove("on");}); b.classList.add("on"); fn(b.dataset[attr]);};});}
seg("cepe","t",function(t){S.type=t;load();});
seg("ivl","i",function(i){S.tf=i;load();});
$("bins").onchange=drawVP;
window.addEventListener("resize",function(){requestAnimationFrame(drawVP);});
$("scrub").oninput=function(e){pause(); S.idx=+e.target.value;
  if(S.data)render(S.data.candles[S.idx].time); setStatus(false,"replay @ "+S.idx);};
$("play").onclick=function(){S.playing?pause():play();};
$("speed").onchange=function(){if(S.playing){pause();play();}};
$("expiry").onchange=load; $("strike").onchange=load;
$("date").onchange=function(){pause(); if($("date").value===S.today)startLive(); else stopLive(); load();};
$("live-btn").onclick=function(){$("date").value=S.today; pause(); startLive(); load();};

(async function(){
  const m=await(await fetch(api("/api/meta"))).json();
  S.today=m.today; $("date").value=m.today;
  $("expiry").innerHTML=m.expiries.map(function(e){return "<option>"+e+"</option>";}).join("");
  $("strike").innerHTML=m.strikes.map(function(s){return "<option "+(s===m.atm?"selected":"")+">"+s+"</option>";}).join("");
  await load(); startLive();
})();
</script>
</body></html>"""


