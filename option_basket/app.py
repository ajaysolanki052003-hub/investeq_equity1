"""Straddle Basket — NIFTY short straddle / strangle combined-premium replay.

Pick a day + expiry and this builds a two-leg short option basket and plots the
**combined premium** (CE + PE) through the session, the way a broker's basket
"combined premium" tab does.

  • Default = ATM short STRADDLE. The ATM strike is chosen from that day's
    OPENING spot (the 09:15 spot → nearest 50-pt strike).
  • Editable strikes: a Call-offset and Put-offset (in strikes) turn it into an
    OTM short STRANGLE (e.g. Call +5 / Put −5).
  • Shows credit collected, live MTM for a short seller (credit − current),
    session high/low of the premium, and the break-evens.

The option feed (DATA/options/*.parquet) is a sparse, ATM-padded capture, so a
requested strike may not exist for a given expiry. Strikes are therefore chosen
DATA-DRIVEN: the ATM strike is the nearest strike present in BOTH CE and PE for
the expiry, and each offset snaps to the nearest available strike of its type —
the response reports exactly which strikes were used and warns when a snap moved
far from the target.

Run:
    APP_BASE=/basket python -m uvicorn option_basket.app:app --host 127.0.0.1 --port 8714
Mounted behind nginx at /basket/ (auth-gated like the other apps).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

APP_BASE = os.environ.get("APP_BASE", "").rstrip("/")
ROOT     = Path(os.environ.get("INVESTEQ_DATA", r"C:\Users\User\Desktop\investeq_ajs\DATA"))
OPT_DIR  = ROOT / "options"
SPOT_DIR = ROOT / "spot"

STRIKE_STEP = 50          # NIFTY strike grid
LOT_SIZE    = 75          # NIFTY lot (current); premium shown in points, ₹ = ×lot

TF_RULE = {"1m": "1min", "3m": "3min", "5m": "5min",
           "15m": "15min", "30m": "30min", "1h": "60min"}

app = FastAPI(title="Straddle Basket")


# ─────────────────────────────── loaders ───────────────────────────────────
@lru_cache(maxsize=64)
def load_options(date: str) -> pd.DataFrame:
    p = OPT_DIR / f"{date}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["expiry"] = df["expiry"].astype(str)
    df["strike"] = df["strike"].astype(int)
    return df


@lru_cache(maxsize=64)
def load_spot(date: str) -> pd.DataFrame:
    p = SPOT_DIR / f"{date}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=1)
def list_trading_days() -> tuple[str, ...]:
    opt = {p.stem for p in OPT_DIR.glob("*.parquet")}
    spot = {p.stem for p in SPOT_DIR.glob("*.parquet")}
    return tuple(sorted(opt & spot))


def _norm_date(date: str) -> str:
    return date.replace("-", "")


def _to_unix(ts: pd.Series) -> np.ndarray:
    # Naive-IST-as-UTC encoding: the same convention the other option apps use so
    # lightweight-charts' time axis lines the bars up on the IST clock.
    return (pd.to_datetime(ts).astype("int64") // 1_000_000_000).to_numpy()


def _resample_close(df: pd.DataFrame, tf: str) -> pd.Series:
    """Resample a single leg's 1m OHLCV to `tf`, return the close series."""
    if df.empty:
        return pd.Series(dtype=float)
    return (df.set_index("timestamp")["close"]
              .resample(TF_RULE[tf], closed="left", label="left")
              .last()
              .dropna())


def _line(idx: pd.Index, vals) -> list[dict]:
    t = _to_unix(pd.Series(idx))
    out = []
    for i, v in enumerate(vals):
        if pd.notna(v):
            out.append({"time": int(t[i]), "value": round(float(v), 2)})
    return out


def _nearest(cands: list[int], target: int):
    return min(cands, key=lambda k: abs(k - target)) if cands else None


# ─────────────────────────────── endpoints ─────────────────────────────────
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/days")
def api_days():
    days = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in list_trading_days()]
    return JSONResponse({"days": days,
                         "min": days[0] if days else None,
                         "max": days[-1] if days else None,
                         "step": STRIKE_STEP, "lot": LOT_SIZE})


@app.get("/api/expiries")
def api_expiries(date: str = ""):
    d = _norm_date(date)
    opt = load_options(d)
    if opt.empty:
        return JSONResponse({"date": date, "expiries": []})
    spot = load_spot(d)
    atm_target = None
    if not spot.empty:
        atm_target = int(round(float(spot.iloc[0]["open"]) / STRIKE_STEP) * STRIKE_STEP)
    exps = []
    for e in sorted(opt["expiry"].unique()):
        sub = opt[opt["expiry"] == e]
        ce = set(sub.loc[sub.option_type == "CE", "strike"])
        pe = set(sub.loc[sub.option_type == "PE", "strike"])
        both = ce & pe
        near = _nearest(sorted(both), atm_target) if (both and atm_target) else None
        within5 = sum(1 for k in both if atm_target and abs(k - atm_target) <= 5 * STRIKE_STEP)
        exps.append({"expiry": e, "n_common": len(both),
                     "n_within5": within5,
                     "atm_dist": (abs(near - atm_target) if near and atm_target else None)})
    return JSONResponse({"date": date, "atm_target": atm_target, "expiries": exps})


@app.get("/api/basket")
def api_basket(date: str = "", expiry: str = "", ce_off: int = 0, pe_off: int = 0,
               tf: str = "1m"):
    if tf not in TF_RULE:
        tf = "1m"
    d = _norm_date(date)
    opt = load_options(d)
    spot = load_spot(d)
    if opt.empty or spot.empty:
        return JSONResponse({"error": "no data for day", "date": date}, status_code=404)

    opening = float(spot.iloc[0]["open"])
    atm_target = int(round(opening / STRIKE_STEP) * STRIKE_STEP)

    # choose expiry (default: the one with the most CE∩PE coverage near ATM)
    exps = sorted(opt["expiry"].unique())
    if expiry not in exps:
        def score(e):
            sub = opt[opt["expiry"] == e]
            both = set(sub.loc[sub.option_type == "CE", "strike"]) & \
                   set(sub.loc[sub.option_type == "PE", "strike"])
            return sum(1 for k in both if abs(k - atm_target) <= 5 * STRIKE_STEP)
        expiry = max(exps, key=score) if exps else ""

    sub = opt[opt["expiry"] == expiry]
    ce_avail = sorted(sub.loc[sub.option_type == "CE", "strike"].unique())
    pe_avail = sorted(sub.loc[sub.option_type == "PE", "strike"].unique())
    both = sorted(set(ce_avail) & set(pe_avail))
    warnings: list[str] = []

    # ATM strike = nearest strike present in BOTH legs
    atm_strike = _nearest(both, atm_target) if both else None
    if atm_strike is None:
        return JSONResponse({"error": "no strike has both CE and PE for this expiry",
                             "date": date, "expiry": expiry}, status_code=422)
    atm_strike = int(atm_strike)
    if abs(atm_strike - atm_target) > STRIKE_STEP:
        warnings.append(f"ATM {atm_target} not in feed for this expiry — "
                        f"nearest common strike {atm_strike} ({atm_strike-atm_target:+d} pts).")

    # apply offsets, then snap each leg to nearest AVAILABLE strike of its type
    ce_target = atm_strike + ce_off * STRIKE_STEP
    pe_target = atm_strike + pe_off * STRIKE_STEP
    ce_strike = int(_nearest(ce_avail, ce_target))
    pe_strike = int(_nearest(pe_avail, pe_target))
    if ce_strike != ce_target:
        warnings.append(f"Call {ce_target} unavailable — snapped to {ce_strike}.")
    if pe_strike != pe_target:
        warnings.append(f"Put {pe_target} unavailable — snapped to {pe_strike}.")

    ce_bars = _resample_close(sub[(sub.option_type == "CE") & (sub.strike == ce_strike)], tf)
    pe_bars = _resample_close(sub[(sub.option_type == "PE") & (sub.strike == pe_strike)], tf)
    if ce_bars.empty or pe_bars.empty:
        return JSONResponse({"error": "leg has no intraday data", "date": date,
                             "expiry": expiry, "ce_strike": ce_strike,
                             "pe_strike": pe_strike}, status_code=422)

    merged = pd.concat({"ce": ce_bars, "pe": pe_bars}, axis=1).dropna()
    if merged.empty:
        return JSONResponse({"error": "legs do not overlap in time"}, status_code=422)
    merged["comb"] = merged["ce"] + merged["pe"]

    spot_bars = (spot.set_index("timestamp")["close"]
                     .resample(TF_RULE[tf], closed="left", label="left").last().dropna())

    credit  = float(merged["comb"].iloc[0])       # premium collected at entry (open)
    current = float(merged["comb"].iloc[-1])
    hi = float(merged["comb"].max())
    lo = float(merged["comb"].min())
    mtm = credit - current                        # short seller: profit as premium decays
    is_straddle = (ce_strike == pe_strike)

    stats = {
        "credit": round(credit, 2), "current": round(current, 2),
        "high": round(hi, 2), "low": round(lo, 2),
        "mtm": round(mtm, 2),
        "mtm_pct": round(mtm / credit * 100, 1) if credit else None,
        "mtm_rupees": round(mtm * LOT_SIZE, 0),
        "max_profit": round(credit, 2),           # full premium if both expire worthless
        "max_profit_rupees": round(credit * LOT_SIZE, 0),
        "breakeven_up": round(ce_strike + credit, 0),
        "breakeven_dn": round(pe_strike - credit, 0),
        "width": ce_strike - pe_strike,
    }
    meta = {
        "date": date, "expiry": expiry, "tf": tf,
        "opening_spot": round(opening, 2), "atm_target": atm_target,
        "atm_strike": atm_strike, "ce_strike": ce_strike, "pe_strike": pe_strike,
        "ce_off": (ce_strike - atm_strike) // STRIKE_STEP,
        "pe_off": (pe_strike - atm_strike) // STRIKE_STEP,
        "is_straddle": is_straddle, "lot": LOT_SIZE,
        "n_bars": len(merged), "warnings": warnings,
        "expiries": exps,
    }
    series = {
        "combined": _line(merged.index, merged["comb"].values),
        "ce": _line(merged.index, merged["ce"].values),
        "pe": _line(merged.index, merged["pe"].values),
        "spot": _line(spot_bars.index, spot_bars.values),
    }
    return JSONResponse({"meta": meta, "stats": stats, "series": series})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML.replace("__APP_BASE__", APP_BASE)


# ──────────────────────────────────── UI ───────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Straddle Basket</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{
    --bg:#0a0e17; --panel:#111726; --panel2:#0d1320; --line:#1e2940;
    --txt:#e6edf6; --mut:#8a98b2; --accent:#fb923c; --accent2:#22d3ee;
    --pos:#34d399; --neg:#f87171; --chip:#172033;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#33240f 0%,var(--bg) 55%);
       color:var(--txt);font:14px/1.5 'Inter',system-ui,Segoe UI,Roboto,sans-serif;min-height:100vh}
  .wrap{max-width:1340px;margin:0 auto;padding:26px 22px 60px}
  header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:18px}
  .title{font-size:25px;font-weight:800;display:flex;align-items:center;gap:12px}
  .title .dot{width:11px;height:11px;border-radius:50%;background:var(--accent);box-shadow:0 0 16px 2px var(--accent)}
  .sub{color:var(--mut);font-size:13px;margin-top:4px;max-width:760px}
  .kind{font-size:12.5px;color:var(--mut);text-align:right}
  .kind b{color:var(--accent);font-size:15px}
  .toolbar{display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap;margin-bottom:16px}
  .grp{display:flex;flex-direction:column;gap:5px}
  .grp .lab{font-size:10.5px;text-transform:uppercase;letter-spacing:.12em;color:var(--mut)}
  input[type=date],select{background:var(--chip);border:1px solid var(--line);color:var(--txt);
        border-radius:9px;padding:8px 10px;font-size:13px;outline:none;color-scheme:dark}
  input:focus,select:focus{border-color:var(--accent)}
  .seg{display:inline-flex;background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:3px}
  .seg button{background:none;border:0;color:var(--mut);padding:7px 12px;border-radius:8px;cursor:pointer;font-weight:600;font-size:12.5px}
  .seg button.on{background:var(--accent);color:#0a0a18}
  .stepper{display:flex;align-items:center;gap:0;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:var(--chip)}
  .stepper button{background:none;border:0;color:var(--txt);width:32px;height:36px;cursor:pointer;font-size:17px;font-weight:700}
  .stepper button:hover{background:#1c2740;color:var(--accent)}
  .stepper .v{min-width:64px;text-align:center;font-weight:700;font-size:13px;font-variant-numeric:tabular-nums}
  .stepper .v small{display:block;color:var(--mut);font-weight:500;font-size:10px}
  .mini{background:var(--chip);border:1px solid var(--line);color:var(--mut);border-radius:9px;padding:8px 12px;cursor:pointer;font-size:12px;font-weight:600}
  .mini:hover{border-color:var(--accent2);color:var(--accent2)}
  .card{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
        border:1px solid var(--line);border-radius:16px;padding:16px;margin-bottom:18px}
  .chips{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
  .ch{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:10px 14px;min-width:120px}
  .ch .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--mut)}
  .ch .v{font-size:19px;font-weight:800;font-variant-numeric:tabular-nums;margin-top:2px}
  .ch .v small{font-size:11px;color:var(--mut);font-weight:600}
  .pos{color:var(--pos)} .neg{color:var(--neg)}
  .legs{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px;font-size:12.5px;color:var(--mut)}
  .tag{background:var(--chip);border:1px solid var(--line);border-radius:999px;padding:3px 11px;color:var(--txt)}
  .tag.ce{border-color:#2dd4bf55} .tag.pe{border-color:#f9731655}
  .warn{background:#3a2a10;border:1px solid #6b4d18;color:#fbbf24;border-radius:10px;padding:8px 12px;font-size:12.5px;margin-bottom:12px}
  .toggles{display:flex;gap:8px;margin:2px 0 12px;flex-wrap:wrap}
  .toggles label{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--mut);cursor:pointer;
      background:var(--chip);border:1px solid var(--line);border-radius:8px;padding:5px 10px}
  .toggles label .sw{width:12px;height:3px;border-radius:2px}
  #chart{height:440px}
  .empty{padding:44px;text-align:center;color:var(--mut)}
  .spin{padding:36px;text-align:center;color:var(--mut)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div class="title"><span class="dot"></span> Straddle Basket</div>
      <div class="sub">Short a NIFTY straddle/strangle and watch the <b>combined premium</b> decay through the day. ATM is taken from the day's opening spot; nudge the Call/Put offsets to build an OTM strangle.</div>
    </div>
    <div class="kind">Basket<br><b id="kindLbl">—</b></div>
  </header>

  <div class="toolbar">
    <div class="grp"><span class="lab">Date</span><input type="date" id="datePick"/></div>
    <div class="grp"><span class="lab">Expiry</span><select id="expiry"></select></div>
    <div class="grp"><span class="lab">Call strike</span>
      <div class="stepper"><button data-leg="ce" data-d="-1">−</button>
        <div class="v" id="ceV">ATM<small id="ceK">—</small></div>
        <button data-leg="ce" data-d="1">+</button></div>
    </div>
    <div class="grp"><span class="lab">Put strike</span>
      <div class="stepper"><button data-leg="pe" data-d="-1">−</button>
        <div class="v" id="peV">ATM<small id="peK">—</small></div>
        <button data-leg="pe" data-d="1">+</button></div>
    </div>
    <div class="grp"><span class="lab">&nbsp;</span><button class="mini" id="resetBtn">Reset ATM</button></div>
    <div class="grp"><span class="lab">Timeframe</span><div class="seg" id="tfSeg"></div></div>
  </div>

  <div id="warnBox"></div>

  <div class="chips" id="chips"></div>

  <div class="card">
    <div class="legs" id="legsBar"></div>
    <div class="toggles">
      <label><input type="checkbox" id="tgCE"/> <span class="sw" style="background:#2dd4bf"></span> Call leg</label>
      <label><input type="checkbox" id="tgPE"/> <span class="sw" style="background:#fb7185"></span> Put leg</label>
      <label><input type="checkbox" id="tgSpot"/> <span class="sw" style="background:#8a98b2"></span> Spot</label>
    </div>
    <div id="chart"></div>
  </div>
</div>

<script>
const BASE="__APP_BASE__";
const TFS=["1m","3m","5m","15m","30m","1h"];
let STATE={date:"",expiry:"",ce_off:0,pe_off:0,tf:"1m"};
let DAYS=null, RESP=null;
let chart=null, sComb=null, sCE=null, sPE=null, sSpot=null;

function j(u){return fetch(BASE+u).then(r=>{if(!r.ok)return r.json().then(e=>{throw e;});return r.json();});}
function fmt(v,d=2){return (v===null||v===undefined||isNaN(v))?"—":Number(v).toFixed(d);}
function signed(v,d=2){return (v>=0?"+":"")+fmt(v,d);}

async function init(){
  DAYS=await j("/api/days");
  const dp=document.getElementById("datePick");
  if(DAYS.min)dp.min=DAYS.min; if(DAYS.max)dp.max=DAYS.max;
  dp.value=DAYS.max||""; STATE.date=dp.value;
  dp.onchange=async()=>{STATE.date=dp.value; STATE.expiry=""; await loadExpiries(); load();};
  const tf=document.getElementById("tfSeg");
  tf.innerHTML=TFS.map(t=>`<button data-t="${t}" class="${t===STATE.tf?'on':''}">${t}</button>`).join("");
  tf.querySelectorAll("button").forEach(b=>b.onclick=()=>{STATE.tf=b.dataset.t;
    tf.querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");load();});
  document.getElementById("expiry").onchange=e=>{STATE.expiry=e.target.value;STATE.ce_off=0;STATE.pe_off=0;load();};
  document.querySelectorAll(".stepper button").forEach(b=>b.onclick=()=>{
    const leg=b.dataset.leg, d=parseInt(b.dataset.d);
    if(leg==="ce")STATE.ce_off+=d; else STATE.pe_off+=d; load();});
  document.getElementById("resetBtn").onclick=()=>{STATE.ce_off=0;STATE.pe_off=0;load();};
  ["tgCE","tgPE","tgSpot"].forEach(id=>document.getElementById(id).onchange=drawExtras);
  buildChart();
  await loadExpiries(); load();
}

async function loadExpiries(){
  const sel=document.getElementById("expiry");
  try{
    const d=await j(`/api/expiries?date=${STATE.date}`);
    if(!d.expiries.length){sel.innerHTML=`<option>—</option>`;return;}
    // default to the best-covered expiry
    d.expiries.sort((a,b)=>b.n_within5-a.n_within5 || b.n_common-a.n_common);
    const best=d.expiries[0].expiry;
    if(!STATE.expiry)STATE.expiry=best;
    d.expiries.sort((a,b)=>a.expiry<b.expiry?-1:1);
    sel.innerHTML=d.expiries.map(e=>`<option value="${e.expiry}" ${e.expiry===STATE.expiry?'selected':''}>${e.expiry} (${e.n_common} strikes)</option>`).join("");
  }catch(e){sel.innerHTML=`<option>—</option>`;}
}

async function load(){
  document.getElementById("chips").innerHTML='<div class="spin">Loading…</div>';
  document.getElementById("warnBox").innerHTML="";
  let d;
  try{
    d=await j(`/api/basket?date=${STATE.date}&expiry=${STATE.expiry}&ce_off=${STATE.ce_off}&pe_off=${STATE.pe_off}&tf=${STATE.tf}`);
  }catch(err){
    document.getElementById("chips").innerHTML=`<div class="empty">${(err&&err.error)||"No data for this selection."}</div>`;
    if(sComb)sComb.setData([]); if(sCE)sCE.setData([]); if(sPE)sPE.setData([]); if(sSpot)sSpot.setData([]);
    return;
  }
  RESP=d; const m=d.meta, s=d.stats;
  // sync stepper displays to the ACTUAL strikes used
  STATE.ce_off=m.ce_off; STATE.pe_off=m.pe_off; STATE.expiry=m.expiry;
  document.getElementById("kindLbl").textContent=m.is_straddle?"ATM STRADDLE":"STRANGLE";
  document.getElementById("ceV").innerHTML=(m.ce_off>=0?"ATM+"+m.ce_off:"ATM"+m.ce_off)+`<small>${m.ce_strike} CE</small>`;
  document.getElementById("peV").innerHTML=(m.pe_off>=0?"ATM+"+m.pe_off:"ATM"+m.pe_off)+`<small>${m.pe_strike} PE</small>`;
  const es=document.getElementById("expiry"); if(es.value!==m.expiry)es.value=m.expiry;

  document.getElementById("legsBar").innerHTML=
    `<span>Opening spot <b style="color:var(--txt)">${fmt(m.opening_spot)}</b> → ATM ${m.atm_strike}</span>`+
    `<span class="tag ce">SELL ${m.ce_strike} CE</span>`+
    `<span class="tag pe">SELL ${m.pe_strike} PE</span>`+
    `<span>· ${m.n_bars} bars · lot ${m.lot}</span>`;

  const mtmCls=s.mtm>=0?"pos":"neg";
  document.getElementById("chips").innerHTML=[
    chip("Credit (entry)", fmt(s.credit), `pts · ₹${Math.round(s.max_profit_rupees).toLocaleString()}`),
    chip("Combined now", fmt(s.current), "pts"),
    chip("Short MTM", signed(s.mtm), `${signed(s.mtm_pct,1)}% · ₹${Math.round(s.mtm_rupees).toLocaleString()}`, mtmCls),
    chip("Session H / L", `${fmt(s.high)} / ${fmt(s.low)}`, "pts"),
    chip("Break-evens", `${s.breakeven_dn} – ${s.breakeven_up}`, `width ${s.width}`),
  ].join("");

  document.getElementById("warnBox").innerHTML=(m.warnings||[]).map(w=>`<div class="warn">⚠ ${w}</div>`).join("");

  sComb.setData(d.series.combined);
  chart.timeScale().fitContent();
  drawExtras();
}

function chip(k,v,sub,cls=""){
  return `<div class="ch"><div class="k">${k}</div><div class="v ${cls}">${v} <small>${sub||""}</small></div></div>`;
}

function buildChart(){
  const box=document.getElementById("chart");
  chart=LightweightCharts.createChart(box,{
    layout:{background:{color:"transparent"},textColor:"#8a98b2"},
    grid:{vertLines:{color:"#16203a"},horzLines:{color:"#16203a"}},
    rightPriceScale:{borderColor:"#1e2940"},timeScale:{borderColor:"#1e2940",timeVisible:true,secondsVisible:false},
    crosshair:{mode:0}, height:440,
  });
  sComb=chart.addAreaSeries({lineColor:"#fb923c",topColor:"rgba(251,146,60,.28)",bottomColor:"rgba(251,146,60,.02)",
    lineWidth:2,priceLineVisible:false,title:"Combined"});
  sCE=chart.addLineSeries({color:"#2dd4bf",lineWidth:1,priceLineVisible:false,lastValueVisible:false,visible:false,title:"CE"});
  sPE=chart.addLineSeries({color:"#fb7185",lineWidth:1,priceLineVisible:false,lastValueVisible:false,visible:false,title:"PE"});
  sSpot=chart.addLineSeries({color:"#8a98b2",lineWidth:1,priceScaleId:"spot",priceLineVisible:false,lastValueVisible:false,visible:false,title:"Spot"});
  chart.priceScale("spot").applyOptions({scaleMargins:{top:0.1,bottom:0.1},visible:false});
  new ResizeObserver(()=>chart.applyOptions({width:box.clientWidth})).observe(box);
}

function drawExtras(){
  if(!RESP)return;
  const ce=document.getElementById("tgCE").checked, pe=document.getElementById("tgPE").checked, sp=document.getElementById("tgSpot").checked;
  sCE.applyOptions({visible:ce}); sCE.setData(ce?RESP.series.ce:[]);
  sPE.applyOptions({visible:pe}); sPE.setData(pe?RESP.series.pe:[]);
  sSpot.applyOptions({visible:sp}); sSpot.setData(sp?RESP.series.spot:[]);
}

init();
</script>
</body>
</html>
"""
