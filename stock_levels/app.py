"""Level Screener — which stocks' daily candle touched a Daily / Weekly /
Monthly round-number price level on a chosen day.

Three level grids, each a set of evenly-spaced horizontal price lines:
  • Daily   — every 100 pts   (…, 24100, 24200, 24300, …)
  • Weekly  — every 300 pts
  • Monthly — every 200 pts
The step of each grid is a default the screen can override per request.

For any trading day you pick, this lists the stocks whose DAILY candle touched a
grid line — i.e. a multiple of the step falls inside that day's high–low range
(round numbers tend to act as support/resistance). Two test conditions:
  • Touched (inside candle) — a grid line lies within [low, high].
  • Within %                — the nearest grid line sits within `tol`% of close.

Reads ema_scanner/data/1d/*.csv (daily OHLC, one row per session). This feed is
refreshed ONCE per day AFTER the NSE close (investeq-candles-1d.timer at 15:35
IST, Mon-Fri) and never during market hours, so the screener always reflects the
last settled session — not a partial intraday candle. The level grids are pure
arithmetic on price, so no volume / profile is needed and the steps can be
changed at screen-time without rebuilding the snapshot.

Run:
    APP_BASE=/levels python -m uvicorn stock_levels.app:app --host 127.0.0.1 --port 8712
Mounted behind nginx at /levels/ (auth-gated like the other apps).
"""

from __future__ import annotations

import glob
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

APP_BASE = os.environ.get("APP_BASE", "").rstrip("/")

ROOT     = Path(__file__).resolve().parent.parent
# Daily candles, refreshed ONCE after the NSE close (investeq-candles-1d.timer @
# 15:35 IST, Mon-Fri) — never intraday. A daily screener only needs the settled
# daily candle, so we read this post-close feed rather than the intraday 1h one.
DATA_DIR = ROOT / "ema_scanner" / "data" / "1d"
SNAP_TAIL  = 1200       # daily bars per symbol for the universe snapshot (~4.7y)
CHART_TAIL = 800        # daily bars for a single-stock chart
CHART_MAX_LINES = 18    # cap grid lines drawn per period so the chart stays readable

# Level grids (key, label, default step) — each is a set of round-number lines.
GRIDS = [
    {"key": "d", "label": "Daily",   "step": 100.0, "color": "#22d3ee"},
    {"key": "w", "label": "Weekly",  "step": 300.0, "color": "#818cf8"},
    {"key": "m", "label": "Monthly", "step": 200.0, "color": "#f59e0b"},
]
GKEYS    = tuple(g["key"] for g in GRIDS)
GLABEL   = {g["key"]: g["label"] for g in GRIDS}
GCOLOR   = {g["key"]: g["color"] for g in GRIDS}
GSTEP    = {g["key"]: g["step"]  for g in GRIDS}


# ─────────────────────────── daily candles per symbol ───────────────────────
def _per_symbol(df: pd.DataFrame) -> pd.DataFrame | None:
    """One row per trading DATE. The 1d feed is already one row per session, so
    the groupby is a harmless no-op that also collapses any stray duplicate
    dates (and still works if ever pointed back at an intraday feed)."""
    if len(df) < 8 or "datetime" not in df.columns:
        return None
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    o = pd.to_numeric(df["open"],  errors="coerce")
    h = pd.to_numeric(df["high"],  errors="coerce")
    l = pd.to_numeric(df["low"],   errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    date = dt.dt.strftime("%Y-%m-%d")
    base = pd.DataFrame({"date": date.values, "o": o.values, "h": h.values,
                         "l": l.values, "c": c.values})
    day = base.dropna(subset=["date"]).groupby("date", sort=True).agg(
        open=("o", "first"), high=("h", "max"), low=("l", "min"),
        close=("c", "last")).reset_index()
    return day


# ─────────────────────────── round-level maths (vectorised) ─────────────────
def _grid_eval(close: pd.Series, low: pd.Series, high: pd.Series, step: float) -> dict:
    """For a step-spaced grid of round levels, per row:
        nearest   — grid line nearest the close
        touched   — a grid line lies inside [low, high]
        level_in  — the in-candle grid line nearest close (else `nearest`)
        dist_in   — |close - level_in| as % of close
        dist_near — |close - nearest|  as % of close
    All returned as pandas Series aligned to `close.index`."""
    eps = 1e-9
    nearest = (close / step).round() * step
    lo_idx = np.ceil(low / step - eps)              # first grid index >= low
    hi_idx = np.floor(high / step + eps)            # last grid index  <= high
    touched = hi_idx >= lo_idx                       # at least one line inside the candle
    k = (close / step).round().clip(lower=lo_idx, upper=hi_idx)   # in-candle line nearest close
    level_in = pd.Series(np.where(touched, k * step, nearest), index=close.index)
    dist_in   = (close - level_in).abs() / close * 100
    dist_near = (close - nearest).abs()  / close * 100
    return {"nearest": nearest, "touched": touched.fillna(False),
            "level_in": level_in, "dist_in": dist_in, "dist_near": dist_near}


# ─────────────────────────── snapshot cache (mtime-keyed) ───────────────────
_CACHE: dict[str, tuple[float, pd.DataFrame, list[str]]] = {}
_BUILD_LOCK = threading.Lock()


def _snapshot() -> tuple[pd.DataFrame, list[str]]:
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        return pd.DataFrame(), []
    sig = max(os.path.getmtime(f) for f in files)
    cached = _CACHE.get("snap")
    if cached and cached[0] == sig:
        return cached[1], cached[2]

    with _BUILD_LOCK:                      # one builder; concurrent callers reuse the result
        cached = _CACHE.get("snap")
        if cached and cached[0] == sig:
            return cached[1], cached[2]
        frames = []
        for f in files:
            sym = os.path.basename(f).replace("_historical.csv", "").replace(".csv", "")
            try:
                per = _per_symbol(pd.read_csv(f).tail(SNAP_TAIL))
            except Exception:
                per = None
            if per is not None and len(per):
                per = per.copy()
                per.insert(0, "symbol", sym)
                frames.append(per)

        big = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        days = sorted(big["date"].dropna().unique().tolist(), reverse=True) if len(big) else []
        _CACHE["snap"] = (sig, big, days)
        return big, days


# ─────────────────────────────────── API ───────────────────────────────────
app = FastAPI(title="Level Screener")


class ScreenReq(BaseModel):
    date: str | None = None      # YYYY-MM-DD; default = latest available day
    periods: list[str] = []      # subset of {"d","w","m"}; empty = all
    steps: dict[str, float] = {} # per-grid step override; missing = default
    logic: str = "ANY"           # ANY | ALL across selected grids
    mode: str = "inside"         # inside (line within candle) | near (within tol %)
    tol: float = 0.5             # tolerance % for "near" mode
    sort_dir: str = "asc"        # closest touch first


def _steps_from(req_steps: dict[str, float]) -> dict[str, float]:
    out = dict(GSTEP)
    for k, v in (req_steps or {}).items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if k in GKEYS and v > 0:
            out[k] = v
    return out


@app.get("/api/meta")
def meta():
    return {"grids": GRIDS}


@app.get("/api/days")
def days():
    _, dlist = _snapshot()
    return {"days": dlist[:600], "latest": dlist[0] if dlist else None}


@app.post("/api/screen")
def screen(req: ScreenReq):
    big, dlist = _snapshot()
    if big.empty:
        return JSONResponse({"count": 0, "universe": 0, "date": "", "periods": [], "rows": []})

    day = req.date if (req.date in dlist) else (dlist[0] if dlist else None)
    sub = big[big["date"] == day]
    universe = int(sub["symbol"].nunique())
    if sub.empty:
        return JSONResponse({"count": 0, "universe": universe, "date": day or "",
                             "periods": [], "rows": []})

    sel = [p for p in req.periods if p in GKEYS] or list(GKEYS)
    steps = _steps_from(req.steps)
    sel_cfg = [{**g, "step": steps[g["key"]]} for g in GRIDS if g["key"] in sel]
    mode = req.mode if req.mode in ("inside", "near") else "inside"
    tol = max(0.0, float(req.tol))

    lo, hi, cl = sub["low"], sub["high"], sub["close"]
    period_hits, all_dists, per_flags = [], [], {}
    for p in sel:
        ev = _grid_eval(cl, lo, hi, steps[p])
        if mode == "inside":
            hit, V, dv = ev["touched"], ev["level_in"], ev["dist_in"]
        else:
            hit, V, dv = (ev["dist_near"] <= tol), ev["nearest"], ev["dist_near"]
        period_hits.append(hit.fillna(False))
        per_flags[p] = (ev["touched"].fillna(False), V, dv)
        all_dists.append(dv)

    H = pd.concat(period_hits, axis=1)
    passed = H.all(axis=1) if req.logic == "ALL" else H.any(axis=1)
    closest = pd.concat(all_dists, axis=1).min(axis=1)

    res_idx = sub.index[passed.values]
    res = sub.loc[res_idx].copy()
    res["_closest"] = closest.loc[res_idx]
    res = res.sort_values("_closest", ascending=(req.sort_dir == "asc"))

    def _num(x):
        return None if (x is None or pd.isna(x)) else round(float(x), 2)

    rows = []
    for i in res.index:
        row = {"symbol": sub.at[i, "symbol"], "close": _num(sub.at[i, "close"])}
        for p in sel:
            touched, V_, dv = per_flags[p]
            row[p] = _num(V_.at[i])
            row[f"{p}_in"] = bool(touched.at[i])
            row[f"{p}_d"] = _num(dv.at[i])
        rows.append(row)

    return {"count": len(rows), "universe": universe, "date": day,
            "periods": sel_cfg, "mode": mode, "rows": rows}


def _grid_lines(step: float, wlo: float, whi: float,
                dlo: float, dhi: float) -> list[float]:
    """Multiples of `step` to draw in the chart window [wlo, whi]. If that spans
    too many lines, narrow to the chosen day's neighbourhood so the chart stays
    legible."""
    if step <= 0 or not np.isfinite(step):
        return []
    lo, hi = wlo, whi
    n = int(np.floor(hi / step)) - int(np.ceil(lo / step)) + 1
    if n > CHART_MAX_LINES:
        pad = 4 * step
        lo, hi = dlo - pad, dhi + pad
    first = int(np.ceil(lo / step))
    last = int(np.floor(hi / step))
    return [k * step for k in range(first, last + 1)][:CHART_MAX_LINES]


@app.get("/api/chart")
def chart(symbol: str, date: str | None = None, periods: str = "d,w,m",
          steps: str | None = None):
    """Daily candles for one symbol + the round-level grid lines to draw."""
    f = DATA_DIR / f"{symbol}_historical.csv"
    if not f.exists():
        return JSONResponse({"error": "unknown symbol"}, status_code=404)
    try:
        per = _per_symbol(pd.read_csv(f).tail(CHART_TAIL))
    except Exception:
        per = None
    if per is None or not len(per):
        return JSONResponse({"error": "no data"}, status_code=404)

    dates = per["date"].tolist()
    idx = dates.index(date) if (date in dates) else (len(dates) - 1)
    lo_i, hi_i = max(0, idx - 90), min(len(per), idx + 12)
    win = per.iloc[lo_i:hi_i]

    candles, wlo, whi = [], np.inf, -np.inf
    for _, r in win.iterrows():
        h, l, c = r["high"], r["low"], r["close"]
        if pd.isna(h) or pd.isna(l) or pd.isna(c):
            continue
        o = c if pd.isna(r["open"]) else r["open"]
        candles.append({"time": str(r["date"])[:10], "open": round(float(o), 2),
                        "high": round(float(h), 2), "low": round(float(l), 2),
                        "close": round(float(c), 2)})
        wlo, whi = min(wlo, float(l)), max(whi, float(h))

    # per-grid step overrides come in as "d:100,w:300,m:200"
    step_map = dict(GSTEP)
    for tok in (steps or "").split(","):
        if ":" in tok:
            k, _, v = tok.partition(":")
            try:
                fv = float(v)
                if k in GKEYS and fv > 0:
                    step_map[k] = fv
            except ValueError:
                pass

    row = per.iloc[idx]
    dlo, dhi = float(row["low"]), float(row["high"])
    sel = [p for p in periods.split(",") if p in GKEYS] or list(GKEYS)
    levels = []
    for p in sel:
        for v in _grid_lines(step_map[p], wlo, whi, dlo, dhi):
            levels.append({"key": p, "label": GLABEL[p], "value": round(float(v), 2),
                           "touched": bool(dlo <= v <= dhi), "color": GCOLOR[p],
                           "step": step_map[p]})
    return {"symbol": symbol, "date": dates[idx], "candles": candles, "levels": levels}


@app.on_event("startup")
def _warm_cache():
    threading.Thread(target=_snapshot, daemon=True).start()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML.replace("__APP_BASE__", APP_BASE)


# ──────────────────────────────────── UI ───────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Level Screener</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{
    --bg:#0a0e17; --panel:#111726; --panel2:#0d1320; --line:#1e2940;
    --txt:#e6edf6; --mut:#8a98b2; --accent:#818cf8; --accent2:#22d3ee;
    --pos:#34d399; --neg:#f87171; --chip:#172033;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#1b2440 0%,var(--bg) 55%);
       color:var(--txt);font:14px/1.5 'Inter',system-ui,Segoe UI,Roboto,sans-serif;min-height:100vh}
  .wrap{max-width:1240px;margin:0 auto;padding:28px 22px 60px}
  header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px}
  .title{font-size:26px;font-weight:800;letter-spacing:.2px;display:flex;align-items:center;gap:12px}
  .title .dot{width:11px;height:11px;border-radius:50%;background:var(--accent);box-shadow:0 0 16px 2px var(--accent)}
  .sub{color:var(--mut);font-size:13px;margin-top:4px;max-width:600px}
  .asof{color:var(--mut);font-size:12.5px;text-align:right}
  .asof b{color:var(--txt)}
  .grid{display:grid;grid-template-columns:360px 1fr;gap:20px}
  @media(max-width:920px){.grid{grid-template-columns:1fr}}
  .card{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
        border:1px solid var(--line);border-radius:16px;padding:18px}
  .card h3{margin:18px 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--mut)}
  .card h3:first-child{margin-top:0}
  .seg{display:inline-flex;background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:3px;flex-wrap:wrap}
  .seg button{background:none;border:0;color:var(--mut);padding:6px 13px;border-radius:8px;cursor:pointer;font-weight:600;font-size:12.5px}
  .seg button.on{background:var(--accent);color:#0a0a18}
  .dayrow{display:flex;gap:8px;align-items:center}
  input[type=date]{flex:1;background:var(--panel2);border:1px solid var(--line);color:var(--txt);
        border-radius:10px;padding:9px 10px;font-size:13.5px;outline:none}
  input[type=date]:focus{border-color:var(--accent)}
  .mini{background:var(--chip);border:1px solid var(--line);color:var(--mut);border-radius:9px;padding:8px 11px;cursor:pointer;font-size:12px}
  .mini:hover{border-color:var(--accent);color:var(--accent)}
  .lines{display:flex;flex-direction:column;gap:8px}
  .ck{display:flex;align-items:center;gap:10px;background:var(--panel2);border:1px solid var(--line);
      border-radius:10px;padding:10px 12px;cursor:pointer;user-select:none}
  .ck.on{border-color:var(--accent);background:#161d36}
  .ck .box{width:16px;height:16px;border-radius:5px;border:1.5px solid var(--mut);flex:none;position:relative}
  .ck.on .box{background:var(--accent);border-color:var(--accent)}
  .ck.on .box::after{content:"";position:absolute;left:5px;top:1px;width:4px;height:9px;border:solid #0a0a18;
      border-width:0 2px 2px 0;transform:rotate(45deg)}
  .ck .lbl{font-weight:600;flex:1}
  .ck .swatch{width:10px;height:10px;border-radius:50%}
  .stepwrap{display:flex;align-items:center;gap:5px;color:var(--mut);font-size:11.5px}
  .stepwrap input{width:62px}
  input[type=number]{background:var(--panel);border:1px solid var(--line);color:var(--txt);
        border-radius:8px;padding:6px 8px;font-size:12.5px;outline:none}
  input[type=number]:focus{border-color:var(--accent)}
  .tolrow{display:flex;align-items:center;gap:8px;margin-top:10px;color:var(--mut);font-size:12.5px}
  .tolrow.off{opacity:.4;pointer-events:none}
  .tolrow input[type=number]{width:74px}
  .run{width:100%;margin-top:18px;background:linear-gradient(90deg,var(--accent),var(--accent2));
        color:#0a0a18;font-weight:800;border:0;border-radius:11px;padding:12px;cursor:pointer;font-size:14px}
  .run:active{transform:translateY(1px)}
  .resbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;flex-wrap:wrap;gap:10px}
  .count{font-size:15px}
  .count b{font-size:22px;color:var(--accent);font-weight:800}
  .dl{background:var(--chip);border:1px solid var(--line);color:var(--txt);border-radius:9px;padding:7px 13px;cursor:pointer;font-size:12.5px}
  .dl:hover{border-color:var(--accent2);color:var(--accent2)}
  .tablewrap{overflow:auto;border:1px solid var(--line);border-radius:12px;max-height:72vh}
  table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
  th,td{padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
  th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--panel)}
  th{position:sticky;top:0;background:var(--panel2);color:var(--mut);font-size:11.5px;text-transform:uppercase;
     letter-spacing:.07em;z-index:2}
  th:first-child{z-index:3}
  tbody tr{cursor:pointer}
  tbody tr:hover{background:#0f1830}
  td.sym{font-weight:700}
  td.sym::after{content:"📈";font-size:11px;margin-left:7px;opacity:.35}
  td.touch{background:#13243a;color:var(--accent);font-weight:700;box-shadow:inset 2px 0 0 var(--accent)}
  /* chart modal */
  .modal{position:fixed;inset:0;background:rgba(4,7,14,.74);backdrop-filter:blur(2px);
         display:none;align-items:center;justify-content:center;z-index:50}
  .modal.show{display:flex}
  .sheet{width:min(1080px,94vw);height:min(78vh,720px);background:var(--panel);
         border:1px solid var(--line);border-radius:16px;display:flex;flex-direction:column;overflow:hidden}
  .sheethd{display:flex;align-items:center;gap:14px;padding:12px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
  .sheethd .nm{font-size:17px;font-weight:800}
  .sheethd .dd{color:var(--mut);font-size:12.5px}
  .legend{display:flex;gap:12px;flex-wrap:wrap;margin-left:auto}
  .lg{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--mut)}
  .lg .ln{width:16px;height:0;border-top:2px solid #888}
  .xbtn{background:var(--chip);border:1px solid var(--line);color:var(--txt);border-radius:9px;
        width:30px;height:30px;cursor:pointer;font-size:16px;line-height:1}
  .xbtn:hover{border-color:var(--neg);color:var(--neg)}
  #chartBox{flex:1;min-height:0}
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
      <div class="title"><span class="dot"></span>Level Screener</div>
      <div class="sub">Stocks whose daily candle touched a round-number level — Daily / Weekly / Monthly grids at your chosen spacing.</div>
    </div>
    <div class="asof" id="asof"></div>
  </header>

  <div class="grid">
    <!-- LEFT -->
    <div class="card">
      <h3>Day</h3>
      <div class="dayrow">
        <button class="mini" id="prevday" title="Previous trading day">◀</button>
        <input type="date" id="day"/>
        <button class="mini" id="cal" title="Open calendar">📅</button>
        <button class="mini" id="nextday" title="Next trading day">▶</button>
        <button class="mini" id="latest">Latest</button>
      </div>

      <h3>Level grids to test</h3>
      <div class="lines" id="lines"></div>

      <h3>Match</h3>
      <div class="seg" id="logic">
        <button data-logic="ANY" class="on">ANY grid</button>
        <button data-logic="ALL">ALL grids</button>
      </div>

      <h3>Condition</h3>
      <div class="seg" id="mode">
        <button data-mode="inside" class="on">Touched (inside candle)</button>
        <button data-mode="near">Within %</button>
      </div>
      <div class="tolrow off" id="tolrow">
        <span>Tolerance</span><input type="number" id="tol" value="0.5" step="0.1" min="0"/><span>% from level</span>
      </div>

      <button class="run" id="run">Run Screen</button>
    </div>

    <!-- RIGHT -->
    <div class="card">
      <div class="resbar">
        <div class="count" id="count">Ready — pick a day and run.</div>
        <button class="dl" id="dl" style="display:none">⬇ Download CSV</button>
      </div>
      <div class="tablewrap"><div id="res"><div class="empty">No results yet.</div></div></div>
    </div>
  </div>
</div>

<div class="modal" id="modal">
  <div class="sheet">
    <div class="sheethd">
      <span class="nm" id="m-sym">—</span>
      <span class="dd" id="m-dd"></span>
      <span class="legend" id="m-legend"></span>
      <button class="xbtn" id="m-close" title="Close">✕</button>
    </div>
    <div id="chartBox"></div>
  </div>
</div>

<script>
const APP_BASE="__APP_BASE__";
const api=(p,o)=>fetch((APP_BASE||"")+p,o).then(r=>r.json());
let META=null, LAST=null, DAYS=[];
const STATE={logic:"ANY",mode:"inside",periods:{d:true,w:true,m:true},steps:{}};

function renderLines(){
  const box=document.getElementById("lines"); box.innerHTML="";
  META.grids.forEach(g=>{
    if(STATE.steps[g.key]==null) STATE.steps[g.key]=g.step;
    const d=document.createElement("div");
    d.className="ck"+(STATE.periods[g.key]?" on":"");
    d.innerHTML=`<span class="box"></span>`
      +`<span class="swatch" style="background:${g.color}"></span>`
      +`<span class="lbl">${g.label}</span>`
      +`<span class="stepwrap">every <input type="number" min="0" step="1" `
      +`value="${STATE.steps[g.key]}" data-step="${g.key}"/> pts</span>`;
    d.onclick=(e)=>{
      if(e.target.tagName==="INPUT") return;          // don't toggle when editing the step
      STATE.periods[g.key]=!STATE.periods[g.key];
      d.classList.toggle("on",STATE.periods[g.key]);
    };
    box.appendChild(d);
  });
  box.querySelectorAll("input[data-step]").forEach(inp=>{
    inp.onclick=e=>e.stopPropagation();
    inp.onchange=()=>{const v=parseFloat(inp.value); if(v>0) STATE.steps[inp.dataset.step]=v;};
  });
}
const selectedPeriods=()=>META.grids.map(g=>g.key).filter(k=>STATE.periods[k]);
const stepStr=()=>META.grids.map(g=>g.key+":"+(STATE.steps[g.key]||g.step)).join(",");

async function loadDays(){
  const r=await api("/api/days"); DAYS=r.days||[];
  const inp=document.getElementById("day");
  if(DAYS.length){inp.max=DAYS[0]; inp.min=DAYS[DAYS.length-1];}
  if(!DAYS.includes(inp.value)) inp.value=r.latest||"";
}
function snapDay(v){
  if(!DAYS.length||!v) return v;
  if(DAYS.includes(v)) return v;
  for(const d of DAYS){ if(d<=v) return d; }
  return DAYS[DAYS.length-1];
}
function stepDay(dir){           // dir +1 = older, -1 = newer
  const el=document.getElementById("day");
  let i=DAYS.indexOf(snapDay(el.value)); if(i<0)i=0;
  const j=Math.min(DAYS.length-1,Math.max(0,i+dir));
  el.value=DAYS[j]; run();
}

async function run(){
  const sel=selectedPeriods();
  if(!sel.length){document.getElementById("count").textContent="Select at least one grid.";return;}
  document.getElementById("count").innerHTML='<span class="spin"></span>Scanning universe…';
  const body={date:document.getElementById("day").value,periods:sel,steps:STATE.steps,
              logic:STATE.logic,mode:STATE.mode,
              tol:parseFloat(document.getElementById("tol").value)||0.5,sort_dir:"asc"};
  const r=await api("/api/screen",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  LAST=r; render(r);
}

function render(r){
  document.getElementById("asof").innerHTML=`Universe <b>${r.universe}</b> · day <b>${r.date||"—"}</b>`;
  const verb=r.mode==="near"?"near a level":"touched";
  document.getElementById("count").innerHTML=`<b>${r.count}</b> stocks ${verb} on ${r.date||"—"}`;
  document.getElementById("dl").style.display=r.count?"block":"none";
  if(!r.count){document.getElementById("res").innerHTML='<div class="empty">No stocks matched on this day. Try ANY instead of ALL, “Within %”, a wider spacing, or another date.</div>';return;}
  const P=r.periods;
  let head="<tr><th>Symbol</th><th>Close</th>";
  P.forEach(p=>head+=`<th>${p.label} ·${p.step}</th><th>Δ%</th>`);
  head+="</tr>";
  const body=r.rows.map(o=>{
    let tds=`<td class="sym">${o.symbol}</td><td>${o.close}</td>`;
    P.forEach(p=>{
      const vv=o[p.key], d=o[p.key+"_d"], hit=o[p.key+"_in"];
      tds+=`<td class="${hit?'touch':''}">${vv==null?'—':vv}</td>`;
      tds+=`<td>${d==null?'—':d}</td>`;
    });
    return `<tr data-sym="${o.symbol}">${tds}</tr>`;
  }).join("");
  document.getElementById("res").innerHTML=`<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function downloadCSV(){
  if(!LAST||!LAST.rows.length)return;
  const cols=["symbol","close"]; LAST.periods.forEach(p=>{cols.push(p.key,p.key+"_d");});
  const lines=[cols.join(",")].concat(LAST.rows.map(o=>cols.map(c=>o[c]).join(",")));
  const blob=new Blob([lines.join("\n")],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download=`round_levels_${LAST.date||""}.csv`;a.click();
}

function bindSeg(id,key,after){
  document.querySelectorAll(`#${id} button`).forEach(b=>b.onclick=async()=>{
    document.querySelectorAll(`#${id} button`).forEach(x=>x.classList.remove("on"));
    b.classList.add("on"); STATE[key]=b.dataset[key]; if(after)await after();
  });
}

// ── chart modal: click a stock → see its candles cut the grid lines ──
let CHART=null, CSER=null, CLINES=[], CRANGE=null;
function ensureChart(){
  if(CHART) return;
  CHART=LightweightCharts.createChart(document.getElementById("chartBox"),{
    autoSize:true,
    layout:{background:{color:"#10131a"},textColor:"#cfd6e4"},
    grid:{vertLines:{color:"#161b27"},horzLines:{color:"#161b27"}},
    timeScale:{borderColor:"#222838",rightOffset:8,barSpacing:8,minBarSpacing:2},
    rightPriceScale:{borderColor:"#222838",scaleMargins:{top:0.12,bottom:0.12}},
    crosshair:{mode:0}});
  CSER=CHART.addCandlestickSeries({upColor:"#26a69a",downColor:"#ef5350",
    borderUpColor:"#26a69a",borderDownColor:"#ef5350",wickUpColor:"#26a69a",wickDownColor:"#ef5350",
    autoscaleInfoProvider:()=>CRANGE?{priceRange:{minValue:CRANGE.min,maxValue:CRANGE.max}}:null});
}
function fitChart(){if(CHART){CHART.timeScale().fitContent();}}
function openChart(sym){
  document.getElementById("modal").classList.add("show");
  document.getElementById("m-sym").textContent=sym;
  document.getElementById("m-dd").textContent="loading…";
  document.getElementById("m-legend").innerHTML="";
  requestAnimationFrame(()=>requestAnimationFrame(()=>loadChart(sym)));
}
async function loadChart(sym){
  ensureChart();
  const sel=selectedPeriods();
  const u="/api/chart?symbol="+encodeURIComponent(sym)+"&date="+document.getElementById("day").value
          +"&periods="+sel.join(",")+"&steps="+encodeURIComponent(stepStr());
  let r; try{r=await api(u);}catch(e){document.getElementById("m-dd").textContent="load error";return;}
  if(r.error){document.getElementById("m-dd").textContent=r.error;return;}
  CLINES.forEach(l=>{try{CSER.removePriceLine(l)}catch(e){}}); CLINES=[];
  let lo=Infinity,hi=-Infinity;
  (r.candles||[]).forEach(c=>{lo=Math.min(lo,c.low);hi=Math.max(hi,c.high);});
  const pad=(hi>lo)?(hi-lo)*0.04:Math.max(hi*0.01,0.5);
  CRANGE={min:lo-pad,max:hi+pad};
  CSER.setData(r.candles||[]);
  (r.levels||[]).forEach(L=>{
    if(L.value<CRANGE.min||L.value>CRANGE.max) return;   // only lines in view
    CLINES.push(CSER.createPriceLine({price:L.value,color:L.color,
      lineWidth:L.touched?2:1,lineStyle:L.touched?0:2,axisLabelVisible:true,
      title:L.label+(L.touched?" ✓":"")}));
  });
  CSER.setMarkers([{time:r.date,position:"aboveBar",color:"#818cf8",shape:"arrowDown",text:"day"}]);
  CHART.priceScale("right").applyOptions({autoScale:true});
  const touched=(r.levels||[]).filter(L=>L.touched).length;
  document.getElementById("m-dd").textContent=r.date+" · "+(r.candles||[]).length+" daily bars · "+touched+" line(s) touched";
  // one legend entry per selected grid
  const seen={};
  document.getElementById("m-legend").innerHTML=(r.levels||[]).filter(L=>{if(seen[L.key])return false;seen[L.key]=1;return true;})
    .map(L=>`<span class="lg"><span class="ln" style="border-color:${L.color}"></span>${L.label} ·${L.step}</span>`).join("");
  requestAnimationFrame(fitChart);
}
function closeChart(){document.getElementById("modal").classList.remove("show");}

(async()=>{
  META=await api("/api/meta");
  bindSeg("logic","logic");
  bindSeg("mode","mode",()=>document.getElementById("tolrow").classList.toggle("off",STATE.mode!=="near"));
  document.getElementById("run").onclick=run;
  document.getElementById("dl").onclick=downloadCSV;
  const dayEl=document.getElementById("day");
  dayEl.onchange=()=>{dayEl.value=snapDay(dayEl.value); run();};
  dayEl.onclick=()=>{try{dayEl.showPicker();}catch(e){}};
  document.getElementById("cal").onclick=()=>{try{dayEl.showPicker();}catch(e){dayEl.focus();}};
  document.getElementById("prevday").onclick=()=>stepDay(1);   // older
  document.getElementById("nextday").onclick=()=>stepDay(-1);  // newer
  document.getElementById("latest").onclick=()=>{if(DAYS.length){dayEl.value=DAYS[0];run();}};
  document.getElementById("res").addEventListener("click",e=>{
    const tr=e.target.closest("tr[data-sym]"); if(tr)openChart(tr.dataset.sym);});
  document.getElementById("m-close").onclick=closeChart;
  document.getElementById("modal").onclick=e=>{if(e.target.id==="modal")closeChart();};
  document.addEventListener("keydown",e=>{if(e.key==="Escape")closeChart();});
  renderLines();
  await loadDays();
  run();
})();
</script>
</body>
</html>"""
