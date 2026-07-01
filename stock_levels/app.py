"""Level Screener — Volume-Profile POC levels (a faithful server-side port of the
TradingView "OB+VP(UTFS)" indicator) across the whole equity universe.

For each period we build a fixed-row-size volume profile from 1-MINUTE bars and
take the POC = the busiest price row, then draw that row's TWO edges as
horizontal lines (POC ± row/2) — exactly what the Pine script does:

    center = floor(hlc3 / row_size) * row_size + row_size/2   # bucket the price
    POC    = centre of the bucket with the most volume
    lines  = POC + row_size/2   and   POC - row_size/2         # the row's edges

Three periods, with the user's exact "OB+VP" inputs:
  • Daily   — row 100, lookback 30 days,   teal
  • Weekly  — row 200, lookback 20 weeks,  green
  • Monthly — row 300, lookback 12 months, red

A period's POC is finalised at its rollover (Pine draws it at the next
boundary), so for any chosen day D the *active* lines are the deduped POCs of
every COMPLETED period strictly before D's own period, within the lookback
window. For any trading day you pick, this lists the stocks whose DAILY candle
touched one of those POC edge lines (line inside the day's high–low range).

Data: ema_scanner/data/1m/*.csv (1-minute OHLCV, gathered by fetch_1m_history.py).
The heavy per-symbol profile is precomputed once into a small parquet cache
(ema_scanner/data/_levels_poc/), keyed by the 1m files' mtime + the row/lookback
config, so restarts are instant and screen-time is a cheap vectorised filter.

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
# POC needs intraday volume-by-price, so we read the 1-minute feed (gathered by
# ema_scanner/fetch_1m_history.py). This is a static deep snapshot; a post-close
# 1m refresh timer can be added later to keep it current.
DATA_DIR  = ROOT / "ema_scanner" / "data" / "1m"
CACHE_DIR = ROOT / "ema_scanner" / "data" / "_levels_poc"   # precomputed profile cache
CHART_TAIL_DAYS = 120       # daily bars around the chosen day in a single-stock chart
CHART_MAX_LINES = 40        # cap POC lines drawn per chart so it stays legible
SESSION_START_MIN = 9 * 60 + 15    # 09:15 — drop pre-open auction prints
SESSION_END_MIN   = 15 * 60 + 30   # 15:30

# OB+VP periods (key, label, row_size, lookback in days, colour) — the user's
# TradingView "OB+VP(UTFS)" inputs. row_size feeds the profile so it is a build-
# time constant (changing it invalidates the cache); lookback is applied cheaply
# at screen time. Weekly 20w ≈ 140d, Monthly 12m ≈ 365d.
PERIODS = [
    {"key": "d", "label": "Daily",   "row": 100.0, "lookback": 30,  "color": "#22d3ee"},
    {"key": "w", "label": "Weekly",  "row": 200.0, "lookback": 140, "color": "#22c55e"},
    {"key": "m", "label": "Monthly", "row": 300.0, "lookback": 365, "color": "#ef4444"},
]
PKEYS    = tuple(p["key"] for p in PERIODS)
PLABEL   = {p["key"]: p["label"]    for p in PERIODS}
PROW     = {p["key"]: p["row"]      for p in PERIODS}
PLOOK    = {p["key"]: p["lookback"] for p in PERIODS}
PCOLOR   = {p["key"]: p["color"]    for p in PERIODS}


# ─────────────────────────── period-key helpers ─────────────────────────────
def _period_key_series(dt: pd.Series, period: str) -> pd.Series:
    """Zero-padded so lexical order == chronological order."""
    if period == "d":
        return dt.dt.strftime("%Y-%m-%d")
    if period == "w":
        iso = dt.dt.isocalendar()
        return iso["year"].astype(str) + "-W" + iso["week"].astype(int).map("{:02d}".format)
    if period == "m":
        return dt.dt.strftime("%Y-%m")
    raise ValueError(period)


def _period_key_scalar(day: str, period: str) -> str:
    ts = pd.Timestamp(day)
    if period == "d":
        return ts.strftime("%Y-%m-%d")
    if period == "w":
        iso = ts.isocalendar()
        return f"{iso[0]}-W{int(iso[1]):02d}"
    if period == "m":
        return ts.strftime("%Y-%m")
    raise ValueError(period)


# ─────────────────────────── fixed-row-size POC (Pine bucketing) ────────────
def _poc_fixed(keys, price, vol, row: float) -> pd.Series:
    """POC per period key using the Pine's fixed-row buckets:
        center = floor(price/row)*row + row/2 ,  POC = busiest bucket centre.
    Returns Series indexed by period key (sorted)."""
    prof = pd.DataFrame({"k": keys, "p": price, "v": vol}).dropna(subset=["k", "p", "v"])
    prof = prof[prof["v"] > 0]
    if prof.empty or row <= 0:
        return pd.Series(dtype=float)
    prof = prof.assign(c=np.floor(prof["p"] / row) * row + row / 2.0)
    agg = prof.groupby(["k", "c"], as_index=False)["v"].sum()
    poc = agg.sort_values("v").groupby("k", as_index=False).tail(1)   # busiest bucket per key
    return pd.Series(poc["c"].values, index=poc["k"].values).sort_index()


def _per_symbol(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """From one symbol's 1-minute bars, return:
        day  — one row per DATE: the regular-session candle (open/high/low/close)
        poc  — long form (period, key, poc, last_date) for every period's POC."""
    if len(df) < 30 or "datetime" not in df.columns:
        return None
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    o = pd.to_numeric(df["open"],   errors="coerce")
    h = pd.to_numeric(df["high"],   errors="coerce")
    l = pd.to_numeric(df["low"],    errors="coerce")
    c = pd.to_numeric(df["close"],  errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")

    mins = dt.dt.hour * 60 + dt.dt.minute
    keep = dt.notna() & (mins >= SESSION_START_MIN) & (mins <= SESSION_END_MIN)
    dt, o, h, l, c, v = dt[keep], o[keep], h[keep], l[keep], c[keep], v[keep]
    if len(dt) < 30:
        return None
    hlc3 = (h + l + c) / 3.0
    date = dt.dt.strftime("%Y-%m-%d")

    day = pd.DataFrame({"date": date.values, "o": o.values, "h": h.values,
                        "l": l.values, "c": c.values}) \
        .dropna(subset=["date"]) \
        .groupby("date", sort=True).agg(
            open=("o", "first"), high=("h", "max"),
            low=("l", "min"), close=("c", "last")).reset_index()

    poc_frames = []
    for p in PKEYS:
        keys = _period_key_series(dt, p)
        poc = _poc_fixed(keys.values, hlc3.values, v.values, PROW[p])
        if poc.empty:
            continue
        last = pd.Series(date.values, index=keys.values)
        last = last.groupby(level=0).max()                # period key -> its last trading date
        pf = pd.DataFrame({"key": poc.index, "poc": poc.values})
        pf["last_date"] = pf["key"].map(last)
        pf["period"] = p
        poc_frames.append(pf)

    poc_long = (pd.concat(poc_frames, ignore_index=True)
                if poc_frames else
                pd.DataFrame(columns=["key", "poc", "last_date", "period"]))
    return day, poc_long


# ─────────────────────────── snapshot cache (mtime-keyed, disk-backed) ───────
_CACHE: dict[str, tuple[str, pd.DataFrame, pd.DataFrame, list[str]]] = {}
_BUILD_LOCK = threading.Lock()


def _sig(files: list[str]) -> str:
    # Keyed on file-count + row-size config, NOT max-mtime: the 1m feed is a
    # static snapshot, and max-mtime is fragile — a single stray file touch
    # would invalidate the whole cache and force a blocking multi-minute
    # rebuild. A future post-close 1m refresh appends rows without changing the
    # file count, so that job must invalidate the cache explicitly by removing
    # ema_scanner/data/_levels_poc/sig.txt (cheap, and it forces one rebuild).
    cfg = "|".join(f"{p['key']}:{p['row']}" for p in PERIODS)   # only row_size affects the profile
    return f"n{len(files)}|{cfg}"


def _load_disk(sig: str):
    try:
        if (CACHE_DIR / "sig.txt").read_text().strip() != sig:
            return None
        big = pd.read_parquet(CACHE_DIR / "candles.parquet")
        poc = pd.read_parquet(CACHE_DIR / "poc.parquet")
        return big, poc
    except Exception:
        return None


def _days_from(big: pd.DataFrame) -> list[str]:
    """Days available for the SCREENER — only those the bulk of the universe
    has. A trailing day held by a handful of symbols (e.g. one stock re-fetched
    ahead of the rest) would otherwise become the default 'latest' with a
    universe of 1. Require a day to cover ≥30% of symbols (floor 5)."""
    if not len(big):
        return []
    cnt = big.groupby("date")["symbol"].nunique()
    thresh = max(5, int(0.30 * big["symbol"].nunique()))
    return sorted(cnt[cnt >= thresh].index.tolist(), reverse=True)


def _save_disk(sig: str, big: pd.DataFrame, poc: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        big.to_parquet(CACHE_DIR / "candles.parquet", index=False)
        poc.to_parquet(CACHE_DIR / "poc.parquet", index=False)
        (CACHE_DIR / "sig.txt").write_text(sig)
    except Exception:
        pass   # cache is an optimisation; never fail the request on a write error


def _snapshot() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        return pd.DataFrame(), pd.DataFrame(), []
    sig = _sig(files)
    cached = _CACHE.get("snap")
    if cached and cached[0] == sig:
        return cached[1], cached[2], cached[3]

    with _BUILD_LOCK:
        cached = _CACHE.get("snap")
        if cached and cached[0] == sig:
            return cached[1], cached[2], cached[3]

        disk = _load_disk(sig)
        if disk is not None:
            big, poc = disk
        else:
            cand_frames, poc_frames = [], []
            for f in files:
                sym = os.path.basename(f).replace("_historical.csv", "").replace(".csv", "")
                try:
                    res = _per_symbol(pd.read_csv(
                        f, usecols=["datetime", "open", "high", "low", "close", "volume"]))
                except Exception:
                    res = None
                if not res:
                    continue
                day, poc_long = res
                if len(day):
                    day = day.copy(); day.insert(0, "symbol", sym)
                    cand_frames.append(day)
                if len(poc_long):
                    poc_long = poc_long.copy(); poc_long.insert(0, "symbol", sym)
                    poc_frames.append(poc_long)
            big = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
            poc = pd.concat(poc_frames, ignore_index=True) if poc_frames else pd.DataFrame()
            _save_disk(sig, big, poc)

        days = _days_from(big)
        _CACHE["snap"] = (sig, big, poc, days)
        return big, poc, days


# ─────────────────────────── active-lines evaluation ────────────────────────
def _active_pocs(poc: pd.DataFrame, p: str, day: str) -> pd.DataFrame:
    """Deduped POCs (columns symbol, poc) active on `day` for period `p`:
    completed periods strictly before day's own period, within the lookback."""
    if poc.empty:
        return pd.DataFrame(columns=["symbol", "poc"])
    pk = _period_key_scalar(day, p)
    wstart = (pd.Timestamp(day) - pd.Timedelta(days=PLOOK[p])).strftime("%Y-%m-%d")
    pp = poc[(poc["period"] == p) & (poc["key"] < pk) & (poc["last_date"] >= wstart)]
    return pp[["symbol", "poc"]].drop_duplicates(["symbol", "poc"])


def _eval_period(sub: pd.DataFrame, pp: pd.DataFrame, row: float, tol: float) -> pd.DataFrame:
    """Per-symbol POC-line evaluation for one period. `sub` has columns
    symbol, low, high, close. Returns DataFrame indexed by symbol with:
        value     — representative line (nearest touched, else nearest overall)
        dist_in   — |close-value| %      touched — any edge line inside the candle
        dist_near — |close-nearest| %    near_hit — nearest line within tol %"""
    idx = pd.Index(sub["symbol"].values, name="symbol")
    out = pd.DataFrame(index=idx)
    half = row / 2.0
    pp = pp[pp["symbol"].isin(sub["symbol"])]
    if pp.empty:
        out["value"] = np.nan; out["dist_in"] = np.nan
        out["touched"] = False; out["dist_near"] = np.nan; out["near_hit"] = False
        return out

    lines = pd.concat([pp.assign(line=pp["poc"] + half)[["symbol", "line"]],
                       pp.assign(line=pp["poc"] - half)[["symbol", "line"]]],
                      ignore_index=True).drop_duplicates(["symbol", "line"])
    m = lines.merge(sub, on="symbol", how="inner")
    m["touched"] = (m["low"] <= m["line"]) & (m["line"] <= m["high"])
    m["dist"] = (m["close"] - m["line"]).abs() / m["close"] * 100.0

    near_all = m.loc[m.groupby("symbol")["dist"].idxmin()].set_index("symbol")
    mt = m[m["touched"]]
    near_t = (mt.loc[mt.groupby("symbol")["dist"].idxmin()].set_index("symbol")
              if len(mt) else m.iloc[0:0].set_index("symbol"))

    out["touched"]   = out.index.isin(near_t.index)
    out["dist_near"] = near_all["dist"].reindex(out.index)
    out["near_hit"]  = out["dist_near"] <= tol
    v_t, d_t = near_t["line"].reindex(out.index), near_t["dist"].reindex(out.index)
    v_a, d_a = near_all["line"].reindex(out.index), near_all["dist"].reindex(out.index)
    out["value"]   = v_t.where(out["touched"], v_a)
    out["dist_in"] = d_t.where(out["touched"], d_a)
    return out


# ─────────────────────────────────── API ───────────────────────────────────
app = FastAPI(title="Level Screener")


class ScreenReq(BaseModel):
    date: str | None = None      # YYYY-MM-DD; default = latest available day
    periods: list[str] = []      # subset of {"d","w","m"}; empty = all
    logic: str = "ANY"           # ANY | ALL across selected periods
    mode: str = "inside"         # inside (line within candle) | near (within tol %)
    tol: float = 0.5             # tolerance % for "near" mode
    sort_dir: str = "asc"        # closest touch first


@app.get("/api/meta")
def meta():
    return {"periods": [{"key": p["key"], "label": p["label"], "row": p["row"],
                         "lookback": p["lookback"], "color": p["color"]} for p in PERIODS]}


@app.get("/api/days")
def days():
    _, _, dlist = _snapshot()
    return {"days": dlist[:600], "latest": dlist[0] if dlist else None}


@app.post("/api/screen")
def screen(req: ScreenReq):
    big, poc, dlist = _snapshot()
    if big.empty:
        return JSONResponse({"count": 0, "universe": 0, "date": "", "periods": [], "rows": []})

    day = req.date if (req.date in dlist) else (dlist[0] if dlist else None)
    sub = big[big["date"] == day][["symbol", "low", "high", "close"]].dropna()
    universe = int(sub["symbol"].nunique())
    if sub.empty:
        return JSONResponse({"count": 0, "universe": universe, "date": day or "",
                             "periods": [], "rows": []})

    sel = [p for p in req.periods if p in PKEYS] or list(PKEYS)
    mode = req.mode if req.mode in ("inside", "near") else "inside"
    tol = max(0.0, float(req.tol))
    sel_cfg = [{"key": p, "label": PLABEL[p], "row": PROW[p],
                "lookback": PLOOK[p], "color": PCOLOR[p]} for p in sel]

    per_res, hits, dists = {}, [], []
    for p in sel:
        r = _eval_period(sub, _active_pocs(poc, p, day), PROW[p], tol)
        per_res[p] = r
        hit = r["touched"] if mode == "inside" else r["near_hit"]
        d = r["dist_in"] if mode == "inside" else r["dist_near"]
        hits.append(hit.fillna(False)); dists.append(d)

    H = pd.concat(hits, axis=1)
    passed = H.all(axis=1) if req.logic == "ALL" else H.any(axis=1)
    closest = pd.concat(dists, axis=1).min(axis=1)

    order = closest[passed].sort_values(ascending=(req.sort_dir == "asc")).index
    close_map = sub.set_index("symbol")["close"]

    def _num(x):
        return None if (x is None or pd.isna(x)) else round(float(x), 2)

    rows = []
    for sym in order:
        row = {"symbol": sym, "close": _num(close_map.get(sym))}
        for p in sel:
            r = per_res[p]
            row[p] = _num(r.at[sym, "value"])
            row[f"{p}_in"] = bool(r.at[sym, "touched"])
            row[f"{p}_d"] = _num(r.at[sym, "dist_in"])
        rows.append(row)

    return {"count": len(rows), "universe": universe, "date": day,
            "periods": sel_cfg, "mode": mode, "rows": rows}


@app.get("/api/chart")
def chart(symbol: str, date: str | None = None, periods: str = "d,w,m"):
    """Daily candles for one symbol + the OB+VP POC edge lines active on `date`."""
    big, poc, dlist = _snapshot()
    sc = big[big["symbol"] == symbol].sort_values("date").reset_index(drop=True)
    if sc.empty:
        return JSONResponse({"error": "unknown symbol"}, status_code=404)

    dates = sc["date"].tolist()
    idx = dates.index(date) if (date in dates) else (len(dates) - 1)
    day = dates[idx]
    lo_i, hi_i = max(0, idx - (CHART_TAIL_DAYS - 12)), min(len(sc), idx + 12)
    win = sc.iloc[lo_i:hi_i]

    candles = []
    for _, r in win.iterrows():
        h, l, c = r["high"], r["low"], r["close"]
        if pd.isna(h) or pd.isna(l) or pd.isna(c):
            continue
        o = c if pd.isna(r["open"]) else r["open"]
        candles.append({"time": str(r["date"])[:10], "open": round(float(o), 2),
                        "high": round(float(h), 2), "low": round(float(l), 2),
                        "close": round(float(c), 2)})

    drow = sc.iloc[idx]
    dlo, dhi, dcl = float(drow["low"]), float(drow["high"]), float(drow["close"])
    sel = [p for p in periods.split(",") if p in PKEYS] or list(PKEYS)

    levels = []
    for p in sel:
        pp = _active_pocs(poc, p, day)
        pp = pp[pp["symbol"] == symbol]
        half = PROW[p] / 2.0
        edges = sorted({round(float(v) + half, 4) for v in pp["poc"]} |
                       {round(float(v) - half, 4) for v in pp["poc"]})
        if len(edges) > CHART_MAX_LINES:                 # keep those nearest the day's price
            edges = sorted(edges, key=lambda x: abs(x - dcl))[:CHART_MAX_LINES]
        for v in edges:
            levels.append({"key": p, "label": PLABEL[p], "value": round(v, 2),
                           "touched": bool(dlo <= v <= dhi), "color": PCOLOR[p],
                           "row": PROW[p]})
    return {"symbol": symbol, "date": day, "candles": candles, "levels": levels}


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
    --txt:#e6edf6; --mut:#8a98b2; --accent:#22d3ee; --accent2:#818cf8;
    --pos:#34d399; --neg:#f87171; --chip:#172033;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#1b2440 0%,var(--bg) 55%);
       color:var(--txt);font:14px/1.5 'Inter',system-ui,Segoe UI,Roboto,sans-serif;min-height:100vh}
  .wrap{max-width:1240px;margin:0 auto;padding:28px 22px 60px}
  header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px}
  .title{font-size:26px;font-weight:800;letter-spacing:.2px;display:flex;align-items:center;gap:12px}
  .title .dot{width:11px;height:11px;border-radius:50%;background:var(--accent);box-shadow:0 0 16px 2px var(--accent)}
  .sub{color:var(--mut);font-size:13px;margin-top:4px;max-width:620px}
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
  .ck .meta{color:var(--mut);font-size:11.5px}
  .ck .swatch{width:10px;height:10px;border-radius:50%}
  .tolrow{display:flex;align-items:center;gap:8px;margin-top:10px;color:var(--mut);font-size:12.5px}
  .tolrow.off{opacity:.4;pointer-events:none}
  .tolrow input[type=number]{width:74px;background:var(--panel);border:1px solid var(--line);color:var(--txt);
        border-radius:8px;padding:6px 8px;font-size:12.5px;outline:none}
  .tolrow input[type=number]:focus{border-color:var(--accent)}
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
      <div class="sub">Stocks whose daily candle touched a <b>Volume-Profile POC</b> level (OB+VP) — Daily / Weekly / Monthly POC edge bands built from 1-minute volume.</div>
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

      <h3>POC levels to test</h3>
      <div class="lines" id="lines"></div>

      <h3>Match</h3>
      <div class="seg" id="logic">
        <button data-logic="ANY" class="on">ANY level</button>
        <button data-logic="ALL">ALL levels</button>
      </div>

      <h3>Condition</h3>
      <div class="seg" id="mode">
        <button data-mode="inside" class="on">Touched (inside candle)</button>
        <button data-mode="near">Within %</button>
      </div>
      <div class="tolrow off" id="tolrow">
        <span>Tolerance</span><input type="number" id="tol" value="0.5" step="0.1" min="0"/><span>% from POC line</span>
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
const STATE={logic:"ANY",mode:"inside",periods:{d:true,w:true,m:true}};

function renderLines(){
  const box=document.getElementById("lines"); box.innerHTML="";
  META.periods.forEach(g=>{
    const wk = g.key==="w" ? Math.round(g.lookback/7)+"w"
             : g.key==="m" ? Math.round(g.lookback/30)+"mo"
             : g.lookback+"d";
    const d=document.createElement("div");
    d.className="ck"+(STATE.periods[g.key]?" on":"");
    d.innerHTML=`<span class="box"></span>`
      +`<span class="swatch" style="background:${g.color}"></span>`
      +`<span class="lbl">${g.label}</span>`
      +`<span class="meta">row ${g.row} · ${wk}</span>`;
    d.onclick=()=>{
      STATE.periods[g.key]=!STATE.periods[g.key];
      d.classList.toggle("on",STATE.periods[g.key]);
    };
    box.appendChild(d);
  });
}
const selectedPeriods=()=>META.periods.map(g=>g.key).filter(k=>STATE.periods[k]);

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
  if(!sel.length){document.getElementById("count").textContent="Select at least one level type.";return;}
  document.getElementById("count").innerHTML='<span class="spin"></span>Scanning universe…';
  const body={date:document.getElementById("day").value,periods:sel,
              logic:STATE.logic,mode:STATE.mode,
              tol:parseFloat(document.getElementById("tol").value)||0.5,sort_dir:"asc"};
  const r=await api("/api/screen",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  LAST=r; render(r);
}

function render(r){
  document.getElementById("asof").innerHTML=`Universe <b>${r.universe}</b> · day <b>${r.date||"—"}</b>`;
  const verb=r.mode==="near"?"near a POC line":"touched a POC line";
  document.getElementById("count").innerHTML=`<b>${r.count}</b> stocks ${verb} on ${r.date||"—"}`;
  document.getElementById("dl").style.display=r.count?"block":"none";
  if(!r.count){document.getElementById("res").innerHTML='<div class="empty">No stocks matched on this day. Try ANY instead of ALL, “Within %”, or another date.</div>';return;}
  const P=r.periods;
  let head="<tr><th>Symbol</th><th>Close</th>";
  P.forEach(p=>head+=`<th>${p.label} POC ·${p.row}</th><th>Δ%</th>`);
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
  a.download=`poc_levels_${LAST.date||""}.csv`;a.click();
}

function bindSeg(id,key,after){
  document.querySelectorAll(`#${id} button`).forEach(b=>b.onclick=async()=>{
    document.querySelectorAll(`#${id} button`).forEach(x=>x.classList.remove("on"));
    b.classList.add("on"); STATE[key]=b.dataset[key]; if(after)await after();
  });
}

// ── chart modal: click a stock → see its candles cut the POC lines ──
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
          +"&periods="+sel.join(",");
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
  CSER.setMarkers([{time:r.date,position:"aboveBar",color:"#22d3ee",shape:"arrowDown",text:"day"}]);
  CHART.priceScale("right").applyOptions({autoScale:true});
  const touched=(r.levels||[]).filter(L=>L.touched).length;
  document.getElementById("m-dd").textContent=r.date+" · "+(r.candles||[]).length+" daily bars · "+touched+" POC line(s) touched";
  const seen={};
  document.getElementById("m-legend").innerHTML=(r.levels||[]).filter(L=>{if(seen[L.key])return false;seen[L.key]=1;return true;})
    .map(L=>`<span class="lg"><span class="ln" style="border-color:${L.color}"></span>${L.label} POC ·${L.row}</span>`).join("");
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
