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

import asyncio
import glob
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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


# ════════════════════════════════════════════════════════════════════════════
#  UNIVERSE SCAN — run the 🎯 Lift-off detector the chart draws across every
#  symbol, server-side, and surface the ones whose TRIGGER bar is fresh (within
#  the last `fresh` bars = an actionable now signal). This is a faithful Python
#  port of the JS in PAGE: same swing-pivot finding, clusterPivots, levelBreaks
#  and detectScenario, with identical constants — so a scan hit matches the chart.
# ════════════════════════════════════════════════════════════════════════════

_arr_cache: dict[tuple, tuple] = {}   # (sym, tf) -> (candles_obj, {high, vwap})


def _arrays(sym: str, tf: str):
    """OHLC highs + session VWAP as numpy arrays, cached on the candle-list
    identity (rebuilds only when _candles refreshes on an mtime change). VWAP
    resets each session for intraday TFs; cumulative anchor on Daily — matching
    computeVWAP() in the chart."""
    cs = _candles(sym, tf)
    hit = _arr_cache.get((sym, tf))
    if hit and hit[0] is cs:
        return cs, hit[1]
    n = len(cs)
    h  = np.fromiter((c["high"]  for c in cs), float, n)
    l  = np.fromiter((c["low"]   for c in cs), float, n)
    cl = np.fromiter((c["close"] for c in cs), float, n)
    v  = np.fromiter((c["volume"] for c in cs), float, n)
    tp = (h + l + cl) / 3.0
    pv = tp * v
    if tf != "1d":                                   # intraday → daily session reset
        day = np.fromiter((c["time"] // 86400 for c in cs), np.int64, n)
        cum_pv = pd.Series(pv).groupby(day).cumsum().to_numpy()
        cum_v  = pd.Series(v).groupby(day).cumsum().to_numpy()
    else:
        cum_pv, cum_v = np.cumsum(pv), np.cumsum(v)
    vw = np.where(cum_v > 0, cum_pv / np.where(cum_v > 0, cum_v, 1.0), tp)
    data = {"high": h, "vwap": vw}
    _arr_cache[(sym, tf)] = (cs, data)
    return cs, data


def _pivots(h: np.ndarray, k: int) -> list[dict]:
    """Swing highs: bar i where high[i] == max(high[i-k : i+k+1]) (ties kept,
    edges excluded) — vectorised equivalent of the chart's pivot loop. Uses a
    sliding-window view (far faster than pandas centred rolling over 500 syms)."""
    n = len(h)
    if n < 2 * k + 1:
        return []
    W = 2 * k + 1
    wmax = np.lib.stride_tricks.sliding_window_view(h, W).max(axis=1)  # len n-W+1
    centers = np.arange(k, n - k)                                      # window centres
    idx = centers[h[centers] >= wmax]
    return [{"i": int(i), "price": float(h[i])} for i in idx]


_zones_cache: dict[tuple, tuple] = {}   # (sym, tf, k) -> (candles_obj, zones)


def _zones(sym: str, tf: str, k: int):
    """Candles + resistance zones (n≥2 touches), cached on the candle-list
    identity so a re-scan with a different kind/fresh skips pivot work."""
    cs, arr = _arrays(sym, tf)
    key = (sym, tf, k)
    hit = _zones_cache.get(key)
    if hit and hit[0] is cs:
        return cs, arr, hit[1]
    piv = _pivots(arr["high"], k)
    zones = [z for z in _cluster_pivots(piv, tf) if z["n"] >= 2] if len(piv) >= 2 else []
    _zones_cache[key] = (cs, zones)
    return cs, arr, zones


def _cluster_pivots(piv: list[dict], tf: str) -> list[dict]:
    """Cluster nearby swing-high pivots into resistance zones — timeframe-scaled
    tolerance that tightens once a level has ≥2 close touches, with a ≥12-bar min
    gap so a pivot belonging to the same swing isn't counted as a new test."""
    piv = sorted(piv, key=lambda p: p["price"])
    TOL = 0.016 if tf == "1d" else (0.008 if tf == "1h" else 0.0035)
    MINTOL, MINBARS = TOL * 0.25, 12
    zones: list[dict] = []
    for p in piv:
        z = zones[-1] if zones else None
        if z:
            eff = TOL
            if z["n"] >= 2:
                sr = (z["top"] - z["bottom"]) / z["avg"]
                eff = min(TOL, max(MINTOL, sr * 1.5))
            if abs(p["price"] - z["avg"]) <= z["avg"] * eff:
                if any(abs(p["i"] - q["i"]) < MINBARS for q in z["pts"]):
                    continue
                z["sum"] += p["price"]; z["n"] += 1; z["avg"] = z["sum"] / z["n"]
                z["top"] = max(z["top"], p["price"]); z["bottom"] = min(z["bottom"], p["price"])
                z["firstIdx"] = min(z["firstIdx"], p["i"]); z["pts"].append(p)
                continue
        zones.append({"sum": p["price"], "n": 1, "avg": p["price"], "top": p["price"],
                      "bottom": p["price"], "firstIdx": p["i"], "pts": [p]})
    return zones


def _level_breaks(cs: list[dict], top: float, from_idx: int) -> dict:
    """Classify how a level was broken, tolerating ONE false breakout (a close
    above that later returns below). terminal = index of the first SUSTAINED
    break, else len(cs)."""
    BREAK = 0.0015
    up, n = top * (1 + BREAK), len(cs)
    above, break_start, false_breaks = False, -1, 0
    for j in range(from_idx + 1, n):
        if not above and cs[j]["close"] > up:
            above, break_start = True, j
        elif above and cs[j]["close"] < top:
            above, false_breaks = False, false_breaks + 1
    return {"falseBreaks": false_breaks, "terminal": break_start if above else n}


def _active_window(cs, vw, z):
    """Shared preamble for both detectors: the touches, and the index where the
    level's active (still-below) window ends — tolerating one false breakout.
    Returns (touches, a, cap) or None if the level can't host a setup."""
    n = len(cs)
    top = z["top"]
    touches = sorted(p["i"] for p in z["pts"])
    if len(touches) < 2:
        return None
    first, second, last = touches[0], touches[1], touches[-1]
    bk = _level_breaks(cs, top, first)
    if bk["falseBreaks"] <= 1 and (bk["falseBreaks"] == 0 or len(touches) >= 3):
        brk = bk["terminal"]
    else:
        brk = n
        for j in range(first + 1, n):
            if cs[j]["close"] > top * (1.0015):
                brk = j
                break
    if last > brk:                       # a touch after the break = role reversal → drop
        return None
    return first, second, brk


def _detect_liftoff(cs, vw, z, min_hug: int = 4, green_only: bool = True) -> list[dict]:
    """🎯 Lift-off: level tested ≥2× → price holds below → ≥`min_hug` closes hug a
    flat VWAP → next candle closes clearly above the VWAP, still below the level.

    Two sweepable variations (exposed in the UI):
      • min_hug    : hug run length. 4 (default/strict) or 3 (looser — catches
                     3- and 4-candle coils).
      • green_only : if True (default) the lift-off candle must be GREEN
                     (close>open). If False, ANY colour qualifies as long as it
                     closes clearly above the VWAP.
    Returns ALL such setups in the active window (not just the first)."""
    win = _active_window(cs, vw, z)
    if not win:
        return []
    first, a, brk = win
    n, top = len(cs), z["top"]
    HUG, FLAT = 0.004, 0.004
    MINHUG = min_hug
    cap = min(brk - 1, n - 1, a + 80)
    below = lambda i: cs[i]["close"] < top * (1.0015)
    out, i = [], a + 1
    while i + MINHUG <= cap:
        h0, h1, j = i, i - 1, i
        while j <= cap and below(j) and abs(cs[j]["close"] - vw[j]) / vw[j] <= HUG:
            h1, j = j, j + 1
        if h1 - h0 + 1 >= MINHUG:
            flat = abs(vw[h1] - vw[h0]) / vw[h0] <= FLAT
            L = j
            if flat and L <= cap:
                c = cs[L]
                is_green = c["close"] > c["open"]
                if (is_green or not green_only) and c["close"] > vw[L] * (1 + HUG) and below(L):
                    out.append({"kind": 3, "li": first, "ri": L, "top": top,
                                "slope": round((vw[h1] - vw[h0]) / vw[h0] * 100, 2),
                                "nCand": h1 - h0 + 1,
                                "trigTime": cs[L]["time"], "trigClose": cs[L]["close"]})
                    i = L + 1
                    continue
            i = h1 + 1
        else:
            i += 1
    return out


SCAN_DIR = Path(__file__).resolve().parent / "scan_cache"
SCAN_NDAYS = 30
_days_cache: dict[tuple, tuple] = {}     # (tf, pivot) -> (sig, data)


def _compute_scan_days(tf: str, pivot_k: int, ndays: int,
                       min_hug: int = 4, green_only: bool = True) -> dict:
    """Scan the universe and bucket every lift-off setup by its TRIGGER date,
    keeping the last `ndays` trading days. Returns {dates:[...desc], by_date:{}}.
    One run serves all days, so day-switching in the UI needs no rescan.
    `min_hug`/`green_only` are the lift-off variation knobs (see _detect_liftoff)."""
    buckets: dict[str, dict] = {}                 # date -> {symbol -> row}
    for sym in _symbols():
        try:
            cs, arr, zones = _zones(sym, tf, pivot_k)
        except Exception:
            continue
        n = len(cs)
        if n < 2 * pivot_k + 12 or not zones:
            continue
        vw = arr["vwap"]
        daynums = sorted({c["time"] // 86400 for c in cs})
        cutoff_day = daynums[-ndays] if len(daynums) >= ndays else daynums[0]
        cutoff_idx = next((i for i, c in enumerate(cs)
                           if c["time"] // 86400 >= cutoff_day), n)
        for z in zones:
            touches = sorted(p["i"] for p in z["pts"])
            if touches[1] < cutoff_idx - 80:          # 2nd touch too old to trigger in window
                continue
            for st in _detect_liftoff(cs, vw, z, min_hug, green_only):
                if st["ri"] < cutoff_idx:
                    continue
                bars_ago = (n - 1) - st["ri"]
                date = time.strftime("%Y-%m-%d", time.gmtime(st["trigTime"]))
                day = buckets.setdefault(date, {})
                prev = day.get(sym)                   # one row per symbol per day (freshest)
                if prev and prev["bars_ago"] <= bars_ago:
                    continue
                day[sym] = {
                    "symbol": sym, "bars_ago": bars_ago,
                    "trig_time": time.strftime("%Y-%m-%d %H:%M",
                                               time.gmtime(st["trigTime"])),
                    "trig_close": round(st["trigClose"], 2),
                    "resistance": round(st["top"], 2),
                    "slope": st["slope"], "n_cand": st["nCand"],
                }
    dates = sorted(buckets.keys(), reverse=True)[:ndays]
    by_date = {d: sorted(buckets[d].values(),
                         key=lambda r: (r["bars_ago"], r["symbol"])) for d in dates}
    return {"dates": dates, "by_date": by_date}


def _variation_suffix(min_hug: int, green_only: bool) -> str:
    """File/cache-key suffix for a lift-off variation. Empty for the default
    (4-bar hug / green-only) so existing default snapshots stay valid."""
    if min_hug == 4 and green_only:
        return ""
    return f"_h{min_hug}_g{1 if green_only else 0}"


def _days_snap_path(tf: str, pivot_k: int, min_hug: int = 4, green_only: bool = True) -> Path:
    return SCAN_DIR / f"{tf}_p{pivot_k}{_variation_suffix(min_hug, green_only)}.json"


def _load_days_snapshot(tf: str, pivot_k: int, min_hug: int = 4, green_only: bool = True):
    p = _days_snap_path(tf, pivot_k, min_hug, green_only)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_days_snapshot(tf: str, pivot_k: int, sig: tuple, data: dict,
                        min_hug: int = 4, green_only: bool = True) -> None:
    try:
        SCAN_DIR.mkdir(exist_ok=True)
        _days_snap_path(tf, pivot_k, min_hug, green_only).write_text(
            json.dumps({"sig": list(sig), "data": data}))
    except Exception:
        pass


def _scan_days(tf: str, pivot_k: int, ndays: int = SCAN_NDAYS,
               min_hug: int = 4, green_only: bool = True) -> dict:
    """Day-bucketed universe scan, cached in memory AND on disk keyed by the data
    mtime AND the lift-off variation — so the heavy scan runs once per
    (data version, variation), restarts reload from disk instantly, and the UI
    can browse any of the last `ndays` days for free."""
    src_tf = "15m" if tf == "30m" else tf
    files = glob.glob(str(DATA_DIR / src_tf / "*_historical.csv"))
    sig = (tf, pivot_k, ndays, min_hug, green_only,
           round(max((os.path.getmtime(f) for f in files), default=0.0), 3))
    key = (tf, pivot_k, min_hug, green_only)
    hit = _days_cache.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    snap = _load_days_snapshot(tf, pivot_k, min_hug, green_only)
    if snap and tuple(snap.get("sig", [])) == sig:
        _days_cache[key] = (sig, snap["data"])
        return snap["data"]
    data = _compute_scan_days(tf, pivot_k, ndays, min_hug, green_only)
    _days_cache[key] = (sig, data)
    _save_days_snapshot(tf, pivot_k, sig, data, min_hug, green_only)
    return data


@app.get("/api/scan")
def scan(tf: str = Query("15m"), pivot: int = Query(8),
         min_hug: int = Query(4, description="Minimum hug run length (≥N candles). UI offers 1..5; the '>5' option sends 6 (≥6 catches 6+). Default 4."),
         green: int = Query(1, description="Lift-off candle colour gate: 1=green-only (default), 0=any colour (just close above VWAP).")):
    tf = tf if tf in ("1d", "1h", "30m", "15m") else "15m"
    pivot = max(2, min(int(pivot), 60))      # any input, clamped to a sane range
    min_hug = max(1, min(int(min_hug), 60))  # 1..5 from the UI; '>5' sends 6 (≥6 = 6+); bounded for safety
    green_only = bool(int(green))
    data = _scan_days(tf, pivot, SCAN_NDAYS, min_hug, green_only)
    return {"tf": tf, "pivot": pivot, "universe": len(_symbols()),
            "min_hug": min_hug, "green": int(green_only),
            "dates": data["dates"], "by_date": data["by_date"]}


# ════════════════════════════════════════════════════════════════════════════
#  LIVE MONITOR — a background scheduler that, during market hours, re-scans a
#  timeframe right after each of its bars CLOSES and ACCUMULATES the day's hits
#  into a persisted per-date list (deduped). 15m fires every 15 min, 30m every
#  30 min, 1h every hour — driven by the data's latest bar advancing, so we
#  never hardcode NSE bucket boundaries. The /api/found list is what the
#  "📋 Found Today" panel shows; click-through reuses the chart.
# ════════════════════════════════════════════════════════════════════════════

FOUND_DIR = Path(__file__).resolve().parent / "found"
SCAN_TFS = ("15m", "30m", "1h")
_last_scanned: dict[str, int] = {tf: 0 for tf in SCAN_TFS}   # last bar time scanned per TF
_monitor: dict[str, str] = {"last_run": "", "status": "starting"}


def _ist_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)   # bars are IST; server may be UTC


def _ist_date_str() -> str:
    return _ist_now().strftime("%Y-%m-%d")


def _market_hours(now: datetime) -> bool:
    if now.weekday() >= 5:                       # Sat/Sun
        return False
    m = now.hour * 60 + now.minute
    return 9 * 60 + 14 <= m <= 15 * 60 + 45      # 09:14–15:45 IST (grace past the 15:30 bar)


def _latest_bar_time(tf: str) -> int:
    """Latest bar timestamp for a TF, read from a liquid reference symbol."""
    for ref in ("RELIANCE", "HDFCBANK", "INFY", "ICICIBANK"):
        try:
            cs = _candles(ref, tf)
            if cs:
                return cs[-1]["time"]
        except Exception:
            continue
    return 0


def _found_path(date_str: str) -> Path:
    return FOUND_DIR / f"{date_str}.json"


def _load_found(date_str: str) -> list[dict]:
    p = _found_path(date_str)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _append_found(tf: str, rows: list[dict]) -> int:
    """Append NEW setups (deduped by symbol+setup+tf+trigger) to today's list."""
    date = _ist_date_str()
    items = _load_found(date)
    seen = {(it["symbol"], it["setup"], it["tf"], it["trig_time"]) for it in items}
    found_at = _ist_now().strftime("%H:%M:%S")
    added = 0
    for r in rows:
        key = (r["symbol"], r["setup"], tf, r["trig_time"])
        if key in seen:
            continue
        seen.add(key)
        items.append({**r, "tf": tf, "found_at": found_at})
        added += 1
    if added:
        FOUND_DIR.mkdir(exist_ok=True)
        _found_path(date).write_text(json.dumps(items))
    return added


def _run_due_scans() -> None:
    """For each TF whose bar has newly closed, refresh + persist the 30-day
    scan cache so the UI never waits for a scan during/after market hours."""
    for tf in SCAN_TFS:
        lt = _latest_bar_time(tf)
        if lt and lt > _last_scanned[tf]:
            data = _scan_days(tf, 8, SCAN_NDAYS)        # compute + persist to disk
            _last_scanned[tf] = lt
            _monitor["last_run"] = _ist_now().strftime("%Y-%m-%d %H:%M:%S")
            latest = data["dates"][0] if data["dates"] else "—"
            cnt = len(data["by_date"].get(latest, [])) if data["dates"] else 0
            print(f"[resistance scanner] {tf}: 30d cache warmed — {latest}: {cnt} setup(s)")


async def _scheduler() -> None:
    await asyncio.sleep(5)
    loop = asyncio.get_event_loop()
    while True:
        try:
            now = _ist_now()
            if _market_hours(now):
                _monitor["status"] = "monitoring"
                await loop.run_in_executor(None, _run_due_scans)   # _scan is blocking → off the event loop
                await asyncio.sleep(60)
            else:
                _monitor["status"] = "closed"
                await asyncio.sleep(300)
        except Exception as e:           # never let the loop die
            print("[resistance monitor] error:", e)
            await asyncio.sleep(60)


@app.on_event("startup")
async def _startup() -> None:
    FOUND_DIR.mkdir(exist_ok=True)
    SCAN_DIR.mkdir(exist_ok=True)
    asyncio.create_task(_scheduler())


@app.get("/api/found")
def found(date: str | None = Query(None)):
    d = date or _ist_date_str()
    items = _load_found(d)
    items.sort(key=lambda it: (it.get("found_at", ""), it["symbol"]), reverse=True)
    dates = sorted((p.stem for p in FOUND_DIR.glob("*.json")), reverse=True) if FOUND_DIR.exists() else []
    last = {tf: (time.strftime("%Y-%m-%d %H:%M", time.gmtime(t)) if t else None)
            for tf, t in _last_scanned.items()}
    return {"date": d, "count": len(items), "items": items, "dates": dates,
            "status": _monitor["status"], "last_run": _monitor["last_run"],
            "last_bar": last}


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
.hint{color:var(--muted);font-size:11px}
.muted{color:var(--muted)}
#scan{background:linear-gradient(135deg,#f59e0b,#b45309);color:#1a1206;border-color:transparent;font-weight:800;padding:6px 16px}
#k{width:64px;text-align:center}
.wrap{padding:16px 18px 48px}
#scanMeta{margin:0 0 10px;font-size:12.5px}
.stbl{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.stbl th,.stbl td{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--border)}
.stbl th:first-child,.stbl td:first-child{text-align:left}
.stbl th:last-child,.stbl td:last-child{text-align:center}
.stbl th{position:sticky;top:0;background:#10131a;color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;z-index:1}
.stbl tbody tr{cursor:pointer}
.stbl tbody tr:hover{background:#141a27}
.stbl .ssym{font-weight:700;color:var(--text)}
.stbl .pos{color:#34d399}.stbl .neg{color:#ef5350}
.seebtn{padding:4px 10px;font-size:12px;background:#172033;border:1px solid var(--border);color:#cfd6e4;border-radius:6px}
.seebtn:hover{border-color:var(--accent);color:var(--accent)}
.sesshd{margin:18px 0 6px;font-size:13px;font-weight:700;color:var(--text);border-left:3px solid var(--accent);padding-left:9px}
.sesshd:first-child{margin-top:4px}
.sesshd .span{font-weight:400;font-size:11px;color:var(--muted)}
.sesshd .cnt{color:var(--accent)}
.sessempty{color:var(--muted);font-size:12px;padding:4px 2px 6px 11px}
#chartwrap{display:none;margin-top:18px;border:1px solid var(--border);border-radius:10px;overflow:hidden}
#chartwrap.show{display:block}
#charthead{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:#10131a;border-bottom:1px solid var(--border)}
#chartsym{font-weight:700;color:var(--accent);font-size:13px}
#chartclose{padding:4px 10px;font-size:12px}
#chart{height:560px;position:relative}
</style></head><body>

<div id="bar">
  <span class="tag">RESISTANCE SCANNER</span>
  <span class="seg" id="tf"><button data-tf="1d">Daily</button><button data-tf="1h">Hourly</button><button data-tf="30m">30m</button><button data-tf="15m" class="on">15m</button></span>
  <span><label>Pivot</label><input id="k" type="number" min="2" max="60" step="1" value="8"></span>
  <span><label>Hug</label><select id="hug" title="Minimum candles that must hug the flat VWAP before the lift-off"><option value="1">≥1</option><option value="2">≥2</option><option value="3">≥3</option><option value="4" selected>≥4</option><option value="5">≥5</option><option value="6">&gt;5 (6+)</option></select></span>
  <span><label>Lift-off</label><select id="lift" title="The candle that pops off the VWAP"><option value="1">Green only</option><option value="0">Any colour (close &gt; VWAP)</option></select></span>
  <button id="scan">🔭 Scan</button>
  <span><label>Day</label><select id="day" style="min-width:118px"><option>—</option></select></span>
  <span class="hint">🎯 lift-off setups across all stocks · last 30 days stored · pick a Day · click a stock (or “See chart”) to view it below.</span>
  <span id="title"></span>
</div>

<div class="wrap">
  <div class="muted" id="scanMeta">Pick a timeframe (15m/30m/1h), set the pivot, then click <b>Scan</b>.</div>
  <div id="scanRes"></div>
  <div id="chartwrap">
    <div id="charthead"><span id="chartsym"></span><button id="chartclose">✕ close chart</button></div>
    <div id="chart"></div>
  </div>
</div>

<script>
const APP_BASE="__APP_BASE__";
const api=function(p){return (APP_BASE||"")+p;};
const $=function(id){return document.getElementById(id);};

const S={candles:[], tf:"15m", zones:[], setups:[], minHug:4, greenOnly:true, sym:""};  // zones=resistance lines · setups=lift-off boxes · minHug/greenOnly = lift-off variation knobs · sym = open chart symbol
let vwapSeries=null;

// The chart lives inside #chartwrap which is display:none until a stock is
// opened. Creating it while hidden gives it a 0×0 box and broken scales, so we
// create it LAZILY on first reveal (when the container has real dimensions).
let chart=null, candle=null, ov=null;
function ensureChart(){
  if(chart) return;
  chart=LightweightCharts.createChart($("chart"),{autoSize:true,
    layout:{background:{color:"#0b0d11"},textColor:"#cfd6e4"},
    grid:{vertLines:{color:"#161b27"},horzLines:{color:"#161b27"}},
    timeScale:{borderColor:"#222838",rightOffset:6,timeVisible:true,secondsVisible:false},
    rightPriceScale:{borderColor:"#222838"},crosshair:{mode:0}});
  candle=chart.addCandlestickSeries({upColor:"#26a69a",downColor:"#ef5350",
    borderUpColor:"#26a69a",borderDownColor:"#ef5350",wickUpColor:"#26a69a",wickDownColor:"#ef5350"});
  // ── overlay canvas: resistance is drawn as shaded BLOCKS, not lines ──
  $("chart").style.position="relative";
  ov=document.createElement("canvas");
  ov.style.cssText="position:absolute;left:0;top:0;pointer-events:none;z-index:2;";
  $("chart").appendChild(ov);
}

function drawBlocks(){
  if(!chart||!candle||!ov) return;
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
async function load(sym){
  sym=(sym||"").trim().toUpperCase(); if(!sym)return false;
  S.zones=[]; S.setups=[]; drawBlocks();
  $("title").textContent="loading "+sym+"…";
  let j; try{j=await(await fetch(api("/api/candles?symbol="+sym+"&tf="+S.tf))).json();}
  catch(e){$("title").textContent="not found";return false;}
  if(j.error||!j.candles||!j.candles.length){$("title").textContent=sym+": no data";candle.setData([]);return false;}
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
  // Force the price (vertical) axis to re-fit the NEW symbol. Without this, if
  // the user dragged/zoomed the price axis on the previous stock, lightweight-
  // charts leaves autoScale OFF and carries that fixed range over — so a stock
  // at a different price level renders off-screen or squished. Re-enabling it
  // on every load makes each chart fit cleanly to its own data.
  candle.priceScale().applyOptions({autoScale:true});
  drawVWAP();
  $("title").textContent=sym+" · "+S.tf+" · "+S.candles.length+" bars";
  return true;
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
  if(!chart) return;
  if(vwapSeries){ try{chart.removeSeries(vwapSeries)}catch(e){} vwapSeries=null; }
  if(!S.candles.length) return;
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
//   • for >= 4 candles the closes HUG the VWAP (sit on the line) while the VWAP is
//     FLAT/slight-slope — price and VWAP coil together, then
//   • the next candle is a GREEN candle that closes CLEARLY ABOVE the VWAP — the
//     lift-off off the line — still below the resistance. That is the trigger.
function detectScenario(vis, vw, z){
  const out=[], n=vis.length, BREAK=0.0015;
  const HUG=0.004;    // a close within 0.4% of the VWAP = "sitting on the line"
  const FLAT=0.004;   // VWAP slope across the hug must stay within +/-0.4% (flat/slight)
  const MINHUG=S.minHug||4;       // hug length variation: 4 (strict) or 3 (3 & 4)
  const GREENONLY=S.greenOnly!==false;  // lift-off candle: green-only (default) or any colour
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

  // scan for: a run of >=4 closes hugging a flat VWAP, immediately followed by a
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
        const isGreen = c.close>c.open;
        const greenLift = (isGreen || !GREENONLY) && c.close > vw[L]*(1+HUG) && belowRes(L);
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
                    label:"tested ≥2× · hug flat VWAP ≥"+MINHUG+" · "+(GREENONLY?"green":"any")+" lift-off ↑",
                    slope:slopePct, nCand:len, vwPts:vwPts,
                    inner:{x0:vis[h0].time, x1:vis[L].time, yTop:hiH*(1+0.002), yBottom:loL*(1-0.002)},
                    pts:[{time:vis[L].time, value:vis[L].close, star:true}]});  // lift-off candle (triangle)
          i=L+1; continue;   // collect ALL setups in the window (matches the universe scan), not just the first
        }
      }
      i=h1+1;          // hug found but no valid lift-off — search on from the run's end
    } else {
      i++;
    }
  }
  return out;
}

// ── scan THIS chart's full history for lift-off setups and draw them ──
function findAndDraw(){
  S.zones=[]; S.setups=[];
  const vis=S.candles; if(vis.length<10){ drawBlocks(); return 0; }
  const k=+$("k").value||8, vw=visVwap(vis), piv=[];
  for(let i=k;i<vis.length-k;i++){
    let hi=true; for(let j=i-k;j<=i+k;j++){ if(vis[j].high>vis[i].high){hi=false;break;} }
    if(hi) piv.push({i:i, price:vis[i].high});
  }
  clusterPivots(piv, S.tf).forEach(function(z){
    if(z.n<2) return;
    const found=detectScenario(vis,vw,z);
    if(!found.length) return;
    found.forEach(function(s){S.setups.push(s);});
    S.zones.push({top:z.top, bottom:z.bottom, n:z.n, fromTime:vis[z.firstIdx].time,
                  touches: z.pts.map(function(p){return {time:vis[p.i].time, value:p.price};})});
  });
  if(S.setups.length){
    let last=S.setups[0]; S.setups.forEach(function(s){ if(s.ri>last.ri) last=s; });
    chart.timeScale().setVisibleLogicalRange({from:Math.max(0,last.li-25), to:last.ri+30});
  }
  drawBlocks();
  return S.setups.length;
}

// ── wire-up ──────────────────────────────────────────────────────────
[].slice.call($("tf").children).forEach(function(b){ b.onclick=function(){
  [].slice.call($("tf").children).forEach(function(x){x.classList.remove("on");});
  b.classList.add("on"); S.tf=b.dataset.tf;
  // tf change auto-refreshes too (matches Pivot/Hug/Lift-off). Unlike those,
  // the candles themselves differ per timeframe, so the open chart is RELOADED
  // for the new TF (showStock re-fetches candles) rather than just re-detected.
  if(LAST) runScan();
  if($("chartwrap").classList.contains("show") && S.sym) showStock(S.sym);
};});

// ── lift-off variation knobs: re-draw the open chart instantly, and re-scan the
//    whole universe so the table reflects the chosen variation ──
function applyVariation(){
  S.minHug    = +$("hug").value || 4;
  S.greenOnly = $("lift").value !== "0";
  if($("chartwrap").classList.contains("show") && S.candles.length){
    const n=findAndDraw();                       // re-detect on the current chart
    const sym=($("chartsym").textContent||"").split(" ")[0];
    $("chartsym").textContent=sym+" · "+S.tf.toUpperCase()+" · "+n+" lift-off setup(s)";
  }
  if(LAST) runScan();                            // refresh the all-stocks list for the new variation
}
$("hug").onchange=applyVariation;
$("lift").onchange=applyVariation;
// Pivot is part of the same detection — changing it must refresh the list (and the
// open chart) WITHOUT waiting for a manual Scan click. onchange (commit/blur/Enter
// or spinner-arrow) avoids re-scanning on every keystroke while typing a number.
$("k").onchange=applyVariation;

// open a stock's chart BELOW the list and draw its lift-off setups
async function showStock(sym){
  S.sym=sym;                                 // remember the open symbol so a tf switch can reload it
  $("chartwrap").classList.add("show");      // container must be visible BEFORE the chart is created
  ensureChart();
  $("chartsym").textContent=sym+" · "+S.tf.toUpperCase()+" — loading…";
  const ok=await load(sym);
  const n = ok ? findAndDraw() : 0;
  $("chartsym").textContent=sym+" · "+S.tf.toUpperCase()+(ok?(" · "+n+" lift-off setup(s)"):" · no data");
  $("chartwrap").scrollIntoView({behavior:"smooth", block:"start"});
}
$("chartclose").onclick=function(){ $("chartwrap").classList.remove("show"); };

// ── scan: lift-off setups bucketed by trigger day (last 30 days, cached) ──
let LAST=null;   // last /api/scan response: {tf,pivot,universe,dates,by_date}

async function runScan(){
  const k=$("k").value||8;
  $("scanMeta").innerHTML="Scanning all stocks on "+S.tf.toUpperCase()+"… (first run builds the 30-day cache)";
  $("scanRes").innerHTML=""; $("chartwrap").classList.remove("show");
  let j;
  try{ j=await(await fetch(api("/api/scan?tf="+S.tf+"&pivot="+k+"&min_hug="+S.minHug+"&green="+(S.greenOnly?1:0)))).json(); }
  catch(e){ $("scanMeta").textContent="scan failed — is the server running?"; return; }
  LAST=j;
  const dd=$("day");
  if(j.dates && j.dates.length){
    dd.innerHTML=j.dates.map(function(d,i){ return "<option"+(i===0?" selected":"")+">"+d+"</option>"; }).join("");
    renderDay(j.dates[0]);                           // default to the most recent day
  }else{
    dd.innerHTML="<option>—</option>";
    $("scanRes").innerHTML="";
    $("scanMeta").innerHTML="Scanned <b>"+j.universe+"</b> stocks · "+j.tf.toUpperCase()+" · pivot "+j.pivot
      +" — no lift-off setups in the last 30 days."
      +(j.tf==="1d" ? " &nbsp;<span style='color:#ef5350'>(intraday recommended — 15m/30m/1h)</span>" : "");
  }
}

// Which intraday session a trigger time falls in (IST, NSE 09:15–15:30):
//   0 = Morning 09:15–11:30 · 1 = After 11:30–13:30 · 2 = Last 13:30–15:30
function sessionOf(trig){
  const hm=(trig||"").slice(11,16);            // "HH:MM" out of "YYYY-MM-DD HH:MM"
  const mins=(+hm.slice(0,2))*60 + (+hm.slice(3,5));
  if(mins < 11*60+30) return 0;                // before 11:30
  if(mins < 13*60+30) return 1;                // 11:30–13:30
  return 2;                                     // 13:30 onward
}
const SESSIONS=[
  {name:"🌅 Morning", span:"09:15–11:30"},
  {name:"☀️ After",   span:"11:30–13:30"},
  {name:"🌆 Last",    span:"13:30–15:30"},
];
const TBL_HEAD='<thead><tr><th>Symbol</th><th>Trigger time</th><th>Resistance</th>'
  +'<th>Close</th><th>VWAP slope</th><th>Hug bars</th><th></th></tr></thead>';

function rowHtml(r){
  const sl=(r.slope>0?"+":"")+r.slope+"%";
  return '<tr data-sym="'+r.symbol+'">'
    +'<td class="ssym">🎯 '+r.symbol+'</td>'
    +'<td class="muted">'+r.trig_time+'</td>'
    +'<td>'+r.resistance+'</td>'
    +'<td>'+r.trig_close+'</td>'
    +'<td class="'+(r.slope>=0?"pos":"neg")+'">'+sl+'</td>'
    +'<td>'+r.n_cand+'</td>'
    +'<td><button class="seebtn">📈 See chart</button></td></tr>';
}

function renderDay(date){
  if(!LAST) return;
  const rows=(LAST.by_date && LAST.by_date[date]) || [];
  // Split the day's setups into the three intraday sessions (intraday TFs only;
  // Daily has no meaningful session, so it keeps a single flat table).
  const isIntraday = LAST.tf!=="1d";
  const buckets=[[],[],[]];
  if(isIntraday) rows.forEach(function(r){ buckets[sessionOf(r.trig_time)].push(r); });
  const sessCounts = isIntraday
    ? " &nbsp;<span class='muted'>("+SESSIONS.map(function(s,i){return s.name.split(" ")[0]+" "+buckets[i].length;}).join(" · ")+")</span>"
    : "";
  $("scanMeta").innerHTML="<b>"+LAST.universe+"</b> stocks · "+LAST.tf.toUpperCase()+" · pivot "+LAST.pivot
    +" · hug ≥"+(LAST.min_hug||4)+" · "+(LAST.green?"green":"any")+" lift-off"
    +" · <b>"+date+"</b> — <b>"+rows.length+"</b> stock(s) with a lift-off setup"+sessCounts
    +(LAST.tf==="1d" ? " &nbsp;<span style='color:#ef5350'>(intraday recommended — 15m/30m/1h)</span>" : "");
  $("chartwrap").classList.remove("show");
  if(!rows.length){ $("scanRes").innerHTML='<div class="muted" style="padding:16px 2px">No lift-off setups triggered on '+date+'.</div>'; return; }

  let html;
  if(!isIntraday){
    html='<table class="stbl">'+TBL_HEAD+'<tbody>'+rows.map(rowHtml).join("")+'</tbody></table>';
  } else {
    html=SESSIONS.map(function(s,i){
      const list=buckets[i];
      const head='<div class="sesshd">'+s.name+' <span class="span">'+s.span+'</span> · '
        +'<span class="cnt">'+list.length+'</span></div>';
      if(!list.length) return head+'<div class="sessempty">— no setups in this session —</div>';
      return head+'<table class="stbl">'+TBL_HEAD+'<tbody>'+list.map(rowHtml).join("")+'</tbody></table>';
    }).join("");
  }
  $("scanRes").innerHTML=html;
  [].slice.call(document.querySelectorAll("#scanRes tr[data-sym]")).forEach(function(tr){
    tr.onclick=function(){ showStock(tr.dataset.sym); };
  });
}

$("scan").onclick=runScan;
$("day").onchange=function(){ if(LAST) renderDay($("day").value); };   // instant day-switch (no rescan)
</script>
</body></html>"""
