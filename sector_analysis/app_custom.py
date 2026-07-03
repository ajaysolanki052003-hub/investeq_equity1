"""Sector Analysis — sector-wise performance & rotation for Nifty and Sensex.

For a chosen scope (Nifty 50 / Sensex 30 / Nifty 500) we group every constituent
by its NSE Industry (sector) and compute equal-weighted returns over 1D / 1W / 1M
/ 3M / 6M / 1Y from the daily candle feed. The UI shows:

  • a ranked sector bar (best→worst on the selected timeframe),
  • a normalised (base-100) rotation chart — one line per sector + the scope
    equal-weight average, so you can see which sectors lead/lag the market,
  • a sortable table across all timeframes with advancers/decliners and the
    best & worst stock per sector,
  • drill-down into a sector → its constituents → a single-stock candle chart.

Sector map + index membership are small reference CSVs in sector_analysis/ref/
(refresh with `python -m sector_analysis.fetch_sector_data`). Prices come from
the shared daily feed (ema_scanner/data/1d/*.csv). The heavy load is cached to a
small parquet snapshot keyed on the feed's file-count + mtime, so restarts and
requests are cheap.

Benchmark note: we compare each sector to the scope's EQUAL-WEIGHT average (not
the cap-weighted index) — a clean, self-consistent relative-strength basis that
works identically for all three scopes.

Run:
    APP_BASE=/sector python -m uvicorn sector_analysis.app:app --host 127.0.0.1 --port 8713
Mounted behind nginx at /sector/ (auth-gated like the other apps).
"""

from __future__ import annotations

import csv
import glob
import os
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

APP_BASE = os.environ.get("APP_BASE", "").rstrip("/")

ROOT      = Path(__file__).resolve().parent.parent
REF_DIR   = Path(__file__).resolve().parent / "ref"           # sector/index reference csvs
DATA_DIR  = ROOT / "ema_scanner" / "data" / "1d"              # daily OHLCV feed
CACHE_DIR = ROOT / "ema_scanner" / "data" / "_sector2_cache"  # precomputed snapshot (custom universe)
CHART_TAIL_DAYS = 400        # daily bars in the single-stock chart modal

# Return windows in TRADING days (approx NSE sessions).
WINDOWS = [
    {"key": "1D", "label": "1D", "days": 1},
    {"key": "1W", "label": "1W", "days": 5},
    {"key": "1M", "label": "1M", "days": 21},
    {"key": "3M", "label": "3M", "days": 63},
    {"key": "6M", "label": "6M", "days": 126},
    {"key": "1Y", "label": "1Y", "days": 252},
]
WIN_DAYS = {w["key"]: w["days"] for w in WINDOWS}
WIN_KEYS = [w["key"] for w in WINDOWS]

# Custom universe: a user-provided Sector -> Sub-sector -> stocks map
# (sector_analysis/ref/custom_sectors.csv). "Scope" here is the grouping level:
#   scope = "all"        -> group every symbol by its Sector
#   scope = "<Sector>"   -> drill into that sector, group its symbols by Sub-sector
ALL_SCOPE = "all"
ALL_LABEL = "All Sectors"

CONFIG_VERSION = "custom-v1"

app = FastAPI(title="Sector Analysis · Custom")
_LOCK = threading.Lock()
_SNAP: dict = {"sig": None}


# ─────────────────────────────── reference data ────────────────────────────
def _read_reference():
    """Load the custom Sector/Sub-sector map.
    Returns (sec_by_sym, sub_by_sym, all_syms, sectors_list, company_by_sym)."""
    sec: dict[str, str] = {}
    sub: dict[str, str] = {}
    p = REF_DIR / "custom_sectors.csv"
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = (r.get("Symbol") or "").strip().upper()
            if not s:
                continue
            sec[s] = (r.get("Sector") or "").strip() or "Unclassified"
            sub[s] = (r.get("Subsector") or "").strip() or sec[s]
    all_syms = list(sec.keys())
    sectors_list = sorted(set(sec.values()))
    comp: dict[str, str] = {}   # custom map carries no company names; UI shows symbol
    return sec, sub, all_syms, sectors_list, comp


# ──────────────────────────────── snapshot ─────────────────────────────────
def _files() -> list[str]:
    return sorted(glob.glob(str(DATA_DIR / "*_historical.csv")))


def _sig(files: list[str]) -> str:
    mt = max((os.path.getmtime(f) for f in files), default=0)
    return f"n{len(files)}|m{int(mt)}|{CONFIG_VERSION}"


def _load_close(union: set[str]) -> pd.DataFrame:
    """Wide close-price matrix: index=date, columns=symbol (only symbols in union
    that have a candle file)."""
    series = {}
    for path in _files():
        sym = os.path.basename(path).replace("_historical.csv", "")
        if sym not in union:
            continue
        try:
            df = pd.read_csv(path, usecols=["datetime", "close"])
        except Exception:
            continue
        if df.empty:
            continue
        df["d"] = pd.to_datetime(df["datetime"]).dt.normalize()
        s = df.dropna(subset=["close"]).groupby("d")["close"].last()
        if not s.empty:
            series[sym] = s
    if not series:
        return pd.DataFrame()
    wide = pd.DataFrame(series).sort_index()
    return wide


def _returns_table(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol returns (%) over each window, plus last close/date."""
    rows = {}
    for sym in close_wide.columns:
        s = close_wide[sym].dropna()
        if len(s) < 2:
            continue
        last = float(s.iloc[-1])
        rec: dict = {"last": last, "last_date": s.index[-1].strftime("%Y-%m-%d")}
        for k, n in WIN_DAYS.items():
            if len(s) > n:
                base = float(s.iloc[-1 - n])
                rec[k] = (last / base - 1.0) * 100.0 if base > 0 else np.nan
            else:
                rec[k] = np.nan
        rows[sym] = rec
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index")


def _build() -> dict:
    sec, sub, all_syms, sectors_list, comp = _read_reference()
    close_wide = _load_close(set(all_syms))
    rets = _returns_table(close_wide)
    as_of = close_wide.index.max().strftime("%Y-%m-%d") if not close_wide.empty else None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not close_wide.empty:
        close_wide.to_parquet(CACHE_DIR / "close.parquet")
        rets.to_parquet(CACHE_DIR / "rets.parquet")

    return {
        "sec": sec, "sub": sub, "all_syms": all_syms, "sectors_list": sectors_list,
        "comp": comp, "close": close_wide, "rets": rets, "as_of": as_of,
    }


def _snapshot() -> dict:
    files = _files()
    sig = _sig(files)
    if _SNAP.get("sig") == sig:
        return _SNAP
    with _LOCK:
        if _SNAP.get("sig") == sig:
            return _SNAP
        # try disk cache first
        cw_p = CACHE_DIR / "close.parquet"
        rt_p = CACHE_DIR / "rets.parquet"
        sig_p = CACHE_DIR / "sig.txt"
        data = None
        if cw_p.exists() and rt_p.exists() and sig_p.exists() and sig_p.read_text().strip() == sig:
            try:
                close_wide = pd.read_parquet(cw_p)
                rets = pd.read_parquet(rt_p)
                sec, sub, all_syms, sectors_list, comp = _read_reference()
                as_of = close_wide.index.max().strftime("%Y-%m-%d") if not close_wide.empty else None
                data = {"sec": sec, "sub": sub, "all_syms": all_syms,
                        "sectors_list": sectors_list, "comp": comp,
                        "close": close_wide, "rets": rets, "as_of": as_of}
            except Exception:
                data = None
        if data is None:
            data = _build()
            try:
                sig_p.write_text(sig)
            except Exception:
                pass
        data["sig"] = sig
        _SNAP.clear()
        _SNAP.update(data)
        return _SNAP


# ─────────────────────────────── computation ───────────────────────────────
def _resolve_day(snap: dict, date: str | None) -> pd.Timestamp | None:
    """Snap a requested date to the last trading session on or before it. None
    means 'latest'. Returns None if the date is before the data starts."""
    close = snap["close"]
    if close.empty:
        return None
    if not date:
        return close.index.max()
    try:
        d = pd.Timestamp(date).normalize()
    except Exception:
        return close.index.max()
    sub = close.loc[:d]
    if sub.empty:
        return None
    return sub.index.max()


def _rets_asof(snap: dict, day: pd.Timestamp | None) -> pd.DataFrame:
    """Per-symbol returns anchored at `day`. Uses the cached table when `day` is
    the latest session; otherwise recomputes on the price matrix truncated to
    sessions <= day (cheap: ~440 symbols)."""
    close = snap["close"]
    if close.empty:
        return snap["rets"]
    if day is None or day >= close.index.max():
        return snap["rets"]
    return _returns_table(close.loc[:day])


def _valid_scope(snap: dict, scope: str) -> str:
    """`all` or a known sector name; anything else falls back to `all`."""
    if scope == ALL_SCOPE or scope in set(snap.get("sectors_list", [])):
        return scope
    return ALL_SCOPE


def _members(snap: dict, scope: str, rets: pd.DataFrame) -> list[str]:
    """Symbols in scope that have price data (as of `rets`). scope=all -> whole
    universe; scope=<Sector> -> only that sector's symbols."""
    sec = snap["sec"]
    have = set(rets.index) if not rets.empty else set()
    if scope == ALL_SCOPE:
        return [s for s in snap["all_syms"] if s in have and s in sec]
    return [s for s in snap["all_syms"] if s in have and sec.get(s) == scope]


def _grp_of(snap: dict, scope: str, sym: str) -> str:
    """Grouping label for a symbol: its Sector at top level, else its Sub-sector."""
    if scope == ALL_SCOPE:
        return snap["sec"].get(sym, "Unclassified")
    return snap["sub"].get(sym, "—")


def _scope_total(snap: dict, scope: str) -> int:
    if scope == ALL_SCOPE:
        return len(snap["all_syms"])
    return sum(1 for s in snap["all_syms"] if snap["sec"].get(s) == scope)


def _sector_groups(snap: dict, scope: str, rets: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for s in _members(snap, scope, rets):
        groups[_grp_of(snap, scope, s)].append(s)
    return groups


def _sector_rows(snap: dict, scope: str, rets: pd.DataFrame) -> tuple[list[dict], dict]:
    comp = snap["comp"]
    groups = _sector_groups(snap, scope, rets)
    total_in_scope = _scope_total(snap, scope)

    out = []
    for sector, members in groups.items():
        sub = rets.loc[members]
        rec = {"sector": sector, "n": len(members), "rets": {}}
        for k in WIN_KEYS:
            col = sub[k].dropna()
            rec["rets"][k] = round(float(col.mean()), 2) if len(col) else None
        # advancers / decliners + best/worst on 1D
        d = sub["1D"].dropna()
        rec["adv"] = int((d > 0).sum())
        rec["dec"] = int((d < 0).sum())
        rec["flat"] = int((d == 0).sum())
        out.append((sector, members, rec))

    # best / worst constituent per sector uses 1M (a steadier signal than 1D)
    rank_win = "1M"
    sectors = []
    for sector, members, rec in out:
        col = rets.loc[members, rank_win].dropna()
        best = worst = None
        if len(col):
            bsym = col.idxmax(); wsym = col.idxmin()
            best = {"sym": bsym, "company": comp.get(bsym, ""), "ret": round(float(col.max()), 2)}
            worst = {"sym": wsym, "company": comp.get(wsym, ""), "ret": round(float(col.min()), 2)}
        rec["best"] = best
        rec["worst"] = worst
        sectors.append(rec)

    # scope equal-weight average across ALL members (the benchmark line)
    members_all = _members(snap, scope, rets)
    scope_avg = {}
    for k in WIN_KEYS:
        col = rets.loc[members_all, k].dropna()
        scope_avg[k] = round(float(col.mean()), 2) if len(col) else None

    meta = {"scope_avg": scope_avg, "n_total": total_in_scope,
            "n_data": len(members_all), "rank_win": rank_win}
    return sectors, meta


def _series(snap: dict, scope: str, win: str, day: pd.Timestamp | None = None) -> dict:
    """Normalised (base-100) equal-weight series per sector over the window's
    lookback ending at `day`, plus the scope average. For the rotation chart."""
    close = snap["close"]
    if close.empty:
        return {"dates": [], "sectors": {}, "scope_avg": []}
    if day is not None:
        close = close.loc[:day]
    if close.empty:
        return {"dates": [], "sectors": {}, "scope_avg": []}
    rets = _rets_asof(snap, day)
    n = max(WIN_DAYS[win], 5)
    tail = close.iloc[-(n + 1):]
    # base = first valid value of each column within the tail
    base = tail.apply(lambda c: c.dropna().iloc[0] if c.dropna().size else np.nan)
    norm = tail.divide(base) * 100.0
    dates = [d.strftime("%Y-%m-%d") for d in tail.index]

    groups = _sector_groups(snap, scope, rets)
    sectors = {}
    for sector, members in groups.items():
        cols = [m for m in members if m in norm.columns]
        if not cols:
            continue
        line = norm[cols].mean(axis=1)
        sectors[sector] = [None if pd.isna(v) else round(float(v), 2) for v in line]
    members_all = [m for m in _members(snap, scope, rets) if m in norm.columns]
    savg = norm[members_all].mean(axis=1) if members_all else pd.Series(dtype=float)
    scope_avg = [None if pd.isna(v) else round(float(v), 2) for v in savg]
    return {"dates": dates, "sectors": sectors, "scope_avg": scope_avg}


# ──────────────────────────────── endpoints ────────────────────────────────
@app.on_event("startup")
def _warm():
    threading.Thread(target=_snapshot, daemon=True).start()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/meta")
def api_meta():
    snap = _snapshot()
    rets_latest = snap["rets"]
    scopes = [{"key": ALL_SCOPE, "label": ALL_LABEL,
               "total": _scope_total(snap, ALL_SCOPE),
               "with_data": len(_members(snap, ALL_SCOPE, rets_latest))}]
    for sec in snap.get("sectors_list", []):
        scopes.append({"key": sec, "label": sec,
                       "total": _scope_total(snap, sec),
                       "with_data": len(_members(snap, sec, rets_latest))})
    close = snap["close"]
    min_date = close.index.min().strftime("%Y-%m-%d") if not close.empty else None
    return JSONResponse({
        "scopes": scopes,
        "windows": WINDOWS,
        "as_of": snap["as_of"],
        "min_date": min_date,
        "max_date": snap["as_of"],
    })


@app.get("/api/sectors")
def api_sectors(scope: str = ALL_SCOPE, win: str = "1M", date: str = ""):
    if win not in WIN_DAYS:
        win = "1M"
    snap = _snapshot()
    scope = _valid_scope(snap, scope)
    day = _resolve_day(snap, date or None)
    if day is None:
        return JSONResponse({"scope": scope, "win": win, "sectors": [], "scope_avg": {},
                             "n_total": _scope_total(snap, scope), "n_data": 0,
                             "as_of": None, "rank_win": "1M",
                             "series": {"dates": [], "sectors": {}, "scope_avg": []}})
    rets = _rets_asof(snap, day)
    sectors, meta = _sector_rows(snap, scope, rets)
    series = _series(snap, scope, win, day)
    # sort by the selected window return (desc, Nones last)
    sectors.sort(key=lambda r: (r["rets"].get(win) is None, -(r["rets"].get(win) or 0)))
    return JSONResponse({
        "scope": scope, "win": win, "sectors": sectors,
        "scope_avg": meta["scope_avg"], "n_total": meta["n_total"],
        "n_data": meta["n_data"], "as_of": day.strftime("%Y-%m-%d"),
        "rank_win": meta["rank_win"], "series": series,
    })


@app.get("/api/constituents")
def api_constituents(scope: str = ALL_SCOPE, sector: str = "", win: str = "1M", date: str = ""):
    if win not in WIN_DAYS:
        win = "1M"
    snap = _snapshot()
    scope = _valid_scope(snap, scope)
    day = _resolve_day(snap, date or None)
    rets = _rets_asof(snap, day)
    comp = snap["comp"]
    # `sector` here is the clicked GROUP label (a Sector at top level, a
    # Sub-sector when drilled into one sector).
    members = [m for m in _members(snap, scope, rets) if _grp_of(snap, scope, m) == sector]
    rows = []
    for s in members:
        r = rets.loc[s]
        rows.append({
            "sym": s, "company": comp.get(s, ""),
            "last": round(float(r["last"]), 2), "last_date": r["last_date"],
            "rets": {k: (None if pd.isna(r[k]) else round(float(r[k]), 2)) for k in WIN_KEYS},
            "ret": (None if pd.isna(r[win]) else round(float(r[win]), 2)),
        })
    rows.sort(key=lambda x: (x["ret"] is None, -(x["ret"] or 0)))
    as_of = day.strftime("%Y-%m-%d") if day is not None else None
    return JSONResponse({"scope": scope, "sector": sector, "win": win,
                         "as_of": as_of, "rows": rows})


@app.get("/api/chart")
def api_chart(sym: str = "", date: str = ""):
    snap = _snapshot()
    sec, comp = snap["sec"], snap["comp"]
    path = DATA_DIR / f"{sym}_historical.csv"
    if not path.exists():
        return JSONResponse({"error": "no data", "sym": sym}, status_code=404)
    df = pd.read_csv(path, usecols=["datetime", "open", "high", "low", "close"])
    df["dt"] = pd.to_datetime(df["datetime"])
    df["d"] = df["dt"].dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if date:  # end the chart at the selected day (inclusive)
        try:
            df = df[df["dt"].dt.normalize() <= pd.Timestamp(date).normalize()]
        except Exception:
            pass
    df = df.tail(CHART_TAIL_DAYS)
    candles = [{"time": r.d, "open": float(r.open), "high": float(r.high),
                "low": float(r.low), "close": float(r.close)}
               for r in df.itertuples()]
    return JSONResponse({"sym": sym, "company": comp.get(sym, ""),
                         "sector": sec.get(sym, ""), "candles": candles})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML.replace("__APP_BASE__", APP_BASE)


# ──────────────────────────────────── UI ───────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Sector Analysis</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{
    --bg:#0a0e17; --panel:#111726; --panel2:#0d1320; --line:#1e2940;
    --txt:#e6edf6; --mut:#8a98b2; --accent:#a78bfa; --accent2:#22d3ee;
    --pos:#34d399; --neg:#f87171; --chip:#172033;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#231b40 0%,var(--bg) 55%);
       color:var(--txt);font:14px/1.5 'Inter',system-ui,Segoe UI,Roboto,sans-serif;min-height:100vh}
  .wrap{max-width:1340px;margin:0 auto;padding:28px 22px 60px}
  header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:20px}
  .title{font-size:26px;font-weight:800;letter-spacing:.2px;display:flex;align-items:center;gap:12px}
  .title .dot{width:11px;height:11px;border-radius:50%;background:var(--accent);box-shadow:0 0 16px 2px var(--accent)}
  .sub{color:var(--mut);font-size:13px;margin-top:4px;max-width:720px}
  .asof{color:var(--mut);font-size:12.5px;text-align:right}
  .asof b{color:var(--txt)}
  .toolbar{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
  .grp{display:flex;flex-direction:column;gap:5px}
  .grp .lab{font-size:10.5px;text-transform:uppercase;letter-spacing:.12em;color:var(--mut)}
  .seg{display:inline-flex;background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:3px;flex-wrap:wrap}
  .seg button{background:none;border:0;color:var(--mut);padding:7px 14px;border-radius:8px;cursor:pointer;font-weight:600;font-size:12.5px}
  .seg button.on{background:var(--accent);color:#0a0a18}
  .seg button .c{opacity:.6;font-weight:500;font-size:11px;margin-left:5px}
  .seg button.on .c{opacity:.8}
  .daterow{display:flex;gap:6px;align-items:center}
  input[type=date]{background:var(--chip);border:1px solid var(--line);color:var(--txt);
        border-radius:9px;padding:7px 10px;font-size:12.5px;outline:none;color-scheme:dark}
  input[type=date]:focus{border-color:var(--accent)}
  .selctl{background:var(--chip);border:1px solid var(--line);color:var(--txt);border-radius:9px;
        padding:7px 10px;font-size:12.5px;outline:none;color-scheme:dark;min-width:190px;max-width:300px}
  .selctl:focus{border-color:var(--accent)}
  .mini{background:var(--chip);border:1px solid var(--line);color:var(--mut);border-radius:9px;
        padding:7px 11px;cursor:pointer;font-size:12px;font-weight:600}
  .mini:hover{border-color:var(--accent2);color:var(--accent2)}
  .card{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
        border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:20px}
  .card h3{margin:0 0 14px;font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--mut);
           display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .card h3 .pill{background:var(--chip);border:1px solid var(--line);border-radius:999px;padding:2px 10px;
                 font-size:11px;letter-spacing:.04em;color:var(--txt);text-transform:none}
  .two{display:grid;grid-template-columns:1.05fr 1fr;gap:20px}
  @media(max-width:1000px){.two{grid-template-columns:1fr}}
  /* ranked bars */
  .bars{display:flex;flex-direction:column;gap:7px}
  .bar{display:grid;grid-template-columns:150px 1fr 66px;align-items:center;gap:10px;cursor:pointer}
  .bar:hover .nm{color:var(--accent2)}
  .bar .nm{font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .bar .track{position:relative;height:22px;background:var(--panel2);border:1px solid var(--line);border-radius:6px;overflow:hidden}
  .bar .fill{position:absolute;top:0;bottom:0;border-radius:5px}
  .bar .mid{position:absolute;top:0;bottom:0;width:1px;left:50%;background:#39466a}
  .bar .val{text-align:right;font-weight:700;font-variant-numeric:tabular-nums;font-size:12.5px}
  #legend{display:flex;flex-wrap:wrap;gap:8px 14px;margin-top:12px;max-height:96px;overflow:auto}
  .lg{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--mut);cursor:pointer;user-select:none}
  .lg.off{opacity:.35}
  .lg .sw{width:14px;height:3px;border-radius:2px}
  #rotChart{height:340px}
  .resbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;flex-wrap:wrap;gap:10px}
  .count{font-size:14px;color:var(--mut)}
  .count b{color:var(--txt)}
  .dl{background:var(--chip);border:1px solid var(--line);color:var(--txt);border-radius:9px;padding:7px 13px;cursor:pointer;font-size:12.5px}
  .dl:hover{border-color:var(--accent2);color:var(--accent2)}
  .tablewrap{overflow:auto;border:1px solid var(--line);border-radius:12px}
  table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
  th,td{padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
  th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--panel)}
  th{position:sticky;top:0;background:var(--panel2);color:var(--mut);font-size:11px;text-transform:uppercase;
     letter-spacing:.06em;z-index:2;cursor:pointer}
  th:first-child{z-index:3}
  th.on{color:var(--accent2)}
  tbody tr{cursor:pointer}
  tbody tr:hover{background:#0f1830}
  td.sec{font-weight:700}
  td.sec .n{color:var(--mut);font-weight:500;font-size:11px;margin-left:6px}
  .num{font-weight:600}
  .pos{color:var(--pos)} .neg{color:var(--neg)} .zero{color:var(--mut)}
  .stock{color:var(--txt)}
  .stock small{color:var(--mut)}
  .modal{position:fixed;inset:0;background:rgba(4,7,14,.74);backdrop-filter:blur(2px);
         display:none;align-items:center;justify-content:center;z-index:50}
  .modal.show{display:flex}
  .sheet{width:min(1080px,94vw);max-height:88vh;background:var(--panel);
         border:1px solid var(--line);border-radius:16px;display:flex;flex-direction:column;overflow:hidden}
  .sheethd{display:flex;align-items:center;gap:14px;padding:13px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
  .sheethd .nm{font-size:17px;font-weight:800}
  .sheethd .dd{color:var(--mut);font-size:12.5px}
  .xbtn{background:var(--chip);border:1px solid var(--line);color:var(--txt);border-radius:9px;
        width:30px;height:30px;cursor:pointer;font-size:16px;line-height:1;margin-left:auto}
  .xbtn:hover{border-color:var(--neg);color:var(--neg)}
  .sheetbody{padding:6px 4px;overflow:auto}
  #chartBox{height:min(64vh,560px)}
  .empty{padding:48px;text-align:center;color:var(--mut)}
  .spin{padding:40px;text-align:center;color:var(--mut)}
  .pie-wrap{display:flex;gap:28px;align-items:center;flex-wrap:wrap;justify-content:center}
  #pie svg{width:236px;height:236px}
  #pie .seg{cursor:pointer;transition:opacity .12s}
  .pie-legend{flex:1 1 340px;display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));gap:6px 18px;min-width:280px}
  .pl{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--mut)}
  .pl .sw{width:11px;height:11px;border-radius:3px;flex:0 0 auto}
  .pl .nm{color:var(--txt);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .pl .pc{margin-left:auto;font-variant-numeric:tabular-nums;white-space:nowrap}
  .quad-wrap{display:flex;gap:26px;align-items:center;flex-wrap:wrap;justify-content:center}
  #quad svg{width:460px;max-width:100%;height:auto}
  #quad circle{cursor:pointer}
  .quad-key{flex:1 1 230px;display:flex;flex-direction:column;gap:9px;font-size:12.5px;color:var(--mut);min-width:220px}
  .quad-key b{font-weight:700}
  .quad-key .qnote{font-size:11.5px;margin-top:2px;border-top:1px solid var(--line);padding-top:9px;line-height:1.5}
  .heat{display:grid;gap:2px;min-width:660px}
  .heat .hc{padding:6px 3px;text-align:center;font-size:11px;font-variant-numeric:tabular-nums;border-radius:3px;font-weight:700}
  .heat .hh{color:var(--mut);font-size:10px;text-transform:uppercase;letter-spacing:.06em;padding:4px;text-align:center;font-weight:600}
  .heat .hn{text-align:left;color:var(--txt);font-size:11.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:6px 8px 6px 2px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div class="title"><span class="dot"></span> Sector Analysis · Custom</div>
      <div class="sub">Your custom Sector → Sub-sector universe — equal-weight returns from the daily feed. Shows <b>All Sectors</b> by default; pick a sector to drill into its sub-sectors. Click any bar to see its stocks.</div>
    </div>
    <div class="asof">As of <b id="asof">—</b><br><span id="cov">—</span></div>
  </header>

  <div class="toolbar">
    <div class="grp">
      <span class="lab">Sector</span>
      <select class="selctl" id="scopeSel"></select>
    </div>
    <div class="grp">
      <span class="lab">Timeframe</span>
      <div class="seg" id="winSeg"></div>
    </div>
    <div class="grp">
      <span class="lab">As-of date</span>
      <div class="daterow">
        <input type="date" id="datePick"/>
        <button class="mini" id="latestBtn" title="Jump to the latest session">Latest</button>
      </div>
    </div>
  </div>

  <div class="two">
    <div class="card">
      <h3>Sector ranking <span class="pill" id="rankPill">1M</span></h3>
      <div class="bars" id="bars"><div class="spin">Loading…</div></div>
    </div>
    <div class="card">
      <h3>Rotation — normalised to 100 <span class="pill" id="rotPill">1M</span></h3>
      <div id="rotChart"></div>
      <div id="legend"></div>
    </div>
  </div>

  <div class="card">
    <h3>Universe composition <span class="pill">stocks per group</span></h3>
    <div class="pie-wrap"><div id="pie"></div><div class="pie-legend" id="pieLegend"></div></div>
  </div>

  <div class="card">
    <h3>Momentum quadrant <span class="pill">1M vs 3M return</span></h3>
    <div class="quad-wrap">
      <div id="quad"></div>
      <div class="quad-key">
        <div><b style="color:var(--pos)">Leading</b> — strong 1M &amp; 3M</div>
        <div><b style="color:#60a5fa">Improving</b> — 1M up, 3M still down</div>
        <div><b style="color:#fbbf24">Weakening</b> — 1M down, 3M up</div>
        <div><b style="color:var(--neg)">Lagging</b> — weak on both</div>
        <div class="qnote">Each dot is a group; size = number of stocks. Top-right leads; bottom-right is turning up. Hover for details.</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h3>Return heatmap <span class="pill">group × timeframe</span> <span class="pill">sorted by 1M</span></h3>
    <div class="tablewrap"><div id="heat"></div></div>
  </div>

  <div class="card">
    <div class="resbar">
      <div class="count"><b id="secCount">0</b> groups · <span id="scopeName">—</span> · avg <span id="scopeAvg">—</span></div>
      <button class="dl" id="dlBtn">⭳ CSV</button>
    </div>
    <div class="tablewrap">
      <table id="tbl">
        <thead><tr id="thead"></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- sector drilldown -->
<div class="modal" id="secModal">
  <div class="sheet">
    <div class="sheethd">
      <div><div class="nm" id="secNm">—</div><div class="dd" id="secDd">—</div></div>
      <button class="xbtn" onclick="closeModal('secModal')">✕</button>
    </div>
    <div class="sheetbody">
      <div class="tablewrap" style="border:0">
        <table id="conTbl"><thead><tr id="conHead"></tr></thead><tbody id="conBody"></tbody></table>
      </div>
    </div>
  </div>
</div>

<!-- single-stock chart -->
<div class="modal" id="chModal">
  <div class="sheet">
    <div class="sheethd">
      <div><div class="nm" id="chNm">—</div><div class="dd" id="chDd">—</div></div>
      <button class="xbtn" onclick="closeModal('chModal')">✕</button>
    </div>
    <div class="sheetbody"><div id="chartBox"></div></div>
  </div>
</div>

<script>
const BASE="__APP_BASE__";
const WINS=["1D","1W","1M","3M","6M","1Y"];
let META=null, STATE={scope:"all", win:"1M", date:""}, DATA=null, SORT={key:"win", dir:-1};
let chart=null, candleSeries=null, rotChart=null, rotSeries={}, hidden={};

const PALETTE=["#a78bfa","#22d3ee","#34d399","#f472b6","#fbbf24","#60a5fa","#f87171",
  "#c084fc","#2dd4bf","#a3e635","#fb923c","#38bdf8","#e879f9","#4ade80","#facc15",
  "#818cf8","#fca5a5","#5eead4","#fdba74","#93c5fd"];
const colorFor={};
function assignColors(names){names.forEach((n,i)=>{colorFor[n]=PALETTE[i%PALETTE.length];});}

function fmtPct(v){ if(v===null||v===undefined||isNaN(v)) return "—"; return (v>=0?"+":"")+v.toFixed(2)+"%"; }
function cls(v){ if(v===null||v===undefined||isNaN(v)) return "zero"; return v>0?"pos":(v<0?"neg":"zero"); }

async function jget(u){ const r=await fetch(BASE+u); if(!r.ok) throw new Error(r.status); return r.json(); }

async function init(){
  META=await jget("/api/meta");
  document.getElementById("asof").textContent=META.as_of||"—";
  const ss=document.getElementById("scopeSel");
  ss.innerHTML=META.scopes.map(s=>`<option value="${s.key}" ${s.key===STATE.scope?'selected':''}>${s.label} (${s.with_data}/${s.total})</option>`).join("");
  ss.onchange=()=>{STATE.scope=ss.value;SORT={key:"win",dir:-1};load();};
  const ws=document.getElementById("winSeg");
  ws.innerHTML=META.windows.map(w=>`<button data-k="${w.key}" class="${w.key===STATE.win?'on':''}">${w.label}</button>`).join("");
  ws.querySelectorAll("button").forEach(b=>b.onclick=()=>{STATE.win=b.dataset.k;syncSeg(ws,b);SORT={key:"win",dir:-1};load();});
  const dp=document.getElementById("datePick");
  if(META.min_date)dp.min=META.min_date;
  if(META.max_date)dp.max=META.max_date;
  dp.value=META.max_date||"";
  STATE.date=dp.value;
  dp.onchange=()=>{STATE.date=dp.value;load();};
  document.getElementById("latestBtn").onclick=()=>{dp.value=META.max_date||"";STATE.date=dp.value;load();};
  buildRotChart();
  load();
}
function syncSeg(seg,btn){ seg.querySelectorAll("button").forEach(x=>x.classList.remove("on")); btn.classList.add("on"); }

async function load(){
  document.getElementById("bars").innerHTML='<div class="spin">Loading…</div>';
  DATA=await jget(`/api/sectors?scope=${STATE.scope}&win=${STATE.win}&date=${STATE.date}`);
  document.getElementById("asof").textContent=DATA.as_of||"—";
  // snap the picker to the actual trading session used (weekend/holiday → prior)
  if(DATA.as_of){const dp=document.getElementById("datePick");dp.value=DATA.as_of;STATE.date=DATA.as_of;}
  const sc=META.scopes.find(s=>s.key===STATE.scope);
  document.getElementById("scopeName").textContent=sc?sc.label:STATE.scope;
  document.getElementById("cov").textContent=`${DATA.n_data} of ${DATA.n_total} constituents priced`;
  document.getElementById("rankPill").textContent=STATE.win;
  document.getElementById("rotPill").textContent=STATE.win;
  document.getElementById("secCount").textContent=DATA.sectors.length;
  const sa=DATA.scope_avg[STATE.win];
  const saEl=document.getElementById("scopeAvg"); saEl.textContent=fmtPct(sa); saEl.className=cls(sa);
  assignColors(DATA.sectors.map(s=>s.sector));
  renderBars(); renderTable(); renderRotation();
  renderPie(); renderQuadrant(); renderHeatmap();
}

const PIE_PAL=["#a78bfa","#22d3ee","#34d399","#f472b6","#fbbf24","#60a5fa","#f87171","#c084fc","#2dd4bf","#fb923c","#818cf8","#4ade80","#e879f9"];
function renderPie(){
  const arr=DATA.sectors.map(s=>({name:s.sector,n:s.n})).sort((a,b)=>b.n-a.n);
  const TOPN=13, top=arr.slice(0,TOPN), rest=arr.slice(TOPN);
  const restN=rest.reduce((a,b)=>a+b.n,0);
  const slices=top.map((s,i)=>({name:s.name,value:s.n,color:PIE_PAL[i%PIE_PAL.length]}));
  if(restN>0) slices.push({name:`Other (${rest.length} groups)`,value:restN,color:"#64748b"});
  const total=slices.reduce((a,b)=>a+b.value,0)||1;
  const r=70,C=2*Math.PI*r,cx=90,cy=90; let cum=0;
  const segs=slices.map(s=>{const f=s.value/total,dash=f*C;
    const el=`<circle class="seg" cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${s.color}" stroke-width="30" stroke-dasharray="${dash.toFixed(2)} ${(C-dash).toFixed(2)}" transform="rotate(${(cum*360-90).toFixed(2)} ${cx} ${cy})" data-v="${s.value}" data-p="${(f*100).toFixed(1)}"><title>${s.name}: ${s.value} (${(f*100).toFixed(1)}%)</title></circle>`;
    cum+=f; return el;}).join("");
  document.getElementById("pie").innerHTML=`<svg viewBox="0 0 180 180" role="img" aria-label="Stocks per group">${segs}<text id="pieC1" x="90" y="86" text-anchor="middle" fill="#e6edf6" font-size="23" font-weight="800">${total}</text><text id="pieC2" x="90" y="103" text-anchor="middle" fill="#8a98b2" font-size="10">priced stocks</text></svg>`;
  document.getElementById("pieLegend").innerHTML=slices.map(s=>`<span class="pl"><span class="sw" style="background:${s.color}"></span><span class="nm" title="${s.name}">${s.name}</span><span class="pc" style="color:var(--mut)">${s.value} · ${(s.value/total*100).toFixed(1)}%</span></span>`).join("");
  const c1=document.getElementById("pieC1"),c2=document.getElementById("pieC2");
  document.querySelectorAll("#pie .seg").forEach(seg=>{
    seg.onmouseenter=()=>{document.querySelectorAll("#pie .seg").forEach(x=>x.style.opacity=x===seg?"1":".3");c1.textContent=seg.dataset.v;c2.textContent=seg.dataset.p+"% of priced";};
    seg.onmouseleave=()=>{document.querySelectorAll("#pie .seg").forEach(x=>x.style.opacity="1");c1.textContent=total;c2.textContent="priced stocks";};
  });
}
function renderQuadrant(){
  const pts=DATA.sectors.filter(s=>s.rets["1M"]!=null&&s.rets["3M"]!=null).map(s=>({name:s.sector,x:s.rets["1M"],y:s.rets["3M"],n:s.n}));
  const el=document.getElementById("quad");
  if(!pts.length){el.innerHTML='<div class="empty">Need 1M &amp; 3M returns</div>';return;}
  const W=460,H=360,m=36;
  const xmax=Math.max(5,...pts.map(p=>Math.abs(p.x)))*1.12, ymax=Math.max(5,...pts.map(p=>Math.abs(p.y)))*1.12;
  const px=x=>m+((x+xmax)/(2*xmax))*(W-2*m), py=y=>m+(1-(y+ymax)/(2*ymax))*(H-2*m);
  const nmax=Math.max(1,...pts.map(p=>p.n)), rad=n=>4+Math.sqrt(n/nmax)*9;
  const x0=px(0),y0=py(0);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Momentum quadrant 1M vs 3M">`;
  s+=`<rect x="${x0}" y="${m}" width="${W-m-x0}" height="${y0-m}" fill="rgba(52,211,153,.055)"/><rect x="${m}" y="${m}" width="${x0-m}" height="${y0-m}" fill="rgba(251,191,36,.05)"/><rect x="${m}" y="${y0}" width="${x0-m}" height="${H-m-y0}" fill="rgba(248,113,113,.055)"/><rect x="${x0}" y="${y0}" width="${W-m-x0}" height="${H-m-y0}" fill="rgba(96,165,250,.055)"/>`;
  s+=`<line x1="${m}" y1="${y0.toFixed(1)}" x2="${W-m}" y2="${y0.toFixed(1)}" stroke="#39466a"/><line x1="${x0.toFixed(1)}" y1="${m}" x2="${x0.toFixed(1)}" y2="${H-m}" stroke="#39466a"/>`;
  s+=`<text x="${W-m-4}" y="${m+11}" text-anchor="end" fill="#34d399" font-size="10" font-weight="700">LEADING</text><text x="${m+4}" y="${m+11}" fill="#fbbf24" font-size="10" font-weight="700">WEAKENING</text><text x="${m+4}" y="${H-m-5}" fill="#f87171" font-size="10" font-weight="700">LAGGING</text><text x="${W-m-4}" y="${H-m-5}" text-anchor="end" fill="#60a5fa" font-size="10" font-weight="700">IMPROVING</text>`;
  s+=`<text x="${W-m}" y="${(y0-5).toFixed(1)}" text-anchor="end" fill="#5c6b8a" font-size="9">1M %  →</text><text x="${(x0+5).toFixed(1)}" y="${m+9}" fill="#5c6b8a" font-size="9">↑ 3M %</text>`;
  pts.forEach(p=>{const c=p.x>=0?(p.y>=0?"#34d399":"#60a5fa"):(p.y>=0?"#fbbf24":"#f87171");
    s+=`<circle cx="${px(p.x).toFixed(1)}" cy="${py(p.y).toFixed(1)}" r="${rad(p.n).toFixed(1)}" fill="${c}" fill-opacity="0.5" stroke="${c}" stroke-width="1"><title>${p.name} — 1M ${p.x.toFixed(1)}%, 3M ${p.y.toFixed(1)}% · ${p.n} stocks</title></circle>`;});
  pts.slice().sort((a,b)=>(Math.abs(b.x)+Math.abs(b.y))-(Math.abs(a.x)+Math.abs(a.y))).slice(0,4).forEach(p=>{
    s+=`<text x="${(px(p.x)+rad(p.n)+3).toFixed(1)}" y="${(py(p.y)+3).toFixed(1)}" fill="#c7d2f0" font-size="9">${p.name.length>15?p.name.slice(0,14)+'…':p.name}</text>`;});
  el.innerHTML=s+`</svg>`;
}
function renderHeatmap(){
  const WK=(META.windows||[]).map(w=>w.key);
  const rows=DATA.sectors.slice().sort((a,b)=>{const x=a.rets["1M"],y=b.rets["1M"];if(x==null)return 1;if(y==null)return -1;return y-x;});
  const cap=18;
  const cell=v=>{if(v==null||isNaN(v))return `<div class="hc" style="background:#141d31;color:#5c6b8a;font-weight:500">—</div>`;
    const a=Math.min(0.92,Math.abs(v)/cap*0.85+0.1);
    const bg=v>=0?`rgba(52,211,153,${a.toFixed(2)})`:`rgba(248,113,113,${a.toFixed(2)})`;
    const fg=a>0.42?"#0a0e17":(v>=0?"#8ff0c8":"#ffb4b4");
    return `<div class="hc" style="background:${bg};color:${fg}">${(v>=0?"+":"")+v.toFixed(1)}</div>`;};
  let h=`<div class="heat" style="grid-template-columns:minmax(150px,1.4fr) repeat(${WK.length},1fr)"><div class="hh" style="text-align:left">Group</div>`+WK.map(w=>`<div class="hh">${w}</div>`).join("");
  rows.forEach(r=>{h+=`<div class="hn" title="${r.sector}">${r.sector} <small style="color:#5c6b8a">${r.n}</small></div>`+WK.map(w=>cell(r.rets[w])).join("");});
  document.getElementById("heat").innerHTML=h+`</div>`;
}

function renderBars(){
  const rows=DATA.sectors.slice();
  const vals=rows.map(r=>r.rets[STATE.win]).filter(v=>v!==null&&!isNaN(v));
  const mx=Math.max(1,...vals.map(Math.abs));
  const avg=DATA.scope_avg[STATE.win];
  const el=document.getElementById("bars");
  el.innerHTML=rows.map(r=>{
    const v=r.rets[STATE.win];
    const w=v===null?0:(Math.abs(v)/mx)*50;
    const left=v>=0?50:50-w, col=v>=0?"var(--pos)":"var(--neg)";
    return `<div class="bar" onclick="openSector('${r.sector.replace(/'/g,"\\'")}')">
      <div class="nm" title="${r.sector}">${r.sector}</div>
      <div class="track"><div class="mid"></div><div class="fill" style="left:${left}%;width:${w}%;background:${col}"></div></div>
      <div class="val ${cls(v)}">${fmtPct(v)}</div></div>`;
  }).join("");
}

function renderTable(){
  const head=document.getElementById("thead");
  const cols=[["sector","Sector"],...WINS.map(w=>[w,w]),["adv","Adv/Dec"],["best","Best (1M)"],["worst","Worst (1M)"]];
  head.innerHTML=cols.map(([k,l])=>{
    const on=(k===STATE.win&&SORT.key==="win")||SORT.key===k;
    return `<th data-k="${k}" class="${on?'on':''}">${l}${on?(SORT.dir<0?' ▾':' ▴'):''}</th>`;
  }).join("");
  head.querySelectorAll("th").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k;
    if(k==="best"||k==="worst")return;
    if((k===STATE.win&&SORT.key==="win")||SORT.key===k){SORT.dir*=-1;}
    else{SORT={key:(k===STATE.win?"win":k),dir:-1};}
    renderTable();
  });
  let rows=DATA.sectors.slice();
  const sk=SORT.key==="win"?STATE.win:SORT.key;
  rows.sort((a,b)=>{
    let av,bv;
    if(sk==="sector"){av=a.sector;bv=b.sector;return SORT.dir*(av<bv?-1:av>bv?1:0);}
    if(sk==="adv"){av=a.adv-a.dec;bv=b.adv-b.dec;}
    else{av=a.rets[sk];bv=b.rets[sk];}
    if(av===null)return 1; if(bv===null)return -1;
    return SORT.dir*(av-bv);
  });
  const body=document.getElementById("tbody");
  body.innerHTML=rows.map(r=>{
    const wins=WINS.map(w=>`<td class="num ${cls(r.rets[w])}">${fmtPct(r.rets[w])}</td>`).join("");
    const best=r.best?`<span class="stock">${r.best.sym} <small>${fmtPct(r.best.ret)}</small></span>`:"—";
    const worst=r.worst?`<span class="stock">${r.worst.sym} <small>${fmtPct(r.worst.ret)}</small></span>`:"—";
    return `<tr onclick="openSector('${r.sector.replace(/'/g,"\\'")}')">
      <td class="sec"><span style="color:${colorFor[r.sector]}">●</span> ${r.sector}<span class="n">${r.n}</span></td>
      ${wins}
      <td class="num"><span class="pos">${r.adv}</span>/<span class="neg">${r.dec}</span></td>
      <td>${best}</td><td>${worst}</td></tr>`;
  }).join("");
}

/* ---- rotation chart ---- */
function buildRotChart(){
  const box=document.getElementById("rotChart");
  rotChart=LightweightCharts.createChart(box,{
    layout:{background:{color:"transparent"},textColor:"#8a98b2"},
    grid:{vertLines:{color:"#16203a"},horzLines:{color:"#16203a"}},
    rightPriceScale:{borderColor:"#1e2940"},timeScale:{borderColor:"#1e2940"},
    crosshair:{mode:0}, height:340,
  });
  new ResizeObserver(()=>rotChart.applyOptions({width:box.clientWidth})).observe(box);
}
function renderRotation(){
  Object.values(rotSeries).forEach(s=>rotChart.removeSeries(s)); rotSeries={};
  const S=DATA.series; const dates=S.dates;
  // scope average — thick white dashed
  const avgSer=rotChart.addLineSeries({color:"#e6edf6",lineWidth:2,lineStyle:2,priceLineVisible:false,lastValueVisible:false});
  avgSer.setData(dates.map((d,i)=>({time:d,value:S.scope_avg[i]})).filter(p=>p.value!==null));
  rotSeries["__avg"]=avgSer;
  Object.keys(S.sectors).forEach(name=>{
    const ser=rotChart.addLineSeries({color:colorFor[name]||"#888",lineWidth:2,priceLineVisible:false,lastValueVisible:false});
    ser.setData(dates.map((d,i)=>({time:d,value:S.sectors[name][i]})).filter(p=>p.value!==null));
    ser.applyOptions({visible:!hidden[name]});
    rotSeries[name]=ser;
  });
  rotChart.timeScale().fitContent();
  renderLegend();
}
function renderLegend(){
  const names=Object.keys(DATA.series.sectors);
  const el=document.getElementById("legend");
  el.innerHTML=`<div class="lg" style="color:var(--txt)"><span class="sw" style="background:#e6edf6"></span>Scope avg</div>`+
    names.map(n=>`<div class="lg ${hidden[n]?'off':''}" data-n="${n}"><span class="sw" style="background:${colorFor[n]}"></span>${n}</div>`).join("");
  el.querySelectorAll(".lg[data-n]").forEach(g=>g.onclick=()=>{
    const n=g.dataset.n; hidden[n]=!hidden[n];
    if(rotSeries[n])rotSeries[n].applyOptions({visible:!hidden[n]});
    g.classList.toggle("off",hidden[n]);
  });
}

/* ---- drilldown ---- */
async function openSector(sector){
  document.getElementById("secNm").textContent=sector;
  document.getElementById("secDd").textContent="Loading…";
  document.getElementById("conBody").innerHTML="";
  showModal("secModal");
  const d=await jget(`/api/constituents?scope=${STATE.scope}&sector=${encodeURIComponent(sector)}&win=${STATE.win}&date=${STATE.date}`);
  const sc=META.scopes.find(s=>s.key===STATE.scope);
  document.getElementById("secDd").textContent=`${d.rows.length} stocks · ${sc?sc.label:STATE.scope} · as of ${d.as_of||STATE.date} · sorted by ${STATE.win}`;
  const head=document.getElementById("conHead");
  head.innerHTML=`<th>Stock</th>`+WINS.map(w=>`<th class="${w===STATE.win?'on':''}">${w}</th>`).join("")+`<th>Last</th>`;
  const body=document.getElementById("conBody");
  body.innerHTML=d.rows.map(r=>{
    const wins=WINS.map(w=>`<td class="num ${cls(r.rets[w])}">${fmtPct(r.rets[w])}</td>`).join("");
    return `<tr onclick="openChart('${r.sym}')"><td class="sec">${r.sym} <span class="n">${r.company||""}</span></td>${wins}<td class="num">${r.last}</td></tr>`;
  }).join("")||`<tr><td colspan="8" class="empty">No priced constituents.</td></tr>`;
}

/* ---- single-stock chart ---- */
async function openChart(sym){
  document.getElementById("chNm").textContent=sym;
  document.getElementById("chDd").textContent="Loading…";
  showModal("chModal");
  const d=await jget(`/api/chart?sym=${encodeURIComponent(sym)}&date=${STATE.date}`);
  document.getElementById("chNm").textContent=sym;
  document.getElementById("chDd").textContent=`${d.company||""} · ${d.sector||""}`;
  const box=document.getElementById("chartBox");
  if(!chart){
    chart=LightweightCharts.createChart(box,{
      layout:{background:{color:"transparent"},textColor:"#8a98b2"},
      grid:{vertLines:{color:"#16203a"},horzLines:{color:"#16203a"}},
      rightPriceScale:{borderColor:"#1e2940"},timeScale:{borderColor:"#1e2940"},
      height:Math.min(window.innerHeight*0.64,560),
    });
    candleSeries=chart.addCandlestickSeries({upColor:"#34d399",downColor:"#f87171",
      wickUpColor:"#34d399",wickDownColor:"#f87171",borderVisible:false});
    new ResizeObserver(()=>chart.applyOptions({width:box.clientWidth})).observe(box);
  }
  candleSeries.setData(d.candles);
  chart.timeScale().fitContent();
  chart.applyOptions({width:box.clientWidth});
}

function showModal(id){document.getElementById(id).classList.add("show");}
function closeModal(id){document.getElementById(id).classList.remove("show");}
document.querySelectorAll(".modal").forEach(m=>m.addEventListener("click",e=>{if(e.target===m)m.classList.remove("show");}));
document.addEventListener("keydown",e=>{if(e.key==="Escape")document.querySelectorAll(".modal.show").forEach(m=>m.classList.remove("show"));});

document.getElementById("dlBtn").onclick=()=>{
  if(!DATA)return;
  const hdr=["Sector","Stocks",...WINS,"Adv","Dec","Best_1M","Best_ret","Worst_1M","Worst_ret"];
  const lines=[hdr.join(",")];
  DATA.sectors.forEach(r=>{
    lines.push([`"${r.sector}"`,r.n,...WINS.map(w=>r.rets[w]??""),r.adv,r.dec,
      r.best?r.best.sym:"",r.best?r.best.ret:"",r.worst?r.worst.sym:"",r.worst?r.worst.ret:""].join(","));
  });
  const blob=new Blob([lines.join("\n")],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download=`sectors_${STATE.scope}_${STATE.win}_${STATE.date||'latest'}.csv`;a.click();
};

init();
</script>
</body>
</html>
"""
