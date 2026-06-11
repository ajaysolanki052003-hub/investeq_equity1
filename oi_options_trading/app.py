"""OI-driven options trading — strategy suite (FastAPI).

For now the suite ships the NIFTY index chart visualization. Strategies will
plug in over time (OI-skew screen, max-pain rebalancer, etc.).

Data source: the continuous 1-min NIFTY master parquet maintained by
fetch_nifty_master.py. Resampling to 1m/5m/15m/1h/1d happens here.

Run:
    python -m uvicorn oi_options_trading.app:app --host 0.0.0.0 --port 8705

Mounted behind nginx at /strategy/ (set APP_BASE=/strategy via env).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse


ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
MASTER     = ROOT / "nifty_1m_master.parquet"
OI_INTRA   = ROOT / "_atm_oi_intraday.parquet"
LTP_JSON   = ROOT / "_nifty_ltp.json"
APP_BASE   = os.environ.get("APP_BASE", "").rstrip("/")


TF_RULE = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "60min", "4h": "240min", "1d": "1D",
}


# ─── Live-aware caching ──────────────────────────────────────────────────────
# The live worker appends to both parquets every minute during market hours.
# Every cache below is keyed on the file's mtime, so a fresh write busts it
# on the next request while repeated requests between writes stay warm.

def _mt(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


def _now_ist():
    from datetime import datetime, timedelta
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


@lru_cache(maxsize=2)
def _load_master_1m_v(v: int) -> pd.DataFrame:
    if not MASTER.exists():
        return pd.DataFrame()
    df = pd.read_parquet(MASTER)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "volume" not in df.columns:
        df["volume"] = 0
    return (df.sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last")
              .reset_index(drop=True))


def _load_master_1m() -> pd.DataFrame:
    return _load_master_1m_v(_mt(MASTER))


@lru_cache(maxsize=2)
def _master_split_v(v: int):
    """(full, hist, today, hist_fingerprint) for the current master version.
    The live worker rewrites the file every minute, but only TODAY's rows
    ever change — the fingerprint keys caches of anything derived from the
    historical part so it survives the per-minute mtime churn."""
    df = _load_master_1m_v(v)
    if df.empty:
        return df, df, df, (0, "")
    today = _now_ist().date()
    dts = df["timestamp"].dt.date
    hist = df[dts < today]
    live = df[dts >= today]
    fp = (len(hist), str(hist["timestamp"].iloc[-1]) if len(hist) else "")
    return df, hist, live, fp


def _resample_frame(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = TF_RULE.get(tf)
    if rule is None:
        raise HTTPException(400, f"unknown tf: {tf}")
    if df.empty:
        return df
    if tf == "1d":
        d = df.copy()
        d["day"] = d["timestamp"].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
        out = (d.groupby("day", as_index=False)
                 .agg(open=("open", "first"), high=("high", "max"),
                      low=("low", "min"),     close=("close", "last"),
                      volume=("volume", "sum")))
        return out.rename(columns={"day": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
    return (df.set_index("timestamp")
              .resample(rule, closed="left", label="left")
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"})
              .dropna(subset=["open", "close"])
              .reset_index())


_RESAMPLED_HIST_CACHE: dict = {}
_DERIVED_LOCK = __import__("threading").Lock()


def _resampled(tf: str) -> pd.DataFrame:
    """Resampled candles: historical days cached per (tf, hist-fingerprint);
    only today's ~375 rows are resampled per request."""
    df, hist, live, fp = _master_split_v(_mt(MASTER))
    if df.empty or tf == "1m":
        return df
    key = (tf,) + fp
    if key not in _RESAMPLED_HIST_CACHE:
        with _DERIVED_LOCK:
            if key not in _RESAMPLED_HIST_CACHE:
                if len(_RESAMPLED_HIST_CACHE) > 24:
                    _RESAMPLED_HIST_CACHE.clear()
                _RESAMPLED_HIST_CACHE[key] = _resample_frame(hist, tf)
    base = _RESAMPLED_HIST_CACHE[key]
    if live.empty:
        return base
    return pd.concat([base, _resample_frame(live, tf)], ignore_index=True)


def to_unix(ts: pd.Series) -> np.ndarray:
    return (ts.view("int64") // 1_000_000_000).to_numpy()


app = FastAPI(title="Strategy Suite — OI Options")


@app.on_event("startup")
def _prewarm():
    """Warm the heavy caches in the background right after boot, so the
    first page load doesn't eat the ~minutes-long historical compute."""
    import threading

    def warm():
        try:
            _px_by_day()
            _resampled("3m")
            _sigma_n_cached("3m", 10)
            _sigma_total_entries_cached(7)
        except Exception:
            pass

    threading.Thread(target=warm, daemon=True).start()


@app.get("/api/nifty")
def nifty(tf: str = Query("1d"),
          since: int | None = Query(None,
              description="Unix time — return only bars at/after this. The "
                          "UI's live refresh uses it to pull just today's "
                          "bars instead of the full multi-year series.")):
    bars = _resampled(tf)
    if bars.empty:
        return JSONResponse({"bars": [], "n": 0, "tf": tf})
    t = to_unix(bars["timestamp"])
    if since is not None:
        i0 = int(np.searchsorted(t, since))
        bars, t = bars.iloc[i0:], t[i0:]
        if bars.empty:
            return JSONResponse({"bars": [], "n": 0, "tf": tf})
    # Vectorized row build — per-row .iloc over 200k bars costs ~20s of CPU
    # on the VM; zipping plain python lists is ~50x cheaper.
    o = bars["open"].astype(float).tolist()
    h = bars["high"].astype(float).tolist()
    l = bars["low"].astype(float).tolist()
    c = bars["close"].astype(float).tolist()
    tl = t.tolist()
    out = [{"time": ti, "open": oi, "high": hi, "low": li, "close": ci}
           for ti, oi, hi, li, ci in zip(tl, o, h, l, c)]
    return JSONResponse({"bars": out, "n": len(out), "tf": tf,
                         "first": out[0]["time"], "last": out[-1]["time"]})


@lru_cache(maxsize=2)
def _load_oi_intraday_v(v: int) -> pd.DataFrame:
    if not OI_INTRA.exists():
        return pd.DataFrame()
    df = pd.read_parquet(OI_INTRA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_oi_intraday() -> pd.DataFrame:
    return _load_oi_intraday_v(_mt(OI_INTRA))


@lru_cache(maxsize=16)
def _oi_resampled_v(tf: str, v: int) -> pd.DataFrame:
    """Resample the per-tick ATM±1 OI series to the requested TF using 'last'
    aggregation. For TFs coarser than the native ~3-min cadence the line is
    smooth; for 1m the line steps (same value held for 3 consecutive 1m bars)."""
    df = _load_oi_intraday_v(v)
    if df.empty:
        return df
    if tf == "1d":
        d = df.copy()
        # Pin the daily timestamp to 09:15 IST so the OI 1d series shares
        # an x-axis bucket with the candle 1d series (which also uses 09:15).
        # The aggregation still picks each day's LAST tick (EOD value).
        d["day"] = d["timestamp"].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
        out = (d.groupby("day", as_index=False)
                 .agg(ce_oi=("ce_oi", "last"),
                      pe_oi=("pe_oi", "last"),
                      atm=("atm", "last"),
                      spot=("spot", "last")))
        return out.rename(columns={"day": "timestamp"})
    rule = TF_RULE.get(tf)
    if rule is None:
        raise HTTPException(400, f"unknown tf: {tf}")
    out = (df.set_index("timestamp")
             .resample(rule, closed="left", label="left")
             .agg({"ce_oi": "last", "pe_oi": "last",
                   "atm":   "last", "spot":  "last"})
             .dropna(subset=["ce_oi", "pe_oi"])
             .reset_index())
    return out


def _oi_resampled(tf: str) -> pd.DataFrame:
    return _oi_resampled_v(tf, _mt(OI_INTRA))


@app.get("/api/nifty_oi")
def nifty_oi(tf: str = Query("1d")):
    """ATM±1 combined OI series (CE + PE) at the requested TF.
    Returns two time-series ready for lightweight-charts line series."""
    df = _oi_resampled(tf)
    if df.empty:
        return JSONResponse({"ce_oi": [], "pe_oi": [], "n": 0, "tf": tf})
    t = to_unix(df["timestamp"])
    ce = [{"time": int(t[i]), "value": float(df["ce_oi"].iloc[i])}
          for i in range(len(df))]
    pe = [{"time": int(t[i]), "value": float(df["pe_oi"].iloc[i])}
          for i in range(len(df))]
    return JSONResponse({"ce_oi": ce, "pe_oi": pe, "n": len(df), "tf": tf})


@app.get("/api/nifty_oi_cumsum")
def nifty_oi_cumsum(tf: str = Query("1d"), window: int = Query(5)):
    """Rolling-window CUMULATIVE SUM of ATM±1 OI (CE & PE), at the requested
    TF. For each candle t the value is sum(OI[t-window+1 .. t]) — a quick
    proxy for OI "build-up" over the trailing N candles. min_periods=1 so
    the first few candles are partial sums rather than NaN."""
    if window < 1:
        raise HTTPException(400, "window must be >= 1")
    df = _oi_resampled(tf)
    if df.empty:
        return JSONResponse({"ce_cum": [], "pe_cum": [], "n": 0,
                             "tf": tf, "window": window})
    ce_cum = df["ce_oi"].rolling(window=window, min_periods=1).sum()
    pe_cum = df["pe_oi"].rolling(window=window, min_periods=1).sum()
    t = to_unix(df["timestamp"])
    ce = [{"time": int(t[i]), "value": float(ce_cum.iloc[i])}
          for i in range(len(df))]
    pe = [{"time": int(t[i]), "value": float(pe_cum.iloc[i])}
          for i in range(len(df))]
    return JSONResponse({"ce_cum": ce, "pe_cum": pe, "n": len(df),
                         "tf": tf, "window": window})


def _oi_intraday_accumulators(tf: str) -> pd.DataFrame:
    """Per-day cumulative OI series used by the ΔOI / ΣOI panes.

    DATA SOURCE RULE:
      • For every INTRADAY TF (1m, 3m, 5m, 15m, 30m, 1h, 4h) we use the
        RAW 1-minute OI parquet, NOT a TF-resampled copy. The strategy
        (sigma_total_oi_entry.py) detects crossovers at 1-min resolution
        on this exact same series — if we resampled here, the panel
        cumsum lines wouldn't cross at the same minutes the strategy
        fires, producing ghost markers (entry arrows with no visible
        cross on the pane).
      • For 1d TF we use the daily aggregate (one row per day) — at that
        zoom the user is looking at a multi-year panorama and an intraday
        saw-tooth at 1-min resolution would be useless visual noise. The
        strategy still fires intraday; for verification the user can
        switch to an intraday TF.

    Columns added per row:
        ce_chg / pe_chg : OI[t] - OI[first tick of that day]   ← cum changes
        ce_tot / pe_tot : per-day cumulative sum of OI[t]      ← cum totals
    Both series reset at each new day's first tick (typically 09:15 IST).
    """
    if tf == "1d":
        df = _oi_resampled("1d")
    else:
        df = _load_oi_intraday()
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    # Per-day delta vs the day's first observed OI
    d["ce_chg"] = (d.groupby("date", sort=False)["ce_oi"]
                    .transform(lambda s: s - s.iloc[0]))
    d["pe_chg"] = (d.groupby("date", sort=False)["pe_oi"]
                    .transform(lambda s: s - s.iloc[0]))
    # Per-day running sum from the day's first tick
    d["ce_tot"] = (d.groupby("date", sort=False)["ce_oi"].cumsum())
    d["pe_tot"] = (d.groupby("date", sort=False)["pe_oi"].cumsum())
    return d


@app.get("/api/nifty_oi_intra_changes")
def nifty_oi_intra_changes(tf: str = Query("1d")):
    """Per-day cumulative CHANGE in ATM±1 OI from the day's first tick.
    value[t] = OI[t] - OI[day_open].
    Resets to 0 at each new day's 09:15 IST."""
    df = _oi_intraday_accumulators(tf)
    if df.empty:
        return JSONResponse({"ce": [], "pe": [], "n": 0, "tf": tf})
    t = to_unix(df["timestamp"])
    ce = [{"time": int(t[i]), "value": float(df["ce_chg"].iloc[i])}
          for i in range(len(df))]
    pe = [{"time": int(t[i]), "value": float(df["pe_chg"].iloc[i])}
          for i in range(len(df))]
    return JSONResponse({"ce": ce, "pe": pe, "n": len(df), "tf": tf})


@app.get("/api/nifty_oi_intra_totals")
def nifty_oi_intra_totals(tf: str = Query("1d")):
    """Per-day RUNNING SUM of ATM±1 OI from the day's first tick.
    value[t] = sum(OI[day_open .. t]).
    Resets to 0 at each new day's 09:15 IST."""
    df = _oi_intraday_accumulators(tf)
    if df.empty:
        return JSONResponse({"ce": [], "pe": [], "n": 0, "tf": tf})
    t = to_unix(df["timestamp"])
    ce = [{"time": int(t[i]), "value": float(df["ce_tot"].iloc[i])}
          for i in range(len(df))]
    pe = [{"time": int(t[i]), "value": float(df["pe_tot"].iloc[i])}
          for i in range(len(df))]
    return JSONResponse({"ce": ce, "pe": pe, "n": len(df), "tf": tf})


@lru_cache(maxsize=1)
def _strategy_signals_cached():
    """Run Steps 1+2 once per process; OI aggregate changes only when the
    daily 5 PM cron rebuilds it, so a process-level cache is fine."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from strategies.multi_strike_oi_crossover import compute_signals
    return compute_signals()


@app.get("/api/strategy/multi_strike_oi_crossover")
def strategy_signals():
    """All days that fired a Step-2 crossover signal — BULLISH / BEARISH +
    surrounding OI snapshot, ready to overlay as chart markers."""
    sigs, stats = _strategy_signals_cached()
    return JSONResponse({"signals": sigs, "stats": stats, "n": len(sigs)})


# Live split: entry computation over the full history takes minutes on the
# 2-vCPU VM, far too slow to redo every time the live worker appends a row.
# Both strategies reset per-day, so entries for past days can never change
# once the day is over — compute them once per historical-data version and
# only re-run today's (tiny) slice on every request. The historical result
# is also persisted to disk so a service restart never recomputes it, and
# a single-flight lock stops concurrent first-requests from stampeding the
# same minutes-long compute.
_HIST_ENTRIES_CACHE: dict[tuple, tuple] = {}
_ENTRIES_LOCK = __import__("threading").Lock()
_ENTRIES_DISK = ROOT / "_strategy_entries_cache.json"


def _disk_entries_load() -> dict:
    try:
        return json.loads(_ENTRIES_DISK.read_text())
    except Exception:
        return {}


def _disk_entries_save(d: dict) -> None:
    try:
        # Keep the file bounded: the fingerprint changes once per day, so a
        # handful of keys is plenty.
        if len(d) > 12:
            for k in list(d)[:-12]:
                d.pop(k, None)
        tmp = _ENTRIES_DISK.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d))
        tmp.replace(_ENTRIES_DISK)
    except OSError:
        pass


def _entries_live(kind: str, tf: str = "1m", window: int = 10,
                  min_gap: int = 7) -> tuple[list, dict]:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from strategies.sigma5_entry_exit import compute_entries as _ce_sigma_n
    from strategies.sigma_total_oi_entry import compute_entries as _ce_sigma_oi

    def _compute(frame):
        if frame is not None and frame.empty:
            return [], {}
        if kind == "sigma_n":
            return _ce_sigma_n(tf=tf, window=window, df=frame)
        return _ce_sigma_oi(min_gap_bars=min_gap, df=frame)

    df = _load_oi_intraday()
    if df.empty:
        return [], {}

    today = _now_ist().date()
    dts = df["timestamp"].dt.date
    hist = df[dts < today]
    live = df[dts >= today]

    # Fingerprint the historical slice by its row count + last timestamp —
    # invariant while the live worker appends today's rows, changes when the
    # morning rebuild lands settled data.
    fp = (kind, tf, window, min_gap, len(hist),
          str(hist["timestamp"].iloc[-1]) if len(hist) else "")
    if fp not in _HIST_ENTRIES_CACHE:
        with _ENTRIES_LOCK:
            if fp not in _HIST_ENTRIES_CACHE:     # double-checked
                key = "|".join(map(str, fp))
                disk = _disk_entries_load()
                if key in disk:
                    res = tuple(disk[key])
                else:
                    res = _compute(hist)
                    disk[key] = res
                    _disk_entries_save(disk)
                if len(_HIST_ENTRIES_CACHE) > 32:
                    _HIST_ENTRIES_CACHE.clear()
                _HIST_ENTRIES_CACHE[fp] = res
    hist_entries, stats = _HIST_ENTRIES_CACHE[fp]

    live_entries = _compute(live)[0] if len(live) else []
    return list(hist_entries) + live_entries, stats


def _sigma5_entries_cached(tf: str):
    """First qualifying Σ-N entry per day on the requested TF. Historical
    days cached; today recomputed live on each request."""
    return _entries_live("sigma_n", tf=tf)


def _sigma_n_cached(tf: str, window: int):
    """Σ-N entries with a sweepable rolling window. Historical days cached
    per (TF, window); today recomputed live on each request."""
    return _entries_live("sigma_n", tf=tf, window=window)


@app.get("/api/strategy/sigma5_entries")
def sigma5_entries(tf: str = Query("1m")):
    """Σ-N entry signals (smoothed OI cross + raw OI confirm), where N
    candles are evaluated on the requested TF (default 1m). Returns
    LONG / SHORT entries with entry_time, entry_spot, day_open_time so
    the UI can place markers on either intraday or daily candles."""
    entries, stats = _sigma5_entries_cached(tf)
    return JSONResponse({"entries": entries, "stats": stats, "n": len(entries)})


def _sigma_total_entries_cached(min_gap_bars: int = 7):
    """ΣOI-cross entries with the min-gap-bars whipsaw filter applied
    inside the strategy module. Default 7 — best-PF setting per the
    notebook backtest. Historical days cached; today recomputed live."""
    return _entries_live("sigma_oi", min_gap=min_gap_bars)


@app.get("/api/strategy/sigma_total_entries")
def sigma_total_entries(min_gap: int = Query(7, description="Min bars between consecutive same-day crosses (whipsaw filter). 0 = disabled.")):
    """ΣOI entry signals — per-day cumulative-OI crossover (since 09:15)
    inside the entry window, debounced by the min-gap filter. Returns
    LONG / SHORT entries with entry_time, entry_spot, day_open_time so
    the UI can place markers."""
    entries, stats = _sigma_total_entries_cached(min_gap)
    return JSONResponse({"entries": entries, "stats": stats, "n": len(entries),
                         "min_gap": min_gap})


@app.get("/api/backtest")
def api_backtest(strategy: str = Query("C", description="A=multi-strike, B=Σ-N rolling, C=ΣOI"),
                 tf: str       = Query("3m"),
                 sl_pct: float = Query(0.20, description="stop-loss as % of entry spot"),
                 tgt_pct: float= Query(1.50, description="target as % of entry spot"),
                 window: int   = Query(10,  description="Σ-N window (Strategy B only)"),
                 confirm_candle: bool = Query(False,
                     description="Candle-color confirm: reject LONG on red candle / SHORT on green candle (uses the prior 1-min NIFTY bar — leak-free)"),
                 min_gap: int = Query(7,
                     description="Minimum gap between consecutive same-day crosses (in bars of the strategy's native resolution: 1m for C, TF-bucket for B). Default 7 — best-PF setting. 0 = disabled.")):
    """Run one of the three OI strategies + walk forward on 1-min NIFTY with
    (SL%, TGT%, EOD-square-off) exit policy. Returns trade list + summary stats.

    All variables are tweakable from the chart UI; this endpoint is hit each
    time the user clicks Run on the backtest panel.
    """
    strategy = (strategy or "C").upper()

    # 1. Entries from the selected strategy. Entries don't depend on SL/TGT,
    # so we hit the existing per-strategy LRU caches — the second click for
    # the same (strategy, tf, window) returns near-instantly and only the
    # exit-walk re-runs against the new SL/TGT.
    if strategy == "A":
        sigs, _stats = _strategy_signals_cached()
        entries = [{
            "day": s["day"],
            "side": "LONG" if s["signal"] == "BULLISH" else "SHORT",
            "entry_time": s["signal_time"],
            "entry_spot": float(s["signal_spot"]),
            "day_open_time": s["day_open_time"],
        } for s in sigs]
    elif strategy == "B":
        sigs, _stats = _sigma_n_cached(tf, window)
        entries = [{
            "day": s["day"],
            "side": s["side"],
            "entry_time": s["entry_time"],
            "entry_spot": float(s["entry_spot"]),
            "day_open_time": s["day_open_time"],
        } for s in sigs]
    elif strategy == "C":
        # Strategy C bakes the min-gap whipsaw filter inside its module so
        # every consumer (this endpoint, the marker overlay endpoint, the
        # notebook backtest) gets the same filtered set. Pass min_gap from
        # the request so users can still vary it from the BACKTEST panel.
        sigs, _stats = _sigma_total_entries_cached(min_gap)
        entries = [{
            "day": s["day"],
            "side": s["side"],
            "entry_time": s["entry_time"],
            "entry_spot": float(s["entry_spot"]),
            "day_open_time": s["day_open_time"],
        } for s in sigs]
    else:
        raise HTTPException(400, f"unknown strategy: {strategy}")

    # NB: Strategy C now applies min_gap inside its own module (see
    # _sigma_total_entries_cached). Strategies A and B are 1-entry-per-day
    # so the gap filter is a no-op for them — no separate pass needed here.

    # 2. 1-min underlying for exit walk + (optional) candle-color confirm ---
    px_df = _load_master_1m()
    if px_df.empty:
        return JSONResponse({"error": "1-min NIFTY master not available"}, status_code=503)
    px_df = px_df[["timestamp", "open", "close"]].copy()
    px_df["date_key"] = px_df["timestamp"].dt.date.astype(str)
    # Pre-bucket per-day numpy arrays for O(1) per-trade slicing. Keep both
    # open and close — open is only needed for the candle-color filter.
    px_by_day = {}
    for d, g in px_df.groupby("date_key", sort=True):
        g = g.sort_values("timestamp")
        px_by_day[d] = (g["timestamp"].values.astype("datetime64[ns]"),
                        g["open"].values.astype("float64"),
                        g["close"].values.astype("float64"))

    # 2b. Candle-color confirmation filter ----------------------------------
    # Reject LONG signals where the PRIOR 1-min NIFTY bar closed red (close<open)
    # and SHORT signals where the prior bar closed green. Using the prior bar
    # is leak-free — at signal time t, the t-1 minute bar has just closed and
    # is what a live trader actually sees on screen.
    if confirm_candle and entries:
        kept = []
        for e in entries:
            day_arrs = px_by_day.get(str(e["day"]))
            if day_arrs is None:
                continue
            ts_arr, op_arr, cl_arr = day_arrs
            ent_ts = np.datetime64(e["entry_time"])
            idx = int(np.searchsorted(ts_arr, ent_ts, side="left"))
            ref = idx - 1 if idx > 0 else idx
            if ref < 0 or ref >= len(cl_arr):
                continue   # can't evaluate, drop conservatively
            is_green = cl_arr[ref] >= op_arr[ref]
            if e["side"] == "LONG"  and not is_green: continue   # LONG on red → reject
            if e["side"] == "SHORT" and is_green:     continue   # SHORT on green → reject
            kept.append(e)
        entries = kept

    # 3. Walk-forward exit on each entry ------------------------------------
    sl_frac, tgt_frac = sl_pct / 100.0, tgt_pct / 100.0
    trades = []
    for e in entries:
        day_arrs = px_by_day.get(str(e["day"]))
        if day_arrs is None:
            continue
        ts_arr, _op_arr, cl_arr = day_arrs
        ent = np.datetime64(e["entry_time"])
        idx = int(np.searchsorted(ts_arr, ent, side="left"))
        if idx >= len(ts_arr):
            continue
        seg_ts = ts_arr[idx:]
        seg_cl = cl_arr[idx:]
        spot = e["entry_spot"]
        if e["side"] == "LONG":
            sl_px, tgt_px = spot * (1 - sl_frac), spot * (1 + tgt_frac)
            sl_hit  = seg_cl <= sl_px
            tgt_hit = seg_cl >= tgt_px
        else:
            sl_px, tgt_px = spot * (1 + sl_frac), spot * (1 - tgt_frac)
            sl_hit  = seg_cl >= sl_px
            tgt_hit = seg_cl <= tgt_px
        i_sl  = int(np.argmax(sl_hit))  if sl_hit.any()  else len(seg_cl)
        i_tgt = int(np.argmax(tgt_hit)) if tgt_hit.any() else len(seg_cl)
        if i_sl == len(seg_cl) and i_tgt == len(seg_cl):
            exit_ts_v = seg_ts[-1]; exit_px = float(seg_cl[-1]); reason = "EOD"
        elif i_sl <= i_tgt:
            exit_ts_v = seg_ts[i_sl]; exit_px = float(sl_px);  reason = "SL"
        else:
            exit_ts_v = seg_ts[i_tgt]; exit_px = float(tgt_px); reason = "TGT"
        pnl_pts = (exit_px - spot) if e["side"] == "LONG" else (spot - exit_px)
        trades.append({
            "day": e["day"],
            "side": e["side"],
            "entry_time": e["entry_time"],
            "entry_spot": spot,
            "exit_time":  pd.Timestamp(exit_ts_v).isoformat(),
            "exit_spot":  exit_px,
            "reason":     reason,
            "pnl_pts":    float(pnl_pts),
            "pnl_pct":    float(pnl_pts / spot * 100.0),
            "day_open_time": e["day_open_time"],
        })

    # 4. Aggregate stats ----------------------------------------------------
    if not trades:
        stats = {"n": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                 "pf": None, "avg_win": 0.0, "avg_loss": 0.0,
                 "long_n": 0, "short_n": 0, "sl_hits": 0, "tgt_hits": 0, "eod_exits": 0,
                 "best": 0.0, "worst": 0.0}
    else:
        arr = np.array([t["pnl_pts"] for t in trades])
        wins  = arr[arr > 0]; losses = arr[arr <= 0]
        reasons = [t["reason"] for t in trades]
        sides   = [t["side"]   for t in trades]
        loss_sum = float(losses.sum())
        stats = {
            "n":          len(trades),
            "win_rate":   float(len(wins) / len(arr) * 100.0),
            "total_pnl":  float(arr.sum()),
            "avg_pnl":    float(arr.mean()),
            "avg_win":    float(wins.mean())  if len(wins)   else 0.0,
            "avg_loss":   float(losses.mean()) if len(losses) else 0.0,
            "pf":         float(wins.sum() / abs(loss_sum)) if loss_sum != 0 else None,
            "long_n":     sides.count("LONG"),
            "short_n":    sides.count("SHORT"),
            "sl_hits":    reasons.count("SL"),
            "tgt_hits":   reasons.count("TGT"),
            "eod_exits":  reasons.count("EOD"),
            "best":       float(arr.max()),
            "worst":      float(arr.min()),
        }

    return JSONResponse({
        "strategy": strategy, "tf": tf, "sl_pct": sl_pct, "tgt_pct": tgt_pct,
        "window": window, "confirm_candle": confirm_candle,
        "min_gap": min_gap, "entries_kept": len(entries),
        "trades": trades, "stats": stats,
    })


def _build_px(px_df: pd.DataFrame) -> dict:
    if px_df.empty:
        return {}
    px_df = px_df[["timestamp", "close"]].copy()
    px_df["date_key"] = px_df["timestamp"].dt.date.astype(str)
    out = {}
    for d, g in px_df.groupby("date_key", sort=True):
        g = g.sort_values("timestamp")
        out[d] = (g["timestamp"].values.astype("datetime64[ns]"),
                  g["close"].values.astype("float64"))
    return out


_PX_HIST_CACHE: dict = {}


def _px_by_day() -> dict:
    """{date_str: (ts_array, close_array)} for the walk-forward exit.
    Historical days cached by fingerprint (the live worker's per-minute
    rewrite only changes today); today's arrays rebuilt per request."""
    df, hist, live, fp = _master_split_v(_mt(MASTER))
    if df.empty:
        return {}
    if fp not in _PX_HIST_CACHE:
        with _DERIVED_LOCK:
            if fp not in _PX_HIST_CACHE:
                base = _build_px(hist)
                _PX_HIST_CACHE.clear()
                _PX_HIST_CACHE[fp] = base
    out = dict(_PX_HIST_CACHE[fp])
    out.update(_build_px(live))
    return out


@app.get("/api/merged_trades")
def api_merged_trades(mode: str   = Query("both", description="sigma10 | sigma_oi | both"),
                      sl_pct: float = Query(0.18),
                      tgt_pct: float= Query(0.50),
                      trail: bool   = Query(True,
                          description="MULTI-STEP trail SL: once in profit by trail_pct points, SL ratchets up with the close — keeps a `trail_distance`-point trail behind the price and never moves against the trade. Default ON."),
                      trail_pct: float = Query(0.135,
                          description="Trail distance as % of entry spot. Default 0.135% ≈ 30 NIFTY pts at 22k spot. Decoupled from sl_pct so the initial stop can be wider than the trail."),
                      position_mode: str = Query("opposite_ok",
                          description="opposite_ok = block same-side only while in trade (opposite-side signals allowed). single = block ALL signals while any position is open.")):
    """Live-trades endpoint backing the simplified VIEW panel.

    mode="sigma10"  → Σ·10 entries only (Strategy B, window=10, 3m TF)
    mode="sigma_oi" → ΣOI entries only (Strategy C, min-gap-7 baked in)
    mode="both"     → Both, merged in time order, with **same-side dedup
                      within 7 minutes** — if two methods fire a same-side
                      signal less than 7 min apart on the same day, the
                      FIRST one wins and the duplicate is dropped.

    Each returned trade carries a `source` field ("sigma10" | "sigma_oi")
    so the UI can color/label them.
    """
    mode = (mode or "both").lower()
    # 1. Pull entries from the cached helpers (warm after first call)
    sig10_raw, _ = _sigma_n_cached("3m", 10)
    sigot_raw, _ = _sigma_total_entries_cached(7)

    def _tagged(entries, src):
        return [{**e, "_source": src} for e in entries]

    if mode == "sigma10":
        merged = _tagged(sig10_raw, "sigma10")
    elif mode == "sigma_oi":
        merged = _tagged(sigot_raw, "sigma_oi")
    elif mode == "both":
        merged = _tagged(sig10_raw, "sigma10") + _tagged(sigot_raw, "sigma_oi")
        # Sort by (day, entry_time) so the FIRST signal wins
        merged.sort(key=lambda e: (str(e["day"]), str(e["entry_time"])))
        # Dedup: same (day, side), within 7 minutes → keep first
        kept = []
        last_per_key = {}    # (day, side) -> last accepted timestamp
        for e in merged:
            day = str(e["day"])
            side = e["side"]
            ent_ts = pd.Timestamp(e["entry_time"])
            key = (day, side)
            prev = last_per_key.get(key)
            if prev is not None and (ent_ts - prev).total_seconds() < 7 * 60:
                continue   # dedup
            kept.append(e)
            last_per_key[key] = ent_ts
        merged = kept
    else:
        raise HTTPException(400, f"unknown mode: {mode}")

    # 2. Walk-forward exit on 1-min NIFTY index
    px_by_day = _px_by_day()
    if not px_by_day:
        return JSONResponse({"error": "1-min NIFTY master not available"}, status_code=503)

    sl_frac, tgt_frac = sl_pct / 100.0, tgt_pct / 100.0
    trades = []
    for e in merged:
        day_arrs = px_by_day.get(str(e["day"]))
        if day_arrs is None:
            continue
        ts_arr, cl_arr = day_arrs
        ent = np.datetime64(e["entry_time"])
        idx = int(np.searchsorted(ts_arr, ent, side="left"))
        if idx >= len(ts_arr):
            continue
        seg_ts = ts_arr[idx:]
        seg_cl = cl_arr[idx:]
        spot = float(e["entry_spot"])
        side = e["side"]
        sl_distance    = spot * sl_frac       # initial SL distance in pts
        trail_distance = spot * (trail_pct / 100.0)   # trail step in pts (decoupled)
        if side == "LONG":
            sl_initial   = spot - sl_distance
            tgt_px       = spot * (1 + tgt_frac)
            trail_trigger= spot + trail_distance      # in profit by trail_distance pts
        else:
            sl_initial   = spot + sl_distance
            tgt_px       = spot * (1 - tgt_frac)
            trail_trigger= spot - trail_distance
        # Bar-by-bar walk with MULTI-STEP (staircase) trail: every time the
        # close pushes deeper into profit, the SL ratchets up to keep an
        # `sl_distance`-point trail behind the close. SL never moves
        # against the trade. First trigger lands the SL at breakeven; every
        # subsequent advance locks in more profit.
        sl_current   = sl_initial
        trail_active = False
        exit_t = exit_p = reason = None
        for i in range(len(seg_cl)):
            c = seg_cl[i]
            if side == "LONG":
                if c >= tgt_px:
                    exit_t, exit_p, reason = seg_ts[i], float(tgt_px), "TGT"; break
                if trail and c >= trail_trigger:
                    if not trail_active: trail_active = True
                    candidate = c - trail_distance
                    if candidate > sl_current: sl_current = candidate
                if c <= sl_current:
                    exit_t = seg_ts[i]
                    exit_p = float(sl_current)
                    reason = "TRAIL" if trail_active else "SL"
                    break
            else:
                if c <= tgt_px:
                    exit_t, exit_p, reason = seg_ts[i], float(tgt_px), "TGT"; break
                if trail and c <= trail_trigger:
                    if not trail_active: trail_active = True
                    candidate = c + trail_distance
                    if candidate < sl_current: sl_current = candidate
                if c >= sl_current:
                    exit_t = seg_ts[i]
                    exit_p = float(sl_current)
                    reason = "TRAIL" if trail_active else "SL"
                    break
        if exit_t is None:
            exit_t = seg_ts[-1]; exit_p = float(seg_cl[-1]); reason = "EOD"
        pnl_pts = (exit_p - spot) if side == "LONG" else (spot - exit_p)
        trades.append({
            "day": e["day"], "side": side,
            "entry_time": e["entry_time"], "entry_spot": spot,
            "exit_time":  pd.Timestamp(exit_t).isoformat(),
            "exit_spot":  exit_p, "reason": reason,
            "pnl_pts":    float(pnl_pts),
            "pnl_pct":    float(pnl_pts / spot * 100.0),
            "day_open_time": e["day_open_time"],
            "source":     e["_source"],
            "trail_triggered": bool(trail_active),
        })

    # 2b. Position-overlap filter — two modes:
    #   opposite_ok: block only SAME-side re-entries while a trade is open
    #                (a SHORT can still fire while a LONG is still in play).
    #   single:      block ALL re-entries while ANY trade is open. Truly
    #                one-position-at-a-time — opposite signal must wait
    #                for the current trade to exit (SL/TGT/EOD/TRAIL).
    # Per-day reset — a new session starts with no open positions.
    trades.sort(key=lambda t: (str(t["day"]), str(t["entry_time"])))
    filtered = []
    open_until = {"LONG": None, "SHORT": None}
    current_day = None
    for t in trades:
        day = str(t["day"])
        if day != current_day:
            open_until = {"LONG": None, "SHORT": None}
            current_day = day
        side = t["side"]
        ent_ts = pd.Timestamp(t["entry_time"])
        if position_mode == "single":
            # ANY side open → block
            blocked = any(ou is not None and ent_ts < ou for ou in open_until.values())
        else:
            # only SAME side open → block
            blocked = open_until[side] is not None and ent_ts < open_until[side]
        if blocked:
            continue
        filtered.append(t)
        open_until[side] = pd.Timestamp(t["exit_time"])
    trades = filtered

    # 3. Stats
    if not trades:
        stats = {"n":0,"win_rate":0.0,"total_pnl":0.0,"avg_pnl":0.0,"pf":None,
                 "long_n":0,"short_n":0,"sl_hits":0,"tgt_hits":0,"eod_exits":0,
                 "trail_hits":0,"sigma10_n":0,"sigma_oi_n":0,
                 "n_days":0,"avg_per_day":0.0}
    else:
        arr = np.array([t["pnl_pts"] for t in trades])
        wins, losses = arr[arr > 0], arr[arr <= 0]
        reasons = [t["reason"] for t in trades]
        sides   = [t["side"]   for t in trades]
        sources = [t["source"] for t in trades]
        loss_sum = float(losses.sum())
        # Avg PnL excludes TRAIL exits — trail trades are breakeven by
        # construction and just drag the average toward 0, hiding the real
        # per-trade edge on resolved (SL/TGT/EOD) trades.
        non_trail_arr = np.array([t["pnl_pts"] for t in trades if t["reason"] != "TRAIL"])
        stats = {
            "n":          len(trades),
            "win_rate":   float(len(wins) / len(arr) * 100.0),
            "total_pnl":  float(arr.sum()),
            "avg_pnl":    float(non_trail_arr.mean()) if len(non_trail_arr) else 0.0,
            "pf":         float(wins.sum() / abs(loss_sum)) if loss_sum != 0 else None,
            "long_n":     sides.count("LONG"),
            "short_n":    sides.count("SHORT"),
            "sl_hits":    reasons.count("SL"),
            "tgt_hits":   reasons.count("TGT"),
            "eod_exits":  reasons.count("EOD"),
            "trail_hits": reasons.count("TRAIL"),
            "sigma10_n":  sources.count("sigma10"),
            "sigma_oi_n": sources.count("sigma_oi"),
            "n_days":     len({str(t["day"]) for t in trades}),
            "avg_per_day":float(len(trades) / max(1, len({str(t["day"]) for t in trades}))),
        }
    return JSONResponse({
        "mode": mode, "sl_pct": sl_pct, "tgt_pct": tgt_pct,
        "trades": trades, "stats": stats,
    })


@app.get("/api/ltp")
def live_ltp():
    """Latest NIFTY tick written by the live worker (~every 4s during the
    session). Tiny file read — never cached, safe to poll."""
    try:
        return JSONResponse(json.loads(LTP_JSON.read_text()))
    except Exception:
        return JSONResponse({"ts": None, "ltp": None})


@app.get("/api/range")
def date_range():
    df = _load_master_1m()
    if df.empty:
        return {"first": None, "last": None, "rows": 0}
    return {
        "first": df["timestamp"].iloc[0].isoformat(),
        "last":  df["timestamp"].iloc[-1].isoformat(),
        "rows":  int(len(df)),
        "days":  int(df["timestamp"].dt.date.nunique()),
    }


# ─────────────────────────────────── UI ─────────────────────────────────
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Suite — NIFTY</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg:#0b0d11; --panel:#10131a; --panel-2:#161a23; --border:#202533;
    --border-hi:#2b3142; --text:#e7eaf1; --muted:#7b8294; --accent:#60a5fa;
    --green:#26a69a; --red:#ef5350;
  }
  * { box-sizing: border-box; }
  html, body { height:100%; margin:0; background:var(--bg); color:var(--text);
               font-family:'JetBrains Mono', 'SF Mono', Consolas, monospace; font-size:13px; }
  .bar {
    display:flex; align-items:center; gap:14px; padding:10px 18px;
    background:var(--panel); border-bottom:1px solid var(--border); flex-wrap:wrap;
  }
  .brand { display:flex; align-items:baseline; gap:8px; margin-right:18px; }
  .brand .logo { color:var(--accent); font-size:18px; }
  .brand .title { font-weight:700; letter-spacing:0.5px; }
  .brand .sub { color:var(--muted); font-size:11px; }
  .bar label { color:var(--muted); font-size:11px; letter-spacing:0.5px; text-transform:uppercase; }
  .tf-grp { display:inline-flex; background:var(--panel-2); border:1px solid var(--border-hi);
            border-radius:6px; overflow:hidden; }
  .tf-grp button {
    background:transparent; color:var(--text); border:none; padding:6px 14px; cursor:pointer;
    font:inherit; font-family:inherit; border-right:1px solid var(--border-hi);
  }
  .tf-grp button:last-child { border-right:none; }
  .tf-grp button.on { background:var(--accent); color:#0a0c10; font-weight:600; }
  .tf-grp button:hover:not(.on) { background:var(--panel); }
  .btn {
    background:var(--panel-2); color:var(--text); border:1px solid var(--border-hi);
    border-radius:6px; padding:6px 14px; cursor:pointer; font:inherit; font-family:inherit;
  }
  .btn:hover { background:var(--panel); }
  .spot {
    display:flex; align-items:baseline; gap:8px; padding:5px 12px;
    background:var(--panel-2); border:1px solid var(--border-hi); border-radius:6px;
  }
  .spot .l { color:var(--muted); font-size:11px; }
  .spot strong { color:var(--accent); font-size:15px; font-weight:700; }
  .meta {
    margin-left:auto; color:var(--muted); font-size:11px; display:flex; gap:14px;
  }
  .meta b { color:var(--text); font-weight:600; }
  /* Two-pane layout when OI sub-pane is visible. The pane's flex-basis is
     driven by JS (initial 35%, drag between 15% and 50%) — the divider
     gives a draggable handle to resize. */
  .charts { display:flex; flex-direction:column; width:100%; height:calc(100vh - 56px); }
  #chart   { flex: 1 1 auto; min-height: 0; width:100%; position:relative; }
  #pane-resizer {
    flex: 0 0 6px; width: 100%; background: var(--border);
    cursor: row-resize; transition: background 0.15s;
    border-top: 1px solid var(--border-hi);
    border-bottom: 1px solid var(--border-hi);
  }
  #pane-resizer:hover, #pane-resizer.dragging { background: var(--accent, #60a5fa); }
  #oi-pane { flex: 0 0 35%; min-height: 0; width:100%; position:relative; }
  #pnl-pane { flex: 0 0 28%; min-height: 0; width:100%; position:relative; }
  #pnl-resizer {
    flex: 0 0 6px; width: 100%; background: var(--border);
    cursor: row-resize; transition: background 0.15s;
    border-top: 1px solid var(--border-hi);
    border-bottom: 1px solid var(--border-hi);
  }
  #pnl-resizer:hover, #pnl-resizer.dragging { background: var(--accent, #60a5fa); }
  #pnl-label {
    position:absolute; top:6px; left:10px; z-index:5; font-size:11px; color:var(--muted);
    background: rgba(11,13,17,0.85); padding:3px 8px; border-radius:4px;
    border: 1px solid var(--border-hi); pointer-events:none;
  }
  #pnl-label b { color: var(--text); font-weight:600; }
  #oi-pane.hidden, #pane-resizer.hidden, #oi-label.hidden { display: none !important; }
  #oi-label {
    position:absolute; left:14px; top:6px; z-index:5;
    color: var(--muted); font-size:11px; letter-spacing:0.6px;
    background: rgba(11,13,17,0.85); padding:3px 8px; border-radius:4px;
    border: 1px solid var(--border-hi); pointer-events:none;
  }
  #oi-label b { color: var(--text); font-weight:600; }
  #oi-label .ce { color: #26a69a; }
  #oi-label .pe { color: #ef5350; }
  /* Third pane — rolling Σ(N) of ATM±1 OI. Same chrome as #pane-resizer/
     #oi-pane so the layout stacks cleanly. Smaller default basis (25%) so
     three panes fit without crushing the candle chart. */
  #cum-resizer {
    flex: 0 0 6px; width: 100%; background: var(--border);
    cursor: row-resize; transition: background 0.15s;
    border-top: 1px solid var(--border-hi);
    border-bottom: 1px solid var(--border-hi);
  }
  #cum-resizer:hover, #cum-resizer.dragging { background: var(--accent, #60a5fa); }
  #cum-pane { flex: 0 0 25%; min-height: 0; width:100%; position:relative; }
  #cum-pane.hidden, #cum-resizer.hidden, #cum-label.hidden { display: none !important; }
  #cum-label {
    position:absolute; left:14px; top:6px; z-index:5;
    color: var(--muted); font-size:11px; letter-spacing:0.6px;
    background: rgba(11,13,17,0.85); padding:3px 8px; border-radius:4px;
    border: 1px solid var(--border-hi); pointer-events:none;
  }
  #cum-label b { color: var(--text); font-weight:600; }
  #cum-label .ce { color: #26a69a; }
  #cum-label .pe { color: #ef5350; }
  /* Intraday accumulator panes (ΔOI = changes since 09:15, ΣOI = running sum
     since 09:15). Default OFF (HTML class="hidden"); the toolbar buttons
     toggle them. Same chrome as #cum-pane for visual consistency. */
  #chg-resizer, #tot-resizer {
    flex: 0 0 6px; width: 100%; background: var(--border);
    cursor: row-resize; transition: background 0.15s;
    border-top: 1px solid var(--border-hi);
    border-bottom: 1px solid var(--border-hi);
  }
  #chg-resizer:hover, #chg-resizer.dragging,
  #tot-resizer:hover, #tot-resizer.dragging { background: var(--accent, #60a5fa); }
  #chg-pane, #tot-pane {
    flex: 0 0 18%; min-height: 0; width:100%; position:relative;
  }
  #chg-pane.hidden, #chg-resizer.hidden, #chg-label.hidden,
  #tot-pane.hidden, #tot-resizer.hidden, #tot-label.hidden { display: none !important; }
  #chg-label, #tot-label {
    position:absolute; left:14px; top:6px; z-index:5;
    color: var(--muted); font-size:11px; letter-spacing:0.6px;
    background: rgba(11,13,17,0.85); padding:3px 8px; border-radius:4px;
    border: 1px solid var(--border-hi); pointer-events:none;
  }
  #chg-label b, #tot-label b { color: var(--text); font-weight:600; }
  #chg-label .ce, #tot-label .ce { color: #26a69a; }
  #chg-label .pe, #tot-label .pe { color: #ef5350; }
</style>
</head>
<body>

<div class="bar">
  <div class="brand">
    <span class="logo">◆</span>
    <span class="title">STRATEGY SUITE</span>
    <span class="sub">NIFTY 50 · index</span>
  </div>

  <label>Timeframe</label>
  <span class="tf-grp" id="tf-grp">
    <button data-tf="3m" class="on">3m</button>
  </span>

  <button class="btn" id="fit-all">FIT ALL</button>
  <!-- Hidden by request — kept in DOM so JS handlers and state continue to work; un-hide here to restore. -->
  <button class="btn" id="oi-toggle"   style="display:none;" title="ATM·OI">ATM·OI</button>
  <button class="btn" id="cum-toggle"  style="display:none;" title="Σ·10">Σ·10</button>
  <button class="btn" id="chg-toggle"  style="display:none;" title="ΔOI">ΔOI</button>
  <button class="btn" id="tot-toggle"  style="display:none;" title="ΣOI">ΣOI</button>
  <button class="btn" id="strat-toggle" style="display:none;" title="STRATEGY">STRATEGY</button>
  <button class="btn" id="sig5-toggle"  style="display:none;" title="Σ·10 ENTRY">Σ·10 ENTRY</button>
  <button class="btn" id="sigt-toggle"  style="display:none;" title="ΣOI ENTRY">ΣOI ENTRY</button>
  <button class="btn" id="bt-toggle" style="display:none;">BACKTEST</button>
  <span id="strat-stat" style="color:var(--muted); font-size:11px; display:none;">
    <b id="ss-bull" style="color:#26a69a;">—</b> bullish ·
    <b id="ss-bear" style="color:#ef5350;">—</b> bearish ·
    <b id="ss-no">—</b> no-cross
  </span>
  <span id="sig5-stat" style="color:var(--muted); font-size:11px; display:none;">
    <b id="s5-long"  style="color:#22c55e;">—</b> long ·
    <b id="s5-short" style="color:#ef4444;">—</b> short ·
    <b id="s5-no">—</b> no-cross
  </span>
  <span id="sigt-stat" style="color:var(--muted); font-size:11px; display:none;">
    <b id="st-long"  style="color:#14b8a6;">—</b> long ·
    <b id="st-short" style="color:#c026d3;">—</b> short ·
    <b id="st-no">—</b> no-cross
  </span>

  <button class="btn" id="pnl-toggle" title="Show / hide the cumulative-PnL pane below the candle chart.">PnL</button>
  <button class="btn" id="live-toggle" title="Auto-refresh candles + trades every 60s while the NSE session is open. The live worker appends fresh 1-min bars and ATM±1 OI every minute.">LIVE</button>

  <div class="spot">
    <span class="l">LAST</span>
    <strong id="last-val">—</strong>
  </div>

  <div class="meta">
    <span><b id="bar-count">—</b> bars</span>
    <span><b id="day-count">—</b> days</span>
    <span><b id="range">—</b></span>
  </div>
</div>

<!-- TRADES PANEL — every parameter is now locked to the winning config:
     view=Both, SL=0.18%, TGT=0.50%, Trail=MULTI 0.135% (≈30 pts),
     position_mode=Single. Only the stats strip is rendered. -->
<div id="v-panel" style="
     display:flex; align-items:center; gap:14px; padding:10px 18px; flex-wrap:wrap;
     background:linear-gradient(180deg,#0d1018 0%, #10131a 100%);
     border-bottom:1px solid var(--border);">
  <span style="color:var(--accent); font-weight:700; letter-spacing:0.5px;">▶ TRADES</span>
  <span style="color:var(--muted); font-size:11px;"
        title="Both methods merged with same-side dedup-7. Single position at a time. Exit rules (SL / target / multi-step trail) are tuned defaults — values intentionally not surfaced in the header.">
    Both · Single
  </span>

  <!-- Hidden: position-mode toggle is locked to Single; markup kept so the
       JS that reads `.querySelector('button.on')` still resolves. -->
  <span id="v-pos-grp" style="display:none;">
    <button data-pm="single" class="on"></button>
  </span>

  <span id="v-stats" style="display:none; margin-left:auto; color:var(--muted); font-size:11px; line-height:1.5;
                              padding:6px 12px; background:var(--panel-2); border:1px solid var(--border-hi); border-radius:6px;">
    <b id="v-n"        style="color:var(--text); font-weight:700;">—</b> trades &nbsp;·&nbsp;
    <b id="v-wr"       style="color:#22c55e;">—</b> win-rate &nbsp;·&nbsp;
    <b id="v-totpnl"   style="color:var(--accent);">—</b> pts total &nbsp;·&nbsp;
    <b id="v-pf"       style="color:#fbbf24;">—</b> PF &nbsp;·&nbsp;
    <b id="v-sl-hits"  style="color:#ef4444;">—</b> SL ·
    <b id="v-tgt-hits" style="color:#22c55e;">—</b> TGT ·
    <b id="v-eod"      style="color:var(--muted);">—</b> EOD ·
    <b id="v-trail-hits" style="color:#fbbf24;">—</b> TRAIL &nbsp;·&nbsp;
    <b id="v-per-day"  style="color:#a78bfa;">—</b> /day &nbsp;·&nbsp;
    L <b id="v-long"   style="color:#06b6d4;">—</b> ·
    S <b id="v-short"  style="color:#f59e0b;">—</b> &nbsp;|&nbsp;
    <b id="v-src-s10"  style="color:#60a5fa;">—</b> Σ·10 ·
    <b id="v-src-soi"  style="color:#a78bfa;">—</b> ΣOI
  </span>
</div>

<div class="charts" id="charts-wrap">
  <div id="chart"></div>
  <div id="pnl-resizer" title="Drag to resize the cumulative-PnL pane (15% – 50%)"></div>
  <div id="pnl-pane">
    <div id="pnl-label">
      CUM PnL &nbsp;·&nbsp; <b id="pnl-final">—</b> pts
      <span style="color:var(--muted)"> &nbsp;·&nbsp;
        max <b id="pnl-max">—</b> &nbsp;·&nbsp;
        min <b id="pnl-min">—</b> &nbsp;·&nbsp;
        DD <b id="pnl-dd" style="color:#ef5350;">—</b>
      </span>
    </div>
  </div>
  <div id="pane-resizer" class="hidden" title="Drag to resize the OI pane (15% – 50%)"></div>
  <div id="oi-pane" class="hidden">
    <div id="oi-label" class="hidden">
      ATM±1 OI &nbsp;·&nbsp; <span class="ce">CE <b id="ce-val">—</b></span>
      &nbsp;·&nbsp; <span class="pe">PE <b id="pe-val">—</b></span>
      &nbsp;·&nbsp; ATM <b id="atm-val">—</b>
    </div>
  </div>
  <div id="cum-resizer" class="hidden" title="Drag to resize the Σ-N pane (10% – 40%)"></div>
  <div id="cum-pane" class="hidden">
    <div id="cum-label" class="hidden">
      ATM±1 OI · Σ(<b id="cum-win">10</b>) &nbsp;·&nbsp;
      <span class="ce">CE <b id="cum-ce-val">—</b></span> &nbsp;·&nbsp;
      <span class="pe">PE <b id="cum-pe-val">—</b></span>
    </div>
  </div>
  <div id="chg-resizer" class="hidden" title="Drag to resize ΔOI pane (8% – 35%)"></div>
  <div id="chg-pane"    class="hidden">
    <div id="chg-label" class="hidden">
      ΔOI since 09:15 &nbsp;·&nbsp;
      <span class="ce">CE <b id="chg-ce-val">—</b></span> &nbsp;·&nbsp;
      <span class="pe">PE <b id="chg-pe-val">—</b></span>
    </div>
  </div>
  <div id="tot-resizer" class="hidden" title="Drag to resize ΣOI pane (8% – 35%)"></div>
  <div id="tot-pane"    class="hidden">
    <div id="tot-label" class="hidden">
      ΣOI since 09:15 &nbsp;·&nbsp;
      <span class="ce">CE <b id="tot-ce-val">—</b></span> &nbsp;·&nbsp;
      <span class="pe">PE <b id="tot-pe-val">—</b></span> &nbsp;·&nbsp;
      <b id="tot-xcount">—</b> ✕
    </div>
  </div>
</div>

<script>
const APP_BASE = "__APP_BASE__";
const apiUrl = (p) => (APP_BASE || "") + p;

const state = {
  tf: "3m",
  chart: null,
  candle: null,
  bars: [],
  // OI sub-pane
  oiChart: null,
  ceLine: null, peLine: null,
  ceBars: [], peBars: [],
  showOi: false,  // hidden by user request — only candle chart + BACKTEST
  // Per-tf cache so toggling/switching is instant after first load
  oiByTf: {},
  // Cumulative-Σ sub-pane (rolling sum of ATM±1 OI over the last N candles)
  cumChart: null,
  ceCumLine: null, peCumLine: null,
  ceCumBars: [], peCumBars: [],
  showCum: false,          // hidden by user request — only ATM·OI + BACKTEST visible
  cumWindow: 10,           // N candles of the active TF (e.g. 10×3min=30min on 3m TF)
  cumByTf: {},             // cache: { "1d": {ce: [...], pe: [...]}, ... }
  // Per-day cumulative OI CHANGES from 09:15 (ΔOI). Default OFF.
  chgChart: null,
  ceChgLine: null, peChgLine: null,
  ceChgBars: [], peChgBars: [],
  showChg: false,
  chgByTf: {},
  // Per-day cumulative OI TOTAL from 09:15 (ΣOI = running sum of snapshots).
  // Default OFF.
  totChart: null,
  ceTotLine: null, peTotLine: null,
  ceTotBars: [], peTotBars: [],
  showTot: false,
  totByTf: {},
  // Strategy markers
  showStrategy: false,
  signals: [],     // list of {day, signal, signal_time, day_open_time, ...}
  // Σ·10 entry markers (TF-aware smoothed-OI crossover + raw OI confirm).
  // Signals depend on the active TF — cached per-TF so a TF switch picks
  // up the right signal set without a refetch round-trip.
  showSig5: false,
  sig5: [],         // current TF's entries
  sig5ByTf: {},     // { "1m": [...], "3m": [...], ... }
  // ΣOI entry markers (per-day cumulative-OI crossover since 09:15)
  showSigT: false,
  sigT: [],
  // VIEW state — replaces the older multi-knob BACKTEST. User picks one
  // of three view modes (Σ·10 only / ΣOI only / both with dedup) and
  // SL/TGT. Trades auto-load on each mode change or RELOAD click.
  bt: {
    mode: 'both',           // 'sigma10' | 'sigma_oi' | 'both'
    sl_pct: 0.18,           // initial SL ≈ 40 pts at 22k spot
    tgt_pct: 0.50,
    trail: true,            // MULTI-STEP staircase trail by trail_pct pts
    trail_pct: 0.135,       // trail step ≈ 30 pts at 22k spot (decoupled from SL)
    position_mode: 'single',// 'opposite_ok' | 'single' (single = block ALL re-entries while in any trade)
    trades: [],
    stats: null,
    running: false,
  },
  // Hover-tooltip indexes — populated by rebuildHoverIndexes() each time
  // applyAllMarkers runs. Initialized empty here so an early crosshair-move
  // (before the first trade fetch resolves) doesn't crash with "Cannot read
  // properties of undefined".
  tradeByTime:  Object.create(null),
  signalByTime: Object.create(null),
  // Cumulative-PnL sub-pane — populated whenever loadTrades resolves.
  pnlChart: null,
  pnlLine:  null,
  pnlBars:  [],   // [{time, value}]
};

const $ = (id) => document.getElementById(id);

function fmtPrice(v) { return v == null ? "—" : v.toFixed(2); }
function fmtDateUnix(t) {
  const d = new Date(t * 1000);
  // The bars carry IST timestamps encoded as UTC seconds at the same wall-clock,
  // so calling getUTCxxx gives back the original IST date.
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth()+1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function setupChart() {
  const baseOpts = {
    autoSize: true,   // chart owns a ResizeObserver — no more manual applyOptions on every mousemove
    layout: { background: { color: "#0b0d11" }, textColor: "#e7eaf1" },
    grid: {
      vertLines: { color: "#1a1f2b", visible: true },
      horzLines: { color: "#1a1f2b" },
    },
    crosshair: { mode: 0 },
    rightPriceScale: { borderColor: "#2b3142" },
  };
  state.chart = LightweightCharts.createChart($("chart"), Object.assign({}, baseOpts, {
    timeScale: {
      borderColor: "#2b3142",
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 12,
      barSpacing: 8,
      minBarSpacing: 1.5,
      tickMarkFormatter: (t, tickType) => {
        const d = new Date(t * 1000);
        const yr = d.getUTCFullYear();
        const mo = d.toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" });
        const day = d.getUTCDate();
        if (state.tf === "1d") {
          // For 1d ticks, show "5 Jan '22"
          return `${day} ${mo} '${String(yr).slice(2)}`;
        }
        // For intraday, show "Jan 5 14:30"
        const hh = String(d.getUTCHours()).padStart(2, "0");
        const mm = String(d.getUTCMinutes()).padStart(2, "0");
        return `${mo} ${day} ${hh}:${mm}`;
      },
    },
  }));
  state.candle = state.chart.addCandlestickSeries({
    upColor: "#26a69a", downColor: "#ef5350",
    borderUpColor: "#26a69a", borderDownColor: "#ef5350",
    wickUpColor: "#26a69a", wickDownColor: "#ef5350",
  });

  // CUMULATIVE-PnL sub-pane — area series painted blue above zero, red
  // below. Wrapped in try/catch so an unexpected lightweight-charts API
  // change can't take the whole page down: if PnL creation fails, the
  // candle chart still renders and trade markers still appear.
  try {
    state.pnlChart = LightweightCharts.createChart($("pnl-pane"), Object.assign({}, baseOpts, {
      // Date axis is ALWAYS visible on the PnL pane, formatted as
      // "<Month> '<yy>" (e.g. "Jan '22"). The custom tickMarkFormatter
      // suppresses the intraday hh:mm format that lightweight-charts
      // defaults to for short ranges.
      timeScale: {
        borderColor: "#2b3142",
        timeVisible: false,
        secondsVisible: false,
        visible: true,
        tickMarkFormatter: (t) => {
          const d = new Date(t * 1000);
          const mo = d.toLocaleDateString("en-GB",
              { month: "short", timeZone: "UTC" });
          const yr = String(d.getUTCFullYear()).slice(2);
          return `${mo} '${yr}`;
        },
      },
      rightPriceScale: { borderColor: "#2b3142" },
    }));
    state.pnlLine = state.pnlChart.addAreaSeries({
      lineColor: "#60a5fa",
      topColor: "rgba(96,165,250,0.35)",
      bottomColor: "rgba(96,165,250,0.02)",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
    });
  } catch (e) {
    console.error('[setupChart] PnL pane init failed:', e);
    state.pnlChart = null;
    state.pnlLine = null;
  }

  // OI sub-pane chart — two line series (CE teal / PE red), x-axis hidden
  // (synced with the main chart's axis via subscribeVisibleLogicalRangeChange)
  state.oiChart = LightweightCharts.createChart($("oi-pane"), Object.assign({}, baseOpts, {
    timeScale: { borderColor: "#2b3142", timeVisible: false, secondsVisible: false,
                 visible: false },
    rightPriceScale: { borderColor: "#2b3142" },
  }));
  // Compact OI formatter: 12,299,040 -> "12.3M", 850000 -> "850K".
  // Without this the OI right-axis labels are 8-10 chars wide vs. ~6 for
  // the candle prices, so the two charts' plot areas don't end at the
  // same X coordinate and time positions visually drift.
  const oiFmt = {
    type: 'custom',
    formatter: (v) => {
      if (v >= 1e7) return (v/1e7).toFixed(1) + 'Cr';
      if (v >= 1e5) return (v/1e5).toFixed(1) + 'L';
      if (v >= 1e3) return (v/1e3).toFixed(0) + 'K';
      return String(v|0);
    },
  };
  state.ceLine = state.oiChart.addLineSeries({
    color: "#26a69a", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: oiFmt,
  });
  state.peLine = state.oiChart.addLineSeries({
    color: "#ef5350", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: oiFmt,
  });
  // Third pane — rolling Σ(N) of ATM±1 OI. Same shape as the OI chart so
  // the visual stack is consistent.
  state.cumChart = LightweightCharts.createChart($("cum-pane"), Object.assign({}, baseOpts, {
    timeScale: { borderColor: "#2b3142", timeVisible: false, secondsVisible: false,
                 visible: false },
    rightPriceScale: { borderColor: "#2b3142" },
  }));
  state.ceCumLine = state.cumChart.addLineSeries({
    color: "#26a69a", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: oiFmt,
  });
  state.peCumLine = state.cumChart.addLineSeries({
    color: "#ef5350", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: oiFmt,
  });

  // Fourth + fifth panes — ΔOI (per-day cum change since 09:15) and
  // ΣOI (per-day cum total since 09:15). Created here even though they're
  // default-hidden so the toggle handlers just need to flip CSS, not
  // lazy-create charts.
  const subPaneOpts = Object.assign({}, baseOpts, {
    timeScale: { borderColor: "#2b3142", timeVisible: false, secondsVisible: false,
                 visible: false },
    rightPriceScale: { borderColor: "#2b3142" },
  });
  state.chgChart = LightweightCharts.createChart($("chg-pane"), subPaneOpts);
  state.ceChgLine = state.chgChart.addLineSeries({
    color: "#26a69a", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false, priceFormat: oiFmt,
  });
  state.peChgLine = state.chgChart.addLineSeries({
    color: "#ef5350", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false, priceFormat: oiFmt,
  });
  state.totChart = LightweightCharts.createChart($("tot-pane"), subPaneOpts);
  state.ceTotLine = state.totChart.addLineSeries({
    color: "#26a69a", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false, priceFormat: oiFmt,
  });
  state.peTotLine = state.totChart.addLineSeries({
    color: "#ef5350", lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false, priceFormat: oiFmt,
  });
  // Lock every pane's right-gutter width so a given time t lands at the
  // same X coordinate everywhere — required for the candle and OI lines
  // to visually align across panes.
  const GUTTER = 64;
  // The cumulative-PnL pane is INTENTIONALLY excluded from the sync loop
  // — the user wants the full multi-year history visible at once, not
  // following the candle chart's pan/zoom. PnL fitContent runs after each
  // setData inside updateCumPnLChart.
  const allCharts = [state.chart, state.oiChart, state.cumChart,
                     state.chgChart, state.totChart].filter(c => c);
  for (const c of allCharts) {
    try { c.priceScale('right').applyOptions({ minimumWidth: GUTTER }); } catch (_) {}
  }
  // Time-axis sync across every pane. We can't use logical range because
  // the series carry different bar counts (1591 daily candles vs 1569
  // daily OI points across 2020-2026). subscribeVisibleTimeRangeChange
  // + setVisibleRange keeps every pane locked to the same wall-clock
  // window regardless. A single shared `syncing` guard prevents the
  // chain from echoing back into a loop.
  let syncing = false;
  const propagate = (src, dsts) => {
    src.timeScale().subscribeVisibleTimeRangeChange((r) => {
      if (syncing || !r || r.from == null || r.to == null) return;
      syncing = true;
      for (const d of dsts) {
        try { d.timeScale().setVisibleRange({ from: r.from, to: r.to }); } catch (_) {}
      }
      syncing = false;
    });
  };
  for (const src of allCharts) {
    propagate(src, allCharts.filter(c => c !== src));
  }

  // Re-render markers when the visible range changes, throttled via rAF
  // so it runs at most once per frame even if the user pans fast. Covers
  // both marker sources — STRATEGY + Σ·5 ENTRY — via applyAllMarkers().
  let markerPending = false;
  state.chart.timeScale().subscribeVisibleTimeRangeChange(() => {
    if ((!state.showStrategy && !state.showSig5 && !state.showSigT) || markerPending) return;
    markerPending = true;
    requestAnimationFrame(() => {
      markerPending = false;
      applyAllMarkers();
    });
  });

  // ── Crosshair sync ─────────────────────────────────────────────
  // Hovering EITHER pane draws the dashed crosshair on BOTH at the same
  // time-cursor. The horizontal line on the "other" chart sits at the
  // value of that other chart's series at the hovered time, so each pane
  // shows its own scale-appropriate reading.
  let xhSyncing = false;
  // Helper: project a `time` cursor onto the OTHER two charts. Each pane
  // displays its own series at the same wall-clock time, so the horizontal
  // line on every pane sits at its own scale-appropriate value.
  const projectXh = (sourceChart, time) => {
    if (xhSyncing) return;
    xhSyncing = true;
    try {
      const targets = [
        { chart: state.chart,    bars: state.bars,
          series: state.candle,  val: (b) => b.close },
        { chart: state.oiChart,  bars: state.ceBars,
          series: state.ceLine,  val: (b) => b.value },
        { chart: state.cumChart, bars: state.ceCumBars,
          series: state.ceCumLine, val: (b) => b.value },
        { chart: state.chgChart, bars: state.ceChgBars,
          series: state.ceChgLine, val: (b) => b.value },
        { chart: state.totChart, bars: state.ceTotBars,
          series: state.ceTotLine, val: (b) => b.value },
      ];
      for (const t of targets) {
        if (!t.chart || t.chart === sourceChart || !t.series) continue;
        if (time == null) { t.chart.clearCrosshairPosition(); continue; }
        const b = findAtTime(t.bars, time);
        if (b != null) t.chart.setCrosshairPosition(t.val(b), time, t.series);
      }
    } catch (_) {}
    xhSyncing = false;
  };
  state.chart.subscribeCrosshairMove((p) => {
    if (p && p.seriesData) {
      const c = p.seriesData.get(state.candle);
      if (c && c.close != null) $("last-val").textContent = fmtPrice(c.close);
    }
    projectXh(state.chart, p && p.time);
  });
  state.oiChart.subscribeCrosshairMove((p) => {
    // OI readouts in the sub-pane label
    if (p && p.seriesData) {
      const ce = p.seriesData.get(state.ceLine);
      const pe = p.seriesData.get(state.peLine);
      if (ce) $("ce-val").textContent = (ce.value/1e5).toFixed(2) + " L";
      if (pe) $("pe-val").textContent = (pe.value/1e5).toFixed(2) + " L";
    }
    projectXh(state.oiChart, p && p.time);
  });
  state.cumChart.subscribeCrosshairMove((p) => {
    // Cumulative-Σ readouts in the third-pane label
    if (p && p.seriesData) {
      const ce = p.seriesData.get(state.ceCumLine);
      const pe = p.seriesData.get(state.peCumLine);
      if (ce) $("cum-ce-val").textContent = (ce.value/1e5).toFixed(2) + " L";
      if (pe) $("cum-pe-val").textContent = (pe.value/1e5).toFixed(2) + " L";
    }
    projectXh(state.cumChart, p && p.time);
  });
  // Helper: format ΔOI values — they can be NEGATIVE (OI unwound below
  // the day's open), so render with sign and the same Lakh formatting
  // used in the other panes.
  const fmtSigned = (v) => (v >= 0 ? "+" : "") + (v / 1e5).toFixed(2) + " L";
  state.chgChart.subscribeCrosshairMove((p) => {
    if (p && p.seriesData) {
      const ce = p.seriesData.get(state.ceChgLine);
      const pe = p.seriesData.get(state.peChgLine);
      if (ce) $("chg-ce-val").textContent = fmtSigned(ce.value);
      if (pe) $("chg-pe-val").textContent = fmtSigned(pe.value);
    }
    projectXh(state.chgChart, p && p.time);
  });
  state.totChart.subscribeCrosshairMove((p) => {
    if (p && p.seriesData) {
      const ce = p.seriesData.get(state.ceTotLine);
      const pe = p.seriesData.get(state.peTotLine);
      if (ce) $("tot-ce-val").textContent = (ce.value/1e5).toFixed(2) + " L";
      if (pe) $("tot-pe-val").textContent = (pe.value/1e5).toFixed(2) + " L";
    }
    projectXh(state.totChart, p && p.time);
  });
}

// ── Marker converters ──────────────────────────────────────────────────
// Both strategy sources (STRATEGY = multi-strike crossover, Σ·5 = smoothed)
// produce candle-overlay markers via the same lightweight-charts setMarkers
// API. We keep two converters because the underlying signal objects have
// different field names + we want their text labels to be distinguishable.

// Marker-time helper: timestamps that come out of the strategy API are
// IST wall-clock encoded as a naive ISO string; on intraday TFs we use
// the exact signal minute, on 1d we anchor to the day's 09:15 candle.
function _markerTime(isoStr) {
  return Math.floor(new Date(isoStr + (isoStr.endsWith('Z') ? '' : 'Z')).getTime() / 1000);
}

// STRATEGY (multi-strike crossover) — green BUY ▲ / red SELL ▼.
function signalToMarker(s) {
  const isDaily = state.tf === '1d';
  const t = _markerTime(isDaily ? s.day_open_time : s.signal_time);
  const spot = Math.round(s.signal_spot).toLocaleString();
  if (s.signal === 'BULLISH') {
    return { time: t, position: 'belowBar', color: '#22c55e',
             shape: 'arrowUp',  text: 'BUY ' + spot, size: 2 };
  }
  return   { time: t, position: 'aboveBar', color: '#ef4444',
             shape: 'arrowDown', text: 'SELL ' + spot, size: 2 };
}

// Σ·5 ENTRY — cyan LONG ▲ / orange SHORT ▼ (distinct hue from STRATEGY).
function entryToMarker(e) {
  const isDaily = state.tf === '1d';
  const t = _markerTime(isDaily ? e.day_open_time : e.entry_time);
  const spot = Math.round(e.entry_spot).toLocaleString();
  if (e.side === 'LONG') {
    return { time: t, position: 'belowBar', color: '#06b6d4',
             shape: 'arrowUp',  text: 'Σ LONG ' + spot, size: 2 };
  }
  return   { time: t, position: 'aboveBar', color: '#f59e0b',
             shape: 'arrowDown', text: 'Σ SHORT ' + spot, size: 2 };
}

// ΣOI ENTRY (per-day cum-OI cross since 09:15). Teal LONG ▲ / magenta
// SHORT ▼ — distinct from BOTH the STRATEGY (green/red) and Σ·5
// (cyan/orange) palettes so all three marker sources can co-exist.
function totEntryToMarker(e) {
  const isDaily = state.tf === '1d';
  const t = _markerTime(isDaily ? e.day_open_time : e.entry_time);
  const spot = Math.round(e.entry_spot).toLocaleString();
  if (e.side === 'LONG') {
    return { time: t, position: 'belowBar', color: '#14b8a6',
             shape: 'arrowUp',  text: 'ΣOI LONG ' + spot, size: 2 };
  }
  return   { time: t, position: 'aboveBar', color: '#c026d3',
             shape: 'arrowDown', text: 'ΣOI SHORT ' + spot, size: 2 };
}

// Drawing 1,400+ markers at once stresses the chart on every pan/zoom — the
// browser repaints all of them even when only a few are visible. We render
// only signals inside the current visible time range (plus a small buffer)
// and re-render whenever the user pans/zooms. Net: ~50-300 markers on
// screen even at the full 5-year zoom — and we merge both marker sources
// (STRATEGY + Σ·5 ENTRY) into a single sorted call to setMarkers.
function applyAllMarkers() {
  if (!state.candle) return;
  const isDaily = state.tf === '1d';
  const ms = [];

  // Two display modes:
  //   (1) Pre-backtest: show raw ΣOI entry signals (auto-loaded on page init).
  //   (2) Post-backtest: hide raw signals, show paired entry+exit trade
  //       markers colored by P&L. Tooltip on hover shows SL/TGT/exit/P&L.
  const hasTrades = state.bt.trades && state.bt.trades.length > 0;
  if (!hasTrades && state.sigT && state.sigT.length) {
    for (const e of state.sigT) ms.push(totEntryToMarker(e));
  }
  // BACKTEST markers — paired entry+exit per trade, colored by P&L. On 1d
  // we anchor both ends to the day's 09:15 candle. On intraday TFs we
  // place each end on its actual timestamp.
  if (hasTrades) {
    for (const tr of state.bt.trades) {
      const win = tr.pnl_pts > 0;
      const entryT = _markerTime(isDaily ? tr.day_open_time : tr.entry_time);
      const exitT  = _markerTime(isDaily ? tr.day_open_time : tr.exit_time);
      const isTrail = tr.reason === 'TRAIL';
      const entryColor = isTrail ? '#fbbf24'
                       : win     ? '#22c55e' : '#ef4444';
      const exitColor  = tr.reason === 'TGT'   ? '#16a34a'
                       : tr.reason === 'SL'    ? '#b91c1c'
                       : tr.reason === 'TRAIL' ? '#fbbf24'
                       :                          '#9ca3af';   // EOD
      const shape  = tr.side === 'LONG' ? 'arrowUp' : 'arrowDown';
      const pos    = tr.side === 'LONG' ? 'belowBar' : 'aboveBar';
      const pnl    = tr.pnl_pts.toFixed(1);
      const sign   = tr.pnl_pts >= 0 ? '+' : '';
      ms.push({
        time: entryT, position: pos, color: entryColor, shape: shape,
        text: `${tr.side[0]} ${Math.round(tr.entry_spot)} → ${sign}${pnl}pt`,
        size: 2,
      });
      // Exit marker — small circle, opposite side of the bar from entry.
      if (!isDaily && exitT !== entryT) {
        ms.push({
          time: exitT,
          position: tr.side === 'LONG' ? 'aboveBar' : 'belowBar',
          color: exitColor, shape: 'circle',
          text: `${tr.reason} ${sign}${pnl}`, size: 1,
        });
      }
    }
  }
  ms.sort((a, b) => a.time - b.time);
  state.candle.setMarkers(ms);
  rebuildHoverIndexes();
}

// ── HOVER TOOLTIP — rich rectangle popup on entry/exit arrow hover ─────
// Lightweight-charts markers only have a `text` field for native tooltip.
// We replace that with an absolute-positioned div populated on each
// crosshair-move from a {bar_time → trade/signal} index. The index is
// rebuilt every time applyAllMarkers() runs.
function rebuildHoverIndexes() {
  state.tradeByTime  = Object.create(null);
  state.signalByTime = Object.create(null);
  const isDaily = state.tf === '1d';
  if (state.bt.trades && state.bt.trades.length) {
    for (const tr of state.bt.trades) {
      const eT = _markerTime(isDaily ? tr.day_open_time : tr.entry_time);
      const xT = _markerTime(isDaily ? tr.day_open_time : tr.exit_time);
      (state.tradeByTime[eT] = state.tradeByTime[eT] || []).push(tr);
      if (xT !== eT)
        (state.tradeByTime[xT] = state.tradeByTime[xT] || []).push(tr);
    }
  } else if (state.sigT && state.sigT.length) {
    for (const s of state.sigT) {
      const t = _markerTime(isDaily ? s.day_open_time : s.entry_time);
      (state.signalByTime[t] = state.signalByTime[t] || []).push(s);
    }
  }
}

function ensureTooltipEl() {
  if (state.tipEl) return state.tipEl;
  const tip = document.createElement('div');
  tip.id = 'bt-tip';
  tip.style.cssText = [
    'position:absolute', 'display:none', 'z-index:1000',
    'background:rgba(11,13,17,0.97)', 'border:1px solid #60a5fa',
    'border-radius:6px', 'padding:10px 14px', 'color:#e7eaf1',
    'font-size:12px', 'line-height:1.65',
    'font-family:inherit', 'pointer-events:none',
    'box-shadow:0 6px 24px rgba(0,0,0,0.5)', 'min-width:230px',
    'white-space:nowrap'
  ].join(';');
  $("chart").appendChild(tip);
  state.tipEl = tip;
  return tip;
}

function positionTip(tip, pt) {
  const rect = $("chart").getBoundingClientRect();
  const tipR = tip.getBoundingClientRect();
  const w = tipR.width  || 240;
  const h = tipR.height || 180;
  let x = pt.x + 16, y = pt.y + 16;
  if (x + w > rect.width)  x = pt.x - w - 16;
  if (y + h > rect.height) y = pt.y - h - 16;
  if (x < 0) x = 4;
  if (y < 0) y = 4;
  tip.style.left = x + 'px';
  tip.style.top  = y + 'px';
  tip.style.display = 'block';
}

function _hhmm(iso) { return (iso || '').slice(11, 16); }
function _day(d)    { return String(d).slice(0, 10); }
function _f(n, d=2) { return (n == null || isNaN(n)) ? '—' : n.toFixed(d); }

function renderTradeTooltip(tip, trades, pt) {
  // Choose first trade — multi-trade collisions are rare on intraday TF.
  const tr = trades[0];
  const isWin = tr.pnl_pts > 0;
  const sl  = state.bt.sl_pct, tgt = state.bt.tgt_pct;
  const sl_px  = tr.side === 'LONG' ? tr.entry_spot * (1 - sl/100)  : tr.entry_spot * (1 + sl/100);
  const tgt_px = tr.side === 'LONG' ? tr.entry_spot * (1 + tgt/100) : tr.entry_spot * (1 - tgt/100);
  const sign   = tr.pnl_pts >= 0 ? '+' : '';
  const pnlCol = isWin ? '#22c55e' : '#ef4444';
  const reasonCol = tr.reason === 'TGT'   ? '#22c55e'
                  : tr.reason === 'SL'    ? '#ef4444'
                  : tr.reason === 'TRAIL' ? '#fbbf24'
                  :                          '#9ca3af';
  const sideCol = tr.side === 'LONG' ? '#22c55e' : '#ef4444';
  const srcLabel = tr.source === 'sigma10' ? 'Σ·10'
                : tr.source === 'sigma_oi' ? 'ΣOI'
                :                              (state.bt.mode || '');
  const stratName = srcLabel;
  const td = (label, val, color) =>
    `<tr><td style="color:#7b8294; padding-right:18px;">${label}</td>
         <td${color?` style="color:${color};"`:''}>${val}</td></tr>`;
  tip.innerHTML = `
    <div style="font-weight:700; color:${sideCol}; font-size:13px; margin-bottom:6px;">
      ${tr.side} · ${_day(tr.day)}
      <span style="color:#7b8294; font-weight:500; font-size:10px; margin-left:6px;">${stratName}</span>
    </div>
    <table style="border-collapse:collapse;">
      ${td('Entry',  `<b>${_f(tr.entry_spot)}</b> @ ${_hhmm(tr.entry_time)}`)}
      ${td('SL',     `${_f(sl_px)}  (-${sl}%)`, '#ef4444')}
      ${td('Target', `${_f(tgt_px)}  (+${tgt}%)`, '#22c55e')}
      ${td('Exit',   `<b>${_f(tr.exit_spot)}</b> @ ${_hhmm(tr.exit_time)} · <span style="color:${reasonCol}">${tr.reason}</span>`)}
    </table>
    <div style="margin-top:6px; padding-top:6px; border-top:1px solid #2b3142;">
      <span style="color:#7b8294;">P&amp;L:</span>
      <b style="color:${pnlCol}; font-size:13px;">${sign}${_f(tr.pnl_pts, 1)} pts</b>
      <span style="color:${pnlCol}; margin-left:6px;">(${sign}${_f(tr.pnl_pct)}%)</span>
    </div>
    ${trades.length > 1
      ? `<div style="margin-top:4px; color:#7b8294; font-size:10px;">+${trades.length - 1} more trade(s) at this time</div>`
      : ''}`;
  positionTip(tip, pt);
}

function renderSignalTooltip(tip, sigs, pt) {
  const s = sigs[0];
  const sideCol = s.side === 'LONG' ? '#14b8a6' : '#c026d3';
  tip.innerHTML = `
    <div style="font-weight:700; color:${sideCol}; font-size:13px; margin-bottom:6px;">
      ΣOI ${s.side} signal · ${_day(s.day)}
    </div>
    <table style="border-collapse:collapse;">
      <tr><td style="color:#7b8294; padding-right:18px;">Time</td><td>${_hhmm(s.entry_time)}</td></tr>
      <tr><td style="color:#7b8294;">Spot</td><td><b>${_f(s.entry_spot)}</b></td></tr>
      <tr><td style="color:#7b8294;">CE OI</td><td>${(s.entry_ce_oi/1e5).toFixed(2)} L</td></tr>
      <tr><td style="color:#7b8294;">PE OI</td><td>${(s.entry_pe_oi/1e5).toFixed(2)} L</td></tr>
    </table>
    <div style="margin-top:6px; padding-top:6px; border-top:1px solid #2b3142; color:#7b8294; font-size:10px;">
      Click BACKTEST → RUN to see SL/Target/PnL outcome
    </div>`;
  positionTip(tip, pt);
}

function setupHoverTooltip() {
  const tip = ensureTooltipEl();
  state.chart.subscribeCrosshairMove((param) => {
    if (!param || !param.point || param.time == null) {
      tip.style.display = 'none'; return;
    }
    const t = typeof param.time === 'number' ? param.time : Number(param.time);
    const trades = state.tradeByTime[t];
    if (trades && trades.length) { renderTradeTooltip(tip, trades, param.point); return; }
    const sigs   = state.signalByTime[t];
    if (sigs   && sigs.length)   { renderSignalTooltip(tip, sigs, param.point); return; }
    tip.style.display = 'none';
  });
}

// ── CUMULATIVE PnL ─────────────────────────────────────────────────────
// Build a running-sum series of trade PnL, sorted by exit_time. Points
// land where each trade RESOLVES (its exit timestamp), so the curve steps
// up/down at SL/TGT/EOD/TRAIL events rather than at entry.
function updateCumPnLChart() {
  if (!state.pnlLine) return;
  const trades = (state.bt.trades || []).slice();
  if (!trades.length) {
    state.pnlBars = [];
    state.pnlLine.setData([]);
    $('pnl-final').textContent = '—';
    $('pnl-max').textContent = '—';
    $('pnl-min').textContent = '—';
    $('pnl-dd').textContent  = '—';
    return;
  }
  trades.sort((a, b) =>
    _markerTime(a.exit_time) - _markerTime(b.exit_time));
  const bars = [];
  let cum = 0, mx = 0, mn = 0, peak = 0, maxDD = 0;
  // Bucket by exit-time (same second collisions get merged)
  let lastT = null, lastV = 0;
  for (const tr of trades) {
    cum += tr.pnl_pts;
    if (cum > mx) mx = cum;
    if (cum < mn) mn = cum;
    if (cum > peak) peak = cum;
    const dd = peak - cum;
    if (dd > maxDD) maxDD = dd;
    const t = _markerTime(tr.exit_time);
    if (t === lastT) {
      // Same-second exit — overwrite the running value
      lastV = cum;
      bars[bars.length - 1] = { time: t, value: cum };
    } else {
      bars.push({ time: t, value: cum });
      lastT = t; lastV = cum;
    }
  }
  state.pnlBars = bars;
  state.pnlLine.setData(bars);
  // Recolor the area top: green if final > 0, red if final < 0
  const final = cum;
  if (final >= 0) {
    state.pnlLine.applyOptions({
      lineColor: "#22c55e",
      topColor: "rgba(34,197,94,0.35)",
      bottomColor: "rgba(34,197,94,0.02)",
    });
  } else {
    state.pnlLine.applyOptions({
      lineColor: "#ef4444",
      topColor: "rgba(239,68,68,0.35)",
      bottomColor: "rgba(239,68,68,0.02)",
    });
  }
  const fmt = (v) => (v >= 0 ? '+' : '') + v.toLocaleString(undefined, {maximumFractionDigits: 0});
  $('pnl-final').textContent = fmt(final);
  $('pnl-max').textContent   = fmt(mx);
  $('pnl-min').textContent   = fmt(mn);
  $('pnl-dd').textContent    = '-' + maxDD.toLocaleString(undefined, {maximumFractionDigits: 0});
  $('pnl-final').style.color = final >= 0 ? '#22c55e' : '#ef4444';
  // Fit the FULL trade history into the pane — the user wants every year
  // visible at once and the pane does NOT follow candle chart pan/zoom.
  try { state.pnlChart.timeScale().fitContent(); } catch (_) {}
}

// ── TRADES VIEW ────────────────────────────────────────────────────────
// Everything except position_mode is locked to the winning defaults; the
// panel only exposes the Opp OK / Single toggle. Mode/SL/TGT/Trail live
// in state.bt only.
async function loadTrades() {
  if (state.bt.running) return;
  state.bt.running = true;
  const activePosBtn = $('v-pos-grp').querySelector('button.on');
  if (activePosBtn) state.bt.position_mode = activePosBtn.dataset.pm;
  try {
    const q = new URLSearchParams({
      mode:    state.bt.mode,
      sl_pct:  state.bt.sl_pct,
      tgt_pct: state.bt.tgt_pct,
      trail:     state.bt.trail ? 'true' : 'false',
      trail_pct: state.bt.trail_pct,
      position_mode: state.bt.position_mode,
    }).toString();
    const r = await fetch(apiUrl('/api/merged_trades?' + q));
    const j = await r.json();
    if (j.error) { alert('Load failed: ' + j.error); return; }
    state.bt.trades = j.trades || [];
    state.bt.stats  = j.stats  || null;
    renderViewStats();
    applyAllMarkers();
    try { updateCumPnLChart(); } catch (e) { console.error('[updateCumPnL] failed:', e); }
    if (state.bt.trades.length) {
      try { state.chart.timeScale().fitContent(); } catch (_) {}
    }
  } catch (e) {
    alert('Load failed: ' + e);
  } finally {
    state.bt.running = false;
  }
}

function renderViewStats() {
  const s = state.bt.stats;
  const box = $('v-stats');
  if (!s) { box.style.display = 'none'; return; }
  box.style.display = 'inline-block';
  const fmt = (v, d=1) => (v == null ? '—' : v.toLocaleString(undefined, {maximumFractionDigits: d}));
  $('v-n').textContent        = fmt(s.n, 0);
  $('v-wr').textContent       = fmt(s.win_rate, 1) + '%';
  $('v-totpnl').textContent   = (s.total_pnl >= 0 ? '+' : '') + fmt(s.total_pnl, 0);
  $('v-pf').textContent       = s.pf == null ? '∞' : fmt(s.pf, 2);
  $('v-sl-hits').textContent  = fmt(s.sl_hits, 0);
  $('v-tgt-hits').textContent = fmt(s.tgt_hits, 0);
  $('v-eod').textContent      = fmt(s.eod_exits, 0);
  $('v-trail-hits').textContent = fmt(s.trail_hits || 0, 0);
  $('v-per-day').textContent  = fmt(s.avg_per_day || 0, 2);
  $('v-long').textContent     = fmt(s.long_n, 0);
  $('v-short').textContent    = fmt(s.short_n, 0);
  $('v-src-s10').textContent  = fmt(s.sigma10_n, 0);
  $('v-src-soi').textContent  = fmt(s.sigma_oi_n, 0);
  $('v-totpnl').style.color  = s.total_pnl >= 0 ? '#22c55e' : '#ef4444';
}

async function loadStrategySignals() {
  if (state.signals.length) { applyAllMarkers(); return; }
  try {
    const r = await fetch(apiUrl('/api/strategy/multi_strike_oi_crossover'));
    const j = await r.json();
    state.signals = j.signals || [];
    const st = j.stats || {};
    $('ss-bull').textContent = (st.bull || 0).toLocaleString();
    $('ss-bear').textContent = (st.bear || 0).toLocaleString();
    $('ss-no').textContent   = (st.no_signal || 0).toLocaleString();
    applyAllMarkers();
  } catch (_) {}
}

async function loadSig5Entries() {
  const tf = state.tf;
  // Hit the cache before the network — most TF switches resolve instantly.
  if (state.sig5ByTf[tf]) {
    state.sig5 = state.sig5ByTf[tf];
    applyAllMarkers();
    // Also repaint the stat badge from the cached counts (recompute cheap)
    let l=0,s=0;
    for (const e of state.sig5) (e.side === 'LONG') ? l++ : s++;
    $('s5-long').textContent  = l.toLocaleString();
    $('s5-short').textContent = s.toLocaleString();
    return;
  }
  try {
    const r = await fetch(apiUrl(`/api/strategy/sigma5_entries?tf=${tf}`));
    const j = await r.json();
    state.sig5 = j.entries || [];
    state.sig5ByTf[tf] = state.sig5;
    const st = j.stats || {};
    $('s5-long').textContent  = (st.long  || 0).toLocaleString();
    $('s5-short').textContent = (st.short || 0).toLocaleString();
    $('s5-no').textContent    = (st.no_signal || 0).toLocaleString();
    applyAllMarkers();
  } catch (_) {}
}

async function loadSigTEntries() {
  if (state.sigT.length) { applyAllMarkers(); return; }
  try {
    const r = await fetch(apiUrl('/api/strategy/sigma_total_entries?min_gap=7'));
    const j = await r.json();
    state.sigT = j.entries || [];
    applyAllMarkers();
  } catch (_) {}
}

function toggleStrategy() {
  state.showStrategy = !state.showStrategy;
  $('strat-stat').style.display = state.showStrategy ? 'inline' : 'none';
  const btn = $('strat-toggle');
  if (state.showStrategy) {
    btn.style.background = 'linear-gradient(135deg, #a78bfa 0%, #7c5fd6 100%)';
    btn.style.color = '#0a0c10';
    btn.style.borderColor = 'transparent';
    btn.style.fontWeight = '700';
    if (!state.showOi) toggleOi();
    loadStrategySignals();
  } else {
    btn.style.background = ''; btn.style.color = '';
    btn.style.borderColor = ''; btn.style.fontWeight = '';
    applyAllMarkers();   // re-render without strategy markers (may still show Σ·5)
  }
}

function toggleSig5() {
  state.showSig5 = !state.showSig5;
  $('sig5-stat').style.display = state.showSig5 ? 'inline' : 'none';
  const btn = $('sig5-toggle');
  if (state.showSig5) {
    btn.style.background = 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)';
    btn.style.color = '#0a0c10';
    btn.style.borderColor = 'transparent';
    btn.style.fontWeight = '700';
    // OI panel + Σ·5 panel are the trigger; auto-open OI so the entry
    // arrows have their OI context visible. Σ·5 pane defaults on already.
    if (!state.showOi) toggleOi();
    loadSig5Entries();
  } else {
    btn.style.background = ''; btn.style.color = '';
    btn.style.borderColor = ''; btn.style.fontWeight = '';
    applyAllMarkers();
  }
}

function toggleSigT() {
  state.showSigT = !state.showSigT;
  $('sigt-stat').style.display = state.showSigT ? 'inline' : 'none';
  const btn = $('sigt-toggle');
  if (state.showSigT) {
    btn.style.background = 'linear-gradient(135deg, #14b8a6 0%, #0f766e 100%)';
    btn.style.color = '#0a0c10';
    btn.style.borderColor = 'transparent';
    btn.style.fontWeight = '700';
    // The trigger PANEL for these entries is ΣOI — auto-open it so the
    // user can see the crossover that produced each marker. Also open OI
    // for raw context.
    if (!state.showTot) toggleTot();
    if (!state.showOi)  toggleOi();
    loadSigTEntries();
  } else {
    btn.style.background = ''; btn.style.color = '';
    btn.style.borderColor = ''; btn.style.fontWeight = '';
    applyAllMarkers();
  }
}

// Binary search for the bar at-or-before `time` in a sorted-by-time array.
function findAtTime(bars, time) {
  if (!bars || !bars.length) return null;
  let lo = 0, hi = bars.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (bars[mid].time <= time) lo = mid + 1;
    else hi = mid - 1;
  }
  return hi >= 0 ? bars[hi] : null;
}

async function loadTF(tf) {
  state.tf = tf;
  $("tf-grp").querySelectorAll("button").forEach(b => {
    b.classList.toggle("on", b.dataset.tf === tf);
  });
  const r = await fetch(apiUrl(`/api/nifty?tf=${tf}`));
  const j = await r.json();
  state.bars = j.bars || [];
  state.candle.setData(state.bars);
  if (state.bars.length) {
    const lastBar = state.bars[state.bars.length - 1];
    $("last-val").textContent = fmtPrice(lastBar.close);
    $("bar-count").textContent = state.bars.length.toLocaleString();
    const first = fmtDateUnix(state.bars[0].time);
    const last  = fmtDateUnix(lastBar.time);
    $("range").textContent = `${first} → ${last}`;
  }
  state.chart.timeScale().fitContent();
  // Re-load every visible sub-pane at the new TF
  if (state.showOi)  await loadOI(tf);
  if (state.showCum) await loadCum(tf);
  if (state.showChg) await loadChg(tf);
  if (state.showTot) await loadTot(tf);
  // Σ·10 entries are TF-aware (the strategy resamples to TF then takes
  // rolling-10 sum), so a TF switch requires re-fetching the signal set.
  // The other two marker sources (STRATEGY, ΣOI ENTRY) are TF-invariant
  // — we just re-anchor their existing markers.
  if (state.showSig5) await loadSig5Entries();
  if (state.showStrategy || state.showSigT) applyAllMarkers();
}

async function loadOI(tf) {
  if (state.oiByTf[tf]) {
    state.ceBars = state.oiByTf[tf].ce;
    state.peBars = state.oiByTf[tf].pe;
    state.ceLine.setData(state.ceBars);
    state.peLine.setData(state.peBars);
    syncOiToIndex();
    return;
  }
  try {
    const r = await fetch(apiUrl(`/api/nifty_oi?tf=${tf}`));
    const j = await r.json();
    state.ceBars = j.ce_oi || [];
    state.peBars = j.pe_oi || [];
    state.oiByTf[tf] = { ce: state.ceBars, pe: state.peBars };
    state.ceLine.setData(state.ceBars);
    state.peLine.setData(state.peBars);
    if (state.ceBars.length) {
      const last = state.ceBars[state.ceBars.length - 1];
      $("ce-val").textContent = (last.value/1e5).toFixed(2) + " L";
    }
    if (state.peBars.length) {
      const last = state.peBars[state.peBars.length - 1];
      $("pe-val").textContent = (last.value/1e5).toFixed(2) + " L";
    }
    syncOiToIndex();
  } catch (e) {
    state.ceBars = []; state.peBars = [];
  }
}

// Force the OI pane to match the index chart's current visible time range.
// Called after setData (when the OI chart auto-fits to its own data, which
// is wider than the index range and breaks the visual lockstep).
function syncOiToIndex() {
  if (!state.chart || !state.oiChart) return;
  try {
    const r = state.chart.timeScale().getVisibleRange();
    if (r && r.from != null && r.to != null) {
      state.oiChart.timeScale().setVisibleRange({ from: r.from, to: r.to });
    }
  } catch (_) {}
}

// Same idea for the cumulative-Σ pane.
function syncCumToIndex() {
  if (!state.chart || !state.cumChart) return;
  try {
    const r = state.chart.timeScale().getVisibleRange();
    if (r && r.from != null && r.to != null) {
      state.cumChart.timeScale().setVisibleRange({ from: r.from, to: r.to });
    }
  } catch (_) {}
}

async function loadCum(tf) {
  if (state.cumByTf[tf]) {
    state.ceCumBars = state.cumByTf[tf].ce;
    state.peCumBars = state.cumByTf[tf].pe;
    state.ceCumLine.setData(state.ceCumBars);
    state.peCumLine.setData(state.peCumBars);
    syncCumToIndex();
    return;
  }
  try {
    const r = await fetch(apiUrl(
      `/api/nifty_oi_cumsum?tf=${tf}&window=${state.cumWindow}`));
    const j = await r.json();
    state.ceCumBars = j.ce_cum || [];
    state.peCumBars = j.pe_cum || [];
    state.cumByTf[tf] = { ce: state.ceCumBars, pe: state.peCumBars };
    state.ceCumLine.setData(state.ceCumBars);
    state.peCumLine.setData(state.peCumBars);
    if (state.ceCumBars.length) {
      const last = state.ceCumBars[state.ceCumBars.length - 1];
      $("cum-ce-val").textContent = (last.value/1e5).toFixed(2) + " L";
    }
    if (state.peCumBars.length) {
      const last = state.peCumBars[state.peCumBars.length - 1];
      $("cum-pe-val").textContent = (last.value/1e5).toFixed(2) + " L";
    }
    syncCumToIndex();
  } catch (e) {
    state.ceCumBars = []; state.peCumBars = [];
  }
}

function toggleCum() {
  state.showCum = !state.showCum;
  $("cum-pane").classList.toggle("hidden", !state.showCum);
  $("cum-resizer").classList.toggle("hidden", !state.showCum);
  $("cum-label").classList.toggle("hidden", !state.showCum);
  $("cum-toggle").classList.toggle("on", state.showCum);
  const btn = $("cum-toggle");
  if (state.showCum) {
    btn.style.background = "linear-gradient(135deg, #fbbf24 0%, #d97706 100%)";
    btn.style.color = "#0a0c10";
    btn.style.borderColor = "transparent";
    btn.style.fontWeight = "700";
    loadCum(state.tf);
  } else {
    btn.style.background = ""; btn.style.color = "";
    btn.style.borderColor = ""; btn.style.fontWeight = "";
  }
}

// ── ΔOI (per-day cumulative OI CHANGES from 09:15) ──────────────────────
function syncChgToIndex() {
  if (!state.chart || !state.chgChart) return;
  try {
    const r = state.chart.timeScale().getVisibleRange();
    if (r && r.from != null && r.to != null) {
      state.chgChart.timeScale().setVisibleRange({ from: r.from, to: r.to });
    }
  } catch (_) {}
}

async function loadChg(tf) {
  if (state.chgByTf[tf]) {
    state.ceChgBars = state.chgByTf[tf].ce;
    state.peChgBars = state.chgByTf[tf].pe;
    state.ceChgLine.setData(state.ceChgBars);
    state.peChgLine.setData(state.peChgBars);
    syncChgToIndex();
    return;
  }
  try {
    const r = await fetch(apiUrl(`/api/nifty_oi_intra_changes?tf=${tf}`));
    const j = await r.json();
    state.ceChgBars = j.ce || [];
    state.peChgBars = j.pe || [];
    state.chgByTf[tf] = { ce: state.ceChgBars, pe: state.peChgBars };
    state.ceChgLine.setData(state.ceChgBars);
    state.peChgLine.setData(state.peChgBars);
    if (state.ceChgBars.length) {
      const last = state.ceChgBars[state.ceChgBars.length - 1];
      const sign = last.value >= 0 ? "+" : "";
      $("chg-ce-val").textContent = sign + (last.value/1e5).toFixed(2) + " L";
    }
    if (state.peChgBars.length) {
      const last = state.peChgBars[state.peChgBars.length - 1];
      const sign = last.value >= 0 ? "+" : "";
      $("chg-pe-val").textContent = sign + (last.value/1e5).toFixed(2) + " L";
    }
    syncChgToIndex();
  } catch (_) {
    state.ceChgBars = []; state.peChgBars = [];
  }
}

function toggleChg() {
  state.showChg = !state.showChg;
  $("chg-pane").classList.toggle("hidden", !state.showChg);
  $("chg-resizer").classList.toggle("hidden", !state.showChg);
  $("chg-label").classList.toggle("hidden", !state.showChg);
  $("chg-toggle").classList.toggle("on", state.showChg);
  const btn = $("chg-toggle");
  if (state.showChg) {
    btn.style.background = "linear-gradient(135deg, #a855f7 0%, #7e22ce 100%)";
    btn.style.color = "#0a0c10";
    btn.style.borderColor = "transparent";
    btn.style.fontWeight = "700";
    loadChg(state.tf);
  } else {
    btn.style.background = ""; btn.style.color = "";
    btn.style.borderColor = ""; btn.style.fontWeight = "";
  }
}

// ── ΣOI (per-day cumulative OI TOTAL from 09:15) ────────────────────────
function syncTotToIndex() {
  if (!state.chart || !state.totChart) return;
  try {
    const r = state.chart.timeScale().getVisibleRange();
    if (r && r.from != null && r.to != null) {
      state.totChart.timeScale().setVisibleRange({ from: r.from, to: r.to });
    }
  } catch (_) {}
}

// Walk two parallel time-series and emit a marker every time the
// difference (pe - ce) changes sign. PE→up (green) = PE total overtook
// CE total; CE→up (red) = mirror. Markers attach to a line series via
// setMarkers(); position 'inBar' anchors them at the line's value.
function findCrossoverMarkers(ceBars, peBars) {
  const out = [];
  const n = Math.min(ceBars.length, peBars.length);
  if (n < 2) return out;
  let prevDiff = peBars[0].value - ceBars[0].value;
  for (let i = 1; i < n; i++) {
    // Skip misaligned ticks defensively (shouldn't happen — same backend)
    if (peBars[i].time !== ceBars[i].time) {
      prevDiff = peBars[i].value - ceBars[i].value;
      continue;
    }
    const currDiff = peBars[i].value - ceBars[i].value;
    if (prevDiff <= 0 && currDiff > 0) {
      out.push({ time: peBars[i].time, position: 'inBar',
                 color: '#22c55e', shape: 'circle',
                 text: 'PE↑', size: 1 });
    } else if (prevDiff >= 0 && currDiff < 0) {
      out.push({ time: peBars[i].time, position: 'inBar',
                 color: '#ef4444', shape: 'circle',
                 text: 'CE↑', size: 1 });
    }
    prevDiff = currDiff;
  }
  return out;
}

async function loadTot(tf) {
  if (state.totByTf[tf]) {
    state.ceTotBars = state.totByTf[tf].ce;
    state.peTotBars = state.totByTf[tf].pe;
    state.ceTotLine.setData(state.ceTotBars);
    state.peTotLine.setData(state.peTotBars);
    state.peTotLine.setMarkers(findCrossoverMarkers(state.ceTotBars, state.peTotBars));
    syncTotToIndex();
    return;
  }
  try {
    const r = await fetch(apiUrl(`/api/nifty_oi_intra_totals?tf=${tf}`));
    const j = await r.json();
    state.ceTotBars = j.ce || [];
    state.peTotBars = j.pe || [];
    state.totByTf[tf] = { ce: state.ceTotBars, pe: state.peTotBars };
    state.ceTotLine.setData(state.ceTotBars);
    state.peTotLine.setData(state.peTotBars);
    // Crossover markers — green ● where PE total crosses above CE total,
    // red ● where CE total crosses above PE total. Attached to the PE
    // line series so the markers sit on top of the wider-range line.
    const xMarkers = findCrossoverMarkers(state.ceTotBars, state.peTotBars);
    state.peTotLine.setMarkers(xMarkers);
    $("tot-xcount").textContent = xMarkers.length.toLocaleString();
    if (state.ceTotBars.length) {
      const last = state.ceTotBars[state.ceTotBars.length - 1];
      $("tot-ce-val").textContent = (last.value/1e5).toFixed(2) + " L";
    }
    if (state.peTotBars.length) {
      const last = state.peTotBars[state.peTotBars.length - 1];
      $("tot-pe-val").textContent = (last.value/1e5).toFixed(2) + " L";
    }
    syncTotToIndex();
  } catch (_) {
    state.ceTotBars = []; state.peTotBars = [];
  }
}

function toggleTot() {
  state.showTot = !state.showTot;
  $("tot-pane").classList.toggle("hidden", !state.showTot);
  $("tot-resizer").classList.toggle("hidden", !state.showTot);
  $("tot-label").classList.toggle("hidden", !state.showTot);
  $("tot-toggle").classList.toggle("on", state.showTot);
  const btn = $("tot-toggle");
  if (state.showTot) {
    btn.style.background = "linear-gradient(135deg, #14b8a6 0%, #0f766e 100%)";
    btn.style.color = "#0a0c10";
    btn.style.borderColor = "transparent";
    btn.style.fontWeight = "700";
    loadTot(state.tf);
  } else {
    btn.style.background = ""; btn.style.color = "";
    btn.style.borderColor = ""; btn.style.fontWeight = "";
  }
}

function toggleOi() {
  state.showOi = !state.showOi;
  $("oi-pane").classList.toggle("hidden",  !state.showOi);
  $("pane-resizer").classList.toggle("hidden", !state.showOi);
  $("oi-label").classList.toggle("hidden", !state.showOi);
  $("oi-toggle").classList.toggle("on", state.showOi);
  // Style the toggle for an active state similar to TF buttons
  const btn = $("oi-toggle");
  if (state.showOi) {
    btn.style.background = "linear-gradient(135deg, #26a69a 0%, #1d8a80 100%)";
    btn.style.color = "#0a0c10";
    btn.style.borderColor = "transparent";
    btn.style.fontWeight = "700";
  } else {
    btn.style.background = "";
    btn.style.color = "";
    btn.style.borderColor = "";
    btn.style.fontWeight = "";
  }
  if (state.showOi) loadOI(state.tf);
  // autoSize is set on each chart at creation, so the ResizeObserver picks
  // up the new flex layout automatically — no manual applyOptions needed.
}

async function loadMeta() {
  try {
    const r = await fetch(apiUrl(`/api/range`));
    const j = await r.json();
    if (j.days) $("day-count").textContent = j.days.toLocaleString();
  } catch (_) {}
}

function init() {
  setupChart();
  $("tf-grp").querySelectorAll("button").forEach(b => {
    b.addEventListener("click", () => loadTF(b.dataset.tf));
  });
  $("fit-all").addEventListener("click", () => {
    state.chart.timeScale().fitContent();
    if (state.oiChart) state.oiChart.timeScale().fitContent();
  });
  $("oi-toggle").addEventListener("click", toggleOi);
  $("cum-toggle").addEventListener("click", toggleCum);
  $("chg-toggle").addEventListener("click", toggleChg);
  $("tot-toggle").addEventListener("click", toggleTot);
  $("strat-toggle").addEventListener("click", toggleStrategy);
  $("sig5-toggle").addEventListener("click", toggleSig5);
  $("sigt-toggle").addEventListener("click", toggleSigT);
  // Position mode is now locked to Single — no toggle event listeners.

  // Cumulative-PnL pane VISIBILITY toggle, parked next to the LAST price
  // in the top bar. Hides the pane + its resizer so the candle chart can
  // claim the full vertical space.
  let pnlVisible = true;
  const _markPnLBtn = () => {
    const btn = $("pnl-toggle");
    if (pnlVisible) {
      btn.style.background = "linear-gradient(135deg, #60a5fa 0%, #2563eb 100%)";
      btn.style.color = "#0a0c10";
      btn.style.borderColor = "transparent";
      btn.style.fontWeight = "700";
    } else {
      btn.style.background = ""; btn.style.color = "";
      btn.style.borderColor = ""; btn.style.fontWeight = "";
    }
  };
  _markPnLBtn();
  $("pnl-toggle").addEventListener("click", () => {
    pnlVisible = !pnlVisible;
    $("pnl-pane").style.display    = pnlVisible ? "" : "none";
    $("pnl-resizer").style.display = pnlVisible ? "" : "none";
    _markPnLBtn();
    // Re-fit the curve when the pane comes back so it lays out cleanly.
    if (pnlVisible && state.pnlChart) {
      try { state.pnlChart.timeScale().fitContent(); } catch (_) {}
    }
  });


  // ── LIVE auto-refresh ────────────────────────────────────────────
  // Every 60s during the NSE session, re-pull candles + merged trades.
  // The live worker appends fresh 1-min bars and ATM±1 OI each minute,
  // so today's entries appear here within ~1 min of the cross.
  let liveOn = true;
  const _isMktHours = () => {
    const ist = new Date(Date.now() + 330 * 60000);   // UTC+5:30
    const d = ist.getUTCDay(), m = ist.getUTCHours() * 60 + ist.getUTCMinutes();
    return d >= 1 && d <= 5 && m >= 555 && m <= 935;  // Mon-Fri 09:15–15:35
  };
  const _markLiveBtn = () => {
    const btn = $("live-toggle");
    if (liveOn) {
      btn.style.background = "linear-gradient(135deg, #34d399 0%, #059669 100%)";
      btn.style.color = "#0a0c10";
      btn.style.borderColor = "transparent";
      btn.style.fontWeight = "700";
    } else {
      btn.style.background = ""; btn.style.color = "";
      btn.style.borderColor = ""; btn.style.fontWeight = "";
    }
  };
  const _liveTick = async () => {
    if (!liveOn || !_isMktHours()) return;
    try {
      // Incremental candle pull: only bars from the last known bucket on.
      // The full multi-year series is fetched once per page load / TF
      // switch (loadTF); shipping it every minute melted the server.
      const lastT = state.bars.length
        ? state.bars[state.bars.length - 1].time : 0;
      const r = await fetch(apiUrl(`/api/nifty?tf=${state.tf}&since=${lastT}`));
      const j = await r.json();
      for (const b of (j.bars || [])) {
        const lb = state.bars[state.bars.length - 1];
        if (lb && lb.time === b.time) {
          state.bars[state.bars.length - 1] = b;     // amend forming bucket
        } else if (!lb || b.time > lb.time) {
          state.bars.push(b);
        }
        try { state.candle.update(b); } catch (_) {}
      }
      if (state.bars.length) {
        $("bar-count").textContent = state.bars.length.toLocaleString();
        $("range").textContent = fmtDateUnix(state.bars[0].time) + " → "
          + fmtDateUnix(state.bars[state.bars.length - 1].time);
      }
      await loadTrades();         // markers/stats on top
    } catch (_) {}
  };
  _markLiveBtn();
  setInterval(_liveTick, 60000);
  $("live-toggle").addEventListener("click", () => {
    liveOn = !liveOn;
    _markLiveBtn();
    if (liveOn) _liveTick();
  });

  // ── Tick-level forming candle ────────────────────────────────────
  // The worker writes the NIFTY LTP to /api/ltp every ~4s. Poll it and
  // amend the current TF bucket in place so the rightmost candle ticks
  // like a real terminal between the 60s full refreshes.
  const TF_SEC = {"1m":60, "3m":180, "5m":300, "15m":900,
                  "30m":1800, "1h":3600, "4h":14400};
  let formBar = null;
  const _tickPoll = async () => {
    if (!liveOn || !_isMktHours()) return;
    const sec = TF_SEC[state.tf];
    if (!sec || !state.candle) return;
    try {
      const r = await fetch(apiUrl('/api/ltp'));
      const j = await r.json();
      if (!j.ltp || !j.ts) return;
      // Chart timestamps are naive-IST-as-UTC (see to_unix server-side):
      // parse the worker's naive IST stamp as UTC to land in the same frame.
      const t = Math.floor(Date.parse(j.ts + "Z") / 1000);
      const bucket = t - (t % sec);
      if (!formBar || formBar.time !== bucket) {
        formBar = {time: bucket, open: j.ltp, high: j.ltp,
                   low: j.ltp, close: j.ltp};
      } else {
        formBar.high  = Math.max(formBar.high, j.ltp);
        formBar.low   = Math.min(formBar.low,  j.ltp);
        formBar.close = j.ltp;
      }
      // series.update throws if the bucket is older than the last bar
      // (e.g. right after a full refresh replaced the data) — ignore.
      try { state.candle.update(formBar); } catch (_) {}
      $("last-val").textContent = fmtPrice(j.ltp);
    } catch (_) {}
  };
  setInterval(_tickPoll, 4000);

  // Rich hover tooltip on entry/exit arrows — shows SL/Target/exit/P&L.
  setupHoverTooltip();
  // Auto-load default view (Both) so the chart shows trades immediately
  // on page open — no clicks required.
  loadTrades();

  // Reflect the default-ON state on the OI and Σ(N) toggle buttons. The
  // HTML panes are also un-hidden by default — here we just paint the
  // button chrome to match the open panels.
  // All toggle buttons (ATM·OI, Σ·10, ΔOI, ΣOI, STRATEGY, Σ·10 ENTRY,
  // ΣOI ENTRY) are now hidden via display:none — no default styling needed.

  // ── Pane drag-to-resize ──────────────────────────────────────────
  // Generic resizer factory: dragging `handle` adjusts `pane`'s flex-basis
  // so it stays between MIN_PCT and MAX_PCT of the wrap container's height.
  // Pane height is computed from the pane's OWN bottom edge so the math
  // works regardless of which sub-panes are visible below it (cum-pane
  // hidden vs visible just changes the absolute Y position, not the
  // bottom edge of the pane being resized).
  function attachResizer(handleId, paneId, MIN_PCT, MAX_PCT) {
    const handle = $(handleId);
    const wrap   = $("charts-wrap");
    const pane   = $(paneId);
    let dragging = false;
    handle.addEventListener("mousedown", (e) => {
      dragging = true;
      handle.classList.add("dragging");
      document.body.style.cursor = "row-resize";
      e.preventDefault();
    });
    // rAF-throttle the drag — without this we were running flex-basis
    // updates on every mousemove (60+ Hz) which made the drag feel laggy
    // because each style mutation forced a synchronous layout on every
    // chart. With autoSize: true each chart's ResizeObserver picks up the
    // new container height, so no per-tick applyOptions is needed.
    let lastY = 0, pending = false;
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      lastY = e.clientY;
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        const paneRect = pane.getBoundingClientRect();
        const wrapRect = wrap.getBoundingClientRect();
        const heightPx = paneRect.bottom - lastY;
        let pct = (heightPx / wrapRect.height) * 100;
        pct = Math.max(MIN_PCT, Math.min(MAX_PCT, pct));
        pane.style.flexBasis = pct.toFixed(2) + "%";
      });
    });
    window.addEventListener("mouseup", () => {
      if (!dragging) return;
      dragging = false;
      handle.classList.remove("dragging");
      document.body.style.cursor = "";
    });
  }
  // Resizer above OI pane: drag controls oi-pane height (15–50%).
  attachResizer("pnl-resizer", "pnl-pane", 15, 55);
  attachResizer("pane-resizer", "oi-pane", 15, 50);
  // Resizer above Σ(N) pane: drag controls cum-pane height (10–40%).
  attachResizer("cum-resizer",  "cum-pane", 10, 40);
  // Resizers for the ΔOI / ΣOI panes (created hidden; resizer logic uses
  // pane.bottom-cursor so it works regardless of which sibling panes are
  // visible above/below).
  attachResizer("chg-resizer",  "chg-pane", 8, 35);
  attachResizer("tot-resizer",  "tot-pane", 8, 35);
  loadTF("3m");
  loadMeta();
  // autoSize: true at chart creation already wires a ResizeObserver, so we
  // don't need a window resize handler at all.
}

document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML.replace("__APP_BASE__", APP_BASE)
