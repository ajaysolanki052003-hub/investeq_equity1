"""
NIFTY Option-Chain Replay — historical browser (6 years).

Pick any day, any expiry, then scrub through the session minute-by-minute. The
chain table refreshes in place; IV is inverted on the fly; PCR and Max-Pain are
recomputed at the cursor.

Run:
    python -m uvicorn app_chain_replay:app --host 0.0.0.0 --port 8704

Open:
    http://localhost:8704/
"""

from __future__ import annotations
import os
from datetime import time as dtime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse


# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(os.environ.get(
    "INVESTEQ_DATA", r"C:\Users\User\Desktop\investeq_ajs\DATA"))
# When mounted behind a path prefix (e.g. nginx /chain/), set APP_BASE so
# fetch('/api/...') calls in the inline JS are rewritten to '/chain/api/...'.
APP_BASE  = os.environ.get("APP_BASE", "").rstrip("/")
OPT_DIR   = ROOT / "options"
OI_DIR    = ROOT / "oi"
SPOT_DIR  = ROOT / "spot"

RISK_FREE_RATE = 0.065
SECONDS_PER_YEAR = 365.0 * 24 * 3600
SESSION_OPEN  = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
SESSION_MIN_COUNT = 375                          # 09:15 inclusive → 15:30 exclusive

TF_RULE = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "60min",
}


BARS_PER_DAY = {"1m": 375, "3m": 125, "5m": 75, "15m": 25, "30m": 13, "1h": 7}
TRADING_DAYS_YEAR = 252


def realized_vol(spot_bars: pd.DataFrame, tf: str, window: int) -> pd.DataFrame:
    """Rolling annualized RV from close-to-close log returns of spot bars."""
    if spot_bars.empty or len(spot_bars) < window + 1:
        return pd.DataFrame(columns=["timestamp", "rv"])
    ann = float(BARS_PER_DAY.get(tf, 75) * TRADING_DAYS_YEAR) ** 0.5
    logret = np.log(spot_bars["close"] / spot_bars["close"].shift(1))
    rv = logret.rolling(window).std(ddof=1) * ann
    return (pd.DataFrame({"timestamp": spot_bars["timestamp"], "rv": rv})
              .dropna(subset=["rv"]).reset_index(drop=True))


def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df.empty:
        return df
    rule = TF_RULE.get(tf, "1min")
    return (df.set_index("timestamp")
              .resample(rule, closed="left", label="left")
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"})
              .dropna(subset=["open", "close"])
              .reset_index())


# ─── BS / IV ─────────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(market_price, S, K, T, opt_type, r=RISK_FREE_RATE):
    if T <= 0 or market_price is None or market_price <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(S - K, 0) if opt_type == "CE" else max(K - S, 0)
    if market_price <= intrinsic + 0.01:
        return None
    try:
        return brentq(
            lambda s: bs_price(S, K, T, r, s, opt_type) - market_price,
            0.001, 5.0, xtol=1e-6, maxiter=80,
        )
    except (ValueError, RuntimeError):
        return None


def bs_delta(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma is None or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1)) if opt_type == "CE" else float(norm.cdf(d1) - 1)


def years_to_expiry(ts: pd.Timestamp, expiry: str) -> float:
    expiry_dt = pd.to_datetime(expiry).replace(hour=15, minute=30)
    return max((expiry_dt - ts).total_seconds() / SECONDS_PER_YEAR, 1e-8)


# ─── Loaders (memoised) ───────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def load_spot(date: str) -> pd.DataFrame:
    p = SPOT_DIR / f"{date}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=64)
def load_options(date: str) -> pd.DataFrame:
    p = OPT_DIR / f"{date}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@lru_cache(maxsize=64)
def load_oi(date: str) -> pd.DataFrame:
    p = OI_DIR / f"{date}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@lru_cache(maxsize=1)
def list_trading_days() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in SPOT_DIR.glob("*.parquet")))


def to_unix(ts: pd.Series) -> np.ndarray:
    return (ts.view("int64") // 1_000_000_000).to_numpy()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="NIFTY Chain Replay")


@app.get("/api/days")
def days():
    return JSONResponse(list(list_trading_days()))


@app.get("/api/expiries")
def expiries(date: str = Query(...)):
    df = load_options(date)
    if df.empty:
        raise HTTPException(404, "no data for day")
    return sorted(pd.to_datetime(df["expiry"]).dt.strftime("%Y-%m-%d").unique().tolist())


@app.get("/api/strikes")
def strikes(date: str = Query(...), expiry: str = Query(...)):
    df = load_options(date)
    if df.empty:
        raise HTTPException(404, "no data for day")
    ks = sorted(df.loc[df["expiry"] == expiry, "strike"].unique().tolist())
    spot = load_spot(date)
    atm = None
    if not spot.empty and ks:
        last_spot = float(spot.iloc[-1]["close"])
        atm = min(ks, key=lambda k: abs(k - last_spot))
    return {"strikes": ks, "atm": atm}


_SPOT_HISTORY_CACHE = ROOT / "_spot_history_1d.parquet"
# Continuous 1-min NIFTY master parquet, maintained by fetch_nifty_master.py.
# When present, INDEX-view chart and all resampled TFs come from this file
# (gap-free, single source of truth). The per-day SPOT_DIR/*.parquet files
# remain the source for everything else (chain replay session scrubbing).
_NIFTY_MASTER = ROOT / "nifty_1m_master.parquet"


def _master_to_tf(tf: str) -> pd.DataFrame:
    """Read the continuous master 1-min parquet and resample to `tf`."""
    df = pd.read_parquet(_NIFTY_MASTER)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "volume" not in df.columns:
        df["volume"] = 0
    df = df.sort_values("timestamp").reset_index(drop=True)
    if tf == "1m":
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    if tf == "1d":
        d = df.copy()
        d["day"] = d["timestamp"].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
        out = (d.groupby("day", as_index=False)
                 .agg(open=("open", "first"), high=("high", "max"),
                      low=("low", "min"),     close=("close", "last")))
        out = out.rename(columns={"day": "timestamp"})
        return out.sort_values("timestamp").reset_index(drop=True)
    return resample_ohlc(df[["timestamp", "open", "high", "low", "close", "volume"]], tf)


@lru_cache(maxsize=2)
def _full_spot_history(tf: str) -> pd.DataFrame:
    """Continuous spot OHLC series at the requested TF.
    Preference order:
        1) DATA/nifty_1m_master.parquet — single continuous 1-min file
           (built by fetch_nifty_master.py; updated by the candles-nifty timer)
        2) On-disk 1d aggregate cache (DATA/_spot_history_1d.parquet)
        3) Per-day SPOT_DIR/*.parquet aggregation (legacy fallback)"""
    if _NIFTY_MASTER.exists():
        try:
            return _master_to_tf(tf)
        except Exception:
            pass   # corrupted master → fall through to legacy path
    if tf == "1d" and _SPOT_HISTORY_CACHE.exists():
        try:
            return pd.read_parquet(_SPOT_HISTORY_CACHE)
        except Exception:
            pass   # corrupted file → fall through and rebuild
    files = sorted(SPOT_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    if tf == "1d":
        # Fast path: aggregate each day's 1-min bars to one OHLC row without
        # paying the full concat-and-resample cost.
        rows = []
        for f in files:
            try:
                df = pd.read_parquet(f, columns=["open", "high", "low", "close"])
                if df.empty:
                    continue
                ts = pd.to_datetime(f.stem, format="%Y%m%d") + pd.Timedelta(hours=9, minutes=15)
                rows.append({
                    "timestamp": ts,
                    "open":  float(df["open"].iloc[0]),
                    "high":  float(df["high"].max()),
                    "low":   float(df["low"].min()),
                    "close": float(df["close"].iloc[-1]),
                })
            except Exception:
                continue
        out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        try:
            out.to_parquet(_SPOT_HISTORY_CACHE, index=False)
        except Exception:
            pass   # disk-write failure is non-fatal — in-process cache still works
        return out
    # Other TFs: read all, concat, resample once
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(
                f, columns=["timestamp", "open", "high", "low", "close"]))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    full = full.sort_values("timestamp").assign(volume=0).reset_index(drop=True)
    return resample_ohlc(
        full[["timestamp", "open", "high", "low", "close", "volume"]], tf)


_ATM_OI_CACHE = ROOT / "_atm_oi_history.parquet"

@lru_cache(maxsize=1)
def _full_atm_oi_history(strike_step: int = 50) -> pd.DataFrame:
    """For every trading day, snapshot the end-of-day combined OI for the
    three nearest strikes (ATM-1, ATM, ATM+1) on the nearest expiry, split
    by CE and PE. Cached on disk + in-process.

    Disk cache lives at DATA/_atm_oi_history.parquet — the underlying OI
    files don't change once a day is closed, so the cache is permanent.
    Delete the file to force a rebuild."""
    if _ATM_OI_CACHE.exists():
        try:
            return pd.read_parquet(_ATM_OI_CACHE)
        except Exception:
            pass   # corrupted cache → rebuild
    oi_files = sorted(OI_DIR.glob("*.parquet"))
    rows = []
    for f in oi_files:
        date_str = f.stem
        spot_p = SPOT_DIR / f"{date_str}.parquet"
        if not spot_p.exists():
            continue
        try:
            spot = pd.read_parquet(spot_p, columns=["close"])
            if spot.empty:
                continue
            close_px = float(spot["close"].iloc[-1])
            atm = round(close_px / strike_step) * strike_step
            three = [atm - strike_step, atm, atm + strike_step]

            oi = pd.read_parquet(f, columns=["timestamp", "expiry", "strike",
                                              "option_type", "oi"])
            if oi.empty:
                continue
            # Nearest expiry on/after the trading day. Multiple expiries may
            # be present in the same file — pick the soonest still-trading one.
            day_dt = pd.to_datetime(date_str, format="%Y%m%d").date()
            oi["_exp_date"] = pd.to_datetime(oi["expiry"]).dt.date
            forward = sorted({d for d in oi["_exp_date"] if d >= day_dt})
            if not forward:
                continue
            nearest = forward[0]

            sub = oi[(oi["_exp_date"] == nearest) & (oi["strike"].isin(three))]
            if sub.empty:
                continue
            # End-of-day snapshot — last OI per (strike, type)
            sub = (sub.sort_values("timestamp")
                      .groupby(["strike", "option_type"])["oi"].last())
            ce_total = float(sub.xs("CE", level="option_type").sum()) \
                if "CE" in sub.index.get_level_values("option_type") else 0.0
            pe_total = float(sub.xs("PE", level="option_type").sum()) \
                if "PE" in sub.index.get_level_values("option_type") else 0.0

            ts = (pd.to_datetime(date_str, format="%Y%m%d")
                  + pd.Timedelta(hours=9, minutes=15))
            rows.append({"timestamp": ts, "ce_oi": ce_total,
                         "pe_oi": pe_total, "atm": int(atm)})
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    try:
        df.to_parquet(_ATM_OI_CACHE, index=False)
    except Exception:
        pass   # disk-write failure is non-fatal — in-process cache still works
    return df


@app.get("/api/index_oi")
def index_oi():
    """ATM±1 combined OI (CE & PE) for every day in the spot history.
    Used by the INDEX view's OI indicator overlay."""
    df = _full_atm_oi_history()
    if df.empty:
        return JSONResponse({"ce_oi": [], "pe_oi": [], "n": 0})
    t = to_unix(df["timestamp"])
    ce = [{"time": int(t[i]), "value": float(df["ce_oi"].iloc[i])}
          for i in range(len(df))]
    pe = [{"time": int(t[i]), "value": float(df["pe_oi"].iloc[i])}
          for i in range(len(df))]
    return JSONResponse({"ce_oi": ce, "pe_oi": pe, "n": len(df)})


@app.get("/api/spot_all")
def spot_all(tf: str = "1d"):
    """Full multi-year spot OHLC for the INDEX view in the chain-replay UI.
    Defaults to 1d (smallest payload, ~1500 bars across 6 yrs)."""
    bars = _full_spot_history(tf)
    if bars.empty:
        return JSONResponse({"bars": [], "n": 0})
    t = to_unix(bars["timestamp"])
    out = [{
        "time" : int(t[i]),
        "open" : float(bars["open"].iloc[i]),
        "high" : float(bars["high"].iloc[i]),
        "low"  : float(bars["low"].iloc[i]),
        "close": float(bars["close"].iloc[i]),
    } for i in range(len(bars))]
    return JSONResponse({"bars": out, "n": len(out), "tf": tf})


@app.get("/api/spot")
def spot(date: str = Query(...), tf: str = "1m"):
    df = load_spot(date)
    if df.empty:
        raise HTTPException(404, "no spot data")
    if "volume" not in df.columns:
        df = df.assign(volume=0)
    bars = resample_ohlc(df[["timestamp", "open", "high", "low", "close", "volume"]], tf)
    t = to_unix(bars["timestamp"])
    out = [{
        "time" : int(t[i]),
        "open" : float(bars["open"].iloc[i]),
        "high" : float(bars["high"].iloc[i]),
        "low"  : float(bars["low"].iloc[i]),
        "close": float(bars["close"].iloc[i]),
    } for i in range(len(bars))]
    return JSONResponse({"bars": out, "n": len(out)})


@app.get("/api/leg")
def leg(date:    str  = Query(...),
        expiry:  str  = Query(...),
        strike:  int  = Query(...),
        type:    str  = Query(..., description="CE or PE"),
        tf:      str  = "1m",
        with_iv: bool = False,
        with_oi: bool = False):
    """OHLCV bars for one option leg on one day, resampled to tf.

    Optional indicator series:
      with_iv  → live BS-inverted implied vol (returned in %)
      with_oi  → leg open-interest line
    """
    df = load_options(date)
    if df.empty:
        raise HTTPException(404, "no options data")
    sub = df[(df["expiry"] == expiry) & (df["strike"] == strike) & (df["option_type"] == type)]
    if sub.empty:
        return JSONResponse({"bars": [], "iv": [], "oi": []})
    bars = resample_ohlc(sub[["timestamp", "open", "high", "low", "close", "volume"]], tf)
    t = to_unix(bars["timestamp"])
    out_bars = [{
        "time" : int(t[i]),
        "open" : float(bars["open"].iloc[i]),
        "high" : float(bars["high"].iloc[i]),
        "low"  : float(bars["low"].iloc[i]),
        "close": float(bars["close"].iloc[i]),
    } for i in range(len(bars))]

    iv_line = []
    if with_iv and not bars.empty:
        spot = load_spot(date)
        if not spot.empty:
            sp = spot.assign(volume=0)[["timestamp", "open", "high", "low", "close", "volume"]]
            spot_bars = resample_ohlc(sp, tf)
            merged = pd.merge_asof(
                bars[["timestamp", "close"]].rename(columns={"close": "opt"}),
                spot_bars[["timestamp", "close"]].rename(columns={"close": "spot"}),
                on="timestamp", direction="backward",
            )
            for _, row in merged.iterrows():
                T = years_to_expiry(row["timestamp"], expiry)
                iv = implied_vol(row["opt"], row["spot"], strike, T, type)
                if iv is not None:
                    iv_line.append({
                        "time": int(row["timestamp"].value // 10**9),
                        "value": float(iv * 100),
                    })

    oi_line = []
    if with_oi:
        oi_df = load_oi(date)
        if not oi_df.empty:
            o = oi_df[(oi_df["expiry"] == expiry) &
                      (oi_df["strike"] == strike) &
                      (oi_df["option_type"] == type)]
            if not o.empty:
                resampled = (o.set_index("timestamp")["oi"]
                              .resample(TF_RULE.get(tf, "1min"),
                                        closed="left", label="left")
                              .last().dropna())
                oi_line = [{"time": int(ts.value // 10**9), "value": float(v)}
                           for ts, v in resampled.items()]

    return JSONResponse({"bars": out_bars, "iv": iv_line, "oi": oi_line})


@app.get("/api/rv")
def rv(date: str = Query(...),
       tf: str = "1m",
       lookback: int = Query(30, ge=3, le=200)):
    """Rolling realized vol of spot, in %, at the given TF."""
    spot = load_spot(date)
    if spot.empty:
        raise HTTPException(404, "no spot data")
    sp = spot.assign(volume=0)[["timestamp", "open", "high", "low", "close", "volume"]]
    spot_bars = resample_ohlc(sp, tf)
    df = realized_vol(spot_bars, tf, max(2, int(lookback)))
    if df.empty:
        return JSONResponse({"rv": []})
    t = to_unix(df["timestamp"])
    return JSONResponse({
        "rv": [{"time": int(t[i]), "value": float(df["rv"].iloc[i] * 100)}
               for i in range(len(df))],
        "lookback": int(lookback),
    })


def _resolve_cursor_ts(date: str, minute: int) -> pd.Timestamp:
    """Build a naive-IST cursor timestamp from day + minute index (0..374)."""
    y, mo, d = int(date[0:4]), int(date[4:6]), int(date[6:8])
    base = pd.Timestamp(year=y, month=mo, day=d, hour=9, minute=15)
    return base + pd.Timedelta(minutes=max(0, min(minute, SESSION_MIN_COUNT - 1)))


@app.get("/api/chain_at")
def chain_at(date:   str = Query(...),
             expiry: str = Query(...),
             minute: int = Query(...,
                description="Bar index 0..374 (09:15 = 0, 15:30 = 374)"),
             band:   int = Query(10, ge=1, le=40)):
    opt  = load_options(date)
    oi   = load_oi(date)
    spt  = load_spot(date)
    if opt.empty or spt.empty:
        raise HTTPException(404, "no data for day")

    cursor = _resolve_cursor_ts(date, minute)
    session_start = pd.Timestamp(year=int(date[0:4]), month=int(date[4:6]),
                                 day=int(date[6:8]), hour=9, minute=15)

    # Spot at cursor (last value <= cursor)
    s_slice = spt[spt["timestamp"] <= cursor]
    if s_slice.empty:
        s_slice = spt
    spot_px = float(s_slice.iloc[-1]["close"])

    # Restrict to selected expiry, snapshot per (strike, option_type) at cursor
    opt_e = opt[opt["expiry"] == expiry]
    if opt_e.empty:
        return JSONResponse({"rows": [], "spot": spot_px, "atm": None, "cursor": str(cursor)})

    snap = (opt_e[opt_e["timestamp"] <= cursor]
            .sort_values("timestamp")
            .groupby(["strike", "option_type"])
            .agg(ltp=("close", "last"), vol_cum=("volume", "sum"))
            .reset_index())
    if snap.empty:
        return JSONResponse({"rows": [], "spot": spot_px, "atm": None, "cursor": str(cursor)})

    # Per-minute volume (volume in that bar, not cumulative) — use last bar's volume
    last_bar = (opt_e[opt_e["timestamp"] == cursor]
                .groupby(["strike", "option_type"])["volume"].sum()
                .reset_index(name="vol_bar"))
    snap = snap.merge(last_bar, on=["strike", "option_type"], how="left")

    # OI at cursor + OI at session open (for ΔOI from open)
    if not oi.empty:
        oi_e = oi[oi["expiry"] == expiry]
        oi_cur = (oi_e[oi_e["timestamp"] <= cursor]
                  .sort_values("timestamp")
                  .groupby(["strike", "option_type"])["oi"].last()
                  .reset_index(name="oi"))
        oi_open = (oi_e[oi_e["timestamp"] <= session_start + pd.Timedelta(minutes=2)]
                   .sort_values("timestamp")
                   .groupby(["strike", "option_type"])["oi"].first()
                   .reset_index(name="oi_open"))
        snap = snap.merge(oi_cur, on=["strike", "option_type"], how="left")
        snap = snap.merge(oi_open, on=["strike", "option_type"], how="left")
    else:
        snap["oi"] = np.nan
        snap["oi_open"] = np.nan

    # ATM = strike closest to spot
    strikes_all = sorted(snap["strike"].unique().tolist())
    atm = min(strikes_all, key=lambda k: abs(k - spot_px))
    step = int(np.median(np.diff(strikes_all))) if len(strikes_all) > 1 else 50
    lo, hi = atm - band * step, atm + band * step
    in_band = snap[(snap["strike"] >= lo) & (snap["strike"] <= hi)]

    ce = in_band[in_band["option_type"] == "CE"].set_index("strike")
    pe = in_band[in_band["option_type"] == "PE"].set_index("strike")
    keys = sorted(set(ce.index).union(pe.index))

    T = years_to_expiry(cursor, expiry)
    rows = []
    pcr_oi_ce_total = 0
    pcr_oi_pe_total = 0
    pcr_vol_ce_total = 0
    pcr_vol_pe_total = 0
    pain_table = {}              # strike -> total writer payoff if expiry @ strike
    for k in keys:
        ce_ltp  = float(ce.loc[k, "ltp"])     if k in ce.index and pd.notna(ce.loc[k, "ltp"]) else None
        pe_ltp  = float(pe.loc[k, "ltp"])     if k in pe.index and pd.notna(pe.loc[k, "ltp"]) else None
        ce_oi   = int(ce.loc[k, "oi"])        if k in ce.index and "oi" in ce.columns and pd.notna(ce.loc[k, "oi"]) else None
        pe_oi   = int(pe.loc[k, "oi"])        if k in pe.index and "oi" in pe.columns and pd.notna(pe.loc[k, "oi"]) else None
        ce_oio  = int(ce.loc[k, "oi_open"])   if k in ce.index and "oi_open" in ce.columns and pd.notna(ce.loc[k, "oi_open"]) else None
        pe_oio  = int(pe.loc[k, "oi_open"])   if k in pe.index and "oi_open" in pe.columns and pd.notna(pe.loc[k, "oi_open"]) else None
        ce_vol  = int(ce.loc[k, "vol_bar"])   if k in ce.index and "vol_bar" in ce.columns and pd.notna(ce.loc[k, "vol_bar"]) else 0
        pe_vol  = int(pe.loc[k, "vol_bar"])   if k in pe.index and "vol_bar" in pe.columns and pd.notna(pe.loc[k, "vol_bar"]) else 0
        ce_iv   = implied_vol(ce_ltp, spot_px, k, T, "CE") if ce_ltp else None
        pe_iv   = implied_vol(pe_ltp, spot_px, k, T, "PE") if pe_ltp else None
        ce_dlt  = bs_delta(spot_px, k, T, RISK_FREE_RATE, ce_iv, "CE") if ce_iv else None
        pe_dlt  = bs_delta(spot_px, k, T, RISK_FREE_RATE, pe_iv, "PE") if pe_iv else None
        rows.append({
            "strike": int(k),
            "ce_ltp"  : ce_ltp, "pe_ltp"  : pe_ltp,
            "ce_oi"   : ce_oi,  "pe_oi"   : pe_oi,
            "ce_dOI"  : (ce_oi - ce_oio) if ce_oi is not None and ce_oio is not None else None,
            "pe_dOI"  : (pe_oi - pe_oio) if pe_oi is not None and pe_oio is not None else None,
            "ce_vol"  : ce_vol, "pe_vol" : pe_vol,
            "ce_iv"   : (None if ce_iv is None else round(ce_iv * 100, 2)),
            "pe_iv"   : (None if pe_iv is None else round(pe_iv * 100, 2)),
            "ce_delta": (None if ce_dlt is None else round(ce_dlt, 2)),
            "pe_delta": (None if pe_dlt is None else round(pe_dlt, 2)),
            "atm"     : bool(k == atm),
            "ce_itm"  : bool(k <= spot_px),
            "pe_itm"  : bool(k >= spot_px),
        })
        if ce_oi: pcr_oi_ce_total  += ce_oi
        if pe_oi: pcr_oi_pe_total  += pe_oi
        pcr_vol_ce_total += ce_vol
        pcr_vol_pe_total += pe_vol

    # Max-pain: scan all band strikes — for each candidate expiry-spot K*, sum
    # OI-weighted writer payoff. Min total payoff = max pain.
    pain = {}
    for K_star in keys:
        total = 0.0
        for r in rows:
            K = r["strike"]
            if r["ce_oi"]: total += max(K_star - K, 0) * r["ce_oi"]
            if r["pe_oi"]: total += max(K - K_star, 0) * r["pe_oi"]
        pain[K_star] = total
    max_pain = min(pain.items(), key=lambda kv: kv[1])[0] if pain else None

    pcr_oi  = (pcr_oi_pe_total  / pcr_oi_ce_total)  if pcr_oi_ce_total  else None
    pcr_vol = (pcr_vol_pe_total / pcr_vol_ce_total) if pcr_vol_ce_total else None

    return JSONResponse({
        "rows"      : rows,
        "spot"      : spot_px,
        "atm"       : int(atm),
        "max_pain"  : int(max_pain) if max_pain else None,
        "pcr_oi"    : round(pcr_oi, 2)  if pcr_oi  else None,
        "pcr_vol"   : round(pcr_vol, 2) if pcr_vol else None,
        "dte_days"  : round((pd.to_datetime(expiry) - cursor).total_seconds() / 86400, 2),
        "cursor"    : cursor.strftime("%Y-%m-%d %H:%M"),
        "minute"    : int(minute),
    })


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aurora</title>
<script>
// Reverse-proxy prefix shim. Empty in local dev, '/chain' on the VM.
window.APP_BASE = "__APP_BASE__";
(function() {
  const p = window.APP_BASE;
  if (!p) return;
  const _f = window.fetch.bind(window);
  window.fetch = (u, i) => (typeof u === 'string' && u.startsWith('/') && !u.startsWith('//') && !u.startsWith(p + '/'))
    ? _f(p + u, i) : _f(u, i);
})();
</script>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0a0c14; --panel: #0f1219; --panel-2: #131722;
  --border: #1e2030; --border-hi: #2a2d3a;
  --text: #e6e9f2; --muted: #6b7280; --muted-hi: #9ca3af;
  --green: #26a69a; --red: #ef5350; --blue: #60a5fa; --gold: #facc15; --purple: #c084fc;
  --ce-bg: rgba(38,166,154,0.04);
  --pe-bg: rgba(239,83,80,0.04);
}
html, body { background: var(--bg); color: var(--text); height: 100vh; }
body { font: 13px/1.4 'Inter', system-ui, sans-serif; display: flex; flex-direction: column; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* Top bar */
.bar {
  display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  padding: 10px 18px; background: var(--panel); border-bottom: 1px solid var(--border);
}
.brand { display: flex; gap: 10px; align-items: center; }
.brand .logo { color: var(--green); font-size: 18px; text-shadow: 0 0 12px rgba(38,166,154,0.5); }
.brand .title { font-weight: 800; letter-spacing: 0.5px; font-size: 13px; }
.brand .sub { color: var(--muted-hi); font-size: 10px; text-transform: uppercase;
              letter-spacing: 1.4px; border-left: 1px solid var(--border); padding-left: 10px;
              font-family: 'JetBrains Mono', monospace; }
label { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
select, button {
  background: var(--panel-2); border: 1px solid var(--border-hi); color: var(--text);
  border-radius: 5px; padding: 5px 10px; cursor: pointer; font: inherit;
}
select:hover, button:hover { border-color: var(--green); }
button.primary { background: var(--green); border-color: var(--green); color: #052b27; font-weight: 700; }
button.primary:hover { background: #2dd4bf; }
.spot {
  background: rgba(250,204,21,0.08); border: 1px solid rgba(250,204,21,0.3);
  border-radius: 6px; padding: 5px 12px; display: flex; gap: 8px; align-items: baseline;
}
.spot span.l { color: var(--gold); font-size: 10px; letter-spacing: 1px; }
.spot strong { color: var(--gold); font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 700; }

/* Time scrubber */
.scrub {
  display: flex; gap: 14px; align-items: center; padding: 10px 18px;
  background: linear-gradient(180deg, var(--panel) 0%, var(--bg) 100%);
  border-bottom: 1px solid var(--border);
}
.scrub-time {
  font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 800;
  color: var(--green); min-width: 88px; letter-spacing: 1px;
}
.scrub-slider { flex: 1; appearance: none; height: 6px; border-radius: 3px;
                background: var(--border); outline: none; cursor: pointer; }
.scrub-slider::-webkit-slider-thumb {
  appearance: none; width: 18px; height: 18px; border-radius: 50%;
  background: var(--green); border: 2px solid var(--bg);
  box-shadow: 0 0 12px rgba(38,166,154,0.6);
}
.scrub-slider::-moz-range-thumb {
  width: 16px; height: 16px; border-radius: 50%;
  background: var(--green); border: 2px solid var(--bg);
}
.scrub-info { display: flex; gap: 6px; align-items: center; color: var(--muted-hi); font-size: 11px;
              font-family: 'JetBrains Mono', monospace; }
.scrub-info .live-dot { width: 6px; height: 6px; background: var(--red); border-radius: 50%;
                        animation: pulse 1.4s infinite; }
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 currentColor; }
  50%      { box-shadow: 0 0 0 6px transparent; }
}
.play {
  width: 32px; height: 32px; padding: 0; display: flex; align-items: center; justify-content: center;
  font-size: 14px;
}

/* Left sidebar — date browser (6 years of trading days) */
.sidebar {
  flex: 0 0 200px; min-width: 140px;
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  display: flex; flex-direction: column; min-height: 0; overflow: hidden;
}
.side-h {
  padding: 11px 14px; font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 1.6px; color: var(--green);
  border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center;
}
.side-h .count { color: var(--muted); font-weight: 400; }
.side-search {
  margin: 8px 10px; padding: 6px 8px; font-size: 12px;
  background: var(--panel-2); border: 1px solid var(--border-hi);
  color: var(--text); border-radius: 4px; outline: none;
  font-family: 'JetBrains Mono', monospace;
}
.side-search:focus { border-color: var(--green); }
.day-list {
  list-style: none; overflow-y: auto; flex: 1; padding: 4px 0;
  scrollbar-width: thin; scrollbar-color: var(--border-hi) transparent;
}
.day-list::-webkit-scrollbar { width: 6px; }
.day-list::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
.day-list li {
  padding: 5px 14px; cursor: pointer; font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--muted-hi); border-left: 2px solid transparent;
  transition: all 0.12s; display: flex; justify-content: space-between; gap: 6px;
}
.day-list li .y { color: var(--muted); font-size: 10px; }
.day-list li:hover { background: var(--panel-2); color: var(--text); }
.day-list li.active {
  background: rgba(38,166,154,0.12); color: var(--green); font-weight: 700;
  border-left-color: var(--green);
}
.day-list .year-sep {
  padding: 6px 14px 2px; font-size: 9px; letter-spacing: 1.4px;
  color: var(--gold); border-top: 1px dashed var(--border);
  position: sticky; top: 0; background: var(--panel); cursor: default;
}
.day-list .year-sep:first-child { border-top: none; }

/* TF buttons */
.tf-grp { display: inline-flex; gap: 2px; }

/* Resizers between panes (horizontal between rows, vertical between cols) */
.resizer { position: relative; z-index: 10; flex: 0 0 6px; background: transparent; }
.resizer.h { height: 6px; cursor: row-resize; }
.resizer.v { width:  6px; cursor: col-resize; }
.resizer::before {
  content: ''; position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%); background: var(--border-hi); border-radius: 2px;
  transition: background 0.15s;
}
.resizer.h::before { width: 36px; height: 3px; }
.resizer.v::before { width: 3px;  height: 36px; }
.resizer:hover::before, .resizer.dragging::before {
  background: var(--green); box-shadow: 0 0 8px var(--green);
}
body.resizing { user-select: none; }
body.resizing.r-row { cursor: row-resize; }
body.resizing.r-col { cursor: col-resize; }
body.resizing iframe, body.resizing canvas { pointer-events: none; }

/* Fullscreen toggle on each pane */
.fs-btn {
  background: transparent; border: 1px solid var(--border); color: var(--muted);
  font-size: 12px; padding: 2px 6px; border-radius: 4px; cursor: pointer;
  margin-left: 6px; line-height: 1;
}
.fs-btn:hover { color: var(--green); border-color: var(--green); }
.pane.fullscreen {
  position: fixed; inset: 0; z-index: 999;
  border-radius: 0; border: none;
}
.pane.fullscreen .fs-btn { color: var(--green); border-color: var(--green); }

/* When a pane is hidden by a layout preset */
.pane.hidden, .col.hidden, .scrub.hidden,
.sidebar.hidden, .resizer.hidden { display: none; }
.tf-grp button {
  padding: 5px 9px; font-size: 11px; font-weight: 700; letter-spacing: 0.4px;
  background: var(--panel-2); border: 1px solid var(--border); color: var(--muted-hi);
}
.tf-grp button.on { background: var(--green); border-color: var(--green); color: #052b27; }
.tf-grp button:hover:not(.on) { color: var(--text); border-color: var(--border-hi); }

/* Layout: sidebar | resizer | main-content (metric strip + option pane). */
.main {
  flex: 1; display: flex; gap: 0;
  padding: 10px 14px; min-height: 0;
}
.main-content {
  flex: 1; min-width: 0; min-height: 0;
  display: flex; flex-direction: column; gap: 8px;
}
.main-content > .pane.pane-opt { flex: 2; min-height: 0; }
.main-content > .pane.pane-oi,
.main-content > .pane.pane-vol { flex: 1; min-height: 120px; }
/* ATM±1 OI sub-pane in INDEX mode — keep a fixed 220px so it's always
   visible and doesn't collapse to a sliver next to the tall index chart. */
.main-content > .pane.pane-idx-oi { flex: 0 0 220px; min-height: 200px; }
.ind-chart { width: 100%; height: 100%; }

/* Horizontal metric strip (above the CE/PE chart split) */
.metric-strip {
  display: flex; gap: 6px; flex-wrap: wrap;
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 12px;
}
.m-chip {
  background: var(--panel-2); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 14px; min-width: 110px; flex: 1 1 110px;
  display: flex; flex-direction: column; gap: 3px;
}
.m-chip .l { color: var(--muted); font-size: 9px; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; }
.m-chip .v {
  color: var(--text); font-family: 'JetBrains Mono', monospace;
  font-size: 17px; font-weight: 700; letter-spacing: -0.2px;
}
.m-chip.spot   .v { color: var(--gold);   }
.m-chip.atm    .v { color: var(--gold);   }
.m-chip.maxpain.v { color: var(--purple); }
.m-chip.maxpain   { border-color: rgba(192,132,252,0.3); }
.m-chip.pcr-oi .v, .m-chip.pcr-vol .v { color: var(--blue); }
.m-chip.pcr-oi, .m-chip.pcr-vol       { border-color: rgba(96,165,250,0.3); }
.m-chip.ce-iv  .v { color: var(--green); }
.m-chip.ce-iv     { border-color: rgba(38,166,154,0.3); }
.m-chip.pe-iv  .v { color: var(--red);   }
.m-chip.pe-iv     { border-color: rgba(239,83,80,0.3); }
.m-chip.dte    .v { color: var(--green); }

.pane {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  display: flex; flex-direction: column; min-height: 0; overflow: hidden;
}
.pane-h {
  display: flex; justify-content: space-between; align-items: center; gap: 12px;
  padding: 9px 12px; border-bottom: 1px solid var(--border);
  font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted-hi);
  font-family: 'JetBrains Mono', monospace;
}
.pane-h .right { color: var(--text); letter-spacing: 0; text-transform: none; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.pane-body { flex: 1; min-height: 0; position: relative; overflow: hidden; }
.pane-body.scroll { overflow-y: auto; }

/* Option pane: split CE | PE — flex (not grid) so each sub-pane actually
   gets a determinate height from align-items: stretch. */
.opt-split {
  display: flex; flex-direction: row; gap: 1px;
  width: 100%; height: 100%;
  background: var(--border);
}
.opt-sub {
  position: relative; background: #0d0f17;
  flex: 1 1 0; min-width: 0; min-height: 0;
  display: flex; flex-direction: column;
}
.opt-sub .opt-label {
  position: absolute; top: 6px; left: 10px; z-index: 5;
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  letter-spacing: 1.2px; padding: 2px 7px; border-radius: 3px;
  font-weight: 700;
}
.opt-sub.ce .opt-label { color: var(--green); background: rgba(38,166,154,0.12); }
.opt-sub.pe .opt-label { color: var(--red);   background: rgba(239,83,80,0.12); }
.opt-sub.idx .opt-label { color: var(--gold); background: rgba(250,204,21,0.14); }
.opt-sub.hidden { display: none; }
.opt-sub > .opt-chart { flex: 1; min-height: 0; width: 100%; }

/* Live cursor line at the scrubber's minute (drawn over the chart panes) */
.cursor-line {
  position: absolute; top: 0; bottom: 0; width: 1px;
  background: var(--gold); opacity: 0.45;
  box-shadow: 0 0 6px var(--gold);
  pointer-events: none; z-index: 4;
  transition: left 0.18s ease-out;
}
.cursor-line::before {
  content: ''; position: absolute; top: -3px; left: -3px;
  width: 7px; height: 7px; border-radius: 50%; background: var(--gold);
}


/* Loading */
.loading { color: var(--muted); padding: 40px; text-align: center; font-size: 12px; }
</style>
</head>
<body>

<div class="bar">
  <div class="brand">
    <span class="logo">◆</span>
    <span class="title">AURORA</span>
    <span class="sub">private workspace</span>
  </div>

  <label>Day</label>
  <span id="day-readout" class="mono" style="background:var(--panel-2); padding:5px 10px; border:1px solid var(--border-hi); border-radius:5px; min-width:108px; display:inline-block; text-align:center;">—</span>

  <label>Expiry</label>
  <select id="expiry" style="min-width:130px"></select>

  <label>CE strike</label>
  <select id="ce-strike" style="min-width:120px"></select>

  <label>PE strike</label>
  <select id="pe-strike" style="min-width:120px"></select>

  <label>TF</label>
  <span class="tf-grp" id="tf-grp">
    <button data-tf="1m" class="on">1m</button>
    <button data-tf="3m">3m</button>
    <button data-tf="5m">5m</button>
    <button data-tf="15m">15m</button>
    <button data-tf="30m">30m</button>
    <button data-tf="1h">1h</button>
  </span>

  <label>Indicators</label>
  <span class="tf-grp" id="ind-grp">
    <button data-ind="oi" title="Open Interest line per leg">OI</button>
    <button data-ind="iv" title="Implied vol % (BS-inverted) per leg">IV</button>
    <button data-ind="rv" title="Realized vol % of spot">RV</button>
    <button data-ind="atmoi" title="ATM ±1 combined OI (CE + PE) — INDEX view only">ATM·OI</button>
  </span>

  <label>View</label>
  <span class="tf-grp" id="view-grp">
    <button data-view="option" class="on" title="CE and PE charts side-by-side">OPTION</button>
    <button data-view="index" title="NIFTY spot chart only">INDEX</button>
  </span>
  <button id="idx-fit" class="tf-grp" title="Fit full 5-year history into view"
          style="display:none; padding:5px 12px; background:var(--panel-2); color:var(--text); border:1px solid var(--border-hi); border-radius:5px; cursor:pointer; font:inherit; font-family:'JetBrains Mono', monospace;">FIT ALL</button>
  <input id="rv-lookback" type="number" min="3" max="200" value="30"
         title="RV lookback (bars)"
         style="width:58px; background:var(--panel-2); border:1px solid var(--border-hi); color:var(--text); border-radius:5px; padding:5px 7px; font:inherit; font-family:'JetBrains Mono', monospace;" />

  <div class="spot">
    <span class="l">SPOT</span>
    <strong id="spot-val">—</strong>
  </div>

  <span style="margin-left:auto; color:var(--muted); font-size:11px;" id="status">—</span>
</div>

<div class="scrub" id="scrub-bar">
  <button class="play" id="play" title="Auto-play minute by minute">▶</button>
  <div class="scrub-time" id="cursor-time">09:15</div>
  <input id="slider" class="scrub-slider" type="range" min="0" max="374" value="0">
  <div class="scrub-info">
    <span class="live-dot"></span>
    <span id="minute-info">1 / 375</span>
  </div>
</div>

<div class="main">

  <!-- SIDEBAR: every trading day, click to load -->
  <aside class="sidebar" id="sidebar">
    <div class="side-h">
      <span>▸ DAYS</span>
      <span class="count" id="side-count">—</span>
    </div>
    <input id="day-search" type="search" class="side-search" placeholder="Search YYYYMMDD" />
    <ul class="day-list" id="day-list">
      <li style="text-align:center; color:var(--muted)">Loading…</li>
    </ul>
  </aside>

  <div class="resizer v" data-prev="sidebar" data-next="main-content" title="Drag to resize sidebar"></div>

  <!-- MAIN CONTENT: metric strip + CE/PE chart split -->
  <div class="main-content" id="main-content">

    <div class="metric-strip" id="metric-strip">
      <div class="m-chip spot"><div class="l">SPOT</div><div class="v" id="m-spot">—</div></div>
      <div class="m-chip atm"><div class="l">ATM strike</div><div class="v" id="m-atm">—</div></div>
      <div class="m-chip dte"><div class="l">Days to exp</div><div class="v" id="m-dte">—</div></div>
      <div class="m-chip pcr-oi"><div class="l">PCR · OI</div><div class="v" id="m-pcr-oi">—</div></div>
      <div class="m-chip pcr-vol"><div class="l">PCR · Vol</div><div class="v" id="m-pcr-vol">—</div></div>
      <div class="m-chip maxpain"><div class="l">Max pain</div><div class="v" id="m-maxpain">—</div></div>
      <div class="m-chip ce-iv"><div class="l">ATM CE IV</div><div class="v" id="m-ce-iv">—</div></div>
      <div class="m-chip pe-iv"><div class="l">ATM PE IV</div><div class="v" id="m-pe-iv">—</div></div>
    </div>

    <div class="pane pane-opt" id="pane-opt">
      <div class="pane-h">
        <span>▸ OPTION · <span id="opt-strike-lbl">—</span> · <span id="opt-tf">1m</span></span>
        <span class="ema-ctl" style="display:inline-flex; align-items:center; gap:6px; margin-left:14px; font-size:11px; color:#9ca3af;">
          <label style="display:inline-flex; align-items:center; gap:4px; cursor:pointer;">
            <input id="ema-on" type="checkbox" style="margin:0;"> EMA
          </label>
          <span style="color:#ff9800;">Fast</span>
          <input id="ema-fast" type="number" min="1" max="500" value="9"
                 style="width:50px; padding:1px 4px; background:#1a1d27; color:#e6ebf5; border:1px solid #2a2f3b; border-radius:3px; font-size:11px;">
          <span style="color:#42a5f5;">Slow</span>
          <input id="ema-slow" type="number" min="1" max="500" value="21"
                 style="width:50px; padding:1px 4px; background:#1a1d27; color:#e6ebf5; border:1px solid #2a2f3b; border-radius:3px; font-size:11px;">
        </span>
        <span class="right" id="opt-info">CE / PE candles · drag scrubber to move cursor</span>
        <button class="fs-btn" data-pane="pane-opt" title="Toggle fullscreen (Esc)">⛶</button>
      </div>
      <div class="pane-body">
        <div class="opt-split">
          <div class="opt-sub ce">
            <div class="opt-label">CE</div>
            <div class="opt-chart" id="opt-chart-ce"></div>
            <div class="cursor-line" id="cur-ce" style="left:-10px"></div>
          </div>
          <div class="opt-sub idx hidden" id="opt-sub-idx">
            <div class="opt-label">INDEX</div>
            <div class="opt-chart" id="opt-chart-idx"></div>
            <div class="cursor-line" id="cur-idx" style="left:-10px"></div>
          </div>
          <div class="opt-sub pe">
            <div class="opt-label">PE</div>
            <div class="opt-chart" id="opt-chart-pe"></div>
            <div class="cursor-line" id="cur-pe" style="left:-10px"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="pane pane-idx-oi hidden" id="pane-idx-oi">
      <div class="pane-h">
        <span>▸ ATM ±1 · combined OI · CE / PE</span>
        <span class="right" style="color: var(--muted)">end-of-day · daily</span>
        <button class="fs-btn" data-pane="pane-idx-oi" title="Toggle fullscreen (Esc)">⛶</button>
      </div>
      <div class="pane-body">
        <div class="ind-chart" id="chart-idx-oi"></div>
        <div class="cursor-line" id="cur-idx-oi" style="left:-10px"></div>
      </div>
    </div>

    <div class="pane pane-oi hidden" id="pane-oi">
      <div class="pane-h">
        <span>▸ OPEN INTEREST · CE / PE</span>
        <span class="right" style="color: var(--muted)">contracts</span>
        <button class="fs-btn" data-pane="pane-oi" title="Toggle fullscreen (Esc)">⛶</button>
      </div>
      <div class="pane-body">
        <div class="ind-chart" id="chart-oi"></div>
        <div class="cursor-line" id="cur-oi" style="left:-10px"></div>
      </div>
    </div>

    <div class="pane pane-vol hidden" id="pane-vol">
      <div class="pane-h">
        <span>▸ IMPLIED &amp; REALIZED VOL · %</span>
        <span class="right" style="color: var(--muted)">CE IV · PE IV · spot RV</span>
        <button class="fs-btn" data-pane="pane-vol" title="Toggle fullscreen (Esc)">⛶</button>
      </div>
      <div class="pane-body">
        <div class="ind-chart" id="chart-vol"></div>
        <div class="cursor-line" id="cur-vol" style="left:-10px"></div>
      </div>
    </div>

  </div>

</div>

<script>
const $ = id => document.getElementById(id);
const fmt   = (v, d=2) => (v == null || !Number.isFinite(v)) ? '—' : v.toFixed(d);
const fmtInt = v => (v == null) ? '—' : Math.round(v).toLocaleString('en-IN');
const fmtOI = v => {
  if (v == null) return '—';
  const a = Math.abs(v), s = v < 0 ? '-' : '';
  if (a >= 1e7) return s + (a/1e7).toFixed(2) + 'Cr';
  if (a >= 1e5) return s + (a/1e5).toFixed(2) + 'L';
  if (a >= 1e3) return s + (a/1e3).toFixed(1) + 'K';
  return s + Math.round(a);
};
const fmtSigned = v => {
  if (v == null) return '—';
  return (v >= 0 ? '+' : '') + fmtOI(v);
};
const istHHMM = t => {
  const d = new Date(t * 1000);
  return String(d.getUTCHours()).padStart(2,'0') + ':' + String(d.getUTCMinutes()).padStart(2,'0');
};
const minuteToHHMM = m => {
  const h = 9 + Math.floor((m + 15) / 60), mm = (m + 15) % 60;
  return String(h).padStart(2,'0') + ':' + String(mm).padStart(2,'0');
};

// State
const state = {
  date: null, expiry: null, band: 10, minute: 0, tf: '1m',
  // Per-side strike selection — both default to ATM, independently changeable
  ceStrike: null, peStrike: null, atm: null,
  ceChart: null, ceCandle: null, ceBars: [],
  peChart: null, peCandle: null, peBars: [],
  // View mode: 'option' (CE+PE side-by-side, default) or 'index' (full 6-yr
  // historical spot chart, navigable like TradingView)
  idxChart: null, idxCandle: null, idxBars: [], idxFullBars: null,
  viewMode: 'option',
  // INDEX-mode ATM±1 combined-OI overlay (independent indicator)
  idxOiChart: null, sIdxOiCE: null, sIdxOiPE: null,
  idxOiCEBars: [], idxOiPEBars: [], showAtmOi: false,
  // Indicator line series (created in makeAllCharts)
  sOiCE: null, sIvCE: null, sRvCE: null,
  sOiPE: null, sIvPE: null, sRvPE: null,
  // Indicator toggles
  showOI: false, showIV: false, showRV: false,
  // EMA overlay on CE/PE candle charts
  emaOn: false, emaFast: 9, emaSlow: 21,
  sEmaFastCE: null, sEmaSlowCE: null, sEmaFastPE: null, sEmaSlowPE: null,
  rvLookback: 30,
  // Cached indicator data so toggles can hide/show without refetch
  ceOI: [], ceIV: [], peOI: [], peIV: [], rvLine: [],
  syncing: false,
  playing: false, playTimer: null, pendingChain: null,
};

// ── Chart factory ──────────────────────────────────────────────────────────
function mkChart(el) {
  return LightweightCharts.createChart(el, {
    layout: { background: { type: 'solid', color: '#0a0c14' }, textColor: '#9ca3af' },
    grid: { vertLines: { color: '#1a1d27' }, horzLines: { color: '#1a1d27' } },
    rightPriceScale: { borderColor: '#1e2030', scaleMargins: { top: 0.08, bottom: 0.08 } },
    timeScale: {
      borderColor: '#1e2030', timeVisible: true, secondsVisible: false,
      tickMarkFormatter: t => istHHMM(t),
    },
    localization: { timeFormatter: istHHMM },
    crosshair: { mode: 1 }, autoSize: true,
  });
}

function mkCandleSeries(chart, side) {
  // Standard candle convention: green for up, red for down — both sides.
  const up = '#26a69a';
  const dn = '#ef5350';
  return chart.addCandlestickSeries({
    upColor: up, downColor: dn,
    borderUpColor: up, borderDownColor: dn,
    wickUpColor: up, wickDownColor: dn,
  });
}

function makeAllCharts() {
  // CE / PE candle charts — pure candles, no indicator overlays
  state.ceChart  = mkChart($('opt-chart-ce'));
  state.ceCandle = mkCandleSeries(state.ceChart, 'CE');
  state.peChart  = mkChart($('opt-chart-pe'));
  state.peCandle = mkCandleSeries(state.peChart, 'PE');

  // INDEX (NIFTY spot) chart — used standalone in INDEX view to show the full
  // 6-yr 1d history. Uses a date-aware tick formatter (Jan '22 / Mar 2024)
  // instead of the intraday HH:MM formatter the CE/PE charts use.
  state.idxChart  = mkChart($('opt-chart-idx'));
  state.idxChart.applyOptions({
    timeScale: {
      timeVisible: false, secondsVisible: false,
      // Wider candles → each day visually distinct without zooming
      barSpacing: 10, minBarSpacing: 2,
      tickMarkFormatter: (t) => {
        const d = new Date(t * 1000);
        const m = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        return `${d.getUTCDate()} ${m[d.getUTCMonth()]} '${String(d.getUTCFullYear()).slice(2)}`;
      },
    },
    localization: {
      timeFormatter: (t) => {
        const d = new Date(t * 1000);
        const m = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        return `${d.getUTCDate()} ${m[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
      },
    },
    // Brighter vertical gridlines so day boundaries are visible at a glance
    grid: { vertLines: { color: '#262a38', visible: true },
            horzLines: { color: '#1a1d27', visible: true } },
  });
  state.idxCandle = mkCandleSeries(state.idxChart, 'IDX');

  // ATM ±1 combined-OI sub-chart for INDEX mode — same date axis as idxChart
  state.idxOiChart = mkChart($('chart-idx-oi'));
  state.idxOiChart.applyOptions({
    timeScale: {
      timeVisible: false, secondsVisible: false,
      tickMarkFormatter: (t) => {
        const d = new Date(t * 1000);
        const m = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        return `${d.getUTCDate()} ${m[d.getUTCMonth()]} '${String(d.getUTCFullYear()).slice(2)}`;
      },
    },
    localization: { timeFormatter: (t) => {
      const d = new Date(t * 1000);
      const m = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      return `${d.getUTCDate()} ${m[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
    }},
  });
  state.sIdxOiCE = state.idxOiChart.addLineSeries({
    color: '#60a5fa', lineWidth: 2, title: 'CE OI (ATM±1)',
    priceFormat: { type: 'volume' }, lastValueVisible: true,
  });
  state.sIdxOiPE = state.idxOiChart.addLineSeries({
    color: '#c084fc', lineWidth: 2, title: 'PE OI (ATM±1)',
    priceFormat: { type: 'volume' }, lastValueVisible: true,
  });

  // Two-way time-axis sync between idxChart (candles) and idxOiChart (OI lines)
  const idxGroup = [state.idxChart, state.idxOiChart];
  idxGroup.forEach(c => {
    c.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (state.syncing || !range) return;
      state.syncing = true;
      idxGroup.forEach(o => { if (o !== c) o.timeScale().setVisibleLogicalRange(range); });
      state.syncing = false;
    });
  });

  // EMA overlays — fast = orange, slow = blue. Hidden until user toggles on.
  const emaFastOpts = { color: '#ff9800', lineWidth: 2, priceLineVisible: false,
                        lastValueVisible: false, crosshairMarkerVisible: false };
  const emaSlowOpts = { color: '#42a5f5', lineWidth: 2, priceLineVisible: false,
                        lastValueVisible: false, crosshairMarkerVisible: false };
  state.sEmaFastCE = state.ceChart.addLineSeries({ ...emaFastOpts, title: 'EMA fast' });
  state.sEmaSlowCE = state.ceChart.addLineSeries({ ...emaSlowOpts, title: 'EMA slow' });
  state.sEmaFastPE = state.peChart.addLineSeries({ ...emaFastOpts, title: 'EMA fast' });
  state.sEmaSlowPE = state.peChart.addLineSeries({ ...emaSlowOpts, title: 'EMA slow' });

  // Dedicated OI sub-pane (CE blue + PE purple)
  state.oiChart = mkChart($('chart-oi'));
  state.sOiCE_p = state.oiChart.addLineSeries({
    color: '#60a5fa', lineWidth: 2, title: 'CE OI',
    priceFormat: { type: 'volume' }, lastValueVisible: true,
  });
  state.sOiPE_p = state.oiChart.addLineSeries({
    color: '#c084fc', lineWidth: 2, title: 'PE OI',
    priceFormat: { type: 'volume' }, lastValueVisible: true,
  });

  // Dedicated Vol sub-pane (CE IV gold + PE IV orange + spot RV cyan)
  state.volChart = mkChart($('chart-vol'));
  state.sIvCE_p = state.volChart.addLineSeries({
    color: '#facc15', lineWidth: 2, title: 'CE IV%',
    priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    lastValueVisible: true,
  });
  state.sIvPE_p = state.volChart.addLineSeries({
    color: '#fb923c', lineWidth: 2, title: 'PE IV%',
    priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    lastValueVisible: true,
  });
  state.sRv_p = state.volChart.addLineSeries({
    color: '#22d3ee', lineWidth: 1.5, lineStyle: 2, title: 'RV%',
    priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    lastValueVisible: true,
  });

  // Time-axis sync across the intraday charts only (CE, PE, OI, Vol).
  // idxChart shows multi-year 1d data — different time scale — kept separate.
  const allCharts = [state.ceChart, state.peChart, state.oiChart, state.volChart];
  allCharts.forEach(c => {
    c.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (state.syncing || !range) return;
      state.syncing = true;
      allCharts.forEach(o => { if (o !== c) o.timeScale().setVisibleLogicalRange(range); });
      state.syncing = false;
    });
  });

  // ── Crosshair sync — hover CE, see PE/OI/Vol crosshair at the same time
  //    and vice-versa. Each chart's series provides the price for that bar
  //    so the horizontal line lands on the opposite-side candle's close.
  //    Without this, you'd only see vertical alignment on one pane at a time.
  const peers = [
    { chart: state.ceChart, series: state.ceCandle, dataKey: 'ceBars' },
    { chart: state.peChart, series: state.peCandle, dataKey: 'peBars' },
    { chart: state.oiChart, series: state.sOiCE_p,  dataKey: 'ceOI'  },
    { chart: state.volChart, series: state.sIvCE_p, dataKey: 'ceIV'  },
  ];
  peers.forEach(src => {
    src.chart.subscribeCrosshairMove(param => {
      if (state.crossSyncing) return;
      state.crossSyncing = true;
      try {
        if (!param || !param.time) {
          peers.forEach(p => { if (p.chart !== src.chart) p.chart.clearCrosshairPosition(); });
          return;
        }
        peers.forEach(p => {
          if (p.chart === src.chart) return;
          const data = state[p.dataKey] || [];
          // Lightweight-charts gives `param.time` as a UTCTimestamp number;
          // find the corresponding bar on the peer series. Fall back to a
          // nearby bar so the line still shows on TFs where bars don't line
          // up exactly (e.g. one side missing a minute).
          let bar = data.find(b => b.time === param.time);
          if (!bar && data.length) {
            let best = data[0], bestDt = Math.abs(data[0].time - param.time);
            for (let i = 1; i < data.length; i++) {
              const dt = Math.abs(data[i].time - param.time);
              if (dt < bestDt) { best = data[i]; bestDt = dt; }
            }
            bar = best;
          }
          if (!bar) { p.chart.clearCrosshairPosition(); return; }
          const price = ('close' in bar) ? bar.close : bar.value;
          try { p.chart.setCrosshairPosition(price, bar.time, p.series); }
          catch (e) { /* series may be empty if user toggled an indicator off */ }
        });
      } finally {
        state.crossSyncing = false;
      }
    });
  });
}

async function loadOption() {
  if (!state.expiry || !state.date) return;
  if (!state.ceStrike && !state.peStrike) return;
  $('opt-strike-lbl').textContent = `CE ${state.ceStrike ?? '—'} · PE ${state.peStrike ?? '—'}`;
  $('opt-tf').textContent = state.tf;

  async function fetchLeg(strike, type) {
    if (!strike) return { bars: [], iv: [], oi: [] };
    const qs = new URLSearchParams({
      date: state.date, expiry: state.expiry, strike: String(strike),
      tf: state.tf, type,
      with_iv: state.showIV ? 'true' : 'false',
      with_oi: state.showOI ? 'true' : 'false',
    });
    return fetch(`/api/leg?${qs}`).then(r => r.ok ? r.json() : { bars: [], iv: [], oi: [] });
  }
  const [rCE, rPE] = await Promise.all([
    fetchLeg(state.ceStrike, 'CE'),
    fetchLeg(state.peStrike, 'PE'),
  ]);
  state.ceBars = rCE.bars || []; state.ceIV = rCE.iv || []; state.ceOI = rCE.oi || [];
  state.peBars = rPE.bars || []; state.peIV = rPE.iv || []; state.peOI = rPE.oi || [];
  state.ceCandle.setData(state.ceBars);
  state.peCandle.setData(state.peBars);
  refreshEMAs();
  // Push indicator data to their dedicated sub-pane charts
  state.sOiCE_p.setData(state.showOI ? state.ceOI : []);
  state.sOiPE_p.setData(state.showOI ? state.peOI : []);
  state.sIvCE_p.setData(state.showIV ? state.ceIV : []);
  state.sIvPE_p.setData(state.showIV ? state.peIV : []);
  // RV is spot-based — fetched once per day/TF/lookback, then reused
  if (state.showRV) await loadRV();
  else { state.sRv_p.setData([]); }
  // Refit indicator panes after data
  syncPaneVisibility();
  refitAllCharts();   // time + price autoscale on all 4 panes
  $('opt-info').textContent = `CE ${state.ceBars.length} · PE ${state.peBars.length} bars`;
  // Two-pass refit: (1) immediately, (2) on the next animation frame after
  // the flex chain has finished resolving. Lightweight-charts' autoSize may
  // measure 0 before the layout settles, leaving the canvas empty — refitting
  // again on the next frame forces it to remeasure and redraw.
  requestAnimationFrame(() => {
    refitAllCharts();
    updateCursors();
  });
  // And once more after a tick in case fonts/scrollbars shifted the layout.
  setTimeout(refitAllCharts, 120);
}

// Classic EMA over OHLC bars — seeds from the first close so it doesn't lag
// off the bottom of the chart at series start.
function computeEMA(bars, period) {
  if (!bars || bars.length === 0 || period < 1) return [];
  const k = 2 / (period + 1);
  const out = new Array(bars.length);
  let ema = bars[0].close;
  for (let i = 0; i < bars.length; i++) {
    ema = i === 0 ? bars[i].close : bars[i].close * k + ema * (1 - k);
    out[i] = { time: bars[i].time, value: ema };
  }
  return out;
}

function refreshEMAs() {
  if (!state.sEmaFastCE) return;
  const f = state.emaOn ? Math.max(1, parseInt(state.emaFast, 10) || 9)  : 0;
  const s = state.emaOn ? Math.max(1, parseInt(state.emaSlow, 10) || 21) : 0;
  state.sEmaFastCE.setData(f ? computeEMA(state.ceBars, f) : []);
  state.sEmaSlowCE.setData(s ? computeEMA(state.ceBars, s) : []);
  state.sEmaFastPE.setData(f ? computeEMA(state.peBars, f) : []);
  state.sEmaSlowPE.setData(s ? computeEMA(state.peBars, s) : []);
}

// Reset time + price scales on every candle/indicator chart so a brand-new
// date's data is always visible regardless of any prior pan/zoom the user did.
// Without the priceScale autoScale reset, lightweight-charts keeps whatever
// vertical range the user dragged it to — that's why new dates' candles can
// land way above or below the visible area.
function refitAllCharts() {
  const charts = [state.ceChart, state.peChart, state.idxChart, state.oiChart, state.volChart];
  charts.forEach(c => {
    if (!c) return;
    try {
      c.applyOptions({ autoSize: true });
      c.timeScale().fitContent();
      c.priceScale('right').applyOptions({ autoScale: true });
    } catch (e) { /* chart not ready yet — next refit will catch it */ }
  });
}

// ── INDEX view (full 6-yr spot, TradingView-style) ────────────────────────
// Loaded ONCE per session, cached in state.idxFullBars. After that, picking
// a different date just scrolls the chart — no refetch, no flicker.
async function loadIndex() {
  if (state.viewMode !== 'index') return;

  // Already loaded → just scroll to the current selected date
  if (state.idxFullBars && state.idxFullBars.length) {
    scrollIndexToDate();
    return;
  }

  // First-time fetch of the full history (1d resolution, ~1.5k bars)
  $('opt-info').textContent = 'loading full history…';
  try {
    const r = await fetch('/api/spot_all?tf=1d');
    if (!r.ok) {
      state.idxFullBars = [];
      return;
    }
    const j = await r.json();
    state.idxFullBars = j.bars || [];
    state.idxBars = state.idxFullBars;
    state.idxCandle.setData(state.idxBars);
    // First-render: show the full 5-year range. User can scroll, zoom, or
    // pick a different day; date changes will re-center via scrollIndexToDate().
    state.idxChart.timeScale().fitContent();
    $('opt-info').textContent = `${state.idxBars.length} bars · ${state.idxBars.length} days`;
  } catch (e) {
    state.idxFullBars = [];
    $('opt-info').textContent = `index load failed: ${e}`;
  }
}

// Fetch the per-day ATM±1 combined OI series once per session; render into
// the dedicated sub-pane (independent of the per-leg OI pane used in OPTION).
async function loadIndexOi() {
  if (!state.showAtmOi || state.viewMode !== 'index') {
    if (state.sIdxOiCE) state.sIdxOiCE.setData([]);
    if (state.sIdxOiPE) state.sIdxOiPE.setData([]);
    return;
  }
  if (state.idxOiCEBars.length && state.idxOiPEBars.length) {
    state.sIdxOiCE.setData(state.idxOiCEBars);
    state.sIdxOiPE.setData(state.idxOiPEBars);
    state.idxOiChart.timeScale().fitContent();
    return;
  }
  try {
    const r = await fetch('/api/index_oi');
    if (!r.ok) return;
    const j = await r.json();
    state.idxOiCEBars = j.ce_oi || [];
    state.idxOiPEBars = j.pe_oi || [];
    state.sIdxOiCE.setData(state.idxOiCEBars);
    state.sIdxOiPE.setData(state.idxOiPEBars);
    state.idxOiChart.timeScale().fitContent();
  } catch (_) {}
}

// Center the index chart on `state.date` with a ~120-day window. The gold
// cursor line marks the date itself; user can pan/zoom freely from there.
function scrollIndexToDate() {
  if (!state.date || !state.idxBars || !state.idxBars.length || !state.idxChart) return;
  // state.date is "YYYYMMDD". Compute the bar timestamp for that day's 09:15 IST
  // (which we encoded as UTC seconds at the same wall-clock — see to_unix).
  const y = parseInt(state.date.slice(0, 4), 10);
  const m = parseInt(state.date.slice(4, 6), 10) - 1;
  const d = parseInt(state.date.slice(6, 8), 10);
  const targetT = Date.UTC(y, m, d, 9, 15) / 1000;

  // Find the bar at-or-before targetT (the date might be a non-trading day)
  let idx = -1;
  for (let i = state.idxBars.length - 1; i >= 0; i--) {
    if (state.idxBars[i].time <= targetT) { idx = i; break; }
  }
  if (idx < 0) idx = 0;

  const half = 60;   // 60 trading days on each side → ~6-month window
  const from = Math.max(0, idx - half);
  const to   = Math.min(state.idxBars.length - 1, idx + half);
  try {
    state.idxChart.timeScale().setVisibleRange({
      from: state.idxBars[from].time,
      to:   state.idxBars[to].time,
    });
  } catch (_) { /* range may not be valid yet — fitContent handles it */ }
  placeCursor($('cur-idx'), state.idxChart, state.idxBars, state.idxBars[idx].time);
  $('opt-info').textContent =
    `INDEX · 6yr · ${state.idxBars.length} bars · centered ${state.date}`;
}

// ── RV (spot realized vol) — drawn on the vol sub-pane ──────────────────
async function loadRV() {
  if (!state.date) return;
  const qs = new URLSearchParams({
    date: state.date, tf: state.tf, lookback: String(state.rvLookback),
  });
  try {
    const r = await fetch(`/api/rv?${qs}`);
    if (!r.ok) { state.rvLine = []; return; }
    const j = await r.json();
    state.rvLine = j.rv || [];
    if (state.showRV) {
      state.sRv_p.setData(state.rvLine);
      state.volChart.timeScale().fitContent();
    }
  } catch (_) {}
}

// Show or hide the indicator sub-panes based on toggle state.
function syncPaneVisibility() {
  $('pane-oi').classList.toggle('hidden', !state.showOI);
  $('pane-vol').classList.toggle('hidden', !state.showIV && !state.showRV);
  // View toggle: OPTION shows CE+PE + scrubber + day-list sidebar.
  // INDEX shows IDX only (full-width), hides CE/PE, scrubber, sidebar.
  const indexMode = state.viewMode === 'index';
  document.querySelector('.opt-sub.ce').classList.toggle('hidden', indexMode);
  document.querySelector('.opt-sub.pe').classList.toggle('hidden', indexMode);
  $('opt-sub-idx').classList.toggle('hidden', !indexMode);
  const scrub = $('scrub-bar'); if (scrub) scrub.classList.toggle('hidden', indexMode);
  // Left sidebar (day-list) is per-day intraday navigation — not useful for
  // the 6-yr historical index view, so hide in INDEX mode.
  const sidebar = $('sidebar'); if (sidebar) sidebar.classList.toggle('hidden', indexMode);
  document.querySelectorAll('.resizer.v').forEach(r => r.classList.toggle('hidden', indexMode));
  // ATM ±1 combined-OI sub-pane shows only in INDEX mode with the toggle on
  $('pane-idx-oi').classList.toggle('hidden', !(indexMode && state.showAtmOi));
  // FIT-ALL button is INDEX-mode only
  const fitBtn = $('idx-fit'); if (fitBtn) fitBtn.style.display = indexMode ? 'inline-block' : 'none';
  // After visibility changes, charts may need a re-measure + a clean fit so
  // the canvas doesn't keep a stale visible range from the previous view.
  setTimeout(() => {
    if (state.oiChart)  state.oiChart.applyOptions({ autoSize: true });
    if (state.volChart) state.volChart.applyOptions({ autoSize: true });
    if (state.idxChart) {
      state.idxChart.applyOptions({ autoSize: true });
      // On entering INDEX, always fit the full 5-yr range. User scrolls/zooms freely
      // from there; the FIT ALL button restores this view at any time.
      if (indexMode && state.idxBars && state.idxBars.length) {
        state.idxChart.timeScale().fitContent();
      }
    }
    if (state.idxOiChart) state.idxOiChart.applyOptions({ autoSize: true });
    refitAllCharts();
    updateCursors();
  }, 60);
}

// ── Cursor line (gold vertical) — placed via pixel offset on the chart ───
function updateCursors() {
  const t = minuteToUnix(state.date, state.minute);
  if (state.viewMode === 'option') {
    placeCursor($('cur-ce'),  state.ceChart,  state.ceBars, t);
    placeCursor($('cur-pe'),  state.peChart,  state.peBars, t);
  } else {
    placeCursor($('cur-idx'), state.idxChart, state.idxBars, t);
  }
  // Indicator panes use any available bar series for x-coordinate lookup
  if (state.showOI)                 placeCursor($('cur-oi'),  state.oiChart,  state.ceOI.length ? state.ceOI : state.peOI, t);
  if (state.showIV || state.showRV) placeCursor($('cur-vol'), state.volChart, state.rvLine.length ? state.rvLine : (state.ceIV.length ? state.ceIV : state.peIV), t);
}

function placeCursor(lineEl, chart, bars, t, pxLabel) {
  if (!chart || !bars.length) return;
  // Snap to nearest bar at-or-before t (cursor sits ON a bar)
  const idx = bisectAtOrBefore(bars, t);
  if (idx < 0) { lineEl.style.left = '-20px'; return; }
  const bar = bars[idx];
  const x = chart.timeScale().timeToCoordinate(bar.time);
  if (x == null) { lineEl.style.left = '-20px'; return; }
  lineEl.style.left = x + 'px';
  if (pxLabel) pxLabel.textContent = bar.close.toFixed(2);
}

function bisectAtOrBefore(bars, t) {
  let lo = 0, hi = bars.length - 1, ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (bars[m].time <= t) { ans = m; lo = m + 1; }
    else hi = m - 1;
  }
  return ans;
}

// minute 0 = 09:15 IST on the day. The parquet stores naive-IST timestamps and
// we convert them to unix-after-IST-shift (treat IST wall-clock as UTC). So:
//   minute → unix = date_at_0915_IST_treated_as_UTC + minute*60.
function minuteToUnix(date, minute) {
  const y = +date.slice(0,4), mo = +date.slice(4,6), d = +date.slice(6,8);
  return Math.floor(Date.UTC(y, mo - 1, d, 9, 15, 0) / 1000) + minute * 60;
}

// ── Pickers ────────────────────────────────────────────────────────────────
// ── Sidebar: every trading day, click to load ────────────────────────────
let _allDays = [];
async function loadDays() {
  const r = await fetch('/api/days');
  _allDays = await r.json();
  renderDayList(_allDays);
  $('side-count').textContent = `${_allDays.length} days`;
  // Default to most-recent day
  state.date = _allDays[_allDays.length - 1];
  $('day-readout').textContent = fmtDay(state.date);
  highlightActiveDay();
}

function fmtDay(d) {
  return `${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6)}`;
}

function renderDayList(days) {
  const ul = $('day-list');
  // Most-recent first, grouped by year with sticky headers
  const desc = [...days].reverse();
  let html = '';
  let lastYear = '';
  for (const d of desc) {
    const y = d.slice(0,4);
    if (y !== lastYear) {
      html += `<li class="year-sep">${y}</li>`;
      lastYear = y;
    }
    html += `<li data-day="${d}"><span>${d.slice(4,6)}-${d.slice(6)}</span><span class="y">${d.slice(0,4)}</span></li>`;
  }
  ul.innerHTML = html;
  ul.querySelectorAll('li[data-day]').forEach(li => {
    li.addEventListener('click', () => selectDay(li.dataset.day));
  });
}

function highlightActiveDay() {
  const ul = $('day-list');
  ul.querySelectorAll('li.active').forEach(li => li.classList.remove('active'));
  const li = ul.querySelector(`li[data-day="${state.date}"]`);
  if (li) {
    li.classList.add('active');
    li.scrollIntoView({ block: 'center' });
  }
}

async function selectDay(d) {
  state.date = d;
  $('day-readout').textContent = fmtDay(d);
  state.ceStrike = null; state.peStrike = null;
  highlightActiveDay();
  await loadExpiries();
  await loadStrikes();
  setMinute(0);
  await loadOption();
  // In INDEX mode the full history is already loaded — just jump the
  // chart to the picked date. No refetch.
  if (state.viewMode === 'index') scrollIndexToDate();
  loadChain();
}

async function loadStrikes() {
  if (!state.date || !state.expiry) return;
  const r = await fetch(`/api/strikes?date=${state.date}&expiry=${encodeURIComponent(state.expiry)}`);
  if (!r.ok) return;
  const j = await r.json();

  function fill(sel) {
    sel.innerHTML = '';
    (j.strikes || []).forEach(k => {
      const o = document.createElement('option');
      o.value = String(k);
      o.textContent = (k === j.atm) ? `${k}  ★ ATM` : String(k);
      sel.appendChild(o);
    });
  }
  const ceSel = $('ce-strike'); const peSel = $('pe-strike');
  fill(ceSel); fill(peSel);

  const defaultK = j.atm ?? (j.strikes ? j.strikes[Math.floor(j.strikes.length / 2)] : null);
  state.atm      = j.atm;
  state.ceStrike = defaultK;
  state.peStrike = defaultK;
  if (defaultK != null) {
    ceSel.value = String(defaultK);
    peSel.value = String(defaultK);
  }
}

async function loadExpiries() {
  const r = await fetch(`/api/expiries?date=${state.date}`);
  if (!r.ok) return;
  const exps = await r.json();
  const sel = $('expiry'); sel.innerHTML = '';
  exps.forEach(e => {
    const o = document.createElement('option'); o.value = e; o.textContent = e;
    sel.appendChild(o);
  });
  state.expiry = exps[0];
  sel.value = state.expiry;
}

// ── Chain fetch → drives the metric strip ─────────────────────────────────
async function loadChain() {
  if (!state.date || !state.expiry) return;
  if (state.pendingFetch) state.pendingFetch.cancelled = true;
  const tag = { cancelled: false }; state.pendingFetch = tag;

  $('status').textContent = 'Loading metrics…';
  try {
    const r = await fetch(`/api/chain_at?date=${state.date}&expiry=${encodeURIComponent(state.expiry)}&minute=${state.minute}&band=${state.band}`);
    if (!r.ok || tag.cancelled) { $('status').textContent = 'error'; return; }
    const j = await r.json();
    if (tag.cancelled) return;
    state.atm = j.atm;
    // First-load fallback: if strike selectors haven't populated yet, default both to ATM
    if ((state.ceStrike == null || state.peStrike == null) && j.atm != null) {
      if (state.ceStrike == null) state.ceStrike = j.atm;
      if (state.peStrike == null) state.peStrike = j.atm;
      loadOption();
    }
    // ATM-row CE/PE IVs (computed on the backend via brentq)
    const atmRow = (j.rows || []).find(r => r.atm) || null;
    const ceIV = atmRow && atmRow.ce_iv != null ? atmRow.ce_iv : null;
    const peIV = atmRow && atmRow.pe_iv != null ? atmRow.pe_iv : null;

    $('spot-val').textContent  = j.spot ? j.spot.toFixed(2) : '—';
    $('m-spot').textContent    = j.spot ? j.spot.toFixed(2) : '—';
    $('m-atm').textContent     = j.atm ?? '—';
    $('m-maxpain').textContent = j.max_pain ?? '—';
    $('m-pcr-oi').textContent  = j.pcr_oi  != null ? j.pcr_oi.toFixed(2)  : '—';
    $('m-pcr-vol').textContent = j.pcr_vol != null ? j.pcr_vol.toFixed(2) : '—';
    $('m-dte').textContent     = j.dte_days != null ? j.dte_days.toFixed(1) + ' d' : '—';
    $('m-ce-iv').textContent   = ceIV != null ? ceIV.toFixed(1) + '%' : '—';
    $('m-pe-iv').textContent   = peIV != null ? peIV.toFixed(1) + '%' : '—';
    $('status').textContent    = `cursor ${j.cursor}`;
  } catch (e) { $('status').textContent = 'fetch error'; }
}

// ── Time scrubber + play ───────────────────────────────────────────────────
let scrubDebounce = null;
function setMinute(m) {
  state.minute = Math.max(0, Math.min(374, m | 0));
  $('slider').value = state.minute;
  $('cursor-time').textContent = minuteToHHMM(state.minute);
  $('minute-info').textContent = `${state.minute + 1} / 375`;
  updateCursors();
  if (scrubDebounce) clearTimeout(scrubDebounce);
  scrubDebounce = setTimeout(loadChain, 100);
}

function togglePlay() {
  state.playing = !state.playing;
  $('play').textContent = state.playing ? '❚❚' : '▶';
  $('play').classList.toggle('primary', state.playing);
  if (state.playing) {
    state.playTimer = setInterval(() => {
      if (state.minute >= 374) { togglePlay(); return; }
      setMinute(state.minute + 1);
    }, 220);
  } else if (state.playTimer) {
    clearInterval(state.playTimer); state.playTimer = null;
  }
}

// ── Wire UI ────────────────────────────────────────────────────────────────
function attach() {
  $('day-search').addEventListener('input', e => {
    const q = e.target.value.trim();
    const filtered = q ? _allDays.filter(d => d.includes(q)) : _allDays;
    renderDayList(filtered);
    highlightActiveDay();
  });
  $('expiry').addEventListener('change', async e => {
    state.expiry = e.target.value;
    state.ceStrike = null; state.peStrike = null;
    await loadStrikes();
    await loadOption();
    loadChain();
  });
  $('ce-strike').addEventListener('change', e => {
    state.ceStrike = parseInt(e.target.value, 10);
    loadOption();
  });
  $('pe-strike').addEventListener('change', e => {
    state.peStrike = parseInt(e.target.value, 10);
    loadOption();
  });

  // EMA controls — toggle + Fast/Slow inputs. All recompute on the client
  // from the already-loaded CE/PE bars, no backend round-trip.
  $('ema-on').addEventListener('change', e => {
    state.emaOn = e.target.checked;
    refreshEMAs();
  });
  $('ema-fast').addEventListener('input', e => {
    const v = parseInt(e.target.value, 10);
    if (v >= 1 && v <= 500) { state.emaFast = v; if (state.emaOn) refreshEMAs(); }
  });
  $('ema-slow').addEventListener('input', e => {
    const v = parseInt(e.target.value, 10);
    if (v >= 1 && v <= 500) { state.emaSlow = v; if (state.emaOn) refreshEMAs(); }
  });

  $('slider').addEventListener('input', e => setMinute(+e.target.value));
  $('play').addEventListener('click', togglePlay);

  // TF buttons → reload CE/PE option charts at new resolution
  document.querySelectorAll('#tf-grp button').forEach(b => {
    b.addEventListener('click', async () => {
      document.querySelectorAll('#tf-grp button').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      state.tf = b.dataset.tf;
      // INDEX mode is locked to 1d full-history; TF buttons only affect
      // CE/PE/OI/Vol when the user is in OPTION view.
      await loadOption();
      updateCursors();
    });
  });

  // Indicator toggle buttons (OI / IV / RV / ATM·OI)
  document.querySelectorAll('#ind-grp button').forEach(b => {
    b.addEventListener('click', async () => {
      const k = b.dataset.ind;
      const stateKey = ({oi:'showOI', iv:'showIV', rv:'showRV', atmoi:'showAtmOi'})[k];
      state[stateKey] = !state[stateKey];
      b.classList.toggle('on', state[stateKey]);
      if (k === 'atmoi') {
        await loadIndexOi();      // INDEX-only ATM±1 combined-OI overlay
      } else if (k === 'rv') {
        if (state.showRV) await loadRV();
        else state.sRv_p.setData([]);
      } else {
        // OI / IV need a fresh /api/leg call with the with_iv / with_oi flag
        await loadOption();
      }
      syncPaneVisibility();
    });
  });

  // View-mode toggle (OPTION ⇄ INDEX). OPTION shows CE+PE side-by-side
  // (default). INDEX shows the NIFTY spot chart full-width with CE/PE hidden.
  document.querySelectorAll('#view-grp button').forEach(b => {
    b.addEventListener('click', async () => {
      const v = b.dataset.view;
      if (state.viewMode === v) return;
      document.querySelectorAll('#view-grp button').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      state.viewMode = v;
      if (v === 'index') await loadIndex();
      syncPaneVisibility();
      updateCursors();
    });
  });

  // FIT ALL — restore full 5-yr view at any zoom level
  $('idx-fit').addEventListener('click', () => {
    if (state.idxChart) state.idxChart.timeScale().fitContent();
    if (state.idxOiChart) state.idxOiChart.timeScale().fitContent();
  });

  // RV lookback editor — only refetches RV when changed (cheap)
  $('rv-lookback').addEventListener('change', async e => {
    const v = parseInt(e.target.value, 10);
    if (!Number.isFinite(v)) return;
    state.rvLookback = Math.max(3, Math.min(200, v));
    e.target.value = state.rvLookback;
    if (state.showRV) await loadRV();
  });

  // Reposition cursor lines on resize (chart pixel coordinates change),
  // and re-fit every chart so a window resize / phone rotation doesn't strand
  // candles off-screen.
  window.addEventListener('resize', () => setTimeout(() => {
    refitAllCharts();
    updateCursors();
  }, 50));
}

// ── Resize handles between panes ──────────────────────────────────────────
function attachResizers() {
  document.querySelectorAll('.resizer').forEach(handle => {
    handle.addEventListener('mousedown', startResizeDrag);
  });
}
function startResizeDrag(e) {
  e.preventDefault();
  const handle  = e.currentTarget;
  const vertical = handle.classList.contains('v');   // v handle = column resize (x-axis)
  const prev = document.getElementById(handle.dataset.prev);
  const next = document.getElementById(handle.dataset.next);
  if (!prev || !next) return;

  const prevR = prev.getBoundingClientRect();
  const nextR = next.getBoundingClientRect();
  const dimKey = vertical ? 'width'   : 'height';
  const evtKey = vertical ? 'clientX' : 'clientY';
  const total  = prevR[dimKey] + nextR[dimKey];
  const startVal = e[evtKey];
  const startPrev = prevR[dimKey];

  handle.classList.add('dragging');
  document.body.classList.add('resizing', vertical ? 'r-col' : 'r-row');

  function onMove(ev) {
    const delta = ev[evtKey] - startVal;
    let newPrev = startPrev + delta;
    const minS = 80;
    if (newPrev < minS) newPrev = minS;
    if (newPrev > total - minS) newPrev = total - minS;
    const ratio = newPrev / total;
    if (vertical) {
      prev.style.flex = `0 0 ${(ratio * 100).toFixed(2)}%`;
      next.style.flex = '1';
    } else {
      prev.style.flex = (ratio).toFixed(3);
      next.style.flex = (1 - ratio).toFixed(3);
    }
  }
  function onUp() {
    handle.classList.remove('dragging');
    document.body.classList.remove('resizing', 'r-row', 'r-col');
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
    setTimeout(updateCursors, 60);
  }
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup',   onUp);
}

// ── Fullscreen toggle on each pane ────────────────────────────────────────
let _fsKeyHandler = null;
function attachFullscreen() {
  document.querySelectorAll('.fs-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const pane = btn.closest('.pane');
      if (!pane) return;
      pane.classList.toggle('fullscreen');
      if (pane.classList.contains('fullscreen')) {
        if (_fsKeyHandler) document.removeEventListener('keydown', _fsKeyHandler);
        _fsKeyHandler = ev => {
          if (ev.key === 'Escape') {
            pane.classList.remove('fullscreen');
            document.removeEventListener('keydown', _fsKeyHandler);
            _fsKeyHandler = null;
            setTimeout(updateCursors, 80);
          }
        };
        document.addEventListener('keydown', _fsKeyHandler);
      }
      setTimeout(updateCursors, 80);
    });
  });
}

async function boot() {
  makeAllCharts();
  attach();
  attachResizers();
  attachFullscreen();

  await loadDays();
  await loadExpiries();
  await loadStrikes();
  setMinute(0);
  await loadOption();   // ATM CE+PE candles populate immediately
  loadChain();          // metric strip
}
boot();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML.replace("__APP_BASE__", APP_BASE))
