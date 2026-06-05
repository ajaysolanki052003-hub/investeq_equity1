"""
NIFTY Option-Chain Replay — TradingView-style.

Run:
    python -m uvicorn app_option_replay_tv:app --port 8703 --host 0.0.0.0

Open:
    http://localhost:8703/

Three synced lightweight-charts panes (candles+spot, IV, OI) plus a volume
histogram on the main pane. Expiry/strike dropdowns auto-populate from the
parquet files under DATA/.
"""

from __future__ import annotations
import os
from datetime import datetime
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
# When mounted behind a path prefix (e.g. nginx /straddle/), set APP_BASE so
# fetch('/api/...') calls in the inline JS are rewritten to '/straddle/api/...'.
APP_BASE  = os.environ.get("APP_BASE", "").rstrip("/")
OPT_DIR   = ROOT / "options"
OI_DIR    = ROOT / "oi"
SPOT_DIR  = ROOT / "spot"

RISK_FREE_RATE = 0.065
SECONDS_PER_YEAR = 365.0 * 24 * 3600

TF_RULE = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "60min", "4h": "240min",
}

# Bars per trading day for each TF — NSE session is 09:15 → 15:30 = 375 minutes.
BARS_PER_DAY = {
    "1m": 375, "3m": 125, "5m": 75, "15m": 25,
    "30m": 13, "1h": 7,  "4h":  2,
}
TRADING_DAYS_YEAR = 252


# ─── Black-Scholes / IV ───────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(market_price, S, K, T, opt_type, r=RISK_FREE_RATE):
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
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


def bs_greeks(S, K, T, r, sigma, opt_type):
    """Returns {delta, gamma, theta_per_day, vega_per_pct} or None."""
    if T <= 0 or sigma is None or sigma <= 0 or S <= 0 or K <= 0:
        return None
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    nprime_d1 = norm.pdf(d1)
    if opt_type == "CE":
        delta = float(norm.cdf(d1))
        theta = (-S * nprime_d1 * sigma / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = float(norm.cdf(d1) - 1.0)
        theta = (-S * nprime_d1 * sigma / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    gamma = float(nprime_d1 / (S * sigma * sqrt_T))
    vega  = float(S * nprime_d1 * sqrt_T / 100.0)        # per 1% vol move
    return {"delta": delta, "gamma": gamma,
            "theta": float(theta), "vega": vega}


def years_to_expiry(ts: pd.Timestamp, expiry: str) -> float:
    # Calendar-time convention (T = wall-clock seconds / seconds-per-year).
    # Mixing calendar seconds with a trading-second denominator inflates T
    # by ~5x and crushes IV — same bug existed in the notebook.
    expiry_dt = pd.to_datetime(expiry).replace(hour=15, minute=30)
    return max((expiry_dt - ts).total_seconds() / SECONDS_PER_YEAR, 1e-8)


# ─── Loaders (memoised in-process) ────────────────────────────────────────────
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


def recent_trading_days(date: str, n: int) -> list[str]:
    """Return up to n most-recent trading days ending at `date` (inclusive)."""
    days = list_trading_days()
    if date not in days:
        return []
    idx = days.index(date)
    n = max(1, min(int(n), 30))
    return list(days[max(0, idx - n + 1): idx + 1])


def load_concat(loader, days: list[str]) -> pd.DataFrame:
    parts = [loader(d) for d in days]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


# ─── Realized vol ─────────────────────────────────────────────────────────────
def realized_vol(spot_bars: pd.DataFrame, tf: str, window: int = 30,
                 estimator: str = "close") -> pd.DataFrame:
    """Rolling annualized realized vol from spot bars.

    estimator:
      "close"     close-to-close log returns (default)
      "parkinson" Parkinson high-low estimator (variance-efficient)
    """
    if spot_bars.empty or len(spot_bars) < window + 1:
        return pd.DataFrame(columns=["timestamp", "rv"])
    ann = float(BARS_PER_DAY.get(tf, 75) * TRADING_DAYS_YEAR) ** 0.5

    if estimator == "parkinson":
        log_hl_sq = (np.log(spot_bars["high"] / spot_bars["low"]) ** 2)
        var = log_hl_sq.rolling(window).mean() / (4.0 * np.log(2.0))
        rv = np.sqrt(var) * ann
    else:
        logret = np.log(spot_bars["close"] / spot_bars["close"].shift(1))
        rv = logret.rolling(window).std(ddof=1) * ann

    return (pd.DataFrame({"timestamp": spot_bars["timestamp"], "rv": rv})
              .dropna(subset=["rv"])
              .reset_index(drop=True))


# ─── Resampling helpers ───────────────────────────────────────────────────────
def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df.empty:
        return df
    rule = TF_RULE[tf]
    return (df.set_index("timestamp")
              .resample(rule, closed="left", label="left")
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"})
              .dropna(subset=["open", "close"])
              .reset_index())


def to_unix(ts: pd.Series) -> np.ndarray:
    return (ts.view("int64") // 1_000_000_000).to_numpy()


def to_candle_payload(bars: pd.DataFrame) -> list[dict]:
    if bars.empty:
        return []
    t = to_unix(bars["timestamp"])
    return [
        {"time": int(t[i]),
         "open":  float(bars["open"].iloc[i]),
         "high":  float(bars["high"].iloc[i]),
         "low":   float(bars["low"].iloc[i]),
         "close": float(bars["close"].iloc[i])}
        for i in range(len(bars))
    ]


def to_line_payload(ts: pd.Series, values: pd.Series, scale=1.0) -> list[dict]:
    t = to_unix(ts)
    return [{"time": int(t[i]), "value": float(values.iloc[i] * scale)}
            for i in range(len(ts)) if pd.notna(values.iloc[i])]


def to_hist_payload(bars: pd.DataFrame, color_up="#26a69a", color_dn="#ef5350") -> list[dict]:
    if bars.empty:
        return []
    t = to_unix(bars["timestamp"])
    up = bars["close"] >= bars["open"]
    return [
        {"time": int(t[i]),
         "value": float(bars["volume"].iloc[i]),
         "color": color_up if bool(up.iloc[i]) else color_dn}
        for i in range(len(bars))
    ]


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="NIFTY Option Replay TV")


@app.get("/api/days")
def days():
    return JSONResponse(list(list_trading_days()))


@app.get("/api/expiries")
def expiries(date: str = Query(...)):
    df = load_options(date)
    if df.empty:
        raise HTTPException(404, "no data for day")
    return sorted(df["expiry"].unique().tolist())


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


@app.get("/api/leg")
def leg(date: str, expiry: str, strike: int, type: str,
        tf: str = "5m", with_iv: bool = True, with_oi: bool = True,
        with_spot: bool = True, with_rv: bool = True,
        rv_window: int = 30, rv_estimator: str = "close",
        lookback_days: int = 3):
    if tf not in TF_RULE:
        raise HTTPException(400, f"bad tf: {tf}")

    days = recent_trading_days(date, lookback_days)
    if not days:
        raise HTTPException(404, "date not in dataset")

    opt = load_concat(load_options, days)
    if opt.empty:
        raise HTTPException(404, "no options data for window")

    leg_df = opt[(opt["expiry"] == expiry) &
                 (opt["strike"] == strike) &
                 (opt["option_type"] == type)]
    if leg_df.empty:
        return JSONResponse({"candles": [], "volume": [], "iv": [],
                             "rv": [], "oi": [], "spot": [], "days": days})

    bars = resample_ohlc(
        leg_df[["timestamp", "open", "high", "low", "close", "volume"]], tf)

    candles = to_candle_payload(bars)
    volume  = to_hist_payload(bars)

    iv_series = []
    rv_series = []
    spot_series = []
    greek_series = {"delta": [], "gamma": [], "theta": [], "vega": []}
    if with_iv or with_spot or with_rv:
        spot = load_concat(load_spot, days)
        if not spot.empty:
            spot_bars = resample_ohlc(
                spot[["timestamp", "open", "high", "low", "close", "volume"]], tf)
            if with_spot:
                spot_series = to_line_payload(spot_bars["timestamp"], spot_bars["close"])
            if with_iv and not bars.empty:
                merged = pd.merge_asof(
                    bars[["timestamp", "close"]].rename(columns={"close": "opt"}),
                    spot_bars[["timestamp", "close"]].rename(columns={"close": "spot"}),
                    on="timestamp", direction="backward",
                )
                ivs, greeks_rows = [], []
                for _, r in merged.iterrows():
                    T = years_to_expiry(r["timestamp"], expiry)
                    iv = implied_vol(r["opt"], r["spot"], strike, T, type)
                    ivs.append(iv)
                    g = bs_greeks(r["spot"], strike, T, RISK_FREE_RATE, iv, type) if iv else None
                    greeks_rows.append(g or {"delta": None, "gamma": None,
                                              "theta": None, "vega": None})
                merged["iv"] = ivs
                merged["delta"] = [g["delta"] for g in greeks_rows]
                merged["gamma"] = [g["gamma"] for g in greeks_rows]
                merged["theta"] = [g["theta"] for g in greeks_rows]
                merged["vega"]  = [g["vega"]  for g in greeks_rows]
                merged_iv = merged.dropna(subset=["iv"])
                iv_series = to_line_payload(merged_iv["timestamp"],
                                            merged_iv["iv"], scale=100.0)
                greek_series = {
                    k: to_line_payload(merged_iv["timestamp"], merged_iv[k])
                    for k in ("delta", "gamma", "theta", "vega")
                }
            if with_rv:
                rv_df = realized_vol(spot_bars, tf=tf,
                                     window=max(2, int(rv_window)),
                                     estimator=rv_estimator)
                rv_series = to_line_payload(rv_df["timestamp"],
                                            rv_df["rv"], scale=100.0)

    oi_series = []
    if with_oi:
        oi_df = load_concat(load_oi, days)
        if not oi_df.empty:
            o = oi_df[(oi_df["expiry"] == expiry) &
                      (oi_df["strike"] == strike) &
                      (oi_df["option_type"] == type)]
            if not o.empty:
                resampled = (o.set_index("timestamp")["oi"]
                              .resample(TF_RULE[tf], closed="left", label="left")
                              .last()
                              .dropna())
                oi_series = to_line_payload(resampled.index.to_series(),
                                            resampled)

    # ── Session stats for the most-recent day in the window ───────────────────
    today = days[-1]

    def session_extent_naive(payload):
        if not payload:
            return None
        # times in payload are unix-after-IST-shift; filter by today's date string
        today_date = pd.Timestamp(f"{today[:4]}-{today[4:6]}-{today[6:]}").date()
        rows = [p for p in payload
                if pd.Timestamp(p["time"], unit="s").date() == today_date]
        if not rows:
            return None
        vals = [p["value"] if "value" in p else p["close"] for p in rows]
        return {
            "open"   : float(rows[0]["value"] if "value" in rows[0] else rows[0]["open"]),
            "current": float(rows[-1]["value"] if "value" in rows[-1] else rows[-1]["close"]),
            "high"   : float(max(vals)),
            "low"    : float(min(vals)),
            "n_bars" : len(rows),
        }

    stats = {
        "today"     : today,
        "iv"        : session_extent_naive(iv_series),
        "rv"        : session_extent_naive(rv_series),
        "oi"        : session_extent_naive(oi_series),
        "spot"      : session_extent_naive(spot_series),
        # Premium decay over today's session
        "premium"   : ({"open": float(candles[0]["open"]) if candles else None,
                        "current": float(candles[-1]["close"]) if candles else None,
                        "high": float(max(c["high"] for c in candles)) if candles else None,
                        "low":  float(min(c["low"]  for c in candles)) if candles else None})
                       if candles else None,
    }
    if stats["iv"] and stats["rv"]:
        stats["iv_rv_spread"] = stats["iv"]["current"] - stats["rv"]["current"]

    return JSONResponse({
        "candles": candles, "volume": volume,
        "iv": iv_series, "rv": rv_series,
        "oi": oi_series, "spot": spot_series,
        "greeks": greek_series,
        "stats": stats,
        "days": days,
    })


@app.get("/api/events")
def events():
    """Return every straddle event from notebook output (DATA/straddle_pnl.parquet),
    augmented with LGBM out-of-fold probability if a model feature file exists."""
    p = ROOT / "straddle_pnl.parquet"
    if not p.exists():
        return JSONResponse({"events": [], "source": None})
    df = pd.read_parquet(p)
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("expiry").reset_index(drop=True)

    # Optional LGBM OOF probabilities — take the most-recent run.
    ml_dl_root = ROOT.parent / "ML_DL" / "features"
    p_oof_map = {}
    src_features = None
    if ml_dl_root.exists():
        feats = sorted(ml_dl_root.glob("straddle_lgbm_*.parquet"))
        if feats:
            src_features = feats[-1].name
            f = pd.read_parquet(feats[-1])
            if "p_lgbm_oof" in f.columns and "expiry" in f.columns:
                f["expiry"] = pd.to_datetime(f["expiry"]).dt.strftime("%Y-%m-%d")
                p_oof_map = dict(zip(f["expiry"], f["p_lgbm_oof"]))

    out = []
    for _, r in df.iterrows():
        entry = str(r["entry_date"])
        out.append({
            "expiry"           : r["expiry"],
            "entry_date"       : entry,
            "atm_strike"       : int(r["atm_strike"]),
            "spot_at_entry"    : float(r["spot_at_entry"]),
            "spot_at_exit"     : float(r["spot_at_exit"]),
            "premium_collected": float(r["premium_collected"]),
            "premium_at_exit"  : float(r["premium_at_exit"]),
            "pnl_points"       : float(r["pnl_points"]),
            "pnl_pct"          : float(r["pnl_pct"]),
            "profitable"       : bool(r["profitable"]),
            "p_lgbm"           : (None if pd.isna(p_oof_map.get(r["expiry"]))
                                  else float(p_oof_map.get(r["expiry"]))),
        })
    return JSONResponse({
        "events"        : out,
        "source"        : "DATA/straddle_pnl.parquet",
        "feature_source": src_features,
    })


@app.get("/api/event_detail")
def event_detail(expiry: str = Query(...),
                 entry_date: str = Query(...),
                 atm_strike: int = Query(...)):
    """Per-leg (CE/PE) entry & exit price for a straddle event.

    Entry  = entry_date @ 15:20 IST   Exit = expiry @ 09:20 IST
    PnL is signed for a *short* straddle (premium collected − cost to close).
    Returned timestamps use the same naive-IST-as-UTC unix encoding that the
    chart's time axis uses, so they line up with chart bars.
    """
    ed = entry_date.replace("-", "")
    xd = expiry.replace("-", "")

    opt_entry = load_options(ed)
    opt_exit  = load_options(xd)
    if opt_entry.empty:
        raise HTTPException(404, f"no options data for entry day {ed}")
    if opt_exit.empty:
        raise HTTPException(404, f"no options data for exit day {xd}")

    entry_ts = pd.Timestamp(f"{ed[:4]}-{ed[4:6]}-{ed[6:]} 15:20:00")
    exit_ts  = pd.Timestamp(f"{xd[:4]}-{xd[4:6]}-{xd[6:]} 09:20:00")

    def lookup(df, ts, otype):
        sub = df[(df["expiry"] == expiry) &
                 (df["strike"] == atm_strike) &
                 (df["option_type"] == otype)]
        if sub.empty:
            return None
        i = (sub["timestamp"] - ts).abs().idxmin()
        row = sub.loc[i]
        # Use the bar's open price (the minute snapshot at ts)
        v = row.get("open")
        return float(v) if pd.notna(v) else None

    ce_entry = lookup(opt_entry, entry_ts, "CE")
    pe_entry = lookup(opt_entry, entry_ts, "PE")
    ce_exit  = lookup(opt_exit,  exit_ts,  "CE")
    pe_exit  = lookup(opt_exit,  exit_ts,  "PE")

    def diff(a, b):
        return None if (a is None or b is None) else float(a - b)

    return JSONResponse({
        "entry_time"          : int(entry_ts.value // 10**9),
        "exit_time"           : int(exit_ts.value  // 10**9),
        "ce_entry_px"         : ce_entry,
        "pe_entry_px"         : pe_entry,
        "ce_exit_px"          : ce_exit,
        "pe_exit_px"          : pe_exit,
        "ce_pnl"              : diff(ce_entry, ce_exit),
        "pe_pnl"              : diff(pe_entry, pe_exit),
        "premium_total_entry" : (ce_entry + pe_entry)
                                if (ce_entry is not None and pe_entry is not None)
                                else None,
        "premium_total_exit"  : (ce_exit + pe_exit)
                                if (ce_exit is not None and pe_exit is not None)
                                else None,
    })


@app.get("/api/smile")
def smile(date: str, expiry: str, time: int = Query(...,
          description="Unix timestamp (after IST-shift as used by the chart)"),
          band: int = 12):
    """ATM-banded IV across strikes (CE + PE) at the given cursor time."""
    opt  = load_options(date)
    spot = load_spot(date)
    if opt.empty or spot.empty:
        raise HTTPException(404, "no data")

    # Our to_unix() returns (naive-IST view as UTC ns)//1e9, so the int we get
    # back from the chart is exactly the parquet's naive wall-clock interpreted
    # in UTC. Recover the parquet timestamp by interpreting the int directly.
    cursor_ist = pd.Timestamp(time, unit="s")

    win = pd.Timedelta("3min")
    chain_slice = opt[(opt["expiry"] == expiry) &
                      (opt["timestamp"] >= cursor_ist - win) &
                      (opt["timestamp"] <= cursor_ist + win)]
    if chain_slice.empty:
        return JSONResponse({"strikes": [], "ce_iv": [], "pe_iv": [], "atm": None,
                             "spot": None, "cursor": str(cursor_ist)})

    spot_slice = spot[spot["timestamp"] <= cursor_ist]
    if spot_slice.empty:
        return JSONResponse({"strikes": [], "ce_iv": [], "pe_iv": [], "atm": None,
                             "spot": None, "cursor": str(cursor_ist)})
    spot_px = float(spot_slice.iloc[-1]["close"])

    # Latest price per (strike, option_type) in the window
    snap = (chain_slice.sort_values("timestamp")
            .groupby(["strike", "option_type"])["close"].last()
            .unstack("option_type").reset_index())  # cols: strike, CE, PE

    strikes_all = sorted(snap["strike"].tolist())
    atm = min(strikes_all, key=lambda k: abs(k - spot_px))
    # Use the median strike spacing — robust to thin tails (15000, 16000, 17000…)
    if len(strikes_all) > 1:
        step = int(np.median(np.diff(strikes_all)))
    else:
        step = 50
    lo, hi = atm - band * step, atm + band * step
    snap = snap[(snap["strike"] >= lo) & (snap["strike"] <= hi)]

    T = years_to_expiry(cursor_ist, expiry)
    out_strikes, ce_ivs, pe_ivs = [], [], []
    for _, row in snap.iterrows():
        K = int(row["strike"])
        ce_iv = implied_vol(row.get("CE"), spot_px, K, T, "CE") if pd.notna(row.get("CE")) else None
        pe_iv = implied_vol(row.get("PE"), spot_px, K, T, "PE") if pd.notna(row.get("PE")) else None
        out_strikes.append(K)
        ce_ivs.append(None if ce_iv is None else ce_iv * 100)
        pe_ivs.append(None if pe_iv is None else pe_iv * 100)

    return JSONResponse({
        "strikes": out_strikes,
        "ce_iv": ce_ivs, "pe_iv": pe_ivs,
        "atm": int(atm), "spot": spot_px,
        "cursor": str(cursor_ist),
    })


@app.get("/api/chain")
def chain(date: str, expiry: str, band: int = 8):
    """Latest-row snapshot of CE/PE around ATM."""
    opt = load_options(date)
    oi  = load_oi(date)
    spot = load_spot(date)
    if opt.empty or spot.empty:
        raise HTTPException(404, "no data")

    spot_px = float(spot.iloc[-1]["close"])
    snap = (opt[opt["expiry"] == expiry]
            .sort_values("timestamp")
            .groupby(["strike", "option_type"])
            .agg(ltp=("close", "last"), vol=("volume", "sum"))
            .reset_index())
    oi_snap = (oi[oi["expiry"] == expiry]
               .sort_values("timestamp")
               .groupby(["strike", "option_type"])
               .agg(oi=("oi", "last"))
               .reset_index()) if not oi.empty else pd.DataFrame()
    if not oi_snap.empty:
        snap = snap.merge(oi_snap, on=["strike", "option_type"], how="left")

    strikes = sorted(snap["strike"].unique())
    if not strikes:
        return JSONResponse({"rows": [], "spot": spot_px, "atm": None})
    atm = min(strikes, key=lambda k: abs(k - spot_px))
    step = int(np.median(np.diff(strikes))) if len(strikes) > 1 else 50
    lo, hi = atm - band * step, atm + band * step

    ce = snap[(snap["option_type"] == "CE") & (snap["strike"] >= lo) & (snap["strike"] <= hi)].set_index("strike")
    pe = snap[(snap["option_type"] == "PE") & (snap["strike"] >= lo) & (snap["strike"] <= hi)].set_index("strike")
    keys = sorted(set(ce.index).union(pe.index))

    rows = []
    for k in keys:
        rows.append({
            "strike": int(k),
            "ce_ltp": float(ce.loc[k, "ltp"])  if k in ce.index else None,
            "ce_vol": int(ce.loc[k, "vol"])    if k in ce.index else None,
            "ce_oi" : int(ce.loc[k, "oi"])     if k in ce.index and "oi" in ce.columns and pd.notna(ce.loc[k, "oi"]) else None,
            "pe_ltp": float(pe.loc[k, "ltp"])  if k in pe.index else None,
            "pe_vol": int(pe.loc[k, "vol"])    if k in pe.index else None,
            "pe_oi" : int(pe.loc[k, "oi"])     if k in pe.index and "oi" in pe.columns and pd.notna(pe.loc[k, "oi"]) else None,
            "atm"   : bool(k == atm),
        })
    return JSONResponse({"rows": rows, "spot": spot_px, "atm": int(atm)})


# ─── Inline HTML/JS ───────────────────────────────────────────────────────────
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Options Chain Replay · TV</title>
<script>
// Reverse-proxy prefix shim. Empty in local dev, '/straddle' on the VM.
window.APP_BASE = "__APP_BASE__";
(function() {
  const p = window.APP_BASE;
  if (!p) return;
  const _f = window.fetch.bind(window);
  window.fetch = (u, i) => (typeof u === 'string' && u.startsWith('/') && !u.startsWith('//') && !u.startsWith(p + '/'))
    ? _f(p + u, i) : _f(u, i);
})();
</script>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background:#0d0f1a; color:#c9d1d9; font:12px/1.4 'Segoe UI',monospace; height:100vh; display:flex; flex-direction:column; }
.bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:10px 16px; background:#13151f; border-bottom:1px solid #2a2d3a; }
h1 { color:#fff; font-size:15px; font-weight:700; margin-right:8px; }
label { color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }
select, button { background:#1a1d27; border:1px solid #3a3d4e; color:#e0e0e0; border-radius:5px; padding:5px 9px; cursor:pointer; font:inherit; }
button:hover, select:hover { background:#252837; }
button.on { background:#1e3a5f; border-color:#2563eb; color:#93c5fd; }
.tf-grp button { padding:4px 10px; font-size:11px; font-weight:600; }
.tf-grp button.on { background:#2563eb; border-color:#2563eb; color:#fff; }
.spot { background:#252837; border-radius:6px; padding:4px 12px; }
.spot span { color:#a0a8c0; font-size:11px; } .spot strong { color:#facc15; font-size:14px; margin-left:6px; }
#charts { flex:1; display:flex; flex-direction:column; padding:6px 8px; gap:0px; min-height:0; }
.resizer {
  height:6px; flex:0 0 6px; cursor:row-resize; background:transparent;
  position:relative; z-index:6;
}
.resizer::before {
  content:""; position:absolute; left:50%; top:50%;
  transform:translate(-50%, -50%);
  width:36px; height:3px; border-radius:2px; background:#3a3d4e;
  transition:background .15s;
}
.resizer:hover::before, .resizer.dragging::before { background:#60a5fa; }
body.resizing { cursor:row-resize; user-select:none; }
body.resizing iframe, body.resizing canvas { pointer-events:none; }
.pane { background:#0d0f1a; border:1px solid #1e2030; border-radius:6px; flex:1; min-height:0; position:relative; }
.pane-label { position:absolute; top:6px; left:10px; color:#6b7280; font-size:10px; text-transform:uppercase; letter-spacing:.5px; z-index:5; pointer-events:none; }
.pane-price { flex:3.0; display:flex; flex-direction:row; gap:4px; background:transparent; border:none; }
.pane-price > .sub { flex:1; min-width:0; position:relative;
                     background:#0d0f1a; border:1px solid #1e2030; border-radius:6px; }
.pane-price.solo-ce > .sub-pe { display:none; }
.pane-price.solo-pe > .sub-ce { display:none; }
#status { color:#6b7280; font-size:11px; margin-left:auto; }
.legend-dot { display:inline-block; width:8px; height:8px; border-radius:50%; vertical-align:middle; margin-right:4px; }

/* ── Live stats bar ───────────────────────────────────────────────────── */
.stats {
  display:flex; gap:10px; align-items:stretch; flex-wrap:wrap;
  padding:6px 16px; background:#0f1119; border-bottom:1px solid #2a2d3a;
  font-size:11px;
}
.chip {
  display:flex; flex-direction:column; gap:2px; padding:5px 10px;
  background:#1a1d27; border:1px solid #2a2d3a; border-radius:6px; min-width:96px;
}
.chip .lbl { color:#6b7280; font-size:9px; text-transform:uppercase; letter-spacing:.5px; }
.chip .val { color:#e6e9f2; font-weight:600; font-size:13px; font-variant-numeric:tabular-nums; }
.chip .sub { color:#7b8294; font-size:10px; font-variant-numeric:tabular-nums; }
.chip.pos .val { color:#4ade80; } .chip.neg .val { color:#f87171; }
.chip.iv  .val { color:#60a5fa; } .chip.rv  .val { color:#fb923c; }
.chip.live { border-color:#2563eb; box-shadow:0 0 0 1px #2563eb40 inset; }
.chip.live .lbl { color:#93c5fd; }

/* Trade-info banner (visible when a strategy event is active) */
.trade {
  display:flex; gap:10px; align-items:stretch; flex-wrap:wrap;
  padding:6px 16px; background:#13182a; border-bottom:1px solid #2a2d3a;
  font-size:11px;
}
.trade.hidden { display:none; }
.t-chip {
  display:flex; flex-direction:column; gap:2px; padding:6px 12px;
  background:#1a1d27; border:1px solid #3a3d4e; border-radius:6px; min-width:120px;
}
.t-chip .lbl { color:#9ca3af; font-size:9px; text-transform:uppercase; letter-spacing:.5px; }
.t-chip .val { color:#e6e9f2; font-weight:700; font-size:13px; font-variant-numeric:tabular-nums; }
.t-chip .sub { color:#7b8294; font-size:10px; font-variant-numeric:tabular-nums; }
.t-chip.entry { border-color:#facc15; box-shadow:0 0 0 1px #facc1540 inset; }
.t-chip.entry .lbl { color:#facc15; }
.t-chip.exit  { border-color:#60a5fa; box-shadow:0 0 0 1px #60a5fa40 inset; }
.t-chip.exit  .lbl { color:#60a5fa; }
.t-chip.pnl-win  { border-color:#4ade80; box-shadow:0 0 0 1px #4ade8040 inset; }
.t-chip.pnl-win  .lbl, .t-chip.pnl-win  .val { color:#4ade80; }
.t-chip.pnl-loss { border-color:#f87171; box-shadow:0 0 0 1px #f8717140 inset; }
.t-chip.pnl-loss .lbl, .t-chip.pnl-loss .val { color:#f87171; }
.t-chip.ce { border-color:#26a69a40; }
.t-chip.pe { border-color:#ef535040; }
.t-chip .leg-pnl     { font-weight:700; font-variant-numeric:tabular-nums; }
.t-chip .leg-pnl.pos { color:#4ade80; }
.t-chip .leg-pnl.neg { color:#f87171; }

</style>
</head>
<body>

<div class="bar">
  <h1>⚡ Option Chain Replay · TV</h1>
  <div class="spot"><span>Spot</span><strong id="spotval">—</strong></div>

  <label>Date</label>
  <select id="date"></select>

  <label>Lookback</label>
  <select id="lookback">
    <option value="1">1 day</option>
    <option value="2">2 days</option>
    <option value="3" selected>3 days</option>
    <option value="5">5 days</option>
    <option value="10">10 days</option>
  </select>

  <label>Strategy event</label>
  <select id="event" style="min-width:240px"></select>
  <button id="tg-signals" class="on" title="Show SELL/BUY markers + LGBM probability">Signals</button>

  <label>Expiry</label>
  <select id="expiry"></select>

  <label>Strike</label>
  <select id="strike"></select>

  <label>Side</label>
  <select id="otype">
    <option value="BOTH" selected>BOTH (CE | PE)</option>
    <option value="CE">CE</option>
    <option value="PE">PE</option>
  </select>

  <label>TF (price)</label>
  <span class="tf-grp" id="tf-price">
    <button data-tf="1m">1m</button>
    <button data-tf="3m">3m</button>
    <button data-tf="5m" class="on">5m</button>
    <button data-tf="15m">15m</button>
    <button data-tf="30m">30m</button>
    <button data-tf="1h">1h</button>
  </span>

  <label>Show</label>
  <button id="tg-spot" class="on">Spot</button>

  <span id="status"></span>
</div>

<div class="trade hidden" id="trade-info">
  <!-- Filled by renderTradeBanner() when a strategy event is active -->
</div>

<div class="stats" id="stats">
  <!-- Filled by JS -->
</div>

<div id="charts">
  <div class="pane pane-price" id="pane-price">
    <div class="sub sub-ce" id="pane-price-ce">
      <span class="pane-label">
        <span class="legend-dot" style="background:#26a69a"></span>CE candles
        <span class="legend-dot" style="background:#facc15;margin-left:8px"></span>Spot
        <span class="legend-dot" style="background:#3b82f6;margin-left:8px"></span>Volume
      </span>
    </div>
    <div class="sub sub-pe" id="pane-price-pe">
      <span class="pane-label">
        <span class="legend-dot" style="background:#ef5350"></span>PE candles
        <span class="legend-dot" style="background:#facc15;margin-left:8px"></span>Spot
        <span class="legend-dot" style="background:#3b82f6;margin-left:8px"></span>Volume
      </span>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const setStatus = s => $('status').textContent = s;

// ── State ───────────────────────────────────────────────────────────────────
const state = {
  date: null, expiry: null, strike: null, otype: 'BOTH',
  tfPrice: '5m',
  showSpot: true,
  lookbackDays: 3,
  // chart handles per side (CE / PE candle panes only)
  cPriceCe: null, cPricePe: null,
  sCandleCe: null, sVolCe: null, sSpotCe: null,
  sCandlePe: null, sVolPe: null, sSpotPe: null,
  syncing: false,
  // Per-leg datasets kept for crosshair-driven stats chips
  lastCe: { candles: [], iv: [], oi: [], greeks: {delta:[],gamma:[],theta:[],vega:[]}, stats: null },
  lastPe: { candles: [], iv: [], oi: [], greeks: {delta:[],gamma:[],theta:[],vega:[]}, stats: null },
  lastShared: { rv: [], spot: [], days: [] },
  // For backward compat with existing helpers expecting state.last
  get last() {
    const leg = (state.otype === 'PE') ? state.lastPe : state.lastCe;
    return {...leg, rv: state.lastShared.rv, spot: state.lastShared.spot};
  },
  // Strategy events
  events: [],
  activeEvent: null,
  activeTrade: null,           // per-leg detail from /api/event_detail
  tradeLinesCe: [],            // priceLine handles on CE candle series
  tradeLinesPe: [],            // priceLine handles on PE candle series
  showSignals: true,
  programmaticChange: false,
  // managed list for time-axis & crosshair sync
  timeAxisCharts: [],
};

// ── Chart factories ────────────────────────────────────────────────────────
// Parquet timestamps are naive IST. Our Unix conversion treats them as UTC,
// so to display the actual NSE wall-clock we format using getUTC* — that way
// the chart shows 09:15–15:30 regardless of the browser's local timezone.
const MONS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const pad2 = n => String(n).padStart(2, '0');
const fmtIST = t => {
  const d = new Date(t * 1000);
  return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`;
};
const tickFmtIST = (time, type) => {
  const d = new Date(time * 1000);
  // LightweightCharts.TickMarkType: 0=year, 1=month, 2=dayOfMonth, 3=time
  if (type === 0) return String(d.getUTCFullYear());
  if (type === 1) return MONS[d.getUTCMonth()];
  if (type === 2) return `${d.getUTCDate()} ${MONS[d.getUTCMonth()]}`;
  return fmtIST(time);
};

function mkChart(el) {
  return LightweightCharts.createChart(el, {
    layout: { background: { type: 'solid', color: '#0d0f1a' }, textColor: '#9ca3af' },
    grid: { vertLines: { color: '#1e2030' }, horzLines: { color: '#1e2030' } },
    rightPriceScale: { borderColor: '#2a2d3a', scaleMargins: { top: 0.06, bottom: 0.06 } },
    timeScale: {
      borderColor: '#2a2d3a', timeVisible: true, secondsVisible: false, rightOffset: 4,
      tickMarkFormatter: tickFmtIST,
    },
    localization: { timeFormatter: fmtIST },
    crosshair: { mode: 0 },
    autoSize: true,
  });
}

function buildPriceChart(el, isPe) {
  const c = mkChart(el);
  const sCandle = c.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });
  const sSpot = c.addLineSeries({
    color: '#facc15', lineWidth: 1, priceScaleId: 'spot',
    lastValueVisible: false, priceLineVisible: false,
  });
  c.priceScale('spot').applyOptions({ scaleMargins: { top: 0.06, bottom: 0.06 } });
  const sVol = c.addHistogramSeries({
    color: '#3b82f6', priceScaleId: 'vol',
    priceFormat: { type: 'volume' },
    lastValueVisible: false, priceLineVisible: false,
  });
  c.priceScale('vol').applyOptions({ scaleMargins: { top: 0.80, bottom: 0.0 } });
  return { chart: c, candle: sCandle, spot: sSpot, vol: sVol };
}

function registerSync(chart) {
  state.timeAxisCharts.push(chart);
  chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (state.syncing || !range) return;
    state.syncing = true;
    state.timeAxisCharts.forEach(c => { if (c !== chart) c.timeScale().setVisibleLogicalRange(range); });
    state.syncing = false;
  });
  chart.subscribeCrosshairMove(p => {
    if (!p || !p.time) {
      state.timeAxisCharts.forEach(c => { if (c !== chart) c.clearCrosshairPosition(); });
      renderStats(null);
      return;
    }
    state.timeAxisCharts.forEach(c => {
      if (c !== chart) c.setCrosshairPosition(NaN, p.time, c.priceScale('right'));
    });
    renderStats(p.time);
  });
}

function buildCharts() {
  const ce = buildPriceChart($('pane-price-ce'), false);
  const pe = buildPriceChart($('pane-price-pe'), true);
  state.cPriceCe = ce.chart; state.sCandleCe = ce.candle; state.sVolCe = ce.vol; state.sSpotCe = ce.spot;
  state.cPricePe = pe.chart; state.sCandlePe = pe.candle; state.sVolPe = pe.vol; state.sSpotPe = pe.spot;

  // Time-axis + crosshair sync between CE and PE candle panes
  [state.cPriceCe, state.cPricePe].forEach(registerSync);
}

// ── Stats / live-snapshot helpers ───────────────────────────────────────────
function fmt(v, dp=2)   { return (v == null || !Number.isFinite(v)) ? '—' : v.toFixed(dp); }
function fmtPct(v)      { return (v == null || !Number.isFinite(v)) ? '—' : v.toFixed(2) + '%'; }
function fmtOI(v) {
  if (v == null || !Number.isFinite(v)) return '—';
  const n = Math.abs(v), s = v < 0 ? '-' : '';
  if (n >= 1e7) return s + (n/1e7).toFixed(2) + 'Cr';
  if (n >= 1e5) return s + (n/1e5).toFixed(2) + 'L';
  if (n >= 1e3) return s + (n/1e3).toFixed(1) + 'K';
  return s + Math.round(n);
}
function fmtSigned(v, dp=2) {
  if (v == null || !Number.isFinite(v)) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(dp);
}

// ── Per-day stats (rebuilt on each load) ────────────────────────────────────
const dayKey = t => {
  const d = new Date(t * 1000);
  return d.getUTCFullYear() + '-'
       + String(d.getUTCMonth()+1).padStart(2,'0') + '-'
       + String(d.getUTCDate()).padStart(2,'0');
};

function buildDayStats() {
  const D = state.last;
  const byDay = {};
  function pushLine(arr, key) {
    if (!arr) return;
    arr.forEach(p => {
      if (p.value == null) return;
      const k = dayKey(p.time);
      (byDay[k] ||= {});
      const slot = (byDay[k][key] ||= {open: p.value, high: p.value, low: p.value, current: p.value});
      slot.current = p.value;
      if (p.value > slot.high) slot.high = p.value;
      if (p.value < slot.low)  slot.low  = p.value;
    });
  }
  function pushCandles(arr) {
    if (!arr) return;
    arr.forEach(c => {
      const k = dayKey(c.time);
      (byDay[k] ||= {});
      const slot = (byDay[k].premium ||= {open: c.open, high: c.high, low: c.low, current: c.close});
      slot.current = c.close;
      if (c.high > slot.high) slot.high = c.high;
      if (c.low  < slot.low)  slot.low  = c.low;
    });
  }
  pushLine(D.iv, 'iv');
  pushLine(D.rv, 'rv');
  pushLine(D.oi, 'oi');
  pushLine(D.spot, 'spot');
  pushCandles(D.candles);
  state.dayStats = byDay;
}

// Bisect by time field (sorted ascending) — return the index of the bar
// at-or-before `t`. Used for crosshair-driven lookups.
function bisectAtOrBefore(arr, t) {
  if (!arr || !arr.length) return -1;
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].time <= t) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return ans;
}
const valAt = (arr, t) => {
  const i = bisectAtOrBefore(arr, t);
  return i < 0 ? null : arr[i].value;
};
const candleAt = (arr, t) => {
  const i = bisectAtOrBefore(arr, t);
  return i < 0 ? null : arr[i];
};

function renderStats(t) {
  // t = cursor time (unix, IST-shifted) — falls back to last bar.
  const D = state.last;
  const tEff = t != null
    ? t
    : (D.candles.length ? D.candles[D.candles.length - 1].time : null);
  if (tEff == null) { document.getElementById('stats').innerHTML = ''; return; }

  const c   = candleAt(D.candles, tEff);
  const iv  = valAt(D.iv,  tEff);
  const rv  = valAt(D.rv,  tEff);
  const oi  = valAt(D.oi,  tEff);
  const spt = valAt(D.spot, tEff);
  const del = valAt(D.greeks.delta, tEff);
  const gam = valAt(D.greeks.gamma, tEff);
  const tht = valAt(D.greeks.theta, tEff);
  const veg = valAt(D.greeks.vega,  tEff);

  // Per-day stats for the day the cursor sits on (so ΔOI / premium-Δ / today's
  // ranges all follow the crosshair instead of being pinned to the latest day).
  const cursorDay = dayKey(tEff);
  const today = (state.dayStats || {})[cursorDay] || {};
  const ivToday = today.iv      || {};
  const rvToday = today.rv      || {};
  const oiToday = today.oi      || {};
  const sptToday = today.spot   || {};
  const prToday = today.premium || {};
  const dOI = (oi != null && oiToday.open != null) ? oi - oiToday.open : null;
  const dPremium = (c && prToday.open != null) ? c.close - prToday.open : null;
  const ivSpread = (iv != null && rv != null) ? iv - rv : null;

  const liveCls = t != null ? ' live' : '';
  const dt = new Date(tEff * 1000);
  const hh = String(dt.getUTCHours()).padStart(2,'0');
  const mm = String(dt.getUTCMinutes()).padStart(2,'0');
  const dd = String(dt.getUTCDate()).padStart(2,'0');
  const mo = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][dt.getUTCMonth()];

  const chips = [
    `<div class="chip${liveCls}"><span class="lbl">${t!=null?'cursor':'now'}</span>
       <span class="val">${hh}:${mm}</span><span class="sub">${dd} ${mo}</span></div>`,
    c ? `<div class="chip"><span class="lbl">O / H / L / C</span>
       <span class="val">${fmt(c.close)}</span>
       <span class="sub">${fmt(c.open)} · ${fmt(c.high)} · ${fmt(c.low)}</span></div>` : '',
    `<div class="chip iv"><span class="lbl">IV</span>
       <span class="val">${fmtPct(iv)}</span>
       <span class="sub">day ${fmtPct(ivToday.low)} → ${fmtPct(ivToday.high)}</span></div>`,
    `<div class="chip rv"><span class="lbl">RV (spot)</span>
       <span class="val">${fmtPct(rv)}</span>
       <span class="sub">day ${fmtPct(rvToday.low)} → ${fmtPct(rvToday.high)}</span></div>`,
    `<div class="chip ${ivSpread != null && ivSpread > 0 ? 'pos' : 'neg'}">
       <span class="lbl">IV − RV</span>
       <span class="val">${fmtSigned(ivSpread)}</span>
       <span class="sub">VRP at cursor</span></div>`,
    `<div class="chip"><span class="lbl">Spot</span>
       <span class="val">${fmt(spt, 1)}</span>
       <span class="sub">day ${fmt(sptToday.low,1)} → ${fmt(sptToday.high,1)}</span></div>`,
    `<div class="chip ${(dOI||0) >= 0 ? 'pos' : 'neg'}">
       <span class="lbl">ΔOI from open</span>
       <span class="val">${fmtSigned(dOI, 0)}</span>
       <span class="sub">${fmtOI(oi)} @ cursor · ${fmtOI(oiToday.open)} @ open</span></div>`,
    `<div class="chip"><span class="lbl">Δ / Γ</span>
       <span class="val">${fmt(del, 2)} / ${fmt(gam, 4)}</span>
       <span class="sub">delta · gamma</span></div>`,
    `<div class="chip"><span class="lbl">Θ / ν</span>
       <span class="val">${fmt(tht, 2)} / ${fmt(veg, 2)}</span>
       <span class="sub">theta·day · vega·%</span></div>`,
    c && prToday.open != null
      ? `<div class="chip ${(dPremium||0) >= 0 ? 'pos' : 'neg'}">
           <span class="lbl">premium Δ today</span>
           <span class="val">${fmtSigned(dPremium, 2)}</span>
           <span class="sub">${fmt(prToday.open)} → ${fmt(c.close)}</span></div>` : '',
  ].filter(Boolean).join('');
  document.getElementById('stats').innerHTML = chips;
}

// ── Strategy events ────────────────────────────────────────────────────────
// Backend stores naive IST timestamps as if UTC. To place a marker at "15:20 IST
// on YYYY-MM-DD", construct the matching Unix value via Date.UTC.
function istUnix(y, mo, da, h, m) {
  return Math.floor(Date.UTC(y, mo - 1, da, h, m, 0) / 1000);
}
const ENTRY_HHMM = [15, 20];   // matches notebook ENTRY_TIME
const EXIT_HHMM  = [9, 20];    // matches notebook EXIT_TIME

async function loadEvents() {
  try {
    const r = await fetch('/api/events');
    if (!r.ok) return;
    const j = await r.json();
    state.events = j.events || [];
    const sel = document.getElementById('event');
    sel.innerHTML = '';
    const opt0 = document.createElement('option');
    opt0.value = ''; opt0.textContent = `— pick an expiry (${state.events.length} events) —`;
    sel.appendChild(opt0);
    state.events.forEach((ev, i) => {
      const o = document.createElement('option');
      o.value = String(i);
      const wl = ev.profitable ? 'WIN ' : 'LOSS';
      const sign = ev.pnl_points >= 0 ? '+' : '';
      const pml = ev.p_lgbm != null ? ` · p=${ev.p_lgbm.toFixed(2)}` : '';
      o.textContent = `${ev.expiry}  ${wl}  ${sign}${ev.pnl_points.toFixed(1)}pts${pml}`;
      sel.appendChild(o);
    });
  } catch (_) {}
}

async function pickEvent(idx) {
  const ev = state.events[idx];
  if (!ev) { state.activeEvent = null; applyAllMarkers(); return; }
  state.activeEvent = ev;

  // Anchor the window on the EXPIRY day so the 09:20 exit marker is visible.
  // Lookback=3 → [entry-day-1, entry-day, expiry-day], so entry @ prev-day 15:20
  // and exit @ expiry-day 09:20 (plus rest of expiry-day candles) are all on-screen.
  const expiryDate = ev.expiry.replace(/-/g, '');   // 'YYYY-MM-DD' → 'YYYYMMDD'

  state.programmaticChange = true;
  document.getElementById('lookback').value = '3';
  state.lookbackDays = 3;

  state.date = expiryDate;
  const dateSel = document.getElementById('date');
  if (![...dateSel.options].some(o => o.value === expiryDate)) {
    const o = document.createElement('option'); o.value = expiryDate;
    o.textContent = `${expiryDate.slice(0,4)}-${expiryDate.slice(4,6)}-${expiryDate.slice(6)}`;
    dateSel.appendChild(o);
  }
  dateSel.value = expiryDate;

  await loadExpiries();
  state.expiry = ev.expiry;
  const exSel = document.getElementById('expiry');
  if (![...exSel.options].some(o => o.value === ev.expiry)) {
    const o = document.createElement('option'); o.value = ev.expiry; o.textContent = ev.expiry;
    exSel.appendChild(o);
  }
  exSel.value = ev.expiry;

  await loadStrikes();
  state.strike = ev.atm_strike;
  const kSel = document.getElementById('strike');
  if (![...kSel.options].some(o => +o.value === ev.atm_strike)) {
    const o = document.createElement('option'); o.value = String(ev.atm_strike);
    o.textContent = `${ev.atm_strike}  ★ ATM (event)`; kSel.appendChild(o);
  }
  kSel.value = String(ev.atm_strike);

  // Show CE + PE split for the straddle
  state.otype = 'BOTH';
  document.getElementById('otype').value = 'BOTH';

  state.programmaticChange = false;
  await loadLeg();
  await loadTradeDetail();
}

// Resolve full per-leg trade detail for the active event, draw banner + lines.
async function loadTradeDetail() {
  clearTradePriceLines();
  if (!state.activeEvent) {
    state.activeTrade = null;
    renderTradeBanner();
    return;
  }
  const ev = state.activeEvent;
  try {
    const qs = new URLSearchParams({
      expiry: ev.expiry,
      entry_date: ev.entry_date,
      atm_strike: ev.atm_strike,
    });
    const r = await fetch('/api/event_detail?' + qs);
    if (!r.ok) { state.activeTrade = null; renderTradeBanner(); return; }
    state.activeTrade = await r.json();
  } catch (_) { state.activeTrade = null; }
  renderTradeBanner();
  applyTradePriceLines();
  applyAllMarkers();   // refresh markers now that per-leg prices are known
}

function renderTradeBanner() {
  const el = document.getElementById('trade-info');
  if (!state.activeEvent) { el.classList.add('hidden'); el.innerHTML = ''; return; }
  const ev = state.activeEvent;
  const td = state.activeTrade || {};

  const entryStamp = ev.entry_date.length === 8
    ? `${ev.entry_date.slice(0,4)}-${ev.entry_date.slice(4,6)}-${ev.entry_date.slice(6)}`
    : ev.entry_date;
  const exitStamp  = ev.expiry;
  const pmlText  = ev.p_lgbm != null ? `p_LGBM ${ev.p_lgbm.toFixed(2)}` : 'no model prob';

  // Net P&L = (CE_entry + PE_entry) − (CE_exit + PE_exit), i.e. ce_pnl + pe_pnl.
  // Prefer the live leg values that are actually shown above; fall back to the
  // parquet's pnl_points only if either leg is missing (different timestamp
  // assumptions in the notebook can disagree with the live bar snapshot).
  const haveLegs   = td.ce_pnl != null && td.pe_pnl != null;
  const netPnl     = haveLegs ? (td.ce_pnl + td.pe_pnl) : ev.pnl_points;
  const entryPrem  = (td.ce_entry_px != null && td.pe_entry_px != null)
                       ? (td.ce_entry_px + td.pe_entry_px)
                       : ev.premium_collected;
  const exitCost   = (td.ce_exit_px != null && td.pe_exit_px != null)
                       ? (td.ce_exit_px + td.pe_exit_px)
                       : ev.premium_at_exit;
  const netPct     = (entryPrem && entryPrem !== 0) ? (netPnl / entryPrem) : ev.pnl_pct;
  const isWin      = netPnl >= 0;
  const winCls     = isWin ? 'pnl-win' : 'pnl-loss';
  const sign       = isWin ? '+' : '';

  const chips = [
    `<div class="t-chip entry"><span class="lbl">ENTRY (sell straddle)</span>
       <span class="val">${entryStamp} · 15:20</span>
       <span class="sub">premium ₹${entryPrem.toFixed(2)} · spot ${ev.spot_at_entry.toFixed(1)}</span></div>`,
    `<div class="t-chip exit"><span class="lbl">EXIT (buy back)</span>
       <span class="val">${exitStamp} · 09:20</span>
       <span class="sub">cost ₹${exitCost.toFixed(2)} · spot ${ev.spot_at_exit.toFixed(1)}</span></div>`,
    `<div class="t-chip ce"><span class="lbl">CE ${ev.atm_strike}</span>
       <span class="val">${td.ce_entry_px != null ? td.ce_entry_px.toFixed(2) : '—'} → ${td.ce_exit_px != null ? td.ce_exit_px.toFixed(2) : '—'}</span>
       <span class="sub">leg pnl ${td.ce_pnl != null ? `<span class="leg-pnl ${td.ce_pnl >= 0 ? 'pos' : 'neg'}">${td.ce_pnl >= 0 ? '+' : ''}${td.ce_pnl.toFixed(2)}</span>` : '—'} pts</span></div>`,
    `<div class="t-chip pe"><span class="lbl">PE ${ev.atm_strike}</span>
       <span class="val">${td.pe_entry_px != null ? td.pe_entry_px.toFixed(2) : '—'} → ${td.pe_exit_px != null ? td.pe_exit_px.toFixed(2) : '—'}</span>
       <span class="sub">leg pnl ${td.pe_pnl != null ? `<span class="leg-pnl ${td.pe_pnl >= 0 ? 'pos' : 'neg'}">${td.pe_pnl >= 0 ? '+' : ''}${td.pe_pnl.toFixed(2)}</span>` : '—'} pts</span></div>`,
    `<div class="t-chip ${winCls}"><span class="lbl">TOTAL P&amp;L</span>
       <span class="val">${sign}${netPnl.toFixed(2)} pts</span>
       <span class="sub">${isWin ? 'WIN' : 'LOSS'} · ${(netPct*100).toFixed(2)}% · ${pmlText}</span></div>`,
  ].join('');
  el.innerHTML = chips;
  el.classList.remove('hidden');
}

function clearTradePriceLines() {
  if (state.sCandleCe) state.tradeLinesCe.forEach(l => { try { state.sCandleCe.removePriceLine(l); } catch(_){} });
  if (state.sCandlePe) state.tradeLinesPe.forEach(l => { try { state.sCandlePe.removePriceLine(l); } catch(_){} });
  state.tradeLinesCe = [];
  state.tradeLinesPe = [];
}

function applyTradePriceLines() {
  clearTradePriceLines();
  const td = state.activeTrade;
  if (!td || !state.sCandleCe || !state.sCandlePe) return;
  const mk = (px, color, title) => ({
    price: px, color, lineWidth: 1, lineStyle: 2,  // dashed
    axisLabelVisible: true, title,
  });
  if (td.ce_entry_px != null)
    state.tradeLinesCe.push(state.sCandleCe.createPriceLine(mk(td.ce_entry_px, '#facc15', `entry ${td.ce_entry_px.toFixed(2)}`)));
  if (td.ce_exit_px != null)
    state.tradeLinesCe.push(state.sCandleCe.createPriceLine(mk(td.ce_exit_px, '#60a5fa', `exit ${td.ce_exit_px.toFixed(2)}`)));
  if (td.pe_entry_px != null)
    state.tradeLinesPe.push(state.sCandlePe.createPriceLine(mk(td.pe_entry_px, '#facc15', `entry ${td.pe_entry_px.toFixed(2)}`)));
  if (td.pe_exit_px != null)
    state.tradeLinesPe.push(state.sCandlePe.createPriceLine(mk(td.pe_exit_px, '#60a5fa', `exit ${td.pe_exit_px.toFixed(2)}`)));
}

// Per-side trade markers. Side = 'CE' | 'PE' (price panes get leg-specific
// prices & per-leg PnL); 'TOTAL' (IV / OI / generic — straddle totals).
function eventMarkersFor(side) {
  if (!state.showSignals || !state.activeEvent) return [];
  const ev = state.activeEvent;
  const td = state.activeTrade || {};
  const [ey, em, ed] = ev.entry_date.length === 8
      ? [+ev.entry_date.slice(0,4), +ev.entry_date.slice(4,6), +ev.entry_date.slice(6)]
      : ev.entry_date.split('-').map(Number);
  const [xy, xm, xd] = ev.expiry.split('-').map(Number);
  const entryT = istUnix(ey, em, ed, ENTRY_HHMM[0], ENTRY_HHMM[1]);
  const exitT  = istUnix(xy, xm, xd, EXIT_HHMM[0],  EXIT_HHMM[1]);

  const fmtSign = v => (v == null) ? '—' : ((v >= 0 ? '+' : '') + v.toFixed(2));
  let entryText, exitText, exitColor;

  if (side === 'CE' || side === 'PE') {
    const px_e   = side === 'CE' ? td.ce_entry_px : td.pe_entry_px;
    const px_x   = side === 'CE' ? td.ce_exit_px  : td.pe_exit_px;
    const legPnl = side === 'CE' ? td.ce_pnl      : td.pe_pnl;
    const legWin = legPnl != null && legPnl >= 0;
    entryText = `SELL ${side} ${px_e != null ? '@' + px_e.toFixed(2) : ''}`;
    exitText  = `BUY ${side} ${px_x != null ? '@' + px_x.toFixed(2) : ''}  ${fmtSign(legPnl)} pts`;
    exitColor = legWin ? '#4ade80' : '#f87171';
  } else {
    const haveLegs  = td.ce_pnl != null && td.pe_pnl != null;
    const netPnl    = haveLegs ? (td.ce_pnl + td.pe_pnl) : ev.pnl_points;
    const entryPrem = (td.ce_entry_px != null && td.pe_entry_px != null)
                        ? (td.ce_entry_px + td.pe_entry_px) : ev.premium_collected;
    const exitCost  = (td.ce_exit_px != null && td.pe_exit_px != null)
                        ? (td.ce_exit_px + td.pe_exit_px)   : ev.premium_at_exit;
    const wl   = netPnl >= 0 ? 'WIN' : 'LOSS';
    const sign = netPnl >= 0 ? '+' : '';
    const pml  = ev.p_lgbm != null ? `  p=${ev.p_lgbm.toFixed(2)}` : '';
    entryText = `SELL straddle  premium ${entryPrem.toFixed(1)}${pml}`;
    exitText  = `BUY back  cost ${exitCost.toFixed(1)}  ${wl} ${sign}${netPnl.toFixed(1)}pts`;
    exitColor = netPnl >= 0 ? '#4ade80' : '#f87171';
  }

  return [
    {time: entryT, position: 'aboveBar', color: '#facc15',  shape: 'arrowDown', text: entryText},
    {time: exitT,  position: 'belowBar', color: exitColor,  shape: 'arrowUp',   text: exitText},
  ];
}

// ── Day-boundary + trade-signal markers ────────────────────────────────────
function buildDayMarkersFor(candles) {
  const out = [];
  if (!candles || !candles.length) return out;
  let lastDate = '';
  const MONS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  for (const c of candles) {
    const dt = new Date(c.time * 1000);
    const key = dt.getUTCFullYear()+'-'+(dt.getUTCMonth()+1)+'-'+dt.getUTCDate();
    if (key !== lastDate) {
      out.push({time: c.time, position: 'aboveBar', color: '#475569', shape: 'square',
                text: `${String(dt.getUTCDate()).padStart(2,'0')} ${MONS[dt.getUTCMonth()]}`});
      lastDate = key;
    }
  }
  return out;
}

function applyAllMarkers() {
  if (!state.sCandleCe || !state.sCandlePe) return;
  const evCE  = eventMarkersFor('CE');
  const evPE  = eventMarkersFor('PE');
  const evTOT = eventMarkersFor('TOTAL');
  const ceMarkers = [...buildDayMarkersFor(state.lastCe.candles), ...evCE]
                    .sort((a, b) => a.time - b.time);
  const peMarkers = [...buildDayMarkersFor(state.lastPe.candles), ...evPE]
                    .sort((a, b) => a.time - b.time);
  try { state.sCandleCe.setMarkers(ceMarkers); } catch (_) {}
  try { state.sCandlePe.setMarkers(peMarkers); } catch (_) {}
}

// ── Data loading ───────────────────────────────────────────────────────────
async function loadDays() {
  const r = await fetch('/api/days');
  const days = await r.json();
  const sel = $('date'); sel.innerHTML = '';
  days.forEach(d => {
    const o = document.createElement('option');
    o.value = d; o.textContent = `${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6)}`;
    sel.appendChild(o);
  });
  const pref = days.includes('20240321') ? '20240321' : days[days.length - 1];
  sel.value = pref; state.date = pref;
}

async function loadExpiries() {
  const r = await fetch(`/api/expiries?date=${state.date}`);
  if (!r.ok) { setStatus('No options data for this day.'); return; }
  const exps = await r.json();
  const sel = $('expiry'); sel.innerHTML = '';
  exps.forEach(e => {
    const o = document.createElement('option'); o.value = e; o.textContent = e; sel.appendChild(o);
  });
  sel.value = exps[0]; state.expiry = exps[0];
}

async function loadStrikes() {
  const r = await fetch(`/api/strikes?date=${state.date}&expiry=${encodeURIComponent(state.expiry)}`);
  if (!r.ok) return;
  const j = await r.json();
  const sel = $('strike'); sel.innerHTML = '';
  j.strikes.forEach(k => {
    const o = document.createElement('option'); o.value = k; o.textContent = k;
    if (k === j.atm) o.textContent += '  ★ ATM';
    sel.appendChild(o);
  });
  state.strike = j.atm ?? j.strikes[Math.floor(j.strikes.length / 2)];
  sel.value = state.strike;
}

function fetchLeg(qsParams) {
  return fetch('/api/leg?' + new URLSearchParams(qsParams))
           .then(r => r.ok ? r.json() : null).catch(() => null);
}

async function loadLeg() {
  if (!state.date || !state.expiry || !state.strike) return;
  setStatus('Loading…');

  // Toggle CE/PE pane visibility based on otype
  const wrap = document.getElementById('pane-price');
  wrap.classList.remove('solo-ce', 'solo-pe');
  if (state.otype === 'CE') wrap.classList.add('solo-pe');     // hide PE
  if (state.otype === 'PE') wrap.classList.add('solo-ce');     // hide CE

  const needCe = state.otype !== 'PE';
  const needPe = state.otype !== 'CE';

  // IV / OI / RV are no longer charted, but the API still computes them so
  // the cursor-driven stats chips (IV / RV / IV-RV / ΔOI / greeks) keep working.
  const baseQs = {
    date: state.date, expiry: state.expiry, strike: state.strike,
    tf: state.tfPrice,
    with_iv: true, with_oi: true,
    with_spot: state.showSpot, with_rv: true,
    rv_window: 30, rv_estimator: 'close',
    lookback_days: state.lookbackDays,
  };
  const [ce, pe] = await Promise.all([
    needCe ? fetchLeg({...baseQs, type: 'CE'}) : null,
    needPe ? fetchLeg({...baseQs, type: 'PE'}) : null,
  ]);

  if (!ce && !pe) { setStatus('Error: leg fetch failed'); return; }

  // ── Candle panes ──
  if (ce) {
    state.sCandleCe.setData(ce.candles);
    state.sVolCe   .setData(ce.volume);
    state.sSpotCe  .setData(state.showSpot ? ce.spot : []);
  } else {
    state.sCandleCe.setData([]); state.sVolCe.setData([]); state.sSpotCe.setData([]);
  }
  if (pe) {
    state.sCandlePe.setData(pe.candles);
    state.sVolPe   .setData(pe.volume);
    state.sSpotPe  .setData(state.showSpot ? pe.spot : []);
  } else {
    state.sCandlePe.setData([]); state.sVolPe.setData([]); state.sSpotPe.setData([]);
  }

  const rvSrc = ce || pe;

  // ── Cache per-leg ──
  state.lastCe = ce ? {candles: ce.candles, iv: ce.iv||[], oi: ce.oi||[],
                       greeks: ce.greeks||{delta:[],gamma:[],theta:[],vega:[]},
                       stats: ce.stats||null}
                    : {candles: [], iv: [], oi: [],
                       greeks:{delta:[],gamma:[],theta:[],vega:[]}, stats: null};
  state.lastPe = pe ? {candles: pe.candles, iv: pe.iv||[], oi: pe.oi||[],
                       greeks: pe.greeks||{delta:[],gamma:[],theta:[],vega:[]},
                       stats: pe.stats||null}
                    : {candles: [], iv: [], oi: [],
                       greeks:{delta:[],gamma:[],theta:[],vega:[]}, stats: null};
  state.lastShared = {
    rv: (rvSrc && rvSrc.rv) || [],
    spot: ((ce && ce.spot) || (pe && pe.spot) || []),
    days: ((ce && ce.days) || (pe && pe.days) || []),
  };

  // Spot label (last close in shared spot series)
  const lastSpot = state.lastShared.spot.length
    ? state.lastShared.spot[state.lastShared.spot.length - 1].value : null;
  $('spotval').textContent = lastSpot != null ? lastSpot.toFixed(1) : '—';

  state.cPriceCe.timeScale().fitContent();
  const dy = state.lastShared.days;
  const dayWindow = dy.length
    ? `${dy[0].slice(0,4)}-${dy[0].slice(4,6)}-${dy[0].slice(6)} → ${dy[dy.length-1].slice(0,4)}-${dy[dy.length-1].slice(4,6)}-${dy[dy.length-1].slice(6)}`
    : '';
  const nCE = state.lastCe.candles.length;
  const nPE = state.lastPe.candles.length;
  setStatus(`CE ${nCE} · PE ${nPE} bars · ${state.tfPrice} · ${dayWindow}`);

  buildDayStats();
  applyAllMarkers();
  applyTradePriceLines();   // re-attach trade lines after series.setData()
  renderStats(null);
}

// ── Wire UI ────────────────────────────────────────────────────────────────
function attachToggles() {
  document.querySelectorAll('#tf-price button').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('#tf-price button').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); state.tfPrice = b.dataset.tf; loadLeg();
  }));
  $('tg-spot').addEventListener('click', () => { state.showSpot = !state.showSpot; $('tg-spot').classList.toggle('on', state.showSpot); loadLeg(); });

  function dropTrade() {
    state.activeEvent = null;
    state.activeTrade = null;
    clearTradePriceLines();
    renderTradeBanner();
    const evSel = document.getElementById('event');
    if (evSel) evSel.value = '';
  }
  $('date').addEventListener('change', async e => {
    if (state.programmaticChange) return;
    state.date = e.target.value;
    dropTrade();
    await loadExpiries(); await loadStrikes(); loadLeg();
  });
  $('expiry').addEventListener('change', async e => {
    if (state.programmaticChange) return;
    state.expiry = e.target.value;
    dropTrade();
    await loadStrikes(); loadLeg();
  });
  $('strike').addEventListener('change', e => {
    if (state.programmaticChange) return;
    state.strike = +e.target.value;
    dropTrade();
    loadLeg();
  });
  $('otype').addEventListener('change',  e => { state.otype  =  e.target.value; loadLeg(); });
  $('lookback').addEventListener('change', e => {
    if (state.programmaticChange) return;
    state.lookbackDays = +e.target.value; loadLeg();
  });

  $('event').addEventListener('change', e => {
    const v = e.target.value;
    if (v === '') {
      state.activeEvent = null;
      state.activeTrade = null;
      clearTradePriceLines();
      renderTradeBanner();
      applyAllMarkers();
      return;
    }
    pickEvent(parseInt(v, 10));
  });
  $('tg-signals').addEventListener('click', () => {
    state.showSignals = !state.showSignals;
    $('tg-signals').classList.toggle('on', state.showSignals);
    applyAllMarkers();
  });
}

// ── Pane resize handles ───────────────────────────────────────────────────
// Drag between two panes to redistribute their flex-grow values. The pair's
// combined size stays constant, so other panes are unaffected.
function attachResizers() {
  document.querySelectorAll('.resizer').forEach(handle => {
    handle.addEventListener('mousedown', startDrag);
  });

  function startDrag(e) {
    e.preventDefault();
    const handle = e.currentTarget;
    const prev = document.getElementById(handle.dataset.prev);
    const next = document.getElementById(handle.dataset.next);
    if (!prev || !next) return;

    const prevH0 = prev.getBoundingClientRect().height;
    const nextH0 = next.getBoundingClientRect().height;
    const total  = prevH0 + nextH0;
    const startY = e.clientY;

    const prevFlex0 = parseFloat(getComputedStyle(prev).flexGrow) || 1;
    const nextFlex0 = parseFloat(getComputedStyle(next).flexGrow) || 1;
    const flexTotal = prevFlex0 + nextFlex0;

    handle.classList.add('dragging');
    document.body.classList.add('resizing');

    function onMove(ev) {
      const dy = ev.clientY - startY;
      let prevH = prevH0 + dy;
      const minH = 40;
      if (prevH < minH) prevH = minH;
      if (prevH > total - minH) prevH = total - minH;
      const ratio = prevH / total;
      prev.style.flex = (flexTotal * ratio).toFixed(3) + ' 1 0';
      next.style.flex = (flexTotal * (1 - ratio)).toFixed(3) + ' 1 0';
    }
    function onUp() {
      handle.classList.remove('dragging');
      document.body.classList.remove('resizing');
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
  }
}

async function boot() {
  buildCharts();
  attachToggles();
  attachResizers();
  await loadDays();
  await loadEvents();
  await loadExpiries();
  await loadStrikes();
  loadLeg();
}
boot();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML.replace("__APP_BASE__", APP_BASE))
