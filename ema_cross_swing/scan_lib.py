"""Headless EMA21/50 swing-low scanner library — the exact same gates the
Streamlit UI uses, but importable from cron / batch scripts without Streamlit.

Kept deliberately stateless: no caching, no widgets, no globals. Streamlit's
@st.cache_data on top is a thin perf wrapper around `signals_on_day_lib()`
in the UI.
"""
from __future__ import annotations
import functools
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import (
    add_emas, scan_buy_signals, scan_sell_signals,
    compute_swing_lows, compute_swing_highs, find_swing_sl, find_swing_sh,
)

SWING_WINDOW = 2
DEFAULT_EMA_PARAMS = {
    "1d": {"fast": 21, "slow": 50, "compulsory": 2},
    "1h": {"fast": 21, "slow": 50, "compulsory": 2},
}
SCAN_COLS = ["Symbol", "Side", "TF", "Touch Time",
             "Entry", "SL", "Risk %", "Current", "P&L %", "Peak %", "Status",
             # Pierce-both EMA flags used by the Telegram alert filter. Kept
             # in the parquet so downstream consumers (alerts, analytics) can
             # filter without reloading OHLC. Internal — hidden from the UI.
             "LowBelowBoth", "HighAboveBoth"]
# Subset shown to the user in the Streamlit grid (everything else stays
# in the parquet but is not rendered).
SCAN_COLS_UI = ["Symbol", "Side", "TF", "Touch Time",
                "Entry", "SL", "Risk %", "Current", "P&L %", "Peak %", "Status"]


def _now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


@functools.lru_cache(maxsize=2048)
def _load_1m_opens(symbol: str, data_root: str, _mtime: float):
    """Cached (datetime, open) lookup table for a symbol's 1m CSV.
    `_mtime` is the file mtime — passed as a key arg so the cache invalidates
    automatically when the 1m file is updated by the live fetcher."""
    path = os.path.join(data_root, "1m", f"{symbol}_historical.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=["datetime", "open"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.dropna(subset=["open"]).sort_values("datetime").reset_index(drop=True)
        return df
    except Exception:
        return None


def _recover_open_from_1m(symbol: str, data_root: str,
                          period_start: pd.Timestamp,
                          period_end: pd.Timestamp) -> float | None:
    """Return the Open of the first 1m candle in [period_start, period_end).
    Used to repair a missing Open on a 1d (period = whole day) or 1h
    (period = one hour) bar. Returns None if no 1m data covers that window."""
    path = os.path.join(data_root, "1m", f"{symbol}_historical.csv")
    if not os.path.exists(path):
        return None
    df = _load_1m_opens(symbol, data_root, os.path.getmtime(path))
    if df is None or df.empty:
        return None
    sub = df[(df["datetime"] >= period_start) & (df["datetime"] < period_end)]
    if sub.empty:
        return None
    return float(sub.iloc[0]["open"])


def load_ohlc(symbol: str, tf: str, data_root: str) -> pd.DataFrame:
    """Read a `<sym>_historical.csv` and return a DateTime-indexed OHLCV df.

    When Open is missing on a 1d/1h bar, repair it by reading the Open of the
    first 1m candle within that day (1d) or hour (1h). Final fallback is the
    prior bar's Close."""
    path = os.path.join(data_root, tf, f"{symbol}_historical.csv")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    if df["Open"].isna().any() and tf in ("1d", "1h"):
        period_delta = pd.Timedelta(days=1) if tf == "1d" else pd.Timedelta(hours=1)
        for idx in df.index[df["Open"].isna()]:
            # 1d bars in the CSV are typically stamped 00:00; the actual
            # period starts at 09:15 IST. 1h bars are already at hour
            # boundaries (09:15, 10:15, ...).
            ps = (idx.normalize() + pd.Timedelta(hours=9, minutes=15)
                  if tf == "1d" else idx)
            v = _recover_open_from_1m(symbol, data_root, ps, ps + period_delta)
            if v is not None:
                df.at[idx, "Open"] = v
    if df["Open"].isna().any():
        df["Open"] = df["Open"].fillna(df["Close"].shift(1))
    return df.dropna(subset=["High", "Low", "Close"])


def _synth_today_1d_from_1h(symbol: str, data_root: str,
                            target_day: date) -> pd.DataFrame | None:
    """When scanning today's 1d before market close, synthesize a forming day
    bar from today's 1h bars so today is scannable."""
    try:
        h = load_ohlc(symbol, "1h", data_root)
    except Exception:
        return None
    today_h = h[h.index.normalize() == pd.Timestamp(target_day)]
    if today_h.empty:
        return None
    return pd.DataFrame({
        "Open":   [float(today_h["Open"].iloc[0])],
        "High":   [float(today_h["High"].max())],
        "Low":    [float(today_h["Low"].min())],
        "Close":  [float(today_h["Close"].iloc[-1])],
        "Volume": [float(today_h["Volume"].sum())],
    }, index=[pd.Timestamp(target_day)])


def _sl_risk_pct(side: str, entry: float, sl: float) -> float:
    return abs((entry - sl) / entry * 100)


def _compute_touch_sl(full, side, ttime, full_emas, swing_low_mask, swing_high_mask):
    if ttime not in full.index:
        return None
    idx = full.index.get_loc(ttime)
    if side == "BUY":
        swing_sl = float(find_swing_sl(full_emas, idx, swing_low_mask))
        return min(swing_sl, float(full["Low"].iloc[idx]))
    swing_sh = float(find_swing_sh(full_emas, idx, swing_high_mask))
    return max(swing_sh, float(full["High"].iloc[idx]))


def signals_on_day(symbol: str, tf: str, data_root: str, target_day: date,
                   fast: int, slow: int, compulsory: int,
                   max_sl_pct: float | None = None) -> list[dict]:
    """Return list of touches that fired on `target_day` for one symbol,
    each enriched with SL, current price, P&L%, peak move and tradability."""
    try:
        full = load_ohlc(symbol, tf, data_root)
    except Exception:
        return []
    if tf == "1d":
        today_ist = _now_ist().date()
        if target_day == today_ist and pd.Timestamp(target_day) not in full.index:
            synth = _synth_today_1d_from_1h(symbol, data_root, target_day)
            if synth is not None:
                full = pd.concat([full, synth]).sort_index()
    if len(full) < slow + 5:
        return []
    cutoff = pd.Timestamp(target_day) + pd.Timedelta(days=1)
    df = full[full.index < cutoff]
    if len(df) < slow + 5:
        return []
    df = add_emas(df, fast, slow)
    full_emas = add_emas(full, fast, slow)
    swing_low_mask  = compute_swing_lows(full_emas, SWING_WINDOW)
    swing_high_mask = compute_swing_highs(full_emas, SWING_WINDOW)
    current_close = float(full["Close"].iloc[-1])

    out = []
    for side, fn in (("BUY", scan_buy_signals), ("SELL", scan_sell_signals)):
        sigs = fn(df, fast, slow, compulsory=compulsory)
        if sigs.empty:
            continue
        sigs = sigs[sigs["Touch Time"].dt.date == target_day]
        for _, r in sigs.iterrows():
            ttime = r["Touch Time"]
            entry = round(float(r["Touch Close"]), 2)
            sl = _compute_touch_sl(full, side, ttime, full_emas,
                                   swing_low_mask, swing_high_mask)
            risk_pct = _sl_risk_pct(side, entry, sl) if sl is not None else None
            tradable = (max_sl_pct is None) or (risk_pct is not None and risk_pct <= max_sl_pct)
            # "Pierce both EMAs" booleans for Telegram alert gating. The touch
            # candle must take its Low below BOTH EMA21 and EMA50 for BUY (or
            # High above both for SELL). EMA columns are named EMA<n> by
            # strategy_core.add_emas. We only compute when the touch time
            # exists in the indexed frames.
            low_below_both = False
            high_above_both = False
            if ttime in full.index and ttime in full_emas.index:
                ema_f = full_emas.at[ttime, f"EMA{fast}"]
                ema_s = full_emas.at[ttime, f"EMA{slow}"]
                low_t = full.at[ttime, "Low"]
                high_t = full.at[ttime, "High"]
                if pd.notna(ema_f) and pd.notna(ema_s):
                    if pd.notna(low_t):
                        low_below_both = bool(low_t <= ema_f and low_t <= ema_s)
                    if pd.notna(high_t):
                        high_above_both = bool(high_t >= ema_f and high_t >= ema_s)
            sl_hit_at = None
            peak_pct  = None
            if sl is not None and ttime in full.index:
                post = full.loc[ttime:].iloc[1:]
                if not post.empty:
                    hit = (post["Low"] <= sl) if side == "BUY" else (post["High"] >= sl)
                    if hit.any():
                        sl_hit_at = hit[hit].index[0]
                    eval_post = post.loc[:sl_hit_at] if sl_hit_at is not None else post
                    if not eval_post.empty:
                        if side == "BUY":
                            peak_pct = (float(eval_post["High"].max()) - entry) / entry * 100
                        else:
                            peak_pct = (entry - float(eval_post["Low"].min())) / entry * 100
            if sl_hit_at is not None and tradable and risk_pct is not None:
                pnl_pct = -risk_pct
            else:
                pnl_pct = ((current_close - entry) if side == "BUY"
                           else (entry - current_close)) / entry * 100
            out.append({
                "Symbol":     symbol,
                "Side":       side,
                "TF":         tf,
                "Touch Time": ttime,
                "Entry":      entry,
                "SL":         round(sl, 2) if sl is not None else None,
                "Risk %":     round(risk_pct, 2) if risk_pct is not None else None,
                "Current":    round(current_close, 2),
                "P&L %":      round(pnl_pct, 2) if pnl_pct is not None else None,
                "Peak %":     round(peak_pct, 2) if peak_pct is not None else None,
                "Status":     ("NO TRADE" if not tradable
                               else ("SL HIT" if sl_hit_at is not None else "OPEN")),
                "LowBelowBoth":  low_below_both,
                "HighAboveBoth": high_above_both,
            })
    return out


def list_symbols(data_root: str, tf: str) -> list[str]:
    folder = os.path.join(data_root, tf)
    if not os.path.isdir(folder):
        return []
    return sorted(f[:-len("_historical.csv")]
                  for f in os.listdir(folder) if f.endswith("_historical.csv"))
