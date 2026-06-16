"""Technical-indicator math for the Option Indicator Lab (/optlab).

Pure pandas/numpy — every function takes an OHLCV DataFrame (columns:
timestamp, open, high, low, close, volume) already sorted by time and returns
plain Python lists of {"time": int, "value": float} (or grouped dicts) ready
for lightweight-charts. NaNs are dropped per-series so a short warm-up window
doesn't poison the chart.

Indicators: EMA, SMA, MACD, RSI, Bollinger Bands, VWAP (intraday-cumulative),
ADX/+DI/-DI, Stochastic, OBV. Volume Profile is built client-side from the
candles so it can evolve during replay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _series(ts, vals):
    """Zip a time index + value array into [{time, value}], dropping NaN/inf."""
    out = []
    for t, v in zip(ts, vals):
        if v is None:
            continue
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            continue
        out.append({"time": int(t), "value": round(fv, 4)})
    return out


def ema(df, span):
    return _series(df["timestamp"], df["close"].ewm(span=span, adjust=False).mean())


def sma(df, window):
    return _series(df["timestamp"], df["close"].rolling(window).mean())


def macd(df, fast=12, slow=26, signal=9):
    c = df["close"]
    macd_line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    ts = df["timestamp"]
    return {
        "macd":   _series(ts, macd_line),
        "signal": _series(ts, sig_line),
        "hist":   [{**d, "color": "#26a69a" if d["value"] >= 0 else "#ef5350"}
                   for d in _series(ts, hist)],
    }


def rsi(df, period=14):
    c = df["close"]
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing (RMA) via ewm(alpha=1/period).
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return _series(df["timestamp"], out)


def bollinger(df, window=20, mult=2.0):
    c = df["close"]
    mid = c.rolling(window).mean()
    sd = c.rolling(window).std(ddof=0)
    ts = df["timestamp"]
    return {
        "mid":   _series(ts, mid),
        "upper": _series(ts, mid + mult * sd),
        "lower": _series(ts, mid - mult * sd),
    }


def vwap(df):
    """Intraday cumulative VWAP. Resets each calendar day in the frame."""
    d = df.copy()
    d["day"] = pd.to_datetime(d["timestamp"], unit="s").dt.date
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    pv = tp * d["volume"]
    cum_pv = pv.groupby(d["day"]).cumsum()
    cum_v = d["volume"].groupby(d["day"]).cumsum().replace(0, np.nan)
    return _series(d["timestamp"], cum_pv / cum_v)


def adx(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    ts = df["timestamp"]
    return {
        "adx":   _series(ts, adx_line),
        "plus":  _series(ts, plus_di),
        "minus": _series(ts, minus_di),
    }


def stochastic(df, k=14, d=3):
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    pct_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    pct_d = pct_k.rolling(d).mean()
    ts = df["timestamp"]
    return {"k": _series(ts, pct_k), "d": _series(ts, pct_d)}


def obv(df):
    sign = np.sign(df["close"].diff().fillna(0.0))
    out = (sign * df["volume"]).cumsum()
    return _series(df["timestamp"], out)


def volume_bars(df):
    """Volume histogram colored by candle direction."""
    out = []
    for t, o, c, v in zip(df["timestamp"], df["open"], df["close"], df["volume"]):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out.append({"time": int(t), "value": float(v),
                    "color": "#26a69a55" if c >= o else "#ef535055"})
    return out


def compute_all(df):
    """Bundle every indicator series for one contract's OHLCV frame."""
    return {
        "bb":    bollinger(df),
        "vwap":  vwap(df),
        "macd":  macd(df),
        "rsi":   rsi(df),
        "adx":   adx(df),
        "stoch": stochastic(df),
        "obv":   obv(df),
    }
