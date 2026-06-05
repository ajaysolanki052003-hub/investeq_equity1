"""Groww live-market client — LTP and latest-minute helpers.

Companion to groww_client.py (which is historical-only). Uses the
`/v1/live-data/ltp` endpoint for tick-style LTP and `/v1/historical/candles` at
1minute granularity for "latest closed candle" when needed.

Designed for the EMA scanner live worker — small, no global state, fast retry
on 429/5xx, returns a flat dict that the worker pickles to parquet.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Mapping

import requests

GROWW_LTP_URL   = "https://api.groww.in/v1/live-data/ltp"
GROWW_QUOTE_URL = "https://api.groww.in/v1/live-data/quote"


def _groww_symbol(symbol: str, exchange: str = "NSE") -> str:
    if "-" in symbol:
        return symbol
    return f"{exchange}-{symbol}"


def _ltp_one(
    session: requests.Session,
    symbol: str,
    token: str,
    *,
    exchange: str = "NSE",
    segment: str = "CASH",
    timeout: float = 5.0,
    retries: int = 2,
) -> float | None:
    """LTP for a single symbol. Returns price or None on permanent failure.
    Retries once on 429/5xx with a small backoff."""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "X-API-VERSION": "1.0",
    }
    params = {
        "exchange": exchange,
        "segment":  segment,
        "groww_symbol": _groww_symbol(symbol, exchange),
    }
    backoff = 0.3
    for attempt in range(retries + 1):
        try:
            r = session.get(GROWW_LTP_URL, headers=headers, params=params, timeout=timeout)
        except requests.RequestException:
            if attempt >= retries:
                return None
            time.sleep(backoff); backoff *= 2; continue
        if r.status_code == 200:
            try:
                payload = r.json().get("payload") or {}
                ltp = payload.get("ltp") or payload.get("last_price")
                return float(ltp) if ltp is not None else None
            except Exception:
                return None
        if r.status_code == 429 or r.status_code >= 500:
            if attempt >= retries:
                return None
            time.sleep(backoff); backoff *= 2
        else:
            return None
    return None


def fetch_ltp(
    symbols: Iterable[str],
    token: str,
    *,
    exchange: str = "NSE",
    segment:  str = "CASH",
    max_workers: int = 8,
) -> dict[str, float]:
    """Concurrent per-symbol LTP fetch. Returns {symbol: ltp}, dropping any
    that failed. max_workers caps simultaneous in-flight requests so the broker
    sees ~8 req/s peak even if the caller is polling tightly."""
    syms = list(symbols)
    out: dict[str, float] = {}
    if not syms:
        return out
    sess = requests.Session()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {
            pool.submit(_ltp_one, sess, s, token, exchange=exchange, segment=segment): s
            for s in syms
        }
        for fut in as_completed(fut_map):
            s = fut_map[fut]
            try:
                ltp = fut.result()
            except Exception:
                ltp = None
            if ltp is not None:
                out[s] = ltp
    return out


def fetch_latest_minute(
    symbol: str,
    token: str,
    *,
    exchange: str = "NSE",
    segment:  str = "CASH",
    timeout:  float = 8.0,
) -> dict | None:
    """Last closed 1-minute candle (O/H/L/C/V) for one symbol, via the
    historical-candles endpoint with a 5-minute lookback window. Useful when
    LTP alone isn't enough to seed a forming hourly candle on a cold start."""
    import datetime as _dt
    now = _dt.datetime.utcnow() + _dt.timedelta(hours=5, minutes=30)
    end = now.strftime("%Y-%m-%d %H:%M:%S")
    start = (now - _dt.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "X-API-VERSION": "1.0",
    }
    params = {
        "exchange": exchange,
        "segment":  segment,
        "groww_symbol":   _groww_symbol(symbol, exchange),
        "start_time":     start,
        "end_time":       end,
        "candle_interval": "1minute",
    }
    try:
        r = requests.get("https://api.groww.in/v1/historical/candles",
                         headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        candles = r.json().get("payload", {}).get("candles") or []
    except Exception:
        return None
    if not candles:
        return None
    # candles are [ts, o, h, l, c, v] arrays
    ts, o, h, l, c, v = candles[-1][:6]
    return {"timestamp": int(ts), "open": float(o), "high": float(h),
            "low": float(l), "close": float(c), "volume": float(v)}
