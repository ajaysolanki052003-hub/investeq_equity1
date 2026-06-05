"""Groww API client — authentication + chunked historical-candle fetching.

Uses the official `/v1/historical/candles` endpoint (the backtesting one, not the
deprecated `/historical/candle/range`). Per Groww docs:

  Max range per request:
    1minute / 2minute / 3minute / 5minute       -> 30 days
    10minute / 15minute / 30minute              -> 90 days
    1hour / 4hour / 1day / 1week / 1month       -> 180 days

  History availability: from 2020 for equities/indices/FNO (ALL intervals).

This module chunks long date ranges into request-sized windows so you can fetch
multi-year histories.

Usage:
    from groww_client import get_access_token, fetch_candles
    token = get_access_token(TOTP_JWT, TOTP_SECRET)
    df = fetch_candles('RELIANCE', '1h', '2020-01-01 09:15:00', '2026-05-08 15:30:00', token)
"""

import time
from datetime import timedelta
from typing import Iterable, Optional

import pandas as pd
import pyotp
import requests

GROWW_TOKEN_URL   = 'https://api.groww.in/v1/token/api/access'
GROWW_CANDLES_URL = 'https://api.groww.in/v1/historical/candles'

# Map our short keys to the names Groww's API expects in `candle_interval`.
INTERVAL_MAP = {
    '1m':    '1minute',
    '2m':    '2minute',
    '3m':    '3minute',
    '5m':    '5minute',
    '10m':   '10minute',
    '15m':   '15minute',
    '30m':   '30minute',
    '60m':   '1hour',
    '1h':    '1hour',
    '4h':    '4hour',
    '1d':    '1day',
    '1w':    '1week',
    '1M':    '1month',
    # also accept the canonical names verbatim
    '1minute':  '1minute',  '2minute': '2minute',  '3minute': '3minute',
    '5minute':  '5minute',  '10minute': '10minute', '15minute': '15minute',
    '30minute': '30minute', '1hour':    '1hour',    '4hour':    '4hour',
    '1day':     '1day',     '1week':    '1week',    '1month':   '1month',
}

# Max chunk size per interval (days) — slightly below the documented cap so we
# never hit boundary errors.
CHUNK_DAYS = {
    '1minute':  29,
    '2minute':  29,
    '3minute':  29,
    '5minute':  29,
    '10minute': 89,
    '15minute': 89,
    '30minute': 89,
    '1hour':    175,
    '4hour':    175,
    '1day':     175,
    '1week':    175,
    '1month':   175,
}


def get_access_token(totp_jwt: str, totp_secret: str, timeout: int = 30) -> str:
    """Exchange the long-lived TOTP JWT + a live TOTP for a short-lived access token."""
    live_totp = pyotp.TOTP(totp_secret).now()
    r = requests.post(
        GROWW_TOKEN_URL,
        headers={
            'Authorization': f'Bearer {totp_jwt}',
            'Content-Type':  'application/json',
        },
        json={'key_type': 'totp', 'totp': live_totp},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    token = data.get('token') or data.get('access_token')
    if not token:
        raise RuntimeError(f'No token in response: {data}')
    return token


def _groww_symbol(symbol: str, exchange: str = 'NSE') -> str:
    """Convert a plain trading symbol like 'RELIANCE' to Groww's namespaced form 'NSE-RELIANCE'."""
    if '-' in symbol:
        return symbol  # already namespaced
    return f'{exchange}-{symbol}'


def _fetch_window(
    groww_sym:    str,
    candle_int:   str,
    start_str:    str,
    end_str:      str,
    access_token: str,
    exchange:     str = 'NSE',
    segment:      str = 'CASH',
    timeout:      int = 30,
) -> list:
    """Single request to /historical/candles. Returns the candles list (list of arrays)."""
    headers = {
        'Accept':        'application/json',
        'Authorization': f'Bearer {access_token}',
        'X-API-VERSION': '1.0',
    }
    params = {
        'exchange':        exchange,
        'segment':         segment,
        'groww_symbol':    groww_sym,
        'start_time':      start_str,
        'end_time':        end_str,
        'candle_interval': candle_int,
    }
    r = requests.get(GROWW_CANDLES_URL, headers=headers, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get('payload', {}).get('candles') or []


def fetch_candles(
    symbol:        str,
    interval:      str,                  # short key like '1d', '1h', '5m' OR canonical '1day', '1hour'
    start:         str,                  # 'YYYY-MM-DD HH:MM:SS'
    end:           str,
    access_token:  str,
    exchange:      str = 'NSE',
    segment:       str = 'CASH',
    delay_s:       float = 0.2,          # pause between chunks (rate-limit friendly)
    verbose:       bool = False,
    chunk_days_override: Optional[int] = None,
) -> pd.DataFrame:
    """Chunked historical fetch — bypasses per-request range caps.

    Iterates forward from `start` to `end` in windows sized per `CHUNK_DAYS`.
    Returns DataFrame: timestamp (epoch sec), open, high, low, close, volume,
    open_interest, datetime (IST).
    """
    candle_int = INTERVAL_MAP.get(interval)
    if candle_int is None:
        raise ValueError(f'Unknown interval: {interval!r}. Use one of {sorted(INTERVAL_MAP)}')

    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    if start_dt >= end_dt:
        raise ValueError('start must be < end')

    days = chunk_days_override or CHUNK_DAYS.get(candle_int, 30)
    groww_sym = _groww_symbol(symbol, exchange)

    all_candles = []
    chunk_no = 0
    cur = start_dt
    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=days), end_dt)
        s_str = cur.strftime('%Y-%m-%d %H:%M:%S')
        e_str = chunk_end.strftime('%Y-%m-%d %H:%M:%S')
        chunk_no += 1
        try:
            window = _fetch_window(groww_sym, candle_int, s_str, e_str, access_token,
                                   exchange=exchange, segment=segment)
        except Exception as e:
            if verbose:
                print(f'  [chunk {chunk_no}] {s_str} -> {e_str}  FAILED: {e}')
            window = []
        if verbose:
            print(f'  [chunk {chunk_no}] {s_str} -> {e_str}  : {len(window)} candles')
        all_candles.extend(window)
        cur = chunk_end + timedelta(seconds=1)
        if delay_s > 0 and cur < end_dt:
            time.sleep(delay_s)

    cols = ['timestamp_raw', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
    if not all_candles:
        out = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close',
                                    'volume', 'open_interest', 'datetime'])
        return out

    # Some payloads return 6-element arrays for equity (no OI); pad to 7.
    norm = []
    for c in all_candles:
        if not c:
            continue
        if len(c) == 6:
            norm.append(list(c) + [None])
        elif len(c) >= 7:
            norm.append(list(c[:7]))
    df = pd.DataFrame(norm, columns=cols)

    # `timestamp_raw` is an ISO string like '2025-09-24T10:30:00' (IST, naive).
    df['datetime'] = pd.to_datetime(df['timestamp_raw'])
    # Unit-agnostic conversion to epoch seconds — pandas may return ns OR us depending
    # on input precision, so don't assume the unit.
    df['timestamp'] = ((df['datetime'] - pd.Timestamp('1970-01-01')) //
                       pd.Timedelta(seconds=1)).astype('int64')

    df = df.drop_duplicates(subset='timestamp', keep='first')
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'datetime']]
    return df


def fetch_for_symbols(
    symbols:       Iterable[str],
    interval:      str,
    start:         str,
    end:           str,
    access_token:  str,
    save_dir:      Optional[str] = None,
    **kwargs,
) -> dict:
    """Convenience: fetch for many symbols, optionally save each to CSV.

    Returns dict[symbol -> DataFrame].
    """
    import os
    out = {}
    for sym in symbols:
        if kwargs.get('verbose'):
            print(f'\n=== {sym} {interval}  {start} -> {end} ===')
        df = fetch_candles(sym, interval, start, end, access_token, **kwargs)
        out[sym] = df
        if save_dir and not df.empty:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'{sym}_historical.csv')
            df.to_csv(path, index=False)
            print(f'  saved {path}  ({len(df)} rows)')
    return out
