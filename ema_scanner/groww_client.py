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

import json
import os
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyotp
import requests

GROWW_TOKEN_URL   = 'https://api.groww.in/v1/token/api/access'
GROWW_CANDLES_URL = 'https://api.groww.in/v1/historical/candles'

# ───────────────────────── shared access-token cache ────────────────────────
# Every service on the VM (scanner, candle workers, optlab, live strategy, …)
# calls get_access_token(). Minting is itself rate-limited by Groww on
# /v1/token/api/access, and a fresh mint per request quickly trips a 429 that
# then starves EVERY other worker of a token (observed 2026-06-23: optlab's
# retry storm 429'd the token endpoint and crashed the 15m candle refresh).
#
# Fix: persist one token to a file all processes share, refresh it at most once
# per _TOKEN_TTL, and on a 429 enter a cooldown (serving the last good token)
# instead of hammering. This collapses VM-wide token mints to ~1 per 50 min.
_TOKEN_TTL     = 3000.0   # reuse a healthy token for ~50 min before refreshing
_MINT_COOLDOWN = 300.0    # after a 429, wait this long before re-minting. Kept
                          # high (≤12 mints/hr VM-wide) so a stuck token can
                          # never re-trip Groww's extended token rate-limit.
_token_lock = threading.Lock()


def _token_cache_path() -> Path:
    base = os.environ.get("INVESTEQ_DATA") or tempfile.gettempdir()
    return Path(base) / ".groww_token.json"


def _read_token_cache() -> dict:
    try:
        return json.loads(_token_cache_path().read_text())
    except Exception:
        return {}


def _write_token_cache(d: dict) -> None:
    try:
        _token_cache_path().write_text(json.dumps(d))
    except Exception:
        pass   # cache is an optimization; never fail the caller on a write error


def _mint_access_token(totp_jwt: str, totp_secret: str, timeout: int = 30) -> str:
    """Raw token mint — one POST to Groww's token endpoint. Rate-limited; call
    via get_access_token(), which caches + backs off, not directly."""
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
    """Return a Groww access token, shared across all processes via an on-disk
    cache. Refreshes at most once per ~50 min; on a 429 it backs off and serves
    the last good token rather than re-hammering the rate-limited endpoint."""
    now = time.time()
    with _token_lock:
        cache = _read_token_cache()
        tok    = cache.get("token")
        minted = cache.get("minted_at", 0.0)
        # 1. Healthy cached token (possibly minted by another process) → reuse.
        if tok and now - minted < _TOKEN_TTL:
            return tok
        # 2. In post-429 cooldown → don't mint; serve stale token if we have one.
        if now - cache.get("last_fail", 0.0) < _MINT_COOLDOWN:
            if tok:
                return tok
            raise RuntimeError(
                f"Groww token endpoint in 429 cooldown ({_MINT_COOLDOWN:.0f}s) "
                "and no cached token available")
        # 3. Mint a fresh one.
        try:
            new = _mint_access_token(totp_jwt, totp_secret, timeout)
        except requests.exceptions.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code == 429:
                cache["last_fail"] = now
                _write_token_cache(cache)
                if tok:
                    return tok   # serve stale rather than fail the caller
            raise
        _write_token_cache({"token": new, "minted_at": now, "last_fail": 0.0})
        return new


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
    # Retry on 429 / 5xx with exponential backoff. Groww throttles bursts even
    # under the documented rate caps, so swallowing a 429 silently (as the old
    # code did) leaves multi-day holes in the master series.
    last_exc = None
    for attempt in range(5):
        try:
            r = requests.get(GROWW_CANDLES_URL, headers=headers,
                             params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 1.0 * (2 ** attempt)
                time.sleep(wait)
                last_exc = RuntimeError(f'{r.status_code} after attempt {attempt+1}')
                continue
            r.raise_for_status()
            return r.json().get('payload', {}).get('candles') or []
        except requests.RequestException as e:
            last_exc = e
            time.sleep(1.0 * (2 ** attempt))
    raise RuntimeError(f'_fetch_window: exhausted retries: {last_exc}')


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
