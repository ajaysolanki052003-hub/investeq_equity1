"""Incremental fetch — extend each existing CSV in data/1d and data/1h
from its last recorded datetime up to a target end-time.

Usage:
    python incremental_fetch.py
    python incremental_fetch.py --limit 5     # smoke test
    python incremental_fetch.py --interval 1d # only one interval
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from groww_client import get_access_token, fetch_candles

# Credentials live in env vars (set in /etc/investeq.env on the VM, or your
# local shell). Never commit raw tokens — anyone with the TOTP_SECRET can mint
# a perpetual stream of OTPs against the broker.
TOTP_TOKEN  = os.environ["GROWW_TOTP_JWT"]
TOTP_SECRET = os.environ["GROWW_TOTP_SECRET"]


# NSE session bounds (IST). Anything past close clamps to close.
NSE_OPEN  = (9, 15)
NSE_CLOSE = (15, 30)


def now_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def default_end_time() -> str:
    """End-time = now in IST, clamped to today's NSE close. If it's pre-open
    today, fall back to yesterday's close so we don't request a future window."""
    n = now_ist()
    close = n.replace(hour=NSE_CLOSE[0], minute=NSE_CLOSE[1], second=0, microsecond=0)
    openn = n.replace(hour=NSE_OPEN[0],  minute=NSE_OPEN[1],  second=0, microsecond=0)
    if n >= close:
        end = close
    elif n < openn:
        end = (close - timedelta(days=1))
    else:
        end = n
    return end.strftime('%Y-%m-%d %H:%M:%S')


END_TIME = default_end_time()
INTERVAL_FOLDER = {'1d': 'data/1d', '1h': 'data/1h', '15m': 'data/15m'}
INTERVAL_STEP = {'1d': timedelta(days=1), '1h': timedelta(hours=1), '15m': timedelta(minutes=15)}
DELAY_BETWEEN_CHUNKS = 0.15
# Groww historical-candles per-token budget is ~5 req/sec. 0.05 (20 req/sec)
# was running 4x over the cap, which produced 30-40% 429 failure on the 523-
# stock 1h refresh (see live_workers.run_candle_refresh log Jun 05 09:46 UTC).
# 0.20s caps us at 5 req/sec — total run goes from ~2 min to ~4 min and the
# error rate drops to ~0%.
DELAY_BETWEEN_STOCKS = 0.20
REAUTH_EVERY = 100


# NSE 1h buckets are :15-aligned (09:15, 10:15, ..., 15:15). Groww's native
# `1hour` candles align to wall-clock (09:00, 10:00, ...), so for the 1h
# interval we fetch 1-minute bars and roll them up ourselves.
def _nse_bucket_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Start of the NSE :15-aligned 1h bucket containing ts.
    09:15-10:14 -> 09:15, ..., 14:15-15:14 -> 14:15, 15:15-15:29 -> 15:15."""
    hour = ts.hour if ts.minute >= 15 else ts.hour - 1
    return ts.replace(hour=hour, minute=15, second=0, microsecond=0)


VALID_1H_HOURS = (9, 10, 11, 12, 13, 14, 15)


def resample_1m_to_1h_nse(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Roll 1-minute bars into NSE :15-aligned 1h candles.
    Returns DataFrame with cols matching fetch_candles' 1h schema."""
    if df_1m.empty:
        return df_1m.iloc[0:0].copy()
    df = df_1m.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['_bucket'] = df['datetime'].apply(_nse_bucket_start)
    df = df[df['_bucket'].dt.hour.isin(VALID_1H_HOURS)]
    if df.empty:
        return df_1m.iloc[0:0].copy()
    agg = df.groupby('_bucket', as_index=False).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    )
    agg = agg.rename(columns={'_bucket': 'datetime'})
    agg['timestamp'] = ((agg['datetime'] - pd.Timestamp('1970-01-01')) //
                        pd.Timedelta(seconds=1)).astype('int64')
    agg['open_interest'] = ''
    agg = agg[['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'open_interest', 'datetime']]
    return agg.sort_values('datetime').reset_index(drop=True)


def parse_last_datetime(path):
    """Read just the last line of a CSV and parse its `datetime` column."""
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return None
        # Walk backwards to find the last newline
        buf = b''
        step = 256
        pos = size
        while pos > 0:
            read = min(step, pos)
            pos -= read
            f.seek(pos)
            buf = f.read(read) + buf
            if buf.count(b'\n') >= 2:
                break
        text = buf.decode('utf-8', errors='ignore').strip()
        last_line = text.split('\n')[-1].strip()
    if not last_line or last_line.startswith('timestamp'):
        return None
    parts = last_line.split(',')
    # datetime is the LAST column in both 1d and 1h schemas
    dt_str = parts[-1].strip()
    try:
        return pd.to_datetime(dt_str)
    except Exception:
        return None


def append_csv(path, df_new, interval):
    """Append df_new rows to path, preserving the file's existing column order."""
    if df_new.empty:
        return 0
    # Read header to determine column layout
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
    cols = header.split(',')
    # df_new comes from fetch_candles: ['timestamp','open','high','low','close','volume','open_interest','datetime']
    # 1d files use: timestamp,open,high,low,close,volume,datetime  (no open_interest)
    # 1h files use: timestamp,open,high,low,close,volume,open_interest,datetime
    out = df_new.copy()
    # Defensive Open fill — Groww's /historical/candles 1day payload has
    # intermittently returned candles with open=None for equity bars. Without
    # this guard the file gets `,,` cells for open, which trips the scanner's
    # green/red bar check. Fill from this batch's prior close first, then
    # fall back to the file's last close so the first new row is also covered.
    if 'open' in out.columns:
        out['open'] = pd.to_numeric(out['open'], errors='coerce')
        if out['open'].isna().any():
            prev_close = out['close'].astype(float).shift(1)
            if pd.isna(prev_close.iloc[0]):
                last_close = _last_close_from_csv(path)
                if last_close is not None:
                    prev_close.iloc[0] = last_close
            out['open'] = out['open'].fillna(prev_close)
            # Final fallback for any still-NaN open: use the same bar's close
            # so the row at least passes downstream validation.
            out['open'] = out['open'].fillna(out['close'])
    # Format datetime per interval to match existing files
    if interval == '1d':
        out['datetime'] = out['datetime'].dt.strftime('%Y-%m-%d')
    else:
        out['datetime'] = out['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Select columns in the order the file uses
    missing = [c for c in cols if c not in out.columns]
    for m in missing:
        out[m] = ''
    out = out[cols]
    out.to_csv(path, mode='a', header=False, index=False)
    return len(out)


def _last_close_from_csv(path):
    """Read just the last data row's `close` field — used to seed Open fill
    when the first appended bar has no preceding row inside the batch."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2); size = f.tell()
            f.seek(max(0, size - 256))
            tail = f.read().decode('utf-8', errors='ignore').strip()
        # First line = header (or partial), last = the actual final row
        last_line = tail.split('\n')[-1]
        # Read header to find close column index
        with open(path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        if 'close' not in header:
            return None
        idx = header.index('close')
        fields = last_line.split(',')
        if idx >= len(fields) or fields[idx] == '':
            return None
        return float(fields[idx])
    except Exception:
        return None


def update_one(path, symbol, interval, token, end_dt):
    last_dt = parse_last_datetime(path)
    if last_dt is None:
        return 'no_last', 0
    step = INTERVAL_STEP[interval]
    # Start the fetch one step after the last bar we have
    start_dt = last_dt + step
    if start_dt >= end_dt:
        return 'up_to_date', 0
    end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
    if interval == '1h':
        # Pull 1m bars covering the gap, roll up to NSE :15-aligned 1h.
        # Always pull from the 09:15 of last_dt's day at minimum so the
        # 15-min stub at 15:15 isn't blocked by partial-bucket overlap.
        fetch_start = (last_dt + timedelta(minutes=1)).replace(second=0, microsecond=0)
        start_str = fetch_start.strftime('%Y-%m-%d %H:%M:%S')
        df_1m = fetch_candles(symbol, '1m', start_str, end_str, token,
                              delay_s=DELAY_BETWEEN_CHUNKS, verbose=False)
        if df_1m.empty:
            return 'empty', 0
        df = resample_1m_to_1h_nse(df_1m)
        # Drop the (now-stale) partial bucket that overlaps last_dt
        df = df[df['datetime'] > last_dt]
        if df.empty:
            return 'no_new', 0
        # If end_dt is mid-session, the trailing bucket is still forming —
        # don't store it. Keep only buckets that have fully closed.
        bucket_close = df['datetime'] + timedelta(hours=1)
        # 15:15 bucket closes at 15:30, not 16:15:
        is_stub = (df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 15)
        bucket_close = bucket_close.where(~is_stub,
                                          df['datetime'] + timedelta(minutes=15))
        df = df[bucket_close <= end_dt]
        if df.empty:
            return 'no_new', 0
        n = append_csv(path, df, interval)
        return 'ok', n
    if interval == '15m':
        # Native Groww 15-minute candles already align to the NSE :15 grid
        # (09:15, 09:30, ...), so no resampling is needed — but default_end_time
        # is "now" during a session, so drop the still-forming bucket: keep only
        # bars whose 15-min window has fully closed at/under end_dt.
        start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        df = fetch_candles(symbol, '15m', start_str, end_str, token,
                           delay_s=DELAY_BETWEEN_CHUNKS, verbose=False)
        if df.empty:
            return 'empty', 0
        df = df[df['datetime'] > last_dt]
        df = df[df['datetime'] + timedelta(minutes=15) <= end_dt]
        if df.empty:
            return 'no_new', 0
        n = append_csv(path, df, interval)
        return 'ok', n
    # 1d (and any other native interval) goes through Groww's endpoint directly
    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    df = fetch_candles(symbol, interval, start_str, end_str, token,
                       delay_s=DELAY_BETWEEN_CHUNKS, verbose=False)
    if df.empty:
        return 'empty', 0
    df = df[df['datetime'] > last_dt]
    if df.empty:
        return 'no_new', 0
    n = append_csv(path, df, interval)
    return 'ok', n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--interval', choices=['1d', '1h'], default=None,
                    help='Restrict to one interval (default: both)')
    args = ap.parse_args()

    print('[auth] requesting access token...', flush=True)
    token = get_access_token(TOTP_TOKEN, TOTP_SECRET)
    print('[auth] OK', flush=True)

    end_dt = pd.to_datetime(END_TIME)
    intervals = [args.interval] if args.interval else ['1d', '1h']

    grand_t0 = time.time()
    grand_summary = []

    for interval in intervals:
        folder = INTERVAL_FOLDER[interval]
        files = sorted(f for f in os.listdir(folder) if f.endswith('_historical.csv'))
        if args.limit:
            files = files[:args.limit]
        print(f'\n=== {interval}  files={len(files)}  end={END_TIME}  -> {folder}', flush=True)

        ok = up = empty = nonew = err = 0
        total_added = 0
        t0 = time.time()
        for i, fname in enumerate(files, start=1):
            if i > 1 and (i - 1) % REAUTH_EVERY == 0:
                try:
                    token = get_access_token(TOTP_TOKEN, TOTP_SECRET)
                    print(f'  [{i:4d}] re-authed', flush=True)
                except Exception as e:
                    print(f'  [{i:4d}] re-auth FAILED: {e}', flush=True)
            symbol = fname.replace('_historical.csv', '')
            path = os.path.join(folder, fname)
            try:
                status, n = update_one(path, symbol, interval, token, end_dt)
            except Exception as e:
                status, n = 'error', 0
                print(f'  [{i:4d}] {symbol}: ERROR {e}', flush=True)
            if   status == 'ok':         ok += 1;    total_added += n
            elif status == 'up_to_date': up += 1
            elif status == 'empty':      empty += 1
            elif status == 'no_new':     nonew += 1
            else:                        err += 1
            if i % 25 == 0 or i == len(files):
                el = time.time() - t0
                print(f'  [{i:4d}/{len(files)}] ok={ok} up_to_date={up} empty={empty} no_new={nonew} err={err}  added={total_added}  ({el:.0f}s)', flush=True)
            if DELAY_BETWEEN_STOCKS > 0:
                time.sleep(DELAY_BETWEEN_STOCKS)
        print(f'  FINAL {interval}: ok={ok} up_to_date={up} empty={empty} no_new={nonew} err={err}  rows_added={total_added}', flush=True)
        grand_summary.append((interval, ok, up, empty, nonew, err, total_added))

    print(f'\n[done] total elapsed: {(time.time()-grand_t0)/60:.1f} min', flush=True)
    for interval, ok, up, empty, nonew, err, total_added in grand_summary:
        print(f'  {interval}: ok={ok} up_to_date={up} empty={empty} no_new={nonew} err={err}  rows_added={total_added}', flush=True)


if __name__ == '__main__':
    main()
