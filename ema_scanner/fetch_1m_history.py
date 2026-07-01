"""Full 1-minute history gather — fetch ~13 months of 1m candles for the whole
equity universe into data/1m, so the OB+VP volume-profile logic has the
intraday data it needs (POC = busiest price row per day/week/month).

Unlike incremental_fetch.py (which only extends 1d/1h/15m from their last bar),
this does a FULL fetch from a fixed --start to now and OVERWRITES each symbol's
1m file. The pre-existing 1m files on the VM are partial (176/524) and only go
back to ~May 2026 — too shallow for the 12-month monthly lookback — so a clean
deep re-fetch is what's wanted.

Universe = the symbols that already have a data/1d/<SYM>_historical.csv.
Output schema matches the existing 1m files: timestamp,open,high,low,close,volume,datetime
(no open_interest — equity 1m has none).

Usage (run on the VM, where GROWW creds live):
    python fetch_1m_history.py                      # full run, default start
    python fetch_1m_history.py --limit 3            # smoke test (first 3 symbols)
    python fetch_1m_history.py --start 2025-05-01   # custom lookback start
    python fetch_1m_history.py --resume             # skip symbols already deep-fetched
"""
import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from groww_client import get_access_token, fetch_candles

TOTP_TOKEN  = os.environ["GROWW_TOTP_JWT"]
TOTP_SECRET = os.environ["GROWW_TOTP_SECRET"]

HERE       = os.path.dirname(os.path.abspath(__file__))
SRC_FOLDER = os.path.join(HERE, "data", "1d")    # universe source
OUT_FOLDER = os.path.join(HERE, "data", "1m")     # where 1m files are written

# Default lookback start. OB+VP monthly lookback is 12 months; start a margin
# before that so the earliest monthly bucket is complete. Today is mid-2026.
DEFAULT_START = "2025-05-01 09:15:00"

NSE_OPEN  = (9, 15)
NSE_CLOSE = (15, 30)

DELAY_BETWEEN_CHUNKS = 0.15
DELAY_BETWEEN_STOCKS = 0.20   # ~5 req/sec ceiling, matches incremental_fetch
REAUTH_EVERY = 100

OUT_COLS = ["timestamp", "open", "high", "low", "close", "volume", "datetime"]


def now_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def default_end_time() -> str:
    """now in IST, clamped to today's NSE close; pre-open falls back to yesterday."""
    n = now_ist()
    close = n.replace(hour=NSE_CLOSE[0], minute=NSE_CLOSE[1], second=0, microsecond=0)
    openn = n.replace(hour=NSE_OPEN[0],  minute=NSE_OPEN[1],  second=0, microsecond=0)
    if n >= close:
        end = close
    elif n < openn:
        end = (close - timedelta(days=1))
    else:
        end = n
    return end.strftime("%Y-%m-%d %H:%M:%S")


def last_complete_session_end() -> str:
    """End = the most recent FULLY-CLOSED NSE session (post-close philosophy —
    never fetch a partial 'today'). A mid-session run falls back to the prior
    session, so --extend during market hours adds nothing for today."""
    n = now_ist()
    close = n.replace(hour=NSE_CLOSE[0], minute=NSE_CLOSE[1], second=0, microsecond=0)
    d = n if n >= close else (n - timedelta(days=1))
    while d.weekday() >= 5:            # skip Sat/Sun (holidays yield empty fetches, harmless)
        d -= timedelta(days=1)
    return d.replace(hour=NSE_CLOSE[0], minute=NSE_CLOSE[1],
                     second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def first_datetime(path):
    """Parse the datetime of the first data row (used by --resume to tell a
    deep-fetched file apart from a shallow legacy one)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.readline()                 # header
            first = f.readline().strip()
        if not first:
            return None
        return pd.to_datetime(first.split(",")[-1].strip())
    except Exception:
        return None


def last_datetime(path):
    """Parse the datetime of the LAST data row (for --extend incremental append)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END); size = f.tell()
            buf, pos = b"", size
            while pos > 0 and buf.count(b"\n") < 2:
                read = min(512, pos); pos -= read
                f.seek(pos); buf = f.read(read) + buf
        last = buf.decode("utf-8", errors="ignore").strip().split("\n")[-1].strip()
        if not last or last.startswith("timestamp"):
            return None
        return pd.to_datetime(last.split(",")[-1].strip())
    except Exception:
        return None


def fetch_one(symbol, start_str, end_str, token):
    df = fetch_candles(symbol, "1m", start_str, end_str, token,
                       delay_s=DELAY_BETWEEN_CHUNKS, verbose=False)
    if df.empty:
        return df, 0
    out = df[OUT_COLS].copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out, len(out)


def compute_fetch(symbol, path, args, start_str, end_str, end_dt, token):
    """Return (out_df, n_rows, write_mode). write_mode 'a'=append (extend),
    'w'=overwrite (full), or None='up to date, nothing to do'."""
    if args.extend and os.path.exists(path):
        last_dt = last_datetime(path)
        if last_dt is not None:
            if last_dt + timedelta(minutes=1) >= end_dt:
                return None, 0, None                         # already current
            fstart = (last_dt + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
            out, n = fetch_one(symbol, fstart, end_str, token)
            if n:
                out = out[pd.to_datetime(out["datetime"]) > last_dt]
                n = len(out)
            return out, n, "a"
    out, n = fetch_one(symbol, start_str, end_str, token)     # full deep (over)write
    return out, n, "w"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", default=DEFAULT_START,
                    help="lookback start 'YYYY-MM-DD [HH:MM:SS]'")
    ap.add_argument("--resume", action="store_true",
                    help="skip symbols whose 1m file already starts at/near --start")
    ap.add_argument("--extend", action="store_true",
                    help="append only new complete sessions to each existing 1m file "
                         "(incremental; no full re-fetch)")
    args = ap.parse_args()

    os.makedirs(OUT_FOLDER, exist_ok=True)
    start_str = args.start if ":" in args.start else f"{args.start} 09:15:00"
    start_dt  = pd.to_datetime(start_str)
    # Post-close philosophy: end at the last fully-closed session, never a
    # partial 'today'. So --extend run after 15:30 IST adds today; run during
    # market hours it adds nothing for today (only backfills prior sessions).
    end_str   = last_complete_session_end()
    end_dt    = pd.to_datetime(end_str)

    files = sorted(f for f in os.listdir(SRC_FOLDER) if f.endswith("_historical.csv"))
    symbols = [f.replace("_historical.csv", "") for f in files]
    if args.limit:
        symbols = symbols[:args.limit]

    print(f"[auth] requesting access token...", flush=True)
    token = get_access_token(TOTP_TOKEN, TOTP_SECRET)
    print(f"[auth] OK", flush=True)
    span = f"{start_str} -> {end_str}" if not args.extend else f"extend -> {end_str}"
    print(f"=== 1m {'extend' if args.extend else 'gather'}  symbols={len(symbols)}  {span}  -> {OUT_FOLDER}", flush=True)

    ok = empty = err = skip = 0
    total_rows = 0
    t0 = time.time()
    for i, symbol in enumerate(symbols, start=1):
        if i > 1 and (i - 1) % REAUTH_EVERY == 0:
            try:
                token = get_access_token(TOTP_TOKEN, TOTP_SECRET)
                print(f"  [{i:4d}] re-authed", flush=True)
            except Exception as e:
                print(f"  [{i:4d}] re-auth FAILED: {e}", flush=True)

        path = os.path.join(OUT_FOLDER, f"{symbol}_historical.csv")

        if args.resume and os.path.exists(path):
            fd = first_datetime(path)
            # already deep-fetched (file begins within a week of the target start)
            if fd is not None and fd <= start_dt + timedelta(days=7):
                skip += 1
                if i % 25 == 0 or i == len(symbols):
                    print(f"  [{i:4d}/{len(symbols)}] ok={ok} skip={skip} empty={empty} err={err} rows={total_rows} ({time.time()-t0:.0f}s)", flush=True)
                continue

        try:
            out, n, wmode = compute_fetch(symbol, path, args, start_str, end_str, end_dt, token)
            if wmode is None:
                skip += 1                      # already current (extend, up to date)
            elif n == 0:
                empty += 1
            else:
                out.to_csv(path, mode=wmode, header=(wmode == "w"), index=False)
                ok += 1
                total_rows += n
        except Exception as e:
            err += 1
            print(f"  [{i:4d}] {symbol}: ERROR {e}", flush=True)

        if i % 25 == 0 or i == len(symbols):
            el = time.time() - t0
            print(f"  [{i:4d}/{len(symbols)}] ok={ok} skip={skip} empty={empty} err={err} rows={total_rows} ({el:.0f}s)", flush=True)
        if DELAY_BETWEEN_STOCKS > 0:
            time.sleep(DELAY_BETWEEN_STOCKS)

    print(f"\n[done] {(time.time()-t0)/60:.1f} min  ok={ok} skip={skip} empty={empty} err={err}  rows_added={total_rows}", flush=True)


if __name__ == "__main__":
    main()
