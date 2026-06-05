"""Live workers for the EMA scanner.

Two entrypoints:

* `python live_workers.py ltp`    — poll LTPs for stocks listed in a watchlist
                                    file. Writes data/live/ltp.parquet every
                                    LTP_POLL_SECONDS. Sleeps outside market hours.

* `python live_workers.py candles --interval 1h|1d`
                                  — single-shot: fetch new bars for every CSV in
                                    data/{1h,1d}/ and append. Use a systemd
                                    timer to run at :15 of every hour (1h) and
                                    once after market close (1d).

Watchlist semantics
-------------------
The watchlist is a plain-text file `data/live/watchlist.txt` containing one
SYMBOL per line. The Streamlit UI writes to this file whenever the user clicks
a row in the scanner — by default it contains all symbols the user has marked
"active". Empty / missing file = poll nothing.

The LTP loop:
  1. Read watchlist (auto-reloads each tick — UI changes take effect within 1
     poll).
  2. Fetch LTP for each symbol via groww_live.fetch_ltp.
  3. Write {symbol -> ltp, ts} to data/live/ltp.parquet (atomic temp+rename).

If `INCLUDE_ALL_FALLBACK=1` is set in the env, an empty watchlist falls back
to "every CSV in data/1d/" so users can still see ticks without explicitly
adding to the watchlist.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from groww_client import get_access_token, fetch_candles                       # noqa: E402
from groww_live import fetch_ltp                                                # noqa: E402

ONE_MINUTE_DIR  = None  # populated below
LAST_1M_FETCH   = {"ts": 0.0}   # mutated by _maybe_fetch_1m


# ── Configuration ────────────────────────────────────────────────────────────
DATA_ROOT       = HERE / "data"
LIVE_DIR        = DATA_ROOT / "live"
LTP_PARQUET     = LIVE_DIR / "ltp.parquet"
WATCHLIST_FILE  = LIVE_DIR / "watchlist.txt"
ONE_MINUTE_DIR  = DATA_ROOT / "1m"

ONE_MIN_INTERVAL_SEC = 60    # how often the LTP loop attempts a 1m refresh

LTP_POLL_SECONDS = float(os.environ.get("LTP_POLL_SECONDS", "2"))
INCLUDE_ALL_FALLBACK = os.environ.get("INCLUDE_ALL_FALLBACK", "0") == "1"
MAX_LTP_CONCURRENT  = int(os.environ.get("LTP_MAX_CONCURRENT", "8"))
REAUTH_EVERY_SEC    = 25 * 60   # Groww access tokens last ~30 min

NSE_OPEN  = (9, 15)
NSE_CLOSE = (15, 30)


# ── Time helpers ─────────────────────────────────────────────────────────────
def now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def is_market_open(ts: datetime | None = None) -> bool:
    ts = ts or now_ist()
    if ts.weekday() >= 5:   # Sat=5, Sun=6
        return False
    open_t  = ts.replace(hour=NSE_OPEN[0],  minute=NSE_OPEN[1],  second=0, microsecond=0)
    close_t = ts.replace(hour=NSE_CLOSE[0], minute=NSE_CLOSE[1], second=0, microsecond=0)
    return open_t <= ts <= close_t


def secs_until_market_open(ts: datetime | None = None) -> float:
    """Seconds to next NSE open. Sleeping this long is safe — re-check after."""
    ts = ts or now_ist()
    target = ts.replace(hour=NSE_OPEN[0], minute=NSE_OPEN[1], second=0, microsecond=0)
    if ts >= target:
        target += timedelta(days=1)
    # Skip weekends
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return (target - ts).total_seconds()


# ── Watchlist ────────────────────────────────────────────────────────────────
def _read_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        syms = [s.strip().upper() for s in WATCHLIST_FILE.read_text().splitlines() if s.strip()]
        if syms:
            return syms
    if INCLUDE_ALL_FALLBACK:
        d1d = DATA_ROOT / "1d"
        if d1d.exists():
            return sorted(f.stem.replace("_historical", "") for f in d1d.glob("*_historical.csv"))
    return []


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _last_1m_ts(symbol: str) -> pd.Timestamp | None:
    """Last stored 1-min bar timestamp for `symbol`, or None if no file."""
    path = ONE_MINUTE_DIR / f"{symbol}_historical.csv"
    if not path.exists():
        return None
    try:
        # Tail-read for speed
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(1024, size)
            f.seek(size - chunk)
            tail = f.read().decode("utf-8", errors="ignore")
        last_line = [l for l in tail.strip().splitlines() if l and not l.startswith("timestamp")]
        if not last_line:
            return None
        cols = last_line[-1].split(",")
        return pd.to_datetime(cols[-1].strip())
    except Exception:
        return None


def _append_1m_csv(symbol: str, df_new: pd.DataFrame) -> int:
    """Append new 1m bars to data/1m/<SYM>_historical.csv, creating the file
    (with header) if it doesn't yet exist."""
    if df_new.empty:
        return 0
    path = ONE_MINUTE_DIR / f"{symbol}_historical.csv"
    ONE_MINUTE_DIR.mkdir(parents=True, exist_ok=True)
    out = df_new.copy()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["timestamp", "open", "high", "low", "close", "volume", "datetime"]
    out = out[[c for c in cols if c in out.columns]]
    write_header = not path.exists()
    out.to_csv(path, mode="a", header=write_header, index=False)
    return len(out)


def _maybe_fetch_1m(symbols: list[str], token: str) -> int:
    """Once per ONE_MIN_INTERVAL_SEC, fetch the latest 1m candles for each
    watchlist symbol since its last stored bar. Returns total bars added."""
    now = time.time()
    if now - LAST_1M_FETCH["ts"] < ONE_MIN_INTERVAL_SEC:
        return 0
    LAST_1M_FETCH["ts"] = now

    n_now = now_ist()
    end_str = n_now.strftime("%Y-%m-%d %H:%M:%S")
    total_added = 0
    for sym in symbols:
        last = _last_1m_ts(sym)
        if last is None:
            # Fresh symbol — just grab the last 30 minutes so the chart has a
            # forming-window basis. Backfill more via a separate one-shot if you
            # want deeper history.
            start = (n_now - timedelta(minutes=30))
        else:
            start = last + timedelta(minutes=1)
        if start >= n_now:
            continue
        start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        try:
            df = fetch_candles(sym, "1minute", start_str, end_str, token, delay_s=0.0)
        except Exception:
            continue
        if df.empty:
            continue
        if last is not None:
            df = df[df["datetime"] > last]
        if df.empty:
            continue
        total_added += _append_1m_csv(sym, df)
    return total_added


# ── LTP poll loop ────────────────────────────────────────────────────────────
def run_ltp_loop() -> None:
    """Long-running. Polls LTPs while market is open, sleeps otherwise."""
    jwt    = os.environ.get("GROWW_TOTP_JWT", "").strip()
    secret = os.environ.get("GROWW_TOTP_SECRET", "").strip()
    if not jwt or not secret:
        # Sleep loop so systemd doesn't restart-crash this every 5s.
        print("[ltp] GROWW_TOTP_JWT / GROWW_TOTP_SECRET not set in env. "
              "Edit /etc/investeq.env and systemctl restart investeq-live-ltp.",
              flush=True)
        while True:
            time.sleep(300)

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ltp] starting. poll={LTP_POLL_SECONDS}s  fallback_all={INCLUDE_ALL_FALLBACK}", flush=True)

    token = None
    token_acquired_at = 0.0

    while True:
        # Outside market hours? sleep till open.
        if not is_market_open():
            wait = secs_until_market_open()
            print(f"[ltp] market closed. sleeping {wait/60:.1f} min till next open.", flush=True)
            time.sleep(min(wait, 600))   # cap a single sleep at 10 min for safety
            continue

        # Re-auth as needed
        if token is None or (time.time() - token_acquired_at) > REAUTH_EVERY_SEC:
            try:
                token = get_access_token(jwt, secret)
                token_acquired_at = time.time()
                print(f"[ltp] re-authed @ {now_ist().strftime('%H:%M:%S')}", flush=True)
            except Exception as e:
                print(f"[ltp] auth FAIL {e!r} — sleeping 30s", flush=True)
                time.sleep(30); continue

        symbols = _read_watchlist()
        if not symbols:
            time.sleep(LTP_POLL_SECONDS); continue

        t0 = time.time()
        ltps = fetch_ltp(symbols, token, max_workers=MAX_LTP_CONCURRENT)
        elapsed = time.time() - t0

        if ltps:
            ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame([{"symbol": s, "ltp": v, "ts": ts} for s, v in ltps.items()])
            _atomic_write_parquet(df, LTP_PARQUET)
            print(f"[ltp] {len(ltps)}/{len(symbols)} ok in {elapsed:.2f}s", flush=True)
        else:
            print(f"[ltp] 0/{len(symbols)} returned (elapsed {elapsed:.2f}s)", flush=True)

        # Piggyback a 1m candle refresh once a minute. Stays in the same process
        # so we share the access token and rate-limit budget.
        added = _maybe_fetch_1m(symbols, token)
        if added:
            print(f"[1m]  +{added} bars across {len(symbols)} symbol(s)", flush=True)

        # Sleep the remainder of the poll window (clamped to >=0.1s).
        rest = LTP_POLL_SECONDS - elapsed
        if rest > 0:
            time.sleep(rest)


# ── 1h / 1d candle refresher (single-shot) ───────────────────────────────────
def run_candle_refresh(interval: str) -> None:
    """Append new bars to every CSV in data/{interval}. Reuses incremental_fetch
    logic but with a dynamic end-time (now, clamped to today's NSE close)."""
    if interval not in ("1d", "1h"):
        raise SystemExit(f"interval must be 1d or 1h, got {interval}")
    jwt    = os.environ["GROWW_TOTP_JWT"]
    secret = os.environ["GROWW_TOTP_SECRET"]
    from incremental_fetch import (
        INTERVAL_FOLDER, default_end_time, update_one, REAUTH_EVERY,
        DELAY_BETWEEN_STOCKS,
    )

    folder = HERE / INTERVAL_FOLDER[interval]
    files = sorted(p for p in folder.glob("*_historical.csv"))
    end_dt = pd.to_datetime(default_end_time())
    print(f"[candles {interval}] starting. files={len(files)}  end={end_dt}", flush=True)

    token = get_access_token(jwt, secret)
    ok = up = err = added = 0
    t0 = time.time()
    for i, path in enumerate(files, start=1):
        if i > 1 and (i - 1) % REAUTH_EVERY == 0:
            try:    token = get_access_token(jwt, secret)
            except Exception as e:  print(f"  [{i:4d}] re-auth FAILED: {e}", flush=True)
        symbol = path.stem.replace("_historical", "")
        try:
            status, n = update_one(str(path), symbol, interval, token, end_dt)
        except Exception as e:
            status, n = "error", 0
            print(f"  [{i:4d}] {symbol}: ERROR {e}", flush=True)
        if   status == "ok":           ok += 1; added += n
        elif status == "up_to_date":   up += 1
        else:                          err += 1
        if i % 50 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  [{i:4d}/{len(files)}] ok={ok} up_to_date={up} err={err}  added={added}  ({el:.0f}s)", flush=True)
        if DELAY_BETWEEN_STOCKS > 0:
            time.sleep(DELAY_BETWEEN_STOCKS)
    print(f"[candles {interval}] done in {(time.time()-t0)/60:.1f} min — ok={ok} up_to_date={up} err={err}  rows_added={added}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ltp")
    cp = sub.add_parser("candles")
    cp.add_argument("--interval", choices=["1d", "1h"], required=True)
    args = ap.parse_args()
    if args.cmd == "ltp":
        run_ltp_loop()
    else:
        run_candle_refresh(args.interval)


if __name__ == "__main__":
    main()
