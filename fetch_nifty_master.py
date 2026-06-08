"""Fetch continuous NIFTY 50 index 1-minute history and maintain a single master
parquet at DATA/nifty_1m_master.parquet.

Initial run: fetches up to 5 years of 1-min bars back from today.
Subsequent runs: append-only — fetch from (last_ts + 1m) up to now.

The master file is what the chain-replay INDEX view reads. Resampling to 5m /
15m / 1h / 1d happens on-demand in the FastAPI app.

Run:
    python fetch_nifty_master.py                 # incremental update
    python fetch_nifty_master.py --years 5       # force a 5-year backfill (no master file yet)
    python fetch_nifty_master.py --rebuild       # wipe master, refetch 5y from scratch

Env (from /etc/investeq.env on the VM):
    GROWW_TOTP_JWT, GROWW_TOTP_SECRET
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# groww_client lives under ema_scanner/ — add it to the import path.
sys.path.insert(0, str(Path(__file__).parent / "ema_scanner"))
from groww_client import get_access_token, fetch_candles  # noqa: E402


ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
MASTER = ROOT / "nifty_1m_master.parquet"

# NIFTY 50 on Groww. The historical-candles endpoint accepts groww_symbol form
# `NSE-NIFTY` with segment `CASH` for the NSE NIFTY 50 index.
SYMBOL  = "NIFTY"
EXCHANGE = "NSE"
SEGMENT  = "CASH"

NSE_OPEN  = (9, 15)
NSE_CLOSE = (15, 30)


def now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def session_end_for(today: datetime) -> datetime:
    """End-time clamped to NSE close: today's close if past close, else now,
    else yesterday's close if before today's open."""
    close = today.replace(hour=NSE_CLOSE[0], minute=NSE_CLOSE[1], second=0, microsecond=0)
    openn = today.replace(hour=NSE_OPEN[0],  minute=NSE_OPEN[1],  second=0, microsecond=0)
    if today >= close:
        return close
    if today < openn:
        return close - timedelta(days=1)
    return today


def _load_master() -> pd.DataFrame:
    if not MASTER.exists():
        return pd.DataFrame()
    df = pd.read_parquet(MASTER)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _save_master(df: pd.DataFrame) -> None:
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    tmp = MASTER.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(MASTER)


def fetch_range(start_dt: datetime, end_dt: datetime, token: str, verbose: bool = True) -> pd.DataFrame:
    """Fetch 1-minute NIFTY candles for [start_dt, end_dt]. Chunks of 29 days
    are handled inside groww_client.fetch_candles."""
    if start_dt >= end_dt:
        return pd.DataFrame()
    s = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    e = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    if verbose:
        print(f"[fetch] {SYMBOL} 1m  {s} -> {e}", flush=True)
    df = fetch_candles(
        SYMBOL, "1m", s, e, token,
        exchange=EXCHANGE, segment=SEGMENT,
        delay_s=0.2, verbose=verbose,
    )
    if df.empty:
        return df
    df = df[["timestamp", "open", "high", "low", "close", "volume", "datetime"]].copy()
    df = df.rename(columns={"datetime": "ts"})
    df["timestamp"] = pd.to_datetime(df["ts"])
    df = df.drop(columns=["ts"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    # Keep only inside NSE session (drop any pre-market/after-hours bars Groww may emit)
    df = df[
        ((df["timestamp"].dt.time >= pd.Timestamp("09:15").time())
         & (df["timestamp"].dt.time <= pd.Timestamp("15:30").time()))
    ]
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="first").reset_index(drop=True)


def _fill_gaps(min_gap_days: int, verbose: bool) -> int:
    """Detect day-level gaps in the master and refetch those windows.
    A 'gap' here is a span >= min_gap_days calendar days between two
    consecutive present days (so normal weekends — 3-day gap Fri->Mon —
    pass with the default min_gap_days=3 actually do NOT, set 4 to skip)."""
    df = _load_master()
    if df.empty:
        print("[fill] no master file — run a normal fetch first.", flush=True)
        return 1
    days = sorted(df["timestamp"].dt.normalize().unique())
    gaps = []
    for i in range(1, len(days)):
        delta = (days[i] - days[i-1]).days
        if delta >= min_gap_days:
            gaps.append((days[i-1].to_pydatetime() + timedelta(days=1, hours=9, minutes=15),
                         days[i].to_pydatetime() + timedelta(hours=15, minutes=30) - timedelta(days=1)))
    if not gaps:
        print(f"[fill] no gaps >= {min_gap_days} days found.", flush=True)
        return 0
    print(f"[fill] found {len(gaps)} gap windows >= {min_gap_days} days. Refetching...", flush=True)

    print("[auth] requesting Groww access token...", flush=True)
    token = get_access_token(
        os.environ["GROWW_TOTP_JWT"],
        os.environ["GROWW_TOTP_SECRET"],
    )
    print("[auth] OK", flush=True)

    all_new = []
    for i, (s, e) in enumerate(gaps, start=1):
        print(f"  [{i}/{len(gaps)}] {s} -> {e}", flush=True)
        try:
            sub = fetch_range(s, e, token, verbose=verbose)
            if not sub.empty:
                all_new.append(sub)
                print(f"     +{len(sub):,} bars", flush=True)
            else:
                print(f"     (no bars returned)", flush=True)
        except Exception as ex:
            print(f"     FAILED: {ex}", flush=True)
        time.sleep(0.3)

    if not all_new:
        print("[fill] no new bars across all gaps.", flush=True)
        return 0
    new = pd.concat(all_new, ignore_index=True)
    merged = pd.concat([df, new], ignore_index=True)
    n_before = len(df)
    _save_master(merged)
    final = _load_master()
    print(f"[save] rows={len(final):,} (+{len(final) - n_before:,})", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years",   type=int, default=5,
                    help="Backfill horizon when master file is missing (default 5)")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete master file and refetch from scratch")
    ap.add_argument("--fill-gaps", action="store_true",
                    help="Detect missing windows in the master and refetch them only")
    ap.add_argument("--gap-days", type=int, default=3,
                    help="Treat any gap >= this many calendar days as missing (default 3)")
    ap.add_argument("--quiet",   action="store_true")
    args = ap.parse_args()
    verbose = not args.quiet

    if args.rebuild and MASTER.exists():
        print(f"[init] rebuild requested — removing {MASTER}", flush=True)
        MASTER.unlink()

    if args.fill_gaps:
        return _fill_gaps(args.gap_days, verbose)

    end_dt = session_end_for(now_ist())

    existing = pd.DataFrame() if args.rebuild else _load_master()
    if existing.empty:
        # First-ever fetch: go back `--years` from today.
        start_dt = end_dt - timedelta(days=365 * args.years)
        # Groww availability is from 2020 onwards.
        floor = datetime(2020, 1, 1, 9, 15)
        if start_dt < floor:
            start_dt = floor
        print(f"[init] no master file — backfilling {args.years}yr from {start_dt}", flush=True)
    else:
        last_ts = existing["timestamp"].iloc[-1].to_pydatetime()
        start_dt = last_ts + timedelta(minutes=1)
        print(f"[incr] master has {len(existing):,} rows, last = {last_ts}", flush=True)
        if start_dt >= end_dt:
            print("[incr] already up-to-date — nothing to fetch.", flush=True)
            return 0

    print("[auth] requesting Groww access token...", flush=True)
    token = get_access_token(
        os.environ["GROWW_TOTP_JWT"],
        os.environ["GROWW_TOTP_SECRET"],
    )
    print("[auth] OK", flush=True)

    t0 = time.time()
    new = fetch_range(start_dt, end_dt, token, verbose=verbose)
    if new.empty:
        print("[fetch] no new bars returned.", flush=True)
        return 0

    if existing.empty:
        merged = new
    else:
        merged = pd.concat([existing, new], ignore_index=True)

    n_before = len(existing)
    _save_master(merged)
    final = _load_master()
    n_after = len(final)
    print(f"[save] wrote {MASTER}  rows={n_after:,} (+{n_after - n_before:,})", flush=True)
    print(f"[done] elapsed {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
