"""Per-day option + spot 1-minute OHLCV backfill (Volatility Desk data feed).

Sibling of fetch_oi_intraday.py — same Groww historical-candles source and the
same strike/expiry targeting, but it keeps the OHLCV of every leg (the OI
fetcher kept only column 6, the OI) and writes the two per-day files the
straddle replay / Volatility Desk reads:

    DATA/spot/YYYYMMDD.parquet     timestamp, open, high, low, close, volume
    DATA/options/YYYYMMDD.parquet  timestamp, expiry, strike, option_type,
                                   open, high, low, close, volume

Groww 1-min candle row = [ts, open, high, low, close, volume, oi].

Run:
    python fetch_option_ohlcv.py --date 2026-05-04
    python fetch_option_ohlcv.py --catchup            # last options file -> today
    python fetch_option_ohlcv.py --start 2026-05-01 --end 2026-06-15
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# Reuse the OI fetcher's verified primitives (auth, symbol build, candle fetch).
from fetch_oi_intraday import (
    ROOT, SPOT_MASTER,
    _atms_touched, _next_weekly_expiries, _gsym, fetch_candles,
)
sys.path.insert(0, str(Path(__file__).parent / "ema_scanner"))
from groww_client import get_access_token  # noqa: E402

OPT_DIR  = ROOT / "options"
SPOT_DIR = ROOT / "spot"


def _spot_day_ohlcv(d: date) -> pd.DataFrame:
    """Full OHLCV spot bars for one day, straight from the NIFTY 1m master."""
    df = pd.read_parquet(SPOT_MASTER)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    day = df.loc[df["timestamp"].dt.date == d].sort_values("timestamp")
    keep = [c for c in ("timestamp", "open", "high", "low", "close", "volume")
            if c in day.columns]
    return day[keep].reset_index(drop=True)


def fetch_one_day(d: date, token: str, pad_strikes: int = 3,
                  delay: float = 0.4, verbose: bool = True) -> int:
    """Write one day's spot + option OHLCV. Returns option rows written."""
    spot = _spot_day_ohlcv(d)
    if spot.empty:
        if verbose:
            print(f"  {d}  no spot bars — skipping")
        return 0

    # 1) per-day spot file (cheap, no API call)
    SPOT_DIR.mkdir(parents=True, exist_ok=True)
    sp_out = SPOT_DIR / f"{d.strftime('%Y%m%d')}.parquet"
    sp_tmp = sp_out.with_suffix(".parquet.tmp")
    spot.to_parquet(sp_tmp, index=False)
    sp_tmp.replace(sp_out)

    # 2) options: ATM band across the session, padded for overnight gaps, the
    #    two nearest weekly expiries (the straddle enters the day before expiry
    #    and exits on expiry morning at the SAME strike, so the band must be
    #    wide enough to still contain the prior-day ATM on expiry morning).
    strikes  = _atms_touched(spot.rename(columns={"close": "close"}),
                             padding=pad_strikes)
    expiries = _next_weekly_expiries(d, n=2)
    if verbose:
        print(f"  {d}  strikes={strikes}  expiries={[str(e) for e in expiries]}")

    s_str = f"{d.isoformat()} 09:15:00"
    e_str = f"{d.isoformat()} 15:30:00"

    rows = []
    for exp in expiries:
        for k in strikes:
            for typ in ("CE", "PE"):
                cands = fetch_candles(_gsym(exp, k, typ), s_str, e_str, token)
                for c in cands:
                    if len(c) < 6:
                        continue
                    rows.append({
                        "timestamp":   c[0],
                        "expiry":      exp.isoformat(),
                        "strike":      int(k),
                        "option_type": typ,
                        "open":        c[1],
                        "high":        c[2],
                        "low":         c[3],
                        "close":       c[4],
                        "volume":      c[5],
                    })
                if delay > 0:
                    time.sleep(delay)

    if not rows:
        if verbose:
            print(f"  {d}  no option rows fetched (spot file still written)")
        return 0

    df = (pd.DataFrame(rows)
          .sort_values(["timestamp", "expiry", "strike", "option_type"]))
    out = OPT_DIR / f"{d.strftime('%Y%m%d')}.parquet"
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    if verbose:
        print(f"  {d}  wrote {len(df):,} option rows -> {out.name}")
    return len(df)


def _today_ist() -> date:
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",  type=str, default=None, help="One day (YYYY-MM-DD)")
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--catchup", action="store_true",
                    help="Every weekday from the last options file's date to today")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end",   type=str, default=None)
    ap.add_argument("--pad",   type=int, default=3,
                    help="Strikes padded each side of the day's ATM band (default 3)")
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args()

    token = get_access_token(os.environ["GROWW_TOTP_JWT"],
                             os.environ["GROWW_TOTP_SECRET"])
    print("[auth] OK", flush=True)

    if args.today:
        d = _today_ist()
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        days = [d]
    elif args.catchup:
        existing = sorted(OPT_DIR.glob("*.parquet"))
        if not existing:
            print("[fatal] no existing options files — use --date or --start/--end first.")
            return 1
        last_have = datetime.strptime(existing[-1].stem, "%Y%m%d").date()
        today = _today_ist()
        days, cur = [], last_have + timedelta(days=1)
        while cur <= today:
            if cur.weekday() < 5:
                days.append(cur)
            cur += timedelta(days=1)
        if not days:
            print(f"[catchup] already current (last options file = {last_have}).")
            return 0
    elif args.date:
        days = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    elif args.start and args.end:
        s = datetime.strptime(args.start, "%Y-%m-%d").date()
        e = datetime.strptime(args.end,   "%Y-%m-%d").date()
        days, cur = [], s
        while cur <= e:
            if cur.weekday() < 5:
                days.append(cur)
            cur += timedelta(days=1)
    else:
        ap.print_help()
        return 1

    print(f"[plan] {len(days)} days", flush=True)
    t0, total = time.time(), 0
    for d in days:
        try:
            total += fetch_one_day(d, token, pad_strikes=args.pad, delay=args.delay)
        except Exception as e:
            print(f"  {d}  ERROR {e}")
    print(f"[done] {total:,} option rows across {len(days)} days "
          f"({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
