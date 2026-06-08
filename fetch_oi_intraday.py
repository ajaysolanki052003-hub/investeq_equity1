"""Intraday OI backfill from Groww historical-candles.

Symbol format (verified against Groww's instruments.csv):
    NSE-NIFTY-DDMmmYY-strike-CE/PE
        DD   2-digit day, leading zero (e.g. 02, 28)
        Mmm  3-letter month, first letter capitalized (Apr, May)
        YY   2-digit year (26)

For each target day:
  1. Read spot for the day → compute range of ATMs touched during session
  2. List candidate strikes = union of ATM±50 across the day
  3. Pick the 2 nearest weekly expiries on/after the day
  4. Fetch 1-min OHLC+OI for each (strike × expiry × CE/PE) instrument
  5. Re-emit into DATA/oi/YYYYMMDD.parquet using the canonical schema:
        timestamp (ISO), expiry, strike, option_type, oi

Run:
    python fetch_oi_intraday.py --date 2026-05-04
    python fetch_oi_intraday.py --flat-days   # do all 29 known-flat days
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent / "ema_scanner"))
from groww_client import get_access_token  # noqa: E402


ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
OI_DIR        = ROOT / "oi"
SPOT_MASTER   = ROOT / "nifty_1m_master.parquet"
INSTR_CSV     = ROOT / "_groww_instruments.csv"

URL  = "https://api.groww.in/v1/historical/candles"
MONS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
STRIKE_STEP = 50


def _spot_for_day(d: date) -> pd.DataFrame:
    """Per-minute spot bars for a single trading day."""
    df = pd.read_parquet(SPOT_MASTER, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    mask = df["timestamp"].dt.date == d
    return df.loc[mask].sort_values("timestamp").reset_index(drop=True)


def _atms_touched(spot: pd.DataFrame, padding: int = 1) -> list[int]:
    """ATM strikes ever touched during the session, padded by ±N strikes."""
    if spot.empty:
        return []
    lo = int(spot["close"].min())
    hi = int(spot["close"].max())
    atm_lo = round(lo / STRIKE_STEP) * STRIKE_STEP - padding * STRIKE_STEP
    atm_hi = round(hi / STRIKE_STEP) * STRIKE_STEP + padding * STRIKE_STEP
    return list(range(atm_lo, atm_hi + 1, STRIKE_STEP))


_RULE_CHANGE = date(2025, 9, 1)   # NSE moved NIFTY weekly Thu → Tue at start of 2025-09

def _next_weekly_expiries(d: date, n: int = 2) -> list[date]:
    """Next N weekly NIFTY expiries on/after d.

    NIFTY weekly expiry day:
      < 2025-09-01: Thursday (weekday=3)
      >= 2025-09-01: Tuesday (weekday=1)

    If both expiries straddle the rule-change boundary, we always pick
    the rule-relevant day. Groww returns empty for any mis-targeted
    expiry and we skip silently."""
    target_wd = 1 if d >= _RULE_CHANGE else 3
    base = d
    while base.weekday() != target_wd:
        base += timedelta(days=1)
    return [base + timedelta(days=7*i) for i in range(n)]


def _gsym(expiry: date, strike: int, opt_type: str) -> str:
    return (f"NSE-NIFTY-{expiry.day:02d}{MONS[expiry.month-1]}"
            f"{str(expiry.year)[2:]}-{strike}-{opt_type}")


def fetch_candles(sym: str, start: str, end: str, token: str,
                  interval: str = "1minute", retries: int = 3) -> list:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json",
               "X-API-VERSION": "1.0"}
    for attempt in range(retries):
        try:
            r = requests.get(URL, headers=headers, timeout=20, params={
                "exchange": "NSE", "segment": "FNO", "groww_symbol": sym,
                "start_time": start, "end_time": end,
                "candle_interval": interval,
            })
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.0 * (2 ** attempt))
                continue
            r.raise_for_status()
            return r.json().get("payload", {}).get("candles") or []
        except requests.RequestException:
            time.sleep(1.0 * (2 ** attempt))
    return []


def fetch_one_day(d: date, token: str,
                  pad_strikes: int = 1,
                  delay: float = 0.4,
                  verbose: bool = True) -> int:
    """Backfill ONE day's intraday OI. Returns the rows written."""
    spot = _spot_for_day(d)
    if spot.empty:
        if verbose: print(f"  {d}  no spot bars — skipping")
        return 0

    strikes  = _atms_touched(spot, padding=pad_strikes)
    expiries = _next_weekly_expiries(d, n=2)
    if verbose:
        print(f"  {d}  strikes={strikes}  expiries={[str(e) for e in expiries]}")

    s_str = f"{d.isoformat()} 09:15:00"
    e_str = f"{d.isoformat()} 15:30:00"

    rows = []
    for exp in expiries:
        for k in strikes:
            for typ in ("CE", "PE"):
                sym = _gsym(exp, k, typ)
                cands = fetch_candles(sym, s_str, e_str, token)
                if not cands:
                    continue
                # Groww 1-min candles report OI only every ~3 mins (None on
                # the other bars). We want one OI value per minute, so:
                #   - forward-fill the last seen OI through intervening minutes
                #   - BACKWARD-fill the leading minutes (before the first real
                #     OI tick) with the first observed value, so we don't drop
                #     09:15..09:17 when the first OI tick lands at 09:18.
                first_real = next(
                    (c[6] for c in cands if len(c) > 6 and c[6] is not None),
                    None)
                if first_real is None:
                    continue  # no OI ever reported for this instrument — skip
                last_oi = int(first_real)
                for c in cands:
                    ts = c[0] if len(c) > 0 else None
                    raw_oi = c[6] if len(c) > 6 else None
                    if raw_oi is not None:
                        last_oi = int(raw_oi)
                    rows.append({
                        "timestamp":   ts,
                        "expiry":      exp.isoformat(),
                        "strike":      int(k),
                        "option_type": typ,
                        "oi":          last_oi,
                    })
                if delay > 0:
                    time.sleep(delay)

    if not rows:
        if verbose: print(f"  {d}  no rows fetched")
        return 0

    df = pd.DataFrame(rows).sort_values(["timestamp", "strike", "option_type"])
    out_path = OI_DIR / f"{d.strftime('%Y%m%d')}.parquet"
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    if verbose:
        print(f"  {d}  wrote {len(df):,} rows -> {out_path.name}")
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",  type=str, default=None,
                    help="Backfill one day (YYYY-MM-DD)")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end",   type=str, default=None)
    ap.add_argument("--flat-days", action="store_true",
                    help="Backfill all known-flat days (1-tick days in the aggregate)")
    ap.add_argument("--pad",   type=int, default=1,
                    help="Padding strikes added on each side of ATM range (default 1)")
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args()

    token = get_access_token(os.environ["GROWW_TOTP_JWT"],
                             os.environ["GROWW_TOTP_SECRET"])
    print("[auth] OK", flush=True)

    if args.flat_days:
        # Discover flat days (≤5 ticks/day in the aggregate). Includes only
        # weekdays with spot data — skips weekends and holidays automatically.
        agg = pd.read_parquet(ROOT / "_atm_oi_intraday.parquet")
        agg["timestamp"] = pd.to_datetime(agg["timestamp"])
        counts = agg.groupby(agg["timestamp"].dt.date).size()
        spot_days = set()
        sp = pd.read_parquet(SPOT_MASTER, columns=["timestamp"])
        sp["timestamp"] = pd.to_datetime(sp["timestamp"])
        spot_days = set(sp["timestamp"].dt.date.unique())
        days = [d for d in sorted(counts[counts <= 5].index) if d in spot_days]
    elif args.date:
        days = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    elif args.start and args.end:
        s = datetime.strptime(args.start, "%Y-%m-%d").date()
        e = datetime.strptime(args.end,   "%Y-%m-%d").date()
        days = []
        cur = s
        while cur <= e:
            if cur.weekday() < 5:
                days.append(cur)
            cur += timedelta(days=1)
    else:
        ap.print_help(); return 1

    print(f"[plan] {len(days)} days to backfill", flush=True)
    t0 = time.time()
    total = 0
    for d in days:
        try:
            total += fetch_one_day(d, token, pad_strikes=args.pad,
                                   delay=args.delay)
        except Exception as e:
            print(f"  {d}  ERROR {e}")
    print(f"[done] {total:,} rows across {len(days)} days "
          f"({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
