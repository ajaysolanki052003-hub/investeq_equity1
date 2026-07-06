"""Backfill 1h history for scanner symbols that have a 1d file but no 1h file.

NSE 1h buckets are :15-aligned (09:15, 10:15, ...). Groww's native 1hour candles
are wall-clock aligned, so — exactly like incremental_fetch.py — we fetch 1-minute
bars and roll them up with resample_1m_to_1h_nse(). Output schema matches the
existing 1h files: timestamp,open,high,low,close,volume,open_interest,datetime
(datetime '%Y-%m-%d %H:%M:%S'). Resumable (skips symbols that already have a 1h
file); a 429-safe probe skips genuinely-dead symbols fast.

Run (VM, env from /etc/investeq.env):
    python -m ema_scanner.backfill_1h --months 12
    python -m ema_scanner.backfill_1h --only IXIGO,MOREPENLAB      # specific
    python -m ema_scanner.backfill_1h --months 12 --limit 20       # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from groww_client import (  # noqa: E402
    get_access_token, fetch_candles, _groww_symbol, GROWW_CANDLES_URL,
)
from incremental_fetch import resample_1m_to_1h_nse  # noqa: E402

DATA = HERE / "data"
D1D = DATA / "1d"
D1H = DATA / "1h"


def now_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def probe_live(sym, token, tries=6):
    """Last ~10 sessions of 1m. True=has data, False=dead(4xx/200-empty),
    None=undetermined (persistent 429). Never treats 429 as dead."""
    end = now_ist()
    start = end - timedelta(days=14)
    h = {"Accept": "application/json", "Authorization": f"Bearer {token}", "X-API-VERSION": "1.0"}
    p = {"exchange": "NSE", "segment": "CASH", "groww_symbol": _groww_symbol(sym, "NSE"),
         "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
         "end_time": end.strftime("%Y-%m-%d %H:%M:%S"), "candle_interval": "1minute"}
    for a in range(tries):
        try:
            r = requests.get(GROWW_CANDLES_URL, timeout=20, headers=h, params=p)
        except requests.RequestException:
            time.sleep(min(8.0, 2 ** a)); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(8.0, 2 ** a)); continue
        if r.status_code != 200:
            return False
        return bool(r.json().get("payload", {}).get("candles"))
    return None


def write_1h(sym, token, start, end, chunk_delay):
    """Fetch 1m -> roll to :15 1h -> write the file. Returns rows written."""
    df_1m = fetch_candles(sym, "1m", start, end, token, delay_s=chunk_delay)
    if df_1m.empty:
        return 0
    h1 = resample_1m_to_1h_nse(df_1m)
    if h1.empty:
        return 0
    h1 = h1.copy()
    h1["datetime"] = h1["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["timestamp", "open", "high", "low", "close", "volume", "open_interest", "datetime"]
    out = D1H / f"{sym}_historical.csv"
    tmp = out.with_suffix(".csv.tmp")
    h1[cols].to_csv(tmp, index=False)
    tmp.replace(out)
    return len(h1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--only", type=str, default="", help="comma list of symbols")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--chunk-delay", type=float, default=0.3)
    args = ap.parse_args()

    have_1d = {p.name[:-len("_historical.csv")] for p in D1D.glob("*_historical.csv")}
    have_1h = {p.name[:-len("_historical.csv")] for p in D1H.glob("*_historical.csv")}
    if args.only:
        todo = [s.strip().upper() for s in args.only.split(",") if s.strip()]
    else:
        todo = sorted(have_1d - have_1h)
    if args.limit:
        todo = todo[:args.limit]

    print(f"[plan] 1d={len(have_1d)} 1h={len(have_1h)} to-fetch={len(todo)} months={args.months}", flush=True)
    token = get_access_token(os.environ["GROWW_TOTP_JWT"], os.environ["GROWW_TOTP_SECRET"])
    print("[auth] OK", flush=True)

    start = (now_ist() - timedelta(days=int(args.months * 30.5) + 5)).strftime("%Y-%m-%d 09:15:00")
    end = now_ist().strftime("%Y-%m-%d 15:30:00")

    D1H.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fetched = skipped = undet = failed = 0
    for i, sym in enumerate(todo):
        if i and i % 50 == 0:
            token = get_access_token(os.environ["GROWW_TOTP_JWT"], os.environ["GROWW_TOTP_SECRET"])
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] fetched={fetched} skipped={skipped} undet={undet} "
                  f"failed={failed} | {el:.0f}s (~{el/max(1,i)*len(todo)/60:.0f}min total)", flush=True)
        res = probe_live(sym, token)
        if res is None:
            undet += 1; continue
        if not res:
            skipped += 1; continue
        try:
            n = write_1h(sym, token, start, end, args.chunk_delay)
            if n:
                fetched += 1
                print(f"[OK] {sym:14} {n} 1h bars", flush=True)
            else:
                skipped += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[fail] {sym}: {e}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    print(f"[done] fetched={fetched} skipped={skipped} undet={undet} failed={failed} "
          f"of {len(todo)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
