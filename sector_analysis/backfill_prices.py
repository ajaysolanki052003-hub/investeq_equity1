"""Backfill daily OHLCV for the custom Sector-Analysis universe.

Reads sector_analysis/ref/custom_sectors.csv (Symbol,Sector,Subsector) and, for
every symbol NOT already present in the shared daily feed (ema_scanner/data/1d/
SYMBOL_historical.csv), fetches ~N years of daily candles from Groww and writes
a byte-compatible *_historical.csv (same columns as the existing feed, so both
the scanner and the sector apps read them identically).

Design notes:
  - RESUMABLE: symbols that already have a file are skipped, so re-running (or a
    crashed background run) just continues where it left off.
  - FAST-SKIP dead names: Groww's _fetch_window retries 5x on 4xx (~31s), so a
    delisted/unknown symbol would otherwise burn half a minute. We first do ONE
    lightweight no-retry probe of the last ~45 days; only symbols that return
    candles get the full multi-year fetch.
  - Rate-limit friendly: serial, ~5 req/sec (fetch_candles' own inter-chunk
    delay + a small inter-symbol sleep). Re-auth periodically (token is cached).

Once backfilled, the existing candles-1d refresh keeps these files current — no
separate timer needed if that job iterates the feed directory.

Usage (on the VM, env from /etc/investeq.env):
    python -m sector_analysis.backfill_prices                 # 2 years (default)
    python -m sector_analysis.backfill_prices --years 5
    python -m sector_analysis.backfill_prices --limit 20      # smoke test
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ema_scanner"))
from groww_client import (  # noqa: E402
    get_access_token, fetch_candles, _groww_symbol, GROWW_CANDLES_URL,
)

FEED_DIR = ROOT / "ema_scanner" / "data" / "1d"
MAP_CSV  = ROOT / "sector_analysis" / "ref" / "custom_sectors.csv"


def now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def probe_live(sym: str, token: str, tries: int = 6):
    """Probe the last ~45 days. Returns:
        True  -> has data (fetch it)
        False -> genuine skip (200-empty, or a non-429 4xx = unknown symbol)
        None  -> undetermined (rate-limited past our retries) — leave for a later
                 pass; do NOT mark it dead.
    CRITICAL: a 429 is rate-limiting, NOT a dead symbol — we wait and retry, never
    skip on it (the first cut of this backfill treated 429 as dead and false-skipped
    ~half the live universe)."""
    end = now_ist()
    start = end - timedelta(days=45)
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}",
               "X-API-VERSION": "1.0"}
    params = {"exchange": "NSE", "segment": "CASH",
              "groww_symbol": _groww_symbol(sym, "NSE"),
              "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
              "end_time":   end.strftime("%Y-%m-%d %H:%M:%S"),
              "candle_interval": "1day"}
    for a in range(tries):
        try:
            r = requests.get(GROWW_CANDLES_URL, timeout=20, headers=headers, params=params)
        except requests.RequestException:
            time.sleep(min(8.0, 1.0 * (2 ** a)))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(8.0, 1.0 * (2 ** a)))       # rate/server — wait, retry
            continue
        if r.status_code != 200:
            return False                                # genuine unknown symbol (4xx)
        return bool(r.json().get("payload", {}).get("candles"))
    return None                                         # persistent 429 — undetermined


def _auth() -> str:
    return get_access_token(os.environ["GROWW_TOTP_JWT"],
                            os.environ["GROWW_TOTP_SECRET"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N missing (smoke test)")
    ap.add_argument("--sleep", type=float, default=0.25, help="pause between symbols")
    ap.add_argument("--chunk-delay", type=float, default=0.35, help="pause between fetch chunks")
    args = ap.parse_args()

    symbols = [row["Symbol"].strip().upper()
               for row in csv.DictReader(open(MAP_CSV, encoding="utf-8"))
               if row.get("Symbol", "").strip()]
    # de-dup preserving order
    seen: set[str] = set()
    symbols = [s for s in symbols if not (s in seen or seen.add(s))]

    existing = {p.name[:-len("_historical.csv")]
                for p in FEED_DIR.glob("*_historical.csv")}
    todo = [s for s in symbols if s not in existing]
    if args.limit:
        todo = todo[:args.limit]

    print(f"[plan] universe={len(symbols)}  already have={len(symbols)-len(todo) if not args.limit else '?'}"
          f"  to fetch={len(todo)}  years={args.years}", flush=True)

    token = _auth()
    print("[auth] OK", flush=True)

    start = (now_ist() - timedelta(days=365 * args.years + 10)).strftime("%Y-%m-%d 09:15:00")
    end   = now_ist().strftime("%Y-%m-%d 15:30:00")

    t0 = time.time()
    fetched = skipped = failed = undet = 0
    FEED_DIR.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(todo):
        if i and i % 100 == 0:
            token = _auth()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] fetched={fetched} skipped={skipped} undet={undet} "
                  f"failed={failed} | {el:.0f}s  (~{el/max(1,i)*len(todo)/60:.0f}min total est)",
                  flush=True)
        res = probe_live(sym, token)
        if res is None:      # rate-limited past retries — leave for a later pass
            undet += 1
            continue
        if not res:          # genuine dead / unknown symbol
            skipped += 1
            continue
        try:
            df = fetch_candles(sym, "1d", start, end, token, delay_s=args.chunk_delay)
            if df.empty:
                skipped += 1
                continue
            out = FEED_DIR / f"{sym}_historical.csv"
            tmp = out.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(out)
            fetched += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[fail] {sym}: {e}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    print(f"[done] fetched={fetched} skipped={skipped} undet={undet} failed={failed} "
          f"of {len(todo)} in {time.time()-t0:.0f}s "
          f"(re-run to retry the {undet} undetermined + any skips)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
