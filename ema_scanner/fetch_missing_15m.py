"""
Fetch 15-minute candles directly from Groww for the small set of stocks that
were missing from the 1m bulk download (so the resampler couldn't build their
15m bars).

Saves to ./data/15m/<SYM>_historical.csv in the same column layout the
resampler produces:  timestamp, open, high, low, close, volume, datetime
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groww_client import get_access_token, fetch_candles
from incremental_fetch import TOTP_TOKEN, TOTP_SECRET


MISSING = ["INFY", "VISHAL"]
START   = "2024-05-14 09:15:00"
END     = "2026-05-19 15:30:00"
DST     = Path(__file__).resolve().parent / "data" / "15m"


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    print(f"[auth] requesting access token...", flush=True)
    token = get_access_token(TOTP_TOKEN, TOTP_SECRET)
    print(f"[auth] OK", flush=True)

    for i, sym in enumerate(MISSING, start=1):
        out_path = DST / f"{sym}_historical.csv"
        if out_path.exists():
            print(f"  [{i}/{len(MISSING)}] {sym:<10} SKIP (already exists)", flush=True)
            continue
        t0 = time.time()
        try:
            df = fetch_candles(sym, "15m", START, END, token,
                               delay_s=0.15, verbose=False)
        except Exception as e:
            print(f"  [{i}/{len(MISSING)}] {sym:<10} ERROR  {e.__class__.__name__}: {e}", flush=True)
            continue
        if df.empty:
            print(f"  [{i}/{len(MISSING)}] {sym:<10} EMPTY  ({time.time()-t0:.1f}s)", flush=True)
            continue
        # Drop open_interest (resampled CSVs don't carry it for equities).
        cols = ["timestamp", "open", "high", "low", "close", "volume", "datetime"]
        df = df[[c for c in cols if c in df.columns]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(out_path, index=False)
        print(f"  [{i}/{len(MISSING)}] {sym:<10} OK     rows={len(df):,}  ({time.time()-t0:.1f}s)", flush=True)

    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
