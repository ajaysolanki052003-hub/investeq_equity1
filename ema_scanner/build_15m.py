"""
Resample 1-minute OHLC CSVs from `C:/Users/User/Desktop/ajs/data/1m/` into
15-minute candles aligned to the NSE open (09:15 IST), and write them to
`./data/15m/` so the scanner can use them.

A 15-min bin [t, t+15) labelled at t.  Empty bins (off-market) are dropped.
Run:  python build_15m.py
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import pandas as pd


SRC = Path(r"C:/Users/User/Desktop/ajs/data/1m")
DST = Path(__file__).resolve().parent / "data" / "15m"


def resample_one(src_path: Path):
    df = pd.read_csv(src_path)
    if df.empty:
        return 0, 0, None
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    cols = ["open", "high", "low", "close", "volume"]
    # Keep only the regular session (09:15 - 15:30 IST). Pre-open and
    # post-close auction ticks would create extra 15-min bins otherwise.
    mins = df.index.hour * 60 + df.index.minute
    df = df[(mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30)]
    if df.empty:
        return 0, 0, None

    agg = df[cols].resample(
        "15min",
        origin="start_day",
        offset="9h15min",
        label="left",
        closed="left",
    ).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })
    agg = agg.dropna(subset=["open"])

    out = agg.reset_index()
    out["timestamp"] = (out["datetime"].astype("int64") // 10**9).astype("int64")
    out = out[["timestamp", "open", "high", "low", "close", "volume", "datetime"]]
    return len(df), len(out), out


def main() -> int:
    if not SRC.is_dir():
        print(f"ERROR: source not found: {SRC}", file=sys.stderr)
        return 2
    DST.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.glob("*_historical.csv"))
    total = len(files)
    if total == 0:
        print(f"ERROR: no CSVs found under {SRC}", file=sys.stderr)
        return 2

    print(f"Resampling 1m -> 15m for {total} stocks")
    print(f"  src: {SRC}")
    print(f"  dst: {DST}")

    t0 = time.time()
    ok = empty = fail = 0
    for i, src in enumerate(files, start=1):
        try:
            in_rows, out_rows, out = resample_one(src)
            if out_rows == 0:
                empty += 1
                status = "EMPTY"
            else:
                out.to_csv(DST / src.name, index=False)
                ok += 1
                status = f"{out_rows:>6} bars"
        except Exception as e:
            fail += 1
            status = f"FAIL {e.__class__.__name__}"
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i)
        print(f"  [{i:>3}/{total}] {src.stem.replace('_historical','').ljust(14)} "
              f"{status:<18}  elapsed={elapsed/60:5.1f}m  eta={eta/60:5.1f}m")

    dur = time.time() - t0
    print(f"\nDONE  ok={ok}  empty={empty}  fail={fail}  took {dur/60:.1f} min")
    print(f"      output -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
