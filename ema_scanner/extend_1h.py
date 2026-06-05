"""
Extend the 09:15-aligned 1h CSVs in ./data/1h/ forward to a target end time.

We fetch native 15-min candles from Groww (because 15m starts exactly at 09:15
each day), resample them to 1h with bins starting at the market open, and
append the new bars to each per-stock CSV.

Run:  python extend_1h.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groww_client import get_access_token, fetch_candles
from incremental_fetch import TOTP_TOKEN, TOTP_SECRET


DATA_1H     = Path(__file__).resolve().parent / "data" / "1h"
FETCH_START = pd.Timestamp("2026-05-15 09:15:00")
FETCH_END   = pd.Timestamp("2026-05-20 15:30:00")
REAUTH_EVERY = 100


def parse_last_datetime(path: Path) -> pd.Timestamp | None:
    """Read the last `datetime` value from the tail of a CSV."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return None
        f.seek(max(0, size - 512))
        buf = f.read().decode("utf-8", errors="ignore").strip()
    last_line = buf.split("\n")[-1]
    if not last_line or last_line.startswith("timestamp"):
        return None
    try:
        return pd.to_datetime(last_line.split(",")[-1].strip())
    except Exception:
        return None


def resample_15m_to_1h_915(df15: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-min bars to 1h aligned to the 09:15 market open."""
    if df15.empty:
        return df15
    out = df15.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.set_index("datetime").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    # Keep regular session only — drop pre-open and the closing-auction tick.
    mins = out.index.hour * 60 + out.index.minute
    out = out[(mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30)]
    if out.empty:
        return out

    agg = out[["open", "high", "low", "close", "volume"]].resample(
        "1h",
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
    # Keep only buckets whose label is inside the session
    mins = agg.index.hour * 60 + agg.index.minute
    agg = agg[(mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30)]
    return agg


def append_rows(path: Path, new_rows: pd.DataFrame) -> int:
    """Append `new_rows` to `path`, matching its column order exactly."""
    if new_rows.empty:
        return 0
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    cols = header.split(",")
    out = new_rows.reset_index()
    out["timestamp"] = (out["datetime"].astype("int64") // 10**9).astype("int64")
    out["datetime"]  = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "open_interest" in cols and "open_interest" not in out.columns:
        out["open_interest"] = ""
    missing = [c for c in cols if c not in out.columns]
    for c in missing:
        out[c] = ""
    out = out[cols]
    out.to_csv(path, mode="a", header=False, index=False)
    return len(out)


def main() -> int:
    files = sorted(DATA_1H.glob("*_historical.csv"))
    if not files:
        print(f"ERROR: no 1h CSVs under {DATA_1H}", file=sys.stderr)
        return 2
    total = len(files)
    print(f"Extending {total} 1h CSVs through {FETCH_END}", flush=True)
    print(f"  data: {DATA_1H}", flush=True)

    print("[auth] requesting access token...", flush=True)
    token = get_access_token(TOTP_TOKEN, TOTP_SECRET)
    print("[auth] OK", flush=True)

    ok = up = empty = fail = 0
    rows_total = 0
    t0 = time.time()
    for i, p in enumerate(files, start=1):
        if i > 1 and (i - 1) % REAUTH_EVERY == 0:
            try:
                token = get_access_token(TOTP_TOKEN, TOTP_SECRET)
                print(f"  [{i:>3}/{total}] re-authed", flush=True)
            except Exception as e:
                print(f"  [{i:>3}/{total}] re-auth FAILED: {e}", flush=True)

        sym = p.stem.replace("_historical", "")
        last_dt = parse_last_datetime(p)
        if last_dt is None:
            fail += 1
            print(f"  [{i:>3}/{total}] {sym:<14} parse-last failed", flush=True)
            continue
        # Start the 15m fetch from the day after the last 1h bar's calendar day.
        next_session = pd.Timestamp(last_dt.date()) + pd.Timedelta(days=1)
        fetch_start = max(FETCH_START, next_session.replace(hour=9, minute=15))
        if fetch_start >= FETCH_END:
            up += 1
            continue
        try:
            df15 = fetch_candles(
                sym, "15m",
                fetch_start.strftime("%Y-%m-%d %H:%M:%S"),
                FETCH_END.strftime("%Y-%m-%d %H:%M:%S"),
                token, delay_s=0.10, verbose=False,
            )
        except Exception as e:
            fail += 1
            print(f"  [{i:>3}/{total}] {sym:<14} fetch FAIL: {e.__class__.__name__}", flush=True)
            continue
        if df15.empty:
            empty += 1
            continue
        hourly = resample_15m_to_1h_915(df15)
        hourly = hourly[hourly.index > last_dt]
        if hourly.empty:
            empty += 1
            continue
        n = append_rows(p, hourly)
        ok += 1
        rows_total += n
        if i % 25 == 0 or i == total:
            el = time.time() - t0
            print(f"  [{i:>3}/{total}] ok={ok} up_to_date={up} empty={empty} "
                  f"fail={fail}  rows_added={rows_total}  ({el:.0f}s)", flush=True)

    dur = (time.time() - t0) / 60
    print(f"\nDONE  ok={ok}  up_to_date={up}  empty={empty}  fail={fail}  "
          f"rows_added={rows_total}  took {dur:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
