"""Headless full-universe EMA21/50 swing-low scan, written for cron use.

Runs the EXACT same `signals_on_day()` function the Streamlit UI calls and
persists the result to `data/scan_cache/scan_<YYYY-MM-DD>.parquet` so the UI
can serve it instantly when the user picks that date.

CLI:
    python scan_universe.py                       # today (IST)
    python scan_universe.py --date 2026-06-02
    python scan_universe.py --last-n 30           # backfill last N trading days
    python scan_universe.py --last-n 30 --skip-existing
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Pure library — no Streamlit dependency on the cron side.
from scan_lib import (    # noqa: E402
    DEFAULT_EMA_PARAMS, SCAN_COLS, list_symbols, signals_on_day,
)
from telegram_alerts import maybe_alert_1h  # noqa: E402

DATA_ROOT_DEFAULT = os.path.abspath(os.path.join(
    HERE.parent, "ema_scanner", "data"))
TIMEFRAMES = ["1d", "1h"]
EMA_PARAMS = DEFAULT_EMA_PARAMS
_SCAN_CACHE_DIR = Path(DATA_ROOT_DEFAULT) / "scan_cache"


def _scan_cache_path(target_day) -> Path:
    return _SCAN_CACHE_DIR / f"scan_{target_day}.parquet"


def _save_scan_cache(target_day, max_sl_pct, df1d, df1h) -> None:
    _SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    d = pd.concat([df1d.assign(_split="1d"),
                   df1h.assign(_split="1h")], ignore_index=True)
    d.to_parquet(_scan_cache_path(target_day), index=False)


def scan_for_day(target_day: date, data_root: str = DATA_ROOT_DEFAULT,
                 max_workers: int = 16, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full universe scan for one date. Returns (df1d, df1h)."""
    rows_per_tf: dict[str, list[dict]] = {tf: [] for tf in TIMEFRAMES}
    jobs = []
    for tf in TIMEFRAMES:
        p = EMA_PARAMS[tf]
        for sym in list_symbols(data_root, tf):
            jobs.append((tf, sym, p))
    n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_tf = {
            ex.submit(signals_on_day, sym, tf, data_root, target_day,
                      p["fast"], p["slow"], p["compulsory"]): (tf, sym)
            for tf, sym, p in jobs
        }
        done = 0
        for fut in as_completed(future_to_tf):
            tf, sym = future_to_tf[fut]
            try:
                rows_per_tf[tf].extend(fut.result())
            except Exception:
                n_err += 1
            done += 1
            if verbose and (done % 50 == 0 or done == len(jobs)):
                print(f"  [{done:4d}/{len(jobs)}]  1d={len(rows_per_tf['1d']):4d}  "
                      f"1h={len(rows_per_tf['1h']):4d}  err={n_err}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
    df1d = pd.DataFrame(rows_per_tf["1d"], columns=SCAN_COLS)
    df1h = pd.DataFrame(rows_per_tf["1h"], columns=SCAN_COLS)
    return df1d, df1h


def trading_days_in_window(end_day: date, n_days: int,
                           data_root: str = DATA_ROOT_DEFAULT) -> list[date]:
    """Last `n_days` trading days (NSE), inferred from any 1d CSV in the data
    root. Falls back to weekdays within the window if no data exists."""
    folder = os.path.join(data_root, "1d")
    sample = next((p for p in os.listdir(folder) if p.endswith("_historical.csv")), None)
    if sample:
        df = pd.read_csv(os.path.join(folder, sample), usecols=["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        days = sorted(set(df["datetime"].dt.date))
        days = [d for d in days if d <= end_day]
        return days[-n_days:]
    # Fallback: weekdays only
    out = []
    d = end_day
    while len(out) < n_days:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return list(reversed(out))


def _now_ist_date() -> date:
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="Target date YYYY-MM-DD. Defaults to today (IST).")
    ap.add_argument("--last-n", type=int, default=None,
                    help="Backfill: scan last N trading days ending today.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="With --last-n, skip dates already cached.")
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    _SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.last_n:
        end_day = pd.to_datetime(args.date).date() if args.date else _now_ist_date()
        days = trading_days_in_window(end_day, args.last_n, args.data_root)
        print(f"[backfill] {len(days)} days: {days[0]} → {days[-1]}", flush=True)
        for i, d in enumerate(days, 1):
            p = _scan_cache_path(d)
            if args.skip_existing and p.exists():
                print(f"  [{i:2d}/{len(days)}] {d}  cached — skip", flush=True)
                continue
            print(f"  [{i:2d}/{len(days)}] {d}  scanning ...", flush=True)
            df1d, df1h = scan_for_day(d, args.data_root,
                                       max_workers=args.workers, verbose=False)
            _save_scan_cache(d, None, df1d, df1h)
            print(f"     saved → {p.name}  ({len(df1d)} 1d, {len(df1h)} 1h)",
                  flush=True)
    else:
        d = pd.to_datetime(args.date).date() if args.date else _now_ist_date()
        print(f"[scan] target_day={d}", flush=True)
        df1d, df1h = scan_for_day(d, args.data_root, max_workers=args.workers)
        _save_scan_cache(d, None, df1d, df1h)
        print(f"[scan] saved → {_scan_cache_path(d).name}  "
              f"({len(df1d)} 1d signals, {len(df1h)} 1h signals)", flush=True)
        # Live-only Telegram alert. The hook itself gates on
        # target_day == today, latest-closed bar, NO-TRADE, pierce-both,
        # and dedupe — see telegram_alerts.maybe_alert_1h.
        try:
            maybe_alert_1h(df1h, d)
        except Exception as e:
            # Alerts are non-fatal — never block the parquet write
            print(f"[telegram] alert hook failed: {type(e).__name__}: {e}",
                  flush=True)


if __name__ == "__main__":
    main()
