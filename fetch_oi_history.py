"""Fetch end-of-day NIFTY options OI from NSE F&O bhavcopy and write per-day
parquets matching the existing DATA/oi/YYYYMMDD.parquet schema:

    timestamp, expiry, strike, option_type, oi

Source: https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip

Modes:
    --backfill           detect missing days in DATA/oi/, download each
    --today              download today's bhavcopy (clamps to last weekday)
    --date YYYY-MM-DD    download a specific day
    --start / --end      explicit range
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
OI_DIR = ROOT / "oi"

BHAV_URL_NEW = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
)
# NSE renamed and reformatted the F&O bhavcopy in early 2024. For older dates
# the file lives under /historical/DERIVATIVES/<YYYY>/<MON>/fo<DD><MON><YYYY>bhav.csv.zip
# with the legacy column layout (INSTRUMENT/SYMBOL/EXPIRY_DT/STRIKE_PR/...).
BHAV_URL_OLD = (
    "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
    "{Y}/{MON}/fo{D:02d}{MON}{Y}bhav.csv.zip"
)
MON3 = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# NSE archive subdomain rejects requests without a UA + Accept header.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "application/octet-stream, */*",
}


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5      # Mon=0 .. Fri=4


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _existing_days() -> set[str]:
    return {p.stem for p in OI_DIR.glob("*.parquet")}


def _download_bhav(d: date, session: requests.Session) -> tuple[pd.DataFrame, str]:
    """Return (parsed bhavcopy DataFrame, schema-tag).
    Tries the NEW URL first (post-2024 format), falls back to the OLD URL
    (pre-2024). schema-tag is 'new', 'old', or '' if nothing worked."""
    # NEW format
    url = BHAV_URL_NEW.format(ymd=_ymd(d))
    r = session.get(url, headers=HEADERS, timeout=25)
    if r.status_code == 200 and len(r.content) > 200:
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            inner = z.namelist()[0]
            df = pd.read_csv(z.open(inner))
            if not df.empty:
                return df, "new"
        except Exception:
            pass
    # OLD format
    url = BHAV_URL_OLD.format(Y=d.year, MON=MON3[d.month - 1], D=d.day)
    r = session.get(url, headers=HEADERS, timeout=25)
    if r.status_code == 200 and len(r.content) > 200:
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            inner = z.namelist()[0]
            df = pd.read_csv(z.open(inner))
            if not df.empty:
                return df, "old"
        except Exception:
            pass
    return pd.DataFrame(), ""


def _to_oi_parquet(df: pd.DataFrame, d: date, schema: str) -> pd.DataFrame | None:
    """Filter the bhavcopy to NIFTY options and reshape into the existing
    DATA/oi/ schema. Handles both NEW (UDIFF) and OLD legacy column layouts."""
    if df.empty:
        return None
    ts = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                      hour=15, minute=30).isoformat()
    if schema == "new":
        sub = df[(df["TckrSymb"] == "NIFTY") &
                 (df["OptnTp"].isin(["CE", "PE"]))].copy()
        if sub.empty:
            return None
        out = pd.DataFrame({
            "timestamp":   ts,
            "expiry":      pd.to_datetime(sub["XpryDt"]).dt.strftime("%Y-%m-%d"),
            "strike":      sub["StrkPric"].astype("int32"),
            "option_type": sub["OptnTp"].astype(str),
            "oi":          sub["OpnIntrst"].fillna(0).astype("int64"),
        })
    elif schema == "old":
        # Legacy format: INSTRUMENT='OPTIDX', SYMBOL='NIFTY', EXPIRY_DT='28-Dec-2023'
        sub = df[(df["INSTRUMENT"] == "OPTIDX") &
                 (df["SYMBOL"] == "NIFTY") &
                 (df["OPTION_TYP"].isin(["CE", "PE"]))].copy()
        if sub.empty:
            return None
        out = pd.DataFrame({
            "timestamp":   ts,
            "expiry":      pd.to_datetime(sub["EXPIRY_DT"], format="%d-%b-%Y").dt.strftime("%Y-%m-%d"),
            "strike":      sub["STRIKE_PR"].astype("float").astype("int32"),
            "option_type": sub["OPTION_TYP"].astype(str),
            "oi":          sub["OPEN_INT"].fillna(0).astype("int64"),
        })
    else:
        return None
    out = (out.drop_duplicates(subset=["expiry", "strike", "option_type"], keep="first")
              .reset_index(drop=True))
    return out


def fetch_one(d: date, session: requests.Session, *, overwrite: bool) -> str:
    """Download + write one day. Returns a short status string."""
    if not _is_weekday(d):
        return "weekend"
    path = OI_DIR / f"{_ymd(d)}.parquet"
    if path.exists() and not overwrite:
        return "exists"
    raw, schema = _download_bhav(d, session)
    if raw.empty:
        return "no_bhav"
    out = _to_oi_parquet(raw, d, schema)
    if out is None or out.empty:
        return "no_nifty"
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return f"ok-{schema}({len(out):,})"


def _date_range(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _today_ist() -> date:
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="Find missing weekdays in DATA/oi/ and fetch them")
    ap.add_argument("--today", action="store_true",
                    help="Fetch today's bhavcopy (clamps to last weekday)")
    ap.add_argument("--date", type=str, default=None,
                    help="Specific date YYYY-MM-DD")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end",   type=str, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download even if the per-day parquet already exists")
    ap.add_argument("--delay", type=float, default=0.6,
                    help="Pause between requests (be a good citizen)")
    args = ap.parse_args()

    OI_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    # Prime once — NSE sometimes 403's the very first call without cookies
    try:
        session.get("https://www.nseindia.com/all-reports",
                    headers=HEADERS, timeout=10)
    except Exception:
        pass

    if args.date:
        days = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    elif args.start and args.end:
        s = datetime.strptime(args.start, "%Y-%m-%d").date()
        e = datetime.strptime(args.end,   "%Y-%m-%d").date()
        days = list(_date_range(s, e))
    elif args.today:
        days = [_last_weekday(_today_ist())]
    elif args.backfill:
        existing = _existing_days()
        existing_dates = sorted(datetime.strptime(s, "%Y%m%d").date() for s in existing)
        # Backfill from the most-recent existing day to today
        if not existing_dates:
            print("[backfill] DATA/oi/ is empty — refusing. Specify --start/--end explicitly.",
                  flush=True)
            return 1
        start = existing_dates[-1] + timedelta(days=1)
        end   = _last_weekday(_today_ist())
        if start > end:
            print(f"[backfill] already current (last={existing_dates[-1]}, today={end})", flush=True)
            return 0
        print(f"[backfill] {start} -> {end}", flush=True)
        days = list(_date_range(start, end))
    else:
        ap.print_help()
        return 1

    ok = exists = weekend = nobhav = nonifty = 0
    t0 = time.time()
    for d in days:
        st = fetch_one(d, session, overwrite=args.overwrite)
        head = st.split("(")[0]
        if   head == "ok":       ok += 1
        elif head == "exists":   exists += 1
        elif head == "weekend":  weekend += 1
        elif head == "no_bhav":  nobhav += 1
        elif head == "no_nifty": nonifty += 1
        print(f"  {d}  {st}", flush=True)
        if args.delay > 0:
            time.sleep(args.delay)
    el = time.time() - t0
    print(f"\n[done] ok={ok} exists={exists} weekend={weekend} "
          f"no_bhav={nobhav} no_nifty={nonifty}  ({el:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
