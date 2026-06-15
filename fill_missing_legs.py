"""Fill missing CE/PE legs near ATM in existing DATA/options/*.parquet files.

For Chain Replay: some older days have a near-ATM strike with only one leg
(CE or PE). This fetches ONLY the missing (expiry, strike, option_type)
symbols within ATM±pad and MERGES them into the existing file — the wider
far-OTM strikes already present are preserved (merge, never overwrite).

Far-OTM single legs are left alone: an option with no trades has no candles,
so there is nothing to fetch.

Run:
    python fill_missing_legs.py            # scan + fill every day
    python fill_missing_legs.py --pad 6
"""

from __future__ import annotations
import argparse, os, sys, time, glob
from datetime import datetime
from pathlib import Path
import pandas as pd

from fetch_oi_intraday import ROOT, _gsym, fetch_candles
sys.path.insert(0, str(Path(__file__).parent / "ema_scanner"))
from groww_client import get_access_token  # noqa: E402

OPT_DIR  = ROOT / "options"
SPOT_DIR = ROOT / "spot"
STRIKE_STEP = 50


def fill_day(day: str, token: str, pad: int = 6, delay: float = 0.35) -> int:
    fp = OPT_DIR / f"{day}.parquet"
    sp_fp = SPOT_DIR / f"{day}.parquet"
    if not fp.exists() or not sp_fp.exists():
        return 0
    df = pd.read_parquet(fp)
    sp = pd.read_parquet(sp_fp, columns=["close"])
    atm = round(sp["close"].mean() / STRIKE_STEP) * STRIKE_STEP
    band = [atm + STRIKE_STEP * i for i in range(-pad, pad + 1)]

    have = set(zip(df["expiry"].astype(str), df["strike"].astype(int),
                   df["option_type"].astype(str)))
    expiries = sorted(df["expiry"].astype(str).unique())
    d = datetime.strptime(day, "%Y%m%d").date()
    s_str, e_str = f"{d.isoformat()} 09:15:00", f"{d.isoformat()} 15:30:00"

    new = []
    for exp in expiries:
        edate = datetime.strptime(exp, "%Y-%m-%d").date()
        for k in band:
            for typ in ("CE", "PE"):
                if (exp, int(k), typ) in have:
                    continue  # leg already present — skip
                cands = fetch_candles(_gsym(edate, int(k), typ), s_str, e_str, token)
                for c in cands:
                    if len(c) < 6:
                        continue
                    new.append({"timestamp": c[0], "expiry": exp,
                                "strike": int(k), "option_type": typ,
                                "open": c[1], "high": c[2], "low": c[3],
                                "close": c[4], "volume": c[5]})
                if delay > 0:
                    time.sleep(delay)

    if not new:
        return 0
    merged = (pd.concat([df, pd.DataFrame(new)], ignore_index=True)
              .drop_duplicates(["timestamp", "expiry", "strike", "option_type"])
              .sort_values(["timestamp", "expiry", "strike", "option_type"]))
    tmp = fp.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(fp)
    return len(new)


def _needs_fill(day: str, pad: int) -> bool:
    """True if any band strike is missing a leg (cheap pre-check, no API)."""
    fp, sp_fp = OPT_DIR / f"{day}.parquet", SPOT_DIR / f"{day}.parquet"
    if not fp.exists() or not sp_fp.exists():
        return False
    df = pd.read_parquet(fp, columns=["expiry", "strike", "option_type"])
    sp = pd.read_parquet(sp_fp, columns=["close"])
    atm = round(sp["close"].mean() / STRIKE_STEP) * STRIKE_STEP
    band = {atm + STRIKE_STEP * i for i in range(-pad, pad + 1)}
    sub = df[df["strike"].isin(band)]
    g = sub.groupby(["expiry", "strike"])["option_type"].nunique()
    return bool((g < 2).any())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, default=6,
                    help="Strikes each side of ATM to complete (default 6 = ±300)")
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--days", type=str, default=None,
                    help="Comma-separated YYYYMMDD list; default scans all")
    args = ap.parse_args()

    token = get_access_token(os.environ["GROWW_TOTP_JWT"],
                             os.environ["GROWW_TOTP_SECRET"])
    print("[auth] OK", flush=True)

    if args.days:
        days = [x.strip() for x in args.days.split(",") if x.strip()]
    else:
        all_days = [os.path.basename(f)[:8] for f in sorted(glob.glob(str(OPT_DIR / "*.parquet")))]
        days = [d for d in all_days if _needs_fill(d, args.pad)]
    print(f"[plan] {len(days)} day(s) with a near-ATM missing leg", flush=True)

    t0, total = time.time(), 0
    for i, d in enumerate(days):
        n = fill_day(d, token, pad=args.pad, delay=args.delay)
        total += n
        if n:
            print(f"  {d}  +{n} rows", flush=True)
    print(f"[done] filled {total:,} rows across {len(days)} days "
          f"({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
