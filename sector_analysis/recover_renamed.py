"""One-off recovery for the custom sector universe: re-fetch 'missing' symbols
that ARE valid in Groww's instrument master but failed to backfill because the
old _groww_symbol() mangled hyphenated/special tickers (BAJAJ-AUTO was queried
as 'BAJAJ-AUTO' instead of 'NSE-BAJAJ-AUTO'). That heuristic is now fixed in
groww_client, but this pass also handles the case by using the master's EXACT
groww_symbol, and covers a small normalized-rename map (OCCL -> OCCLLTD).

Only exact trading_symbol matches are auto-recovered (100% same company);
everything else is genuinely absent from Groww and left alone.

Run (on the VM, env from /etc/investeq.env):
    python -m sector_analysis.recover_renamed
"""
import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ema_scanner"))
import groww_client as gc  # noqa: E402
from groww_client import get_access_token, fetch_candles  # noqa: E402

FEED = ROOT / "ema_scanner" / "data" / "1d"
INSTR = ROOT / "DATA" / "_groww_instruments.csv"
MAP = ROOT / "sector_analysis" / "ref" / "custom_sectors.csv"

# Verified normalized renames (old ticker in the map -> current Groww symbol).
RENAME = {"OCCL": "OCCLLTD"}


def now_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def main():
    # master: NSE cash trading_symbol -> (groww_symbol, name)
    gsym, name = {}, {}
    for r in csv.DictReader(open(INSTR, encoding="utf-8")):
        if r.get("exchange") == "NSE" and r.get("segment") == "CASH":
            ts = (r.get("trading_symbol") or "").strip().upper()
            if ts:
                gsym[ts] = (r.get("groww_symbol") or "").strip()
                name[ts] = r.get("name", "")

    _orig = gc._groww_symbol

    def _patched(symbol, exchange="NSE"):
        s = symbol.strip().upper()
        if s in RENAME and RENAME[s] in gsym:
            return gsym[RENAME[s]]
        if gsym.get(s):
            return gsym[s]
        return _orig(symbol, exchange)
    gc._groww_symbol = _patched   # fetch_candles resolves the name at call time

    tok = get_access_token(os.environ["GROWW_TOTP_JWT"], os.environ["GROWW_TOTP_SECRET"])
    print("[auth] OK", flush=True)

    rows = list(csv.DictReader(open(MAP, encoding="utf-8")))
    have = {p.name[:-len("_historical.csv")] for p in FEED.glob("*_historical.csv")}
    miss = [r["Symbol"].strip().upper() for r in rows if r["Symbol"].strip().upper() not in have]
    targets = [s for s in miss if s in gsym or (s in RENAME and RENAME[s] in gsym)]
    print(f"missing={len(miss)}  recoverable targets={len(targets)}", flush=True)

    start = (now_ist() - timedelta(days=365 * 2 + 10)).strftime("%Y-%m-%d 09:15:00")
    end = now_ist().strftime("%Y-%m-%d 15:30:00")

    rec, nodata = [], []
    for s in targets:
        src = RENAME.get(s, s)
        df = fetch_candles(s, "1d", start, end, tok, delay_s=0.35)
        if df.empty:
            nodata.append(s)
            continue
        out = FEED / f"{s}_historical.csv"
        tmp = out.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(out)
        rec.append((s, len(df), str(df["datetime"].iloc[-1])[:10]))
        tag = f"{s} <- {src}" if src != s else s
        print(f"[OK] {tag:22} {len(df):4} rows, last {rec[-1][2]}  ({name.get(src,'')[:34]})", flush=True)

    print(f"\nrecovered={len(rec)}  still-no-data={len(nodata)} {nodata}", flush=True)


if __name__ == "__main__":
    main()
