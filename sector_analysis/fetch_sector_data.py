#!/usr/bin/env python3
"""Fetch & refresh the sector / index reference data for the Sector Analysis app.

Sources (authoritative, free, no key):
  - Nifty 500 constituents + Industry (sector) column   -> niftyindices.com
  - Nifty 50 constituents                               -> niftyindices.com
  - Sensex 30 constituents                              -> BSE API, curated fallback

Writes small CSVs into sector_analysis/ref/ :
  - nifty500_industry.csv   Symbol, Company, Industry   (the master sector map)
  - nifty50.csv             Symbol
  - sensex30.csv            Symbol

These files are tiny reference data (a few KB) and ARE committed / deployed —
unlike the large gitignored OHLCV data under ema_scanner/data/. Re-run this
after an index reshuffle to refresh; the app reads whatever is on disk.

Usage:
    python -m sector_analysis.fetch_sector_data
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "ref"
DATA.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# --- Sensex 30 curated fallback (last verified 2026-07-01). All are Nifty-500
# members, so their sector comes from nifty500_industry.csv automatically. Used
# only if the live BSE fetch fails. ------------------------------------------
SENSEX30_FALLBACK = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC", "LT",
    "KOTAKBANK", "AXISBANK", "SBIN", "BHARTIARTL", "HINDUNILVR", "BAJFINANCE",
    "M&M", "SUNPHARMA", "NTPC", "MARUTI", "TITAN", "ULTRACEMCO", "ASIANPAINT",
    "TATAMOTORS", "POWERGRID", "TATASTEEL", "NESTLEIND", "BAJAJFINSV",
    "TECHM", "INDUSINDBK", "ADANIPORTS", "WIPRO", "JSWSTEEL",
]


def _get(url: str, referer: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/csv,application/csv,application/json,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_nifty_index(list_file: str) -> list[dict]:
    """Return list of {Symbol, Company, Industry} for a niftyindices list csv."""
    url = f"https://niftyindices.com/IndexConstituent/{list_file}"
    raw = _get(url, "https://niftyindices.com/").decode("utf-8-sig", "replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    out = []
    for r in rows:
        sym = (r.get("Symbol") or "").strip()
        if not sym:
            continue
        out.append({
            "Symbol": sym,
            "Company": (r.get("Company Name") or "").strip(),
            "Industry": (r.get("Industry") or "").strip(),
        })
    return out


def fetch_sensex30() -> list[str]:
    """Try the BSE constituents API; fall back to the curated list."""
    try:
        # BSE index-constituent endpoint (SENSEX scrip code 16)
        url = "https://api.bseindia.com/BseIndiaAPI/api/IndexArray/w?index=16&type=D"
        import json
        raw = _get(url, "https://www.bseindia.com/").decode("utf-8", "replace")
        data = json.loads(raw)
        syms = []
        # response shape varies; be defensive
        table = data.get("Table") if isinstance(data, dict) else data
        for row in (table or []):
            s = (row.get("scrip_cd_name") or row.get("Scrip_Name")
                 or row.get("SYMBOL") or row.get("Symbol") or "").strip()
            if s:
                syms.append(s.upper())
        if len(syms) >= 25:
            print(f"  [sensex] BSE API -> {len(syms)} symbols")
            return syms
        raise ValueError(f"unexpected shape, {len(syms)} syms")
    except Exception as e:  # noqa: BLE001
        print(f"  [sensex] BSE fetch failed ({type(e).__name__}: {e}); "
              f"using curated fallback ({len(SENSEX30_FALLBACK)})")
        return list(SENSEX30_FALLBACK)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path.name}: {len(rows)} rows")


def main() -> int:
    print("Fetching sector / index reference data ...")

    n500 = fetch_nifty_index("ind_nifty500list.csv")
    if len(n500) < 400:
        print(f"ERROR: nifty500 returned only {len(n500)} rows; aborting.")
        return 1
    _write_csv(DATA / "nifty500_industry.csv",
               ["Symbol", "Company", "Industry"],
               [[r["Symbol"], r["Company"], r["Industry"]] for r in n500])

    n50 = fetch_nifty_index("ind_nifty50list.csv")
    _write_csv(DATA / "nifty50.csv", ["Symbol"], [[r["Symbol"]] for r in n50])

    sensex = fetch_sensex30()
    _write_csv(DATA / "sensex30.csv", ["Symbol"], [[s] for s in sensex])

    # quick industry summary
    from collections import Counter
    ind = Counter(r["Industry"] for r in n500)
    print("\nSectors (Nifty 500):")
    for name, cnt in ind.most_common():
        print(f"  {cnt:3d}  {name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
