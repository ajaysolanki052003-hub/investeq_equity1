"""Backfill rows in data/{1d,1h}/*_historical.csv where the `open` column is
empty. Fills from the previous row's `close`, which is what the scanner
already does in-memory — this just makes the on-disk data match.

Run from anywhere:
    python ema_scanner/backfill_missing_open.py
    python ema_scanner/backfill_missing_open.py --interval 1d
    python ema_scanner/backfill_missing_open.py --dry-run

Backs up the original to data/{interval}.bak_open_backfill/ before writing.
Safe to re-run (no-op on files that have no empty Opens).
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def backfill_one(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Returns (rows_total, rows_fixed)."""
    df = pd.read_csv(path)
    if "open" not in df.columns or "close" not in df.columns:
        return len(df), 0
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    if not df["open"].isna().any():
        return len(df), 0
    fixed = df["open"].isna().sum()
    # Fill from previous bar's close; for the very first row, fall back to
    # the same-bar's close. Iterate to propagate across consecutive gaps.
    prev_close = df["close"].astype(float).shift(1)
    df["open"] = df["open"].fillna(prev_close)
    # Still-NaN happens only when the first row(s) have empty Open; use the
    # same-bar close so the row is at least valid (close >= open → neutral).
    df["open"] = df["open"].fillna(df["close"])
    if dry_run:
        return len(df), int(fixed)
    # Backup once per file the first time we touch it.
    bak_dir = path.parent.parent / f"{path.parent.name}.bak_open_backfill"
    bak_dir.mkdir(exist_ok=True)
    bak_path = bak_dir / path.name
    if not bak_path.exists():
        shutil.copy2(path, bak_path)
    # Preserve the original column order and integer-style formatting.
    df.to_csv(path, index=False)
    return len(df), int(fixed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", choices=["1d", "1h", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be fixed, don't write.")
    args = ap.parse_args()
    intervals = ["1d", "1h"] if args.interval == "both" else [args.interval]
    grand_total = grand_fixed = files_touched = 0
    for iv in intervals:
        folder = DATA / iv
        if not folder.is_dir():
            print(f"  ! {folder} not found, skipping")
            continue
        files = sorted(folder.glob("*_historical.csv"))
        print(f"\n=== {iv}  ({len(files)} files)  {'[DRY RUN]' if args.dry_run else ''} ===")
        iv_total = iv_fixed = iv_files = 0
        for i, p in enumerate(files, 1):
            total, fixed = backfill_one(p, dry_run=args.dry_run)
            iv_total += total
            iv_fixed += fixed
            if fixed:
                iv_files += 1
            if i % 50 == 0 or i == len(files):
                print(f"  [{i:4d}/{len(files)}]  files_with_fills={iv_files}  "
                      f"rows_fixed={iv_fixed}/{iv_total}")
        print(f"  {iv}: {iv_files} files touched, {iv_fixed} rows backfilled")
        grand_total += iv_total
        grand_fixed += iv_fixed
        files_touched += iv_files
    print(f"\n[done] files_touched={files_touched}  "
          f"rows_backfilled={grand_fixed}/{grand_total}")
    if not args.dry_run and grand_fixed:
        print("Originals backed up to data/{interval}.bak_open_backfill/")


if __name__ == "__main__":
    main()
