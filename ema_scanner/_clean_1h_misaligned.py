"""One-shot: strip rows from data/1h/*_historical.csv whose datetime
minute != 15 (i.e. Groww's wall-clock 1h candles polluting the NSE-aligned
file). The next incremental_fetch run will refill the gap with properly
resampled 1m→1h bars.

Backs each file up to data/1h.bak_misaligned/ before mutating.
"""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC  = HERE / 'data' / '1h'
BAK  = HERE / 'data' / '1h.bak_misaligned'


def main():
    files = sorted(SRC.glob('*_historical.csv'))
    print(f'scanning {len(files)} files in {SRC}', flush=True)
    BAK.mkdir(parents=True, exist_ok=True)

    total_kept = total_dropped = changed = 0
    for i, p in enumerate(files, start=1):
        df = pd.read_csv(p)
        df['datetime'] = pd.to_datetime(df['datetime'])
        good = df['datetime'].dt.minute == 15
        n_drop = int((~good).sum())
        if n_drop == 0:
            total_kept += len(df)
            continue
        # backup once
        bak_path = BAK / p.name
        if not bak_path.exists():
            shutil.copy2(p, bak_path)
        cleaned = df[good].copy()
        # write back, preserving the integer columns where possible
        cleaned['datetime'] = cleaned['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        cleaned.to_csv(p, index=False)
        total_kept += len(cleaned)
        total_dropped += n_drop
        changed += 1
        if i % 50 == 0 or n_drop:
            print(f'  [{i:4d}/{len(files)}] {p.name}: dropped {n_drop}', flush=True)
    print(f'done. changed files={changed}  kept rows={total_kept}  dropped rows={total_dropped}', flush=True)
    print(f'backups: {BAK}')


if __name__ == '__main__':
    main()
