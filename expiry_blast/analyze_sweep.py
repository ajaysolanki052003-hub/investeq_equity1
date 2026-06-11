"""Summarize the relaxation sweep: trade count vs entry quality per variant.

Run after the sweep backtests:
    python -m expiry_blast.analyze_sweep
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SWEEP = Path(__file__).parent / "sweep"


def summarize(name: str, d: Path) -> dict | None:
    csv = d / "signals.csv"
    if not csv.exists():
        return None
    cfg = {}
    summ = d / "summary.txt"
    if summ.exists():
        for line in summ.read_text().splitlines():
            if line.startswith("config"):
                cfg = json.loads(line.split(":", 1)[1].strip())
    try:
        df = pd.read_csv(csv)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    n = len(df)
    out = {"variant": name, "n": n,
           "per_year": round(n / 6.4, 1),
           "knobs": (f"C:{cfg.get('put_strikes_required')}of2"
                     f"@{cfg.get('put_oi_buildup_pct')}% "
                     f"B:{cfg.get('call_oi_unwind_pct')}% "
                     f"A:{cfg.get('spot_proximity_pct')}% "
                     f"max/day:{cfg.get('max_signals_per_day')}")}
    if n and "premium_eod" in df.columns:
        ok = df.dropna(subset=["entry_premium", "premium_eod"])
        if len(ok):
            e = ok["entry_premium"]
            out["win_eod_pct"] = round((ok["premium_eod"] > e).mean() * 100)
            out["med_eod_ret"] = round(((ok["premium_eod"] / e) - 1).median() * 100)
            out["med_mfe_x"] = round((ok["premium_max_after"] / e).median(), 2)
            if "premium_min_after" in ok.columns:
                out["med_mae_pct"] = round(
                    ((ok["premium_min_after"] / e) - 1).median() * 100)
    return out


def main():
    rows = []
    for d in sorted(SWEEP.iterdir()):
        if d.is_dir():
            r = summarize(d.name, d)
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["variant", "knobs", "n", "per_year", "win_eod_pct",
            "med_eod_ret", "med_mfe_x", "med_mae_pct"]
    df = df[[c for c in cols if c in df.columns]]
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
