"""Bracket exit simulation over sweep signals — the 'good trades' test.

For each entry: walk the bought CE's 1-min bars forward; TP at entry*(1+tp),
SL at entry*(1-sl), whichever prints first (same bar -> SL, conservative);
EOD close if neither. Reports win rate and expectancy in R (risk units).

    python -m expiry_blast.bracket_sim [--tp 0.5] [--sl 0.5]
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from .data import DATA_ROOT

SWEEP = Path(__file__).parent / "sweep"


def walk(sig: pd.Series, tp: float, sl: float) -> str | None:
    day = pd.Timestamp(sig["timestamp"]).date()
    p = DATA_ROOT / "options" / f"{day:%Y%m%d}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    leg = df[(df["strike"] == sig["entry_strike"])
             & (df["option_type"] == "CE")
             & (pd.to_datetime(df["expiry"]).dt.date == day)
             & (df["timestamp"] > pd.Timestamp(sig["timestamp"]))
             ].sort_values("timestamp")
    e = sig["entry_premium"]
    if leg.empty or not e or pd.isna(e):
        return None
    tp_px, sl_px = e * (1 + tp), e * (1 - sl)
    for _, b in leg.iterrows():
        hit_tp = b["high"] >= tp_px
        hit_sl = b["low"] <= sl_px
        if hit_tp and hit_sl:
            return "SL"          # both in one bar — assume worst
        if hit_tp:
            return "TP"
        if hit_sl:
            return "SL"
    last = float(leg["close"].iloc[-1])
    return "EOD+" if last > e else "EOD-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=float, default=0.5)
    ap.add_argument("--sl", type=float, default=0.5)
    args = ap.parse_args()

    rows = []
    for d in sorted(SWEEP.iterdir()):
        csv = d / "signals.csv"
        if not csv.exists():
            continue
        try:
            sigs = pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            continue
        if sigs.empty:
            continue
        outcomes = [walk(s, args.tp, args.sl) for _, s in sigs.iterrows()]
        outcomes = [o for o in outcomes if o]
        n = len(outcomes)
        if not n:
            continue
        tp_n = outcomes.count("TP")
        sl_n = outcomes.count("SL")
        eodp = outcomes.count("EOD+")
        eodm = outcomes.count("EOD-")
        # R expectancy with risk = sl fraction of premium: TP = +tp/sl R,
        # SL = -1 R, EOD counted at 0 (unknown partial) for a floor metric.
        r = (tp_n * (args.tp / args.sl) - sl_n) / n
        rows.append({"variant": d.name, "n": n,
                     "TP": tp_n, "SL": sl_n, "EOD+": eodp, "EOD-": eodm,
                     "win_pct": round(100 * (tp_n + eodp) / n),
                     "expectancy_R": round(r, 2)})
    print(f"bracket: TP +{args.tp:.0%} / SL -{args.sl:.0%} on premium "
          f"(same-bar -> SL)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
