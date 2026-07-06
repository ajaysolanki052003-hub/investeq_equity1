"""Parameter-sensitivity sweeps for the Gamma Blast long-straddle strategy.

Reuses backtest_gamma_blast's data loader (combined_straddle) to build the ATM
combined-premium once per day, then sweeps ONE knob at a time (others held at
baseline) so each effect is clean and not overfit:

  A. Entry time gate  (R1: 12:00 .. 13:30)
  B. EMA period       (9, 13, 21, 34, 50)
  C. Spot filter      (R2: none / 0.4 / 0.5 / 0.6 / 0.8 / 1.0 %)
  D. Stop / exit      (fixed note+EOD  vs  trailing stop at various distances)

Baseline: entry after 12:30, EMA21, spot<0.60%, SL=candle-low/25pt, exit EOD.
Reports all-days AND expiry-day(DTE=0) avg (the cost-cushion metric).

    python gamma_blast_sweep.py --tf 5m
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

import backtest_gamma_blast as bt

TF = "5m"


def _hm(ts):
    return ts.hour * 60 + ts.minute


def ema(closes, n):
    out = np.full(len(closes), np.nan)
    if len(closes) < n:
        return out
    k = 2.0 / (n + 1)
    e = closes[:n].mean()
    out[n - 1] = e
    for i in range(n, len(closes)):
        e = closes[i] * k + e * (1 - k)
        out[i] = e
    return out


def find_entry(comb, ema_period, after_min):
    closes = comb["close"].to_numpy()
    highs = comb["high"].to_numpy()
    lows = comb["low"].to_numpy()
    e = ema(closes, ema_period)
    times = comb.index
    for i in range(1, len(comb)):
        if _hm(times[i]) < after_min:
            continue
        if np.isnan(e[i - 1]):
            continue
        if closes[i - 1] > e[i - 1] and highs[i] > highs[i - 1]:
            return i, float(highs[i - 1]), float(lows[i - 1])
    return None


def simulate(path_after, level, setup_low, trail_pts=None):
    """Long straddle. Initial stop = candle-low/25pt (the note rule). If trail_pts
    given, the stop trails at running_high - trail_pts (never loosens). EOD else."""
    sl = max(setup_low, level - 25)
    vals = path_after.to_numpy()
    if len(vals) == 0:
        return 0.0
    run_high = level
    for v in vals:
        if trail_pts is not None:
            if v > run_high:
                run_high = v
            sl = max(sl, run_high - trail_pts)
        if v <= sl:
            return sl - level
    return float(vals[-1]) - level


def build_cache(tf):
    days = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(bt.OPT_DIR, "*.parquet")))
    cache = []
    tfd = pd.Timedelta(bt.TF_RULE[tf])
    for d in days:
        r = bt.combined_straddle(d, tf)
        if r is None:
            continue
        comb, path, atm, expiry, sm = r
        dte = (datetime.strptime(expiry, "%Y-%m-%d").date()
               - datetime.strptime(d, "%Y%m%d").date()).days
        cache.append((d, dte, sm, comb, path, tfd))
    return cache


def run(cache, after_min=750, ema_period=21, spot_thr=0.60, trail_pts=None):
    pnls, dtes = [], []
    for (d, dte, sm, comb, path, tfd) in cache:
        if spot_thr is not None and not (sm < spot_thr):
            continue
        fe = find_entry(comb, ema_period, after_min)
        if fe is None:
            continue
        ei, lvl, slow = fe
        pa = path[path.index >= comb.index[ei] + tfd]
        pnls.append(simulate(pa, lvl, slow, trail_pts))
        dtes.append(dte)
    return np.array(pnls, float), np.array(dtes, int)


def row(label, pnls, dtes):
    if len(pnls) == 0:
        return f"  {label:<16} no trades"
    w = pnls[pnls > 0]
    d0 = pnls[dtes == 0]
    d0avg = d0.mean() if len(d0) else float("nan")
    return (f"  {label:<16} n={len(pnls):>4} win={len(w)/len(pnls)*100:>4.1f}% "
            f"avg={pnls.mean():>+5.2f} total={pnls.sum():>+6.0f}  "
            f"DTE0 avg={d0avg:>+5.2f} (n={len(d0)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="5m")
    args = ap.parse_args()
    global TF
    TF = args.tf

    print(f"building per-day cache (TF={TF}) ...", flush=True)
    cache = build_cache(TF)
    print(f"{len(cache)} days cached\n")

    base = dict(after_min=750, ema_period=21, spot_thr=0.60, trail_pts=None)
    p, d = run(cache, **base)
    print("BASELINE (after 12:30, EMA21, spot<0.60%, note SL, EOD):")
    print(row("baseline", p, d) + "\n")

    print("A. ENTRY TIME (R1):")
    for hm, lbl in [(720, "12:00"), (735, "12:15"), (750, "12:30"),
                    (765, "12:45"), (780, "13:00"), (810, "13:30")]:
        p, d = run(cache, **{**base, "after_min": hm})
        print(row(lbl, p, d))

    print("\nB. EMA PERIOD:")
    for n in [9, 13, 21, 34, 50]:
        p, d = run(cache, **{**base, "ema_period": n})
        print(row(f"EMA{n}", p, d))

    print("\nC. SPOT FILTER (R2, net move till 12:30):")
    for thr, lbl in [(None, "none"), (0.4, "<0.40%"), (0.5, "<0.50%"),
                     (0.6, "<0.60%"), (0.8, "<0.80%"), (1.0, "<1.00%")]:
        p, d = run(cache, **{**base, "spot_thr": thr})
        print(row(lbl, p, d))

    print("\nD. STOP / EXIT:")
    p, d = run(cache, **base)
    print(row("fixed+EOD", p, d))
    for t in [15, 20, 25, 30, 40, 50]:
        p, d = run(cache, **{**base, "trail_pts": t})
        print(row(f"trail {t}pt", p, d))


if __name__ == "__main__":
    main()
