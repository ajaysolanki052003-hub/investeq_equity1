"""Compare trail-SL variants for the merged Σ·10 + ΣOI strategy.

Locks every other parameter to the live-app defaults:
  - mode  = both (Σ·10 + ΣOI, same-side 7-min dedup, position_mode=single)
  - SL    = 0.18 %  (≈ 40 NIFTY points at 22k spot)
  - TGT   = 0.50 %  (≈ 110 NIFTY points)
  - next-candle entry on both strategies
  - min-gap-7 on ΣOI (baked in)

Then sweeps trail variants:
  A) trail OFF                 — initial SL stays, no trailing
  B) trail 40pt  ONE-SHOT     — once in profit by 40 pts, SL → breakeven
  C) trail 30pt  ONE-SHOT     — once in profit by 30 pts, SL → breakeven
  D) trail 40pt  MULTI-STEP   — staircase: SL = max(SL, close - 40) once triggered
  E) trail 30pt  MULTI-STEP   — staircase: SL = max(SL, close - 30) once triggered

ONE-SHOT  = SL moves to entry once, never again (current live behavior).
MULTI     = SL keeps climbing with price, locking in profits incrementally.

Run from project root:
    python analyze_trail_variants.py
"""

from __future__ import annotations

import os
import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from strategies.sigma5_entry_exit import compute_entries as sigma_n_entries
from strategies.sigma_total_oi_entry import compute_entries as sigma_oi_entries


ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))

PRICE_MASTER = ROOT / "nifty_1m_master.parquet"

SL_PCT  = 0.0018     # 0.18 %  → ~40 pts at 22k spot
TGT_PCT = 0.0050     # 0.50 %  → ~110 pts
DEDUP_MIN = 7        # same-side dedup window (min)


def merged_entries() -> list[dict]:
    """Both methods merged with 7-min same-side dedup. Mirrors /api/merged_trades."""
    sig10, _ = sigma_n_entries(tf="3m", window=10)
    sigot, _ = sigma_oi_entries()
    tagged = [{**e, "_source": "sigma10"} for e in sig10] + \
             [{**e, "_source": "sigma_oi"} for e in sigot]
    tagged.sort(key=lambda e: (str(e["day"]), str(e["entry_time"])))
    out, last = [], {}
    for e in tagged:
        key = (str(e["day"]), e["side"])
        ent_ts = pd.Timestamp(e["entry_time"])
        if last.get(key) is not None and \
           (ent_ts - last[key]).total_seconds() < DEDUP_MIN * 60:
            continue
        out.append(e)
        last[key] = ent_ts
    return out


def load_px_by_day() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    px = pd.read_parquet(PRICE_MASTER)[["timestamp", "close"]]
    px["timestamp"] = pd.to_datetime(px["timestamp"])
    px = px.sort_values("timestamp")
    px["date_key"] = px["timestamp"].dt.date.astype(str)
    out = {}
    for d, g in px.groupby("date_key", sort=True):
        g = g.sort_values("timestamp")
        out[d] = (g["timestamp"].values.astype("datetime64[ns]"),
                  g["close"].values.astype("float64"))
    return out


def walk_one_shot(seg_cl, side, spot, sl_initial, tgt_px, trail_pts):
    """One-shot trail: when in profit by trail_pts, SL → entry. Stays there."""
    sl_current = sl_initial
    trail_active = False
    trigger = spot + trail_pts if side == "LONG" else spot - trail_pts
    for i, c in enumerate(seg_cl):
        if side == "LONG":
            if c >= tgt_px: return i, tgt_px, "TGT"
            if not trail_active and c >= trigger:
                trail_active = True; sl_current = spot
            if c <= sl_current:
                return i, sl_current, ("TRAIL" if trail_active else "SL")
        else:
            if c <= tgt_px: return i, tgt_px, "TGT"
            if not trail_active and c <= trigger:
                trail_active = True; sl_current = spot
            if c >= sl_current:
                return i, sl_current, ("TRAIL" if trail_active else "SL")
    return -1, None, None


def walk_multi(seg_cl, side, spot, sl_initial, tgt_px, trail_pts):
    """Multi-step trail: once activated, SL trails price by trail_pts (never moves
    against the trade)."""
    sl_current = sl_initial
    trail_active = False
    trigger = spot + trail_pts if side == "LONG" else spot - trail_pts
    for i, c in enumerate(seg_cl):
        if side == "LONG":
            if c >= tgt_px: return i, tgt_px, "TGT"
            if c >= trigger:
                if not trail_active: trail_active = True
                candidate = c - trail_pts
                if candidate > sl_current: sl_current = candidate
            if c <= sl_current:
                return i, sl_current, ("TRAIL" if trail_active else "SL")
        else:
            if c <= tgt_px: return i, tgt_px, "TGT"
            if c <= trigger:
                if not trail_active: trail_active = True
                candidate = c + trail_pts
                if candidate < sl_current: sl_current = candidate
            if c >= sl_current:
                return i, sl_current, ("TRAIL" if trail_active else "SL")
    return -1, None, None


def walk_off(seg_cl, side, spot, sl_initial, tgt_px, trail_pts):
    """No trail — first hit of SL or TGT wins."""
    for i, c in enumerate(seg_cl):
        if side == "LONG":
            if c >= tgt_px:    return i, tgt_px,     "TGT"
            if c <= sl_initial:return i, sl_initial, "SL"
        else:
            if c <= tgt_px:    return i, tgt_px,     "TGT"
            if c >= sl_initial:return i, sl_initial, "SL"
    return -1, None, None


def backtest(entries, px_by_day, walker, trail_pts):
    """For each entry, run walker → trade. Apply position-overlap (single) filter."""
    raw = []
    for e in entries:
        day = str(e["day"])
        arrs = px_by_day.get(day)
        if arrs is None: continue
        ts_arr, cl_arr = arrs
        idx = int(np.searchsorted(ts_arr, np.datetime64(e["entry_time"]), "left"))
        if idx >= len(ts_arr): continue
        seg_ts, seg_cl = ts_arr[idx:], cl_arr[idx:]
        spot = float(e["entry_spot"])
        side = e["side"]
        if side == "LONG":
            sl_initial, tgt_px = spot * (1 - SL_PCT), spot * (1 + TGT_PCT)
        else:
            sl_initial, tgt_px = spot * (1 + SL_PCT), spot * (1 - TGT_PCT)
        i, exit_p, reason = walker(seg_cl, side, spot, sl_initial, tgt_px, trail_pts)
        if i == -1:
            exit_p = float(seg_cl[-1]); exit_t = seg_ts[-1]; reason = "EOD"
        else:
            exit_t = seg_ts[i]
        pnl = (exit_p - spot) if side == "LONG" else (spot - exit_p)
        raw.append({"day": day, "side": side,
                    "entry_time": pd.Timestamp(e["entry_time"]),
                    "exit_time":  pd.Timestamp(exit_t),
                    "pnl_pts": float(pnl), "reason": reason,
                    "source": e["_source"]})
    # Position-overlap (single) — per-day filter
    raw.sort(key=lambda t: (t["day"], t["entry_time"]))
    filtered, open_until, cur_day = [], {"LONG": None, "SHORT": None}, None
    for t in raw:
        if t["day"] != cur_day:
            open_until = {"LONG": None, "SHORT": None}; cur_day = t["day"]
        if any(ou is not None and t["entry_time"] < ou for ou in open_until.values()):
            continue
        filtered.append(t)
        open_until[t["side"]] = t["exit_time"]
    return filtered


def stats(trades, label):
    if not trades:
        return {"label": label, "n": 0, "n_days": 0}
    arr = np.array([t["pnl_pts"] for t in trades])
    wins, losses = arr[arr > 0], arr[arr <= 0]
    rs = [t["reason"] for t in trades]
    non_trail = arr[arr != 0] if "TRAIL" in rs else arr
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None
    days = len({t["day"] for t in trades})
    return {
        "label": label, "n": len(trades), "n_days": days,
        "per_day": len(trades) / max(1, days),
        "win_rate": len(wins) / len(arr) * 100.0,
        "total_pnl": float(arr.sum()),
        "avg_ex_trail": float(arr[arr != 0].mean()) if (arr != 0).any() else 0.0,
        "pf": pf,
        "sl": rs.count("SL"), "tgt": rs.count("TGT"),
        "eod": rs.count("EOD"), "trail": rs.count("TRAIL"),
    }


def main():
    print("[1/3] Loading entries (merged Σ·10 + ΣOI, dedup-7) ...", flush=True)
    entries = merged_entries()
    print(f"      {len(entries)} entries", flush=True)

    print("[2/3] Loading 1-min NIFTY ...", flush=True)
    px_by_day = load_px_by_day()
    print(f"      {len(px_by_day)} days indexed", flush=True)

    print("[3/3] Running 5 trail variants ...", flush=True)
    runs = [
        ("OFF",            walk_off,       0,  "no trail"),
        ("40pt ONE-SHOT",  walk_one_shot,  40, "SL → entry once in profit by 40 pts"),
        ("30pt ONE-SHOT",  walk_one_shot,  30, "SL → entry once in profit by 30 pts"),
        ("40pt MULTI",     walk_multi,     40, "staircase: SL trails price by 40 pts"),
        ("30pt MULTI",     walk_multi,     30, "staircase: SL trails price by 30 pts"),
    ]
    rows = []
    for label, walker, pts, desc in runs:
        trades = backtest(entries, px_by_day, walker, pts)
        rows.append(stats(trades, label))
        print(f"      {label:<16} → {len(trades)} trades  | {desc}", flush=True)

    # Comparison table
    print("\n" + "=" * 95)
    print("FINAL ANALYSIS — merged Σ·10 + ΣOI, SL=0.18% / TGT=0.50%, position=single, 3-min")
    print("=" * 95)
    print(f"{'Variant':<16} {'n':>5} {'/day':>5} {'WR%':>6} {'TotPnL':>9} {'avg*':>7} {'PF':>5} "
          f"{'SL/TGT/EOD/TRL':>15}")
    print('-' * 95)
    for r in rows:
        if not r["n"]:
            print(f"{r['label']:<16} (no trades)"); continue
        pf = f"{r['pf']:.2f}" if r["pf"] is not None else "inf"
        print(f"{r['label']:<16} {r['n']:>5} {r['per_day']:>5.2f} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+9.0f} {r['avg_ex_trail']:>+7.2f} {pf:>5} "
              f"{r['sl']:>3}/{r['tgt']:>3}/{r['eod']:>3}/{r['trail']:>3}")
    print("=" * 95)

    # Save raw rows
    out = Path("trail_variants_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved → {out.resolve()}")


if __name__ == "__main__":
    main()
