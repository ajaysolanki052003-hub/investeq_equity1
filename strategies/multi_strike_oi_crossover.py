"""Multi-Strike OI Crossover (Index Intraday) — Steps 1 & 2.

For each trading day:
  • Step 1 — Sum_Call_OI and Sum_Put_OI at every tick are already pre-computed
             in DATA/_atm_oi_intraday.parquet (sum of CE/PE OI on the live
             ATM and its two flanking strikes, ATM recomputed each tick).
  • Step 2 — At ENTRY_WINDOW_START (default 09:45 IST), record the morning
             baseline as CALL_HEAVY or PUT_HEAVY. Then walk forward and fire
             the FIRST-time crossover during the entry window:
                BULLISH  = CALL_HEAVY bias + Sum_Put_OI  >= Sum_Call_OI
                BEARISH  = PUT_HEAVY  bias + Sum_Call_OI >= Sum_Put_OI

Step 3 (price-filter via Volume Profile POC) is intentionally NOT here;
the user asked for Steps 1 & 2 only at this stage, then to visualize.
"""

from __future__ import annotations

import os
from datetime import time as dtime
from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
OI_INTRA = ROOT / "_atm_oi_intraday.parquet"


# Defaults — exposed as function args so they can be tuned without code change.
ENTRY_WINDOW_START = dtime(9, 45)
ENTRY_WINDOW_END   = dtime(14, 45)


@lru_cache(maxsize=1)
def _load_oi() -> pd.DataFrame:
    df = pd.read_parquet(OI_INTRA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_signals(window_start: dtime = ENTRY_WINDOW_START,
                    window_end:   dtime = ENTRY_WINDOW_END) -> list[dict]:
    """Run Steps 1 + 2 across every day in the OI aggregate.

    Returns a list of dicts, one entry per day THAT FIRED a signal. Days
    without a crossover are silently skipped (still reported in summary)."""
    df = _load_oi()
    df["date"] = df["timestamp"].dt.date
    df["t"]    = df["timestamp"].dt.time

    out = []
    stats = {"bull": 0, "bear": 0, "no_signal": 0,
             "no_baseline": 0, "no_intraday": 0}

    for day, grp in df.groupby("date"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        # Need intraday ticks (more than just the EOD snapshot)
        if len(grp) <= 2:
            stats["no_intraday"] += 1
            continue

        baseline_rows = grp[grp["t"] >= window_start]
        if baseline_rows.empty:
            stats["no_baseline"] += 1
            continue
        baseline = baseline_rows.iloc[0]
        if baseline["ce_oi"] > baseline["pe_oi"]:
            bias = "CALL_HEAVY"
        elif baseline["pe_oi"] > baseline["ce_oi"]:
            bias = "PUT_HEAVY"
        else:
            # Exactly equal at baseline — call it neutral, skip.
            stats["no_signal"] += 1
            continue

        # Walk forward inside the entry window for first crossover.
        future = grp[(grp["timestamp"] > baseline["timestamp"]) &
                     (grp["t"] <= window_end)]
        signal_row, signal_kind = None, None
        for _, row in future.iterrows():
            if bias == "CALL_HEAVY" and row["pe_oi"] >= row["ce_oi"]:
                signal_row, signal_kind = row, "BULLISH"
                break
            if bias == "PUT_HEAVY" and row["ce_oi"] >= row["pe_oi"]:
                signal_row, signal_kind = row, "BEARISH"
                break

        if signal_row is None:
            stats["no_signal"] += 1
            continue

        stats["bull" if signal_kind == "BULLISH" else "bear"] += 1
        out.append({
            "day":           str(day),
            "bias":          bias,
            "signal":        signal_kind,
            "baseline_time": baseline["timestamp"].isoformat(),
            "baseline_ce":   float(baseline["ce_oi"]),
            "baseline_pe":   float(baseline["pe_oi"]),
            "signal_time":   signal_row["timestamp"].isoformat(),
            "signal_ce":     float(signal_row["ce_oi"]),
            "signal_pe":     float(signal_row["pe_oi"]),
            "signal_spot":   float(signal_row["spot"]),
            "signal_atm":    int(signal_row["atm"]),
            # Day's 09:15 IST timestamp — used to anchor markers on the 1d
            # candle (which is bucketed at 09:15).
            "day_open_time": pd.Timestamp(day).replace(hour=9, minute=15).isoformat(),
        })
    return out, stats


if __name__ == "__main__":
    sigs, stats = compute_signals()
    print(f"Days that fired:")
    print(f"  BULLISH  = {stats['bull']:>4}")
    print(f"  BEARISH  = {stats['bear']:>4}")
    print(f"Skipped:")
    print(f"  no signal (no crossover) = {stats['no_signal']}")
    print(f"  no baseline (no 09:45 tick) = {stats['no_baseline']}")
    print(f"  no intraday (flat day) = {stats['no_intraday']}")
    print(f"Total signals: {len(sigs)}")
    if sigs:
        print(f"First: {sigs[0]}")
        print(f"Last : {sigs[-1]}")
