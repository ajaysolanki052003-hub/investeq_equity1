"""ΣOI Entry (Index Intraday) — ENTRIES ONLY.

Trigger:
  • At every tick, compute ΣOI = per-day cumulative sum of ATM±1 OI from
    09:15 IST (reset each day):
        ce_tot[t] = sum(ce_oi[09:15 .. t])
        pe_tot[t] = sum(pe_oi[09:15 .. t])
  • LONG  fires when pe_tot crosses ABOVE ce_tot inside the entry window
          [09:24–14:45 IST] — PE side's accumulated exposure since open
          has overtaken CE's.
  • SHORT fires on the mirror cross.
  • EVERY qualifying cross inside the entry window fires an entry — multiple
    entries per day are allowed (back-to-back LONG/SHORT/LONG flips as the
    ΣOI lines oscillate). No exit logic yet — by user request, entries are
    validated first.

Why ΣOI cross is a stronger signal than a raw-OI cross:
  Both ce_tot and pe_tot accumulate forward, so for one to overtake the
  other the trailing side needs SUSTAINED per-minute build-up — a single
  noisy minute can't flip the sign. Crosses here mean "the dominant side
  has flipped on a since-open basis," which is rarer and higher-conviction
  than a raw OI snapshot flip.
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


ENTRY_WINDOW_START = dtime(9, 24)   # 9 min after open — first few real OI updates
                                    # have landed (Groww OI ticks every ~3 min).
ENTRY_WINDOW_END   = dtime(14, 45)


@lru_cache(maxsize=1)
def _load_oi() -> pd.DataFrame:
    df = pd.read_parquet(OI_INTRA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _add_tot(df: pd.DataFrame) -> pd.DataFrame:
    """Add ce_tot / pe_tot = per-day cumulative sum of OI from the day's
    first tick (typically 09:15 IST). Resets at each new date."""
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    d["ce_tot"] = d.groupby("date", sort=False)["ce_oi"].cumsum()
    d["pe_tot"] = d.groupby("date", sort=False)["pe_oi"].cumsum()
    return d


def compute_entries(entry_start: dtime = ENTRY_WINDOW_START,
                    entry_end:   dtime = ENTRY_WINDOW_END,
                    min_gap_bars: int = 7
                    ) -> tuple[list[dict], dict]:
    """EVERY qualifying ΣOI cross per day inside the entry window, with a
    whipsaw filter applied: a new cross is only valid if at least
    `min_gap_bars` 1-min bars have passed since the previously-accepted
    cross on the same day. Default 7 — best by backtest (lifts profit
    factor from 1.34 → 1.41 with minimal trade count loss). Set
    min_gap_bars=0 to disable.

    Multiple entries per day are still allowed; the filter only debounces
    flip-flop signals fired within the gap window."""
    df = _add_tot(_load_oi())
    df["t"]   = df["timestamp"].dt.time
    df["pe_minus_ce_tot"] = df["pe_tot"] - df["ce_tot"]

    entries = []
    stats = {"long": 0, "short": 0, "no_signal": 0, "no_intraday": 0,
             "days_fired": 0, "multi_entry_days": 0}

    for day, grp in df.groupby("date", sort=True):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        if len(grp) < 10:
            stats["no_intraday"] += 1
            continue

        day_count = 0
        last_accepted_ts = None    # for the min-gap whipsaw filter
        for i in range(1, len(grp)):
            curr = grp.iloc[i]
            t    = curr["t"]
            if t < entry_start: continue
            if t > entry_end:   break

            prev_diff = grp.iloc[i-1]["pe_minus_ce_tot"]
            curr_diff = curr["pe_minus_ce_tot"]

            side = None
            if prev_diff <= 0 and curr_diff > 0:
                side = "LONG"     # PE cumulative crossed above CE cumulative
            elif prev_diff >= 0 and curr_diff < 0:
                side = "SHORT"    # CE cumulative crossed above PE cumulative
            if side is None:
                continue

            # Next-candle entry: the cross is detected on bar i (when bar i
            # closes); the actual trade fills on the FOLLOWING bar's
            # timestamp + spot. Matches how a live trader reacts — see the
            # cross at the end of the minute, enter on next minute's open.
            if i + 1 >= len(grp):
                continue   # no next bar in this day
            nxt = grp.iloc[i + 1]

            # Min-gap whipsaw filter uses the actual entry timestamp (nxt).
            if min_gap_bars > 0 and last_accepted_ts is not None:
                gap_min = (nxt["timestamp"] - last_accepted_ts).total_seconds() / 60.0
                if gap_min < min_gap_bars:
                    continue
            last_accepted_ts = nxt["timestamp"]

            entries.append({
                "day":            str(day),
                "side":           side,
                "entry_time":     nxt["timestamp"].isoformat(),
                "entry_spot":     float(nxt["spot"]),
                "entry_atm":      int(nxt["atm"]),
                "entry_ce_oi":    float(nxt["ce_oi"]),
                "entry_pe_oi":    float(nxt["pe_oi"]),
                "entry_ce_tot":   float(nxt["ce_tot"]),
                "entry_pe_tot":   float(nxt["pe_tot"]),
                "day_open_time":  pd.Timestamp(day).replace(hour=9, minute=15).isoformat(),
            })
            stats["long" if side == "LONG" else "short"] += 1
            day_count += 1

        if day_count == 0:
            stats["no_signal"] += 1
        else:
            stats["days_fired"] += 1
            if day_count > 1:
                stats["multi_entry_days"] += 1

    return entries, stats


if __name__ == "__main__":
    entries, stats = compute_entries()
    print(f"ΣOI Entries — {len(entries)} days fired")
    print(f"  LONG  = {stats['long']:>4}   SHORT = {stats['short']:>4}")
    print(f"  skipped: no_signal={stats['no_signal']}  "
          f"no_intraday={stats['no_intraday']}")
    if entries:
        print(f"\nFirst: {entries[0]}")
        print(f"Last : {entries[-1]}")
