"""Σ-N OI Entry (Index Intraday) — ENTRIES ONLY, TF-aware.

Trigger:
  • Resample raw 1-min OI to the requested TF (last-aggregation per bucket).
  • At every TF-bucket, compute Σ-N = rolling N-bucket sum of CE / PE OI
    (ATM±1), reset per day. Default N = 10.
  • LONG  fires when Σ-N PE crosses ABOVE Σ-N CE, inside the entry window
          [09:45–14:45 IST], AND raw PE OI > raw CE OI at the same tick
          (smoothed direction confirmed by current state).
  • SHORT mirror: Σ-N CE crosses above Σ-N PE + raw CE > raw PE.
  • First qualifying cross of the day fires the entry; rest of the day
    is ignored (one entry per day). No exit logic yet — by user request,
    entries are validated first.

TF semantics:
    "10 candles of current TF":
        1m TF  → last 10 minutes of OI
        3m TF  → last 30 minutes
        5m TF  → last 50 minutes
        15m TF → last 150 minutes  (2.5 trading hours)
        1h TF  → last 10 hours     (covers ~1.5 trading days)
        1d TF  → last 10 trading days
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


ENTRY_WINDOW_START = dtime(9, 45)
ENTRY_WINDOW_END   = dtime(14, 45)
SIGMA_WINDOW       = 10

# Same TF rules as oi_options_trading/app.py — kept here so the strategy
# is self-contained (no cross-module import dance).
TF_RULE = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "60min", "4h": "240min", "1d": "1D",
}


@lru_cache(maxsize=1)
def _load_oi() -> pd.DataFrame:
    df = pd.read_parquet(OI_INTRA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _resample_oi(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample raw 1-min OI to the requested TF, last-aggregation per bucket.
    1d pins each row to that day's 09:15 IST so marker placement on a 1d
    candle is straightforward."""
    if tf == "1m":
        return df.copy()
    if tf == "1d":
        d = df.copy()
        d["day"] = d["timestamp"].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
        out = (d.groupby("day", as_index=False)
                 .agg(ce_oi=("ce_oi", "last"),
                      pe_oi=("pe_oi", "last"),
                      spot=("spot",  "last"),
                      atm=("atm",   "last")))
        return out.rename(columns={"day": "timestamp"})
    rule = TF_RULE.get(tf)
    if rule is None:
        raise ValueError(f"unknown tf: {tf}")
    out = (df.set_index("timestamp")
             .resample(rule, closed="left", label="left")
             .agg({"ce_oi": "last", "pe_oi": "last",
                   "spot":  "last", "atm":  "last"})
             .dropna(subset=["ce_oi", "pe_oi"])
             .reset_index())
    return out


def _add_sigma(df: pd.DataFrame, window: int = SIGMA_WINDOW) -> pd.DataFrame:
    """Add ce_sum / pe_sum = rolling N-bucket sum, reset per-day so the open
    of a new day doesn't sum across the prior session."""
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    d["ce_sum"] = (d.groupby("date", sort=False)["ce_oi"]
                    .transform(lambda s: s.rolling(window, min_periods=1).sum()))
    d["pe_sum"] = (d.groupby("date", sort=False)["pe_oi"]
                    .transform(lambda s: s.rolling(window, min_periods=1).sum()))
    return d


def _cross_up(prev_diff: float, curr_diff: float) -> bool:
    return prev_diff <= 0 and curr_diff > 0


def compute_entries(tf: str = "1m",
                    window: int = SIGMA_WINDOW,
                    entry_start: dtime = ENTRY_WINDOW_START,
                    entry_end:   dtime = ENTRY_WINDOW_END) -> tuple[list[dict], dict]:
    """Σ-N entries on the requested TF.

    Intraday TFs:
      Per-day reset rolling sum; one entry per day; gated to the
      09:45–14:45 IST entry window.
    1d TF:
      Continuous rolling sum across the full multi-year series; every
      qualifying cross between adjacent daily buckets fires an entry; NO
      time-of-day gate (each daily bucket is pinned at 09:15 — a wall-
      clock gate would reject all of them).
    """
    df = _resample_oi(_load_oi(), tf).sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date

    # Rolling sum strategy depends on TF: per-day reset for intraday so the
    # open of a new session doesn't sum across the prior day; continuous
    # for 1d so the 10-bucket window actually spans 10 days.
    if tf == "1d":
        df["ce_sum"] = df["ce_oi"].rolling(window, min_periods=1).sum()
        df["pe_sum"] = df["pe_oi"].rolling(window, min_periods=1).sum()
    else:
        df["ce_sum"] = (df.groupby("date", sort=False)["ce_oi"]
                          .transform(lambda s: s.rolling(window, min_periods=1).sum()))
        df["pe_sum"] = (df.groupby("date", sort=False)["pe_oi"]
                          .transform(lambda s: s.rolling(window, min_periods=1).sum()))

    df["t"] = df["timestamp"].dt.time
    df["pe_minus_ce_sigma"] = df["pe_sum"] - df["ce_sum"]

    entries = []
    stats = {"long": 0, "short": 0, "no_signal": 0, "no_intraday": 0,
             "tf": tf, "window": window}

    def _emit(curr_row):
        entries.append({
            "day":           str(curr_row["date"]),
            "side":          curr_row["__side"],
            "entry_time":    curr_row["timestamp"].isoformat(),
            "entry_spot":    float(curr_row["spot"]),
            "entry_atm":     int(curr_row["atm"]),
            "entry_ce_oi":   float(curr_row["ce_oi"]),
            "entry_pe_oi":   float(curr_row["pe_oi"]),
            "entry_ce_sum":  float(curr_row["ce_sum"]),
            "entry_pe_sum":  float(curr_row["pe_sum"]),
            "day_open_time": pd.Timestamp(curr_row["date"])
                              .replace(hour=9, minute=15).isoformat(),
        })
        stats["long" if curr_row["__side"] == "LONG" else "short"] += 1

    def _side_for(prev_diff, curr_diff, pe_oi, ce_oi):
        pe_up = _cross_up(prev_diff,  curr_diff)
        ce_up = _cross_up(-prev_diff, -curr_diff)
        if pe_up and pe_oi > ce_oi: return "LONG"
        if ce_up and ce_oi > pe_oi: return "SHORT"
        return None

    if tf == "1d":
        # Walk every daily bucket in date order — no per-day grouping,
        # no time-gate, no one-entry-per-day cap.
        for i in range(1, len(df)):
            prev_diff = df.iloc[i-1]["pe_minus_ce_sigma"]
            curr      = df.iloc[i]
            side = _side_for(prev_diff, curr["pe_minus_ce_sigma"],
                             curr["pe_oi"], curr["ce_oi"])
            if side is None:
                continue
            row = curr.copy(); row["__side"] = side
            _emit(row)
    else:
        # Per-day reset, time-gated, one entry per day.
        # Next-candle entry: the cross is detected on bar i; the trade
        # fills on bar i+1's timestamp + spot.
        for day, grp in df.groupby("date", sort=True):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            if len(grp) < max(3, window // 2):
                stats["no_intraday"] += 1
                continue
            fired = False
            for i in range(1, len(grp)):
                curr = grp.iloc[i]
                t    = curr["t"]
                if t < entry_start: continue
                if t > entry_end:   break
                side = _side_for(grp.iloc[i-1]["pe_minus_ce_sigma"],
                                 curr["pe_minus_ce_sigma"],
                                 curr["pe_oi"], curr["ce_oi"])
                if side is None: continue
                # Take next bar after the cross for actual entry
                if i + 1 >= len(grp): continue
                nxt = grp.iloc[i + 1]
                row = nxt.copy()
                row["date"] = curr["date"]   # preserve day grouping
                row["__side"] = side
                _emit(row)
                fired = True
                break
            if not fired:
                stats["no_signal"] += 1

    return entries, stats


if __name__ == "__main__":
    import sys
    tf = sys.argv[1] if len(sys.argv) > 1 else "1m"
    entries, stats = compute_entries(tf=tf)
    print(f"Σ·{SIGMA_WINDOW} OI Entries (tf={tf}) — {len(entries)} days fired")
    print(f"  LONG  = {stats['long']:>4}   SHORT = {stats['short']:>4}")
    print(f"  skipped: no_signal={stats['no_signal']}  "
          f"no_intraday={stats['no_intraday']}")
    if entries:
        print(f"\nFirst: {entries[0]}")
        print(f"Last : {entries[-1]}")
