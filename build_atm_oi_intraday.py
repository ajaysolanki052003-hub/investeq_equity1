"""Build DATA/_atm_oi_intraday.parquet — for every OI timestamp in the per-day
files, sum CE/PE OI at the 3 strikes around the *live* ATM (ATM is recomputed
at each timestamp from the spot master so the rolling window walks with price).

Schema:
    timestamp   datetime64[ns]
    ce_oi       float (sum CE OI at ATM-50, ATM, ATM+50 on nearest expiry)
    pe_oi       float
    atm         int    (the strike used as ATM at that moment)
    spot        float  (spot price used to compute ATM)

Run after fetch_oi_history.py to keep the aggregate current. Cheap — re-builds
from scratch in ~30s on the VM.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
OI_DIR        = ROOT / "oi"
SPOT_MASTER   = ROOT / "nifty_1m_master.parquet"
OUT           = ROOT / "_atm_oi_intraday.parquet"

STRIKE_STEP   = 50


def _load_spot() -> pd.DataFrame:
    if not SPOT_MASTER.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SPOT_MASTER, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _spot_lookup_array(spot: pd.DataFrame):
    """Build numpy arrays for fast as-of-timestamp spot lookup via searchsorted."""
    if spot.empty:
        return np.array([], dtype="int64"), np.array([], dtype="float64")
    ts_ns = spot["timestamp"].view("int64").to_numpy()
    cl    = spot["close"].astype("float64").to_numpy()
    return ts_ns, cl


def _process_day(p: Path, spot_ts: np.ndarray, spot_cl: np.ndarray) -> pd.DataFrame:
    """Aggregate one day's OI parquet into a (timestamp, ce_oi, pe_oi, atm,
    spot) row per OI tick."""
    try:
        df = pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()
    if df.empty or len(spot_ts) == 0:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["expiry_dt"] = pd.to_datetime(df["expiry"]).dt.date

    # Pick the nearest expiry on/after this trading day.
    day = p.stem
    day_date = pd.to_datetime(day, format="%Y%m%d").date()
    forward = sorted({d for d in df["expiry_dt"] if d >= day_date})
    if not forward:
        return pd.DataFrame()
    near = forward[0]
    df = df[df["expiry_dt"] == near]
    if df.empty:
        return pd.DataFrame()

    # Resolve spot at each timestamp via searchsorted (as-of nearest preceding bar)
    tick_ts = df["timestamp"].view("int64").to_numpy()
    idx = np.searchsorted(spot_ts, tick_ts, side="right") - 1
    idx = np.clip(idx, 0, len(spot_ts) - 1)
    spot_at_tick = spot_cl[idx]

    # Compute ATM per tick
    atm = (np.round(spot_at_tick / STRIKE_STEP) * STRIKE_STEP).astype("int64")
    df = df.assign(_atm=np.repeat(atm, 1)[:len(df)])
    # NB: np.repeat(atm, 1) is a no-op cast; we computed atm aligned to df rows.

    # Keep only rows where strike ∈ {atm-50, atm, atm+50}
    in_window = df["strike"].astype("int64").to_numpy()
    keep = (np.abs(in_window - df["_atm"].to_numpy()) <= STRIKE_STEP)
    df = df[keep]
    if df.empty:
        return pd.DataFrame()

    # Sum OI per (timestamp, option_type)
    agg = (df.groupby(["timestamp", "option_type"], as_index=False)["oi"]
              .sum())
    ce = agg[agg["option_type"] == "CE"][["timestamp", "oi"]].rename(columns={"oi": "ce_oi"})
    pe = agg[agg["option_type"] == "PE"][["timestamp", "oi"]].rename(columns={"oi": "pe_oi"})
    merged = pd.merge(ce, pe, on="timestamp", how="outer").fillna(0)

    # Attach the ATM used at each timestamp (first row per timestamp is fine —
    # the same ATM was applied to every strike at that tick)
    ts_to_atm = (df[["timestamp", "_atm"]]
                   .drop_duplicates(subset=["timestamp"]))
    merged = pd.merge(merged, ts_to_atm.rename(columns={"_atm": "atm"}),
                      on="timestamp", how="left")

    # Spot at each timestamp
    ttn = merged["timestamp"].view("int64").to_numpy()
    sidx = np.searchsorted(spot_ts, ttn, side="right") - 1
    sidx = np.clip(sidx, 0, len(spot_ts) - 1)
    merged["spot"] = spot_cl[sidx]

    return merged[["timestamp", "ce_oi", "pe_oi", "atm", "spot"]]


def main() -> int:
    print(f"[load] spot master ...", flush=True)
    spot = _load_spot()
    if spot.empty:
        print("[fatal] no spot master found — run fetch_nifty_master.py first.",
              flush=True)
        return 1
    spot_ts, spot_cl = _spot_lookup_array(spot)
    print(f"        {len(spot):,} 1m bars  ({spot['timestamp'].min()} → "
          f"{spot['timestamp'].max()})", flush=True)

    files = sorted(OI_DIR.glob("*.parquet"))
    print(f"[scan] {len(files):,} OI files", flush=True)

    rows = []
    t0 = time.time()
    for i, p in enumerate(files, start=1):
        out = _process_day(p, spot_ts, spot_cl)
        if not out.empty:
            rows.append(out)
        if i % 250 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  [{i:5d}/{len(files):5d}]  rows={sum(len(r) for r in rows):,}  "
                  f"({el:.1f}s)", flush=True)

    if not rows:
        print("[fatal] no rows aggregated — aborting (no OI files matched).",
              flush=True)
        return 1
    full = (pd.concat(rows, ignore_index=True)
              .sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last")
              .reset_index(drop=True))

    tmp = OUT.with_suffix(".parquet.tmp")
    full.to_parquet(tmp, index=False)
    tmp.replace(OUT)
    print(f"[save] {OUT}  rows={len(full):,}  ({OUT.stat().st_size/1024:.0f} KB)",
          flush=True)
    print(f"        range: {full['timestamp'].min()}  →  {full['timestamp'].max()}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
