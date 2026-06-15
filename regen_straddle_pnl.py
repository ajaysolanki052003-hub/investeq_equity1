"""Rebuild DATA/straddle_pnl.parquet (Volatility Desk) from the per-day
spot/options files. Deployable version of _regen_straddle_pnl.py — paths come
from INVESTEQ_DATA and there is no notebook patching.

Strategy (weekly short straddle):
  * Entry : sell ATM straddle at 15:20 on the day BEFORE expiry.
  * Exit  : buy back the SAME strike at 09:20 on expiry morning.
  * Bar OPEN at the nearest timestamp is used (reconciles with the replay UI).

Run:
    INVESTEQ_DATA=/home/ajay/investeq_ajs/DATA python regen_straddle_pnl.py
"""

from __future__ import annotations
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT      = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
OPT_DIR   = ROOT / "options"
SPOT_DIR  = ROOT / "spot"
INDEX_DIR = ROOT / "index"
PNL_PARQUET = ROOT / "straddle_pnl.parquet"
PNL_CSV     = ROOT / "straddle_pnl.csv"

ENTRY_TIME = "15:20:00"
EXIT_TIME  = "09:20:00"


def _atm_strike(opt_exp_df, spot_price):
    strikes = opt_exp_df["strike"].unique()
    return int(strikes[np.argmin(np.abs(strikes - spot_price))])


def _leg_open_at(sub_df, otype, target_ts):
    s = sub_df[sub_df["option_type"] == otype]
    if s.empty:
        return None
    i = (s["timestamp"] - target_ts).abs().idxmin()
    v = s.loc[i, "open"]
    return float(v) if pd.notna(v) else None


def _spot_at(spot_df, target_ts):
    spot_at = spot_df[spot_df["timestamp"] <= target_ts]
    return float((spot_at if not spot_at.empty else spot_df).iloc[-1]["close"])


def _load(date_str):
    try:
        opt  = pd.read_parquet(OPT_DIR  / f"{date_str}.parquet")
        spot = pd.read_parquet(SPOT_DIR / f"{date_str}.parquet")
    except FileNotFoundError:
        return None, None
    opt["timestamp"]  = pd.to_datetime(opt["timestamp"])
    spot["timestamp"] = pd.to_datetime(spot["timestamp"])
    return opt, spot


def get_atm_straddle_snapshot(date_str, expiry_str, target_time):
    opt, spot = _load(date_str)
    if opt is None:
        return None
    target_ts = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {target_time}")
    opt_exp = opt[opt["expiry"] == expiry_str]
    if opt_exp.empty:
        return None
    spot_price = _spot_at(spot, target_ts)
    strike     = _atm_strike(opt_exp, spot_price)
    sub        = opt_exp[opt_exp["strike"] == strike]
    ce = _leg_open_at(sub, "CE", target_ts)
    pe = _leg_open_at(sub, "PE", target_ts)
    if ce is None or pe is None:
        return None
    return {"atm_strike": strike, "ce_price": ce, "pe_price": pe,
            "spot": spot_price, "straddle_premium": ce + pe}


def get_leg_prices_at_strike(date_str, expiry_str, strike, target_time):
    opt, spot = _load(date_str)
    if opt is None:
        return None
    target_ts = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {target_time}")
    sub = opt[(opt["expiry"] == expiry_str) & (opt["strike"] == strike)]
    if sub.empty:
        return None
    ce = _leg_open_at(sub, "CE", target_ts)
    pe = _leg_open_at(sub, "PE", target_ts)
    if ce is None or pe is None:
        return None
    return {"atm_strike": strike, "ce_price": ce, "pe_price": pe,
            "spot": _spot_at(spot, target_ts), "straddle_premium": ce + pe}


def regenerate_pnl():
    expiries = pd.read_parquet(INDEX_DIR / "expiries.parquet")
    expiries["expiry"] = pd.to_datetime(expiries["expiry"])
    expiries = expiries.sort_values("expiry").reset_index(drop=True)

    trading_days = sorted(os.path.basename(f)[:8]
                          for f in glob.glob(str(SPOT_DIR / "*.parquet")))
    td_dt = pd.to_datetime(trading_days)

    events = []
    for _, row in expiries.iterrows():
        exp_str = row["expiry"].strftime("%Y%m%d")
        if exp_str not in trading_days:
            continue
        prev = td_dt[td_dt < row["expiry"]]
        if len(prev) == 0:
            continue
        entry_str = prev[-1].strftime("%Y%m%d")
        if entry_str not in trading_days:
            continue
        events.append({"expiry": row["expiry"], "expiry_str": exp_str,
                       "entry_str": entry_str})

    print(f"events to process: {len(events)}")
    results = []
    for i, row in enumerate(events):
        if i % 20 == 0:
            print(f"  {i}/{len(events)}  {row['entry_str']} -> {row['expiry_str']}")
        expiry_str = row["expiry"].strftime("%Y-%m-%d")
        entry = get_atm_straddle_snapshot(row["entry_str"], expiry_str, ENTRY_TIME)
        if entry is None:
            continue
        exit_ = get_leg_prices_at_strike(row["expiry_str"], expiry_str,
                                         entry["atm_strike"], EXIT_TIME)
        if exit_ is None:
            continue
        pnl = entry["straddle_premium"] - exit_["straddle_premium"]
        results.append({
            "expiry"           : row["expiry"],
            "entry_date"       : row["entry_str"],
            "atm_strike"       : entry["atm_strike"],
            "spot_at_entry"    : entry["spot"],
            "spot_at_exit"     : exit_["spot"],
            "premium_collected": entry["straddle_premium"],
            "premium_at_exit"  : exit_["straddle_premium"],
            "pnl_points"       : pnl,
            "pnl_pct"          : pnl / entry["straddle_premium"],
            "profitable"       : int(pnl > 0),
        })

    df = pd.DataFrame(results).sort_values("expiry").reset_index(drop=True)
    tmp = PNL_PARQUET.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False); tmp.replace(PNL_PARQUET)
    df.to_csv(PNL_CSV, index=False)
    print(f"\nWrote {len(df)} events to {PNL_PARQUET}")
    if len(df):
        print(f"  wins {int(df['profitable'].sum())} ({df['profitable'].mean()*100:.1f}%)  "
              f"cum {df['pnl_points'].sum():+.1f} pts  "
              f"last expiry {df['expiry'].max().date()}")
    return df


if __name__ == "__main__":
    regenerate_pnl()
