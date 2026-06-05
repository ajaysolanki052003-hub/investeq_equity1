"""
One-off: patch nifty_options_ml.ipynb cells 14 + 15 with the corrected
straddle-P&L logic and regenerate DATA/straddle_pnl.parquet.

Fix summary
-----------
The earlier notebook called get_atm_straddle_snapshot() at BOTH entry and exit,
which re-ATM'd the strike to the spot price at each moment. For a real short
straddle you must buy back the SAME strike you sold — re-ATM-ing under-states
tail losses whenever spot gaps through the original strike overnight.

The fix:
  1. Entry: pick ATM from entry-day spot at 15:20:00 IST.
  2. Exit : read CE+PE prices for THAT SAME STRIKE on expiry day at 09:20:00 IST.
  3. Use the bar OPEN at the timestamp nearest the target minute (matches the
     /api/event_detail endpoint in app_option_replay_tv.py so the dropdown,
     chart marker, and detail panel all reconcile).
"""

from __future__ import annotations
import os, sys, glob, json, io
import numpy as np
import pandas as pd

ROOT_DIR = r"C:\Users\User\Desktop\investeq_ajs"
DATA_DIR = os.path.join(ROOT_DIR, "DATA")
OPT_DIR  = os.path.join(DATA_DIR, "options")
SPOT_DIR = os.path.join(DATA_DIR, "spot")
INDEX_DIR = os.path.join(DATA_DIR, "index")
NB_PATH  = os.path.join(ROOT_DIR, "nifty_options_ml.ipynb")
PNL_PARQUET = os.path.join(DATA_DIR, "straddle_pnl.parquet")
PNL_CSV     = os.path.join(DATA_DIR, "straddle_pnl.csv")

ENTRY_TIME = "15:20:00"
EXIT_TIME  = "09:20:00"


# ─────────────────────────── price-lookup helpers ───────────────────────────
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


def get_atm_straddle_snapshot(date_str, expiry_str, target_time):
    try:
        opt  = pd.read_parquet(os.path.join(OPT_DIR,  f"{date_str}.parquet"))
        spot = pd.read_parquet(os.path.join(SPOT_DIR, f"{date_str}.parquet"))
    except FileNotFoundError:
        return None
    opt["timestamp"]  = pd.to_datetime(opt["timestamp"])
    spot["timestamp"] = pd.to_datetime(spot["timestamp"])
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
    try:
        opt  = pd.read_parquet(os.path.join(OPT_DIR,  f"{date_str}.parquet"))
        spot = pd.read_parquet(os.path.join(SPOT_DIR, f"{date_str}.parquet"))
    except FileNotFoundError:
        return None
    opt["timestamp"]  = pd.to_datetime(opt["timestamp"])
    spot["timestamp"] = pd.to_datetime(spot["timestamp"])
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


# ─────────────────────────── regenerate parquet ─────────────────────────────
def regenerate_pnl():
    expiries = pd.read_parquet(os.path.join(INDEX_DIR, "expiries.parquet"))
    expiries["expiry"] = pd.to_datetime(expiries["expiry"])
    expiries = expiries.sort_values("expiry").reset_index(drop=True)

    trading_days = sorted(os.path.basename(f)[:8] for f in glob.glob(os.path.join(SPOT_DIR, "*.parquet")))
    td_dt = pd.to_datetime(trading_days)

    events = []
    for _, row in expiries.iterrows():
        exp_date = row["expiry"]
        exp_str  = exp_date.strftime("%Y%m%d")
        if exp_str not in trading_days:
            continue
        prev = td_dt[td_dt < exp_date]
        if len(prev) == 0:
            continue
        entry_str = prev[-1].strftime("%Y%m%d")
        if entry_str not in trading_days:
            continue
        events.append({"expiry": exp_date, "expiry_str": exp_str, "entry_str": entry_str})

    print(f"events to process: {len(events)}")

    results = []
    for i, row in enumerate(events):
        if i % 20 == 0:
            print(f"  {i}/{len(events)}  {row['entry_str']} -> {row['expiry_str']}")
        expiry_str = row["expiry"].strftime("%Y-%m-%d")
        entry = get_atm_straddle_snapshot(row["entry_str"], expiry_str, ENTRY_TIME)
        if entry is None:
            continue
        exit_ = get_leg_prices_at_strike(row["expiry_str"], expiry_str, entry["atm_strike"], EXIT_TIME)
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
    df.to_parquet(PNL_PARQUET, index=False)
    df.to_csv(PNL_CSV, index=False)

    print(f"\nWrote {len(df)} events to {PNL_PARQUET}")
    print(f"  wins  : {int(df['profitable'].sum()):>4d}   ({df['profitable'].mean()*100:.1f}%)")
    print(f"  losses: {int((1 - df['profitable']).sum()):>4d}")
    print(f"  cum PnL pts : {df['pnl_points'].sum():+.2f}")
    print(f"  mean / median : {df['pnl_points'].mean():+.2f} / {df['pnl_points'].median():+.2f}")
    print(f"  worst trade   : {df['pnl_points'].min():+.2f}  (was -125.22 before fix)")
    print(f"  best  trade   : {df['pnl_points'].max():+.2f}")
    return df


# ─────────────────────────── patch notebook source ──────────────────────────
NEW_CELL_14 = """\
# ── 3.2 Extract straddle entry and exit prices ─────────────────────────────────
# Convention used everywhere downstream (matches the /api/event_detail endpoint
# in app_option_replay_tv.py): the bar OPEN at the timestamp nearest the target
# minute. The exit leg is read from the SAME strike that was sold at entry —
# you cannot re-ATM at expiry, you must buy back what you sold.

ENTRY_TIME = '15:20:00'   # sell straddle
EXIT_TIME  = '09:20:00'   # close on expiry morning


def _atm_strike(opt_exp_df, spot_price):
    strikes = opt_exp_df['strike'].unique()
    return int(strikes[np.argmin(np.abs(strikes - spot_price))])


def _leg_open_at(sub_df, otype, target_ts):
    \"\"\"Bar OPEN price for one option_type at the timestamp nearest target_ts.\"\"\"
    s = sub_df[sub_df['option_type'] == otype]
    if s.empty:
        return None
    i = (s['timestamp'] - target_ts).abs().idxmin()
    v = s.loc[i, 'open']
    return float(v) if pd.notna(v) else None


def _spot_at(spot_df, target_ts):
    \"\"\"Most-recent spot close at or before target_ts.\"\"\"
    spot_at = spot_df[spot_df['timestamp'] <= target_ts]
    return float((spot_at if not spot_at.empty else spot_df).iloc[-1]['close'])


def get_atm_straddle_snapshot(date_str, expiry_str, target_time):
    \"\"\"Pick ATM at target_time; return strike + CE/PE OPEN prices.\"\"\"
    try:
        opt  = pd.read_parquet(os.path.join(OPT_DIR,  f'{date_str}.parquet'))
        spot = pd.read_parquet(os.path.join(SPOT_DIR, f'{date_str}.parquet'))
    except FileNotFoundError:
        return None
    opt['timestamp']  = pd.to_datetime(opt['timestamp'])
    spot['timestamp'] = pd.to_datetime(spot['timestamp'])
    target_ts = pd.to_datetime(f\"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {target_time}\")
    opt_exp = opt[opt['expiry'] == expiry_str]
    if opt_exp.empty:
        return None
    spot_price = _spot_at(spot, target_ts)
    strike     = _atm_strike(opt_exp, spot_price)
    sub        = opt_exp[opt_exp['strike'] == strike]
    ce = _leg_open_at(sub, 'CE', target_ts)
    pe = _leg_open_at(sub, 'PE', target_ts)
    if ce is None or pe is None:
        return None
    return {
        'atm_strike'      : strike,
        'ce_price'        : ce,
        'pe_price'        : pe,
        'spot'            : spot_price,
        'straddle_premium': ce + pe,
    }


def get_leg_prices_at_strike(date_str, expiry_str, strike, target_time):
    \"\"\"Read CE+PE OPEN prices for a FIXED strike at target_time.
    Used at exit so the buy-back happens on the same contracts that were sold.\"\"\"
    try:
        opt  = pd.read_parquet(os.path.join(OPT_DIR,  f'{date_str}.parquet'))
        spot = pd.read_parquet(os.path.join(SPOT_DIR, f'{date_str}.parquet'))
    except FileNotFoundError:
        return None
    opt['timestamp']  = pd.to_datetime(opt['timestamp'])
    spot['timestamp'] = pd.to_datetime(spot['timestamp'])
    target_ts = pd.to_datetime(f\"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {target_time}\")
    sub = opt[(opt['expiry'] == expiry_str) & (opt['strike'] == strike)]
    if sub.empty:
        return None
    ce = _leg_open_at(sub, 'CE', target_ts)
    pe = _leg_open_at(sub, 'PE', target_ts)
    if ce is None or pe is None:
        return None
    return {
        'atm_strike'      : strike,
        'ce_price'        : ce,
        'pe_price'        : pe,
        'spot'            : _spot_at(spot, target_ts),
        'straddle_premium': ce + pe,
    }


print('Helpers defined. Cell 3.3 builds the per-event P&L table.')
"""

NEW_CELL_15 = """\
# ── 3.3 Build full P&L table ──────────────────────────────────────────────────
# Slow loop (5–15 min). Cached at DATA/straddle_pnl.parquet — re-uses if present.
#
# Bug-fix vs. older versions of this notebook: at exit we read the SAME strike
# we sold at entry. The previous code re-ATM'd at expiry-day spot, which
# under-stated tail losses when spot gapped through the original strike
# overnight (e.g. 2026-02-03 worst day: -125.22 → -523.35 pts after fix).

PNL_CACHE = os.path.join(ROOT, 'straddle_pnl.parquet')

if os.path.exists(PNL_CACHE):
    pnl_df = pd.read_parquet(PNL_CACHE)
    pnl_df['expiry'] = pd.to_datetime(pnl_df['expiry'])
    print(f\"Loaded cache: {len(pnl_df)} events from {PNL_CACHE}\")
else:
    results = []
    total = len(events_df)
    for i, row in events_df.iterrows():
        if i % 20 == 0:
            print(f\"  Processing {i}/{total} — {row['entry_str']}\")
        expiry_str = row['expiry'].strftime('%Y-%m-%d')
        entry = get_atm_straddle_snapshot(row['entry_str'], expiry_str, ENTRY_TIME)
        if entry is None:
            continue
        exit_ = get_leg_prices_at_strike(
            row['expiry_str'], expiry_str, entry['atm_strike'], EXIT_TIME)
        if exit_ is None:
            continue
        pnl = entry['straddle_premium'] - exit_['straddle_premium']
        results.append({
            'expiry'           : row['expiry'],
            'entry_date'       : row['entry_str'],
            'atm_strike'       : entry['atm_strike'],
            'spot_at_entry'    : entry['spot'],
            'spot_at_exit'     : exit_['spot'],
            'premium_collected': entry['straddle_premium'],
            'premium_at_exit'  : exit_['straddle_premium'],
            'pnl_points'       : pnl,
            'pnl_pct'          : pnl / entry['straddle_premium'],
            'profitable'       : int(pnl > 0),
        })
    pnl_df = pd.DataFrame(results).sort_values('expiry').reset_index(drop=True)
    pnl_df.to_parquet(PNL_CACHE, index=False)
    pnl_df.to_csv(os.path.join(ROOT, 'straddle_pnl.csv'), index=False)
    print(f\"\\nDone. {len(pnl_df)} events processed and cached.\")

print(pnl_df.head())
"""


def patch_notebook():
    with io.open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    # cell 14 = 3.2 (helpers), cell 15 = 3.3 (build table)
    nb["cells"][14]["source"] = NEW_CELL_14.splitlines(keepends=True)
    nb["cells"][14]["outputs"] = []
    nb["cells"][14]["execution_count"] = None
    nb["cells"][15]["source"] = NEW_CELL_15.splitlines(keepends=True)
    nb["cells"][15]["outputs"] = []
    nb["cells"][15]["execution_count"] = None
    with io.open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"Patched cells 14 + 15 in {NB_PATH}")


if __name__ == "__main__":
    print("Step 1: patch notebook cells 14 + 15")
    patch_notebook()
    print()

    if os.path.exists(PNL_PARQUET):
        os.remove(PNL_PARQUET)
        print(f"Step 2: removed stale {PNL_PARQUET}")
    if os.path.exists(PNL_CSV):
        os.remove(PNL_CSV)
        print(f"        removed stale {PNL_CSV}")
    print()

    print("Step 3: regenerate parquet with corrected logic")
    regenerate_pnl()
