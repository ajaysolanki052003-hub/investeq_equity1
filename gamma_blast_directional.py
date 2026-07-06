"""Directional variant of Gamma Blast: at the combined-premium breakout, buy only
ONE leg — the one in the direction of the NIFTY move (CE if spot up, PE if down)
— instead of the full long straddle.

Same entry trigger (combined premium closes above EMA21 then next candle breaks
its high, after the gate). Direction from spot momentum at entry. Compares the
single directional leg vs the straddle (both legs) on the SAME entries, and
reports how often the direction call was right.

    python gamma_blast_directional.py --tf 5m
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import backtest_gamma_blast as bt
import gamma_blast_sweep as gs

TF = "5m"


def _hm(ts):
    return ts.hour * 60 + ts.minute


def load_day(day_str, tf):
    op = os.path.join(bt.OPT_DIR, f"{day_str}.parquet")
    sp = os.path.join(bt.SPOT_DIR, f"{day_str}.parquet")
    if not (os.path.exists(op) and os.path.exists(sp)):
        return None
    opt = pd.read_parquet(op); spot = pd.read_parquet(sp)
    if opt.empty or spot.empty:
        return None
    opt["ts"] = pd.to_datetime(opt["timestamp"]); spot["ts"] = pd.to_datetime(spot["timestamp"])
    spot = spot.sort_values("ts")
    open_spot = float(spot.iloc[0]["open"])
    till = spot[spot["ts"].apply(_hm) <= 750]
    at1230 = float(till.iloc[-1]["close"]) if len(till) else open_spot
    spot_move = abs(at1230 - open_spot) / open_spot * 100 if open_spot else np.nan
    exps = sorted(opt["expiry"].astype(str).unique())
    if not exps:
        return None
    expiry = exps[0]; sub = opt[opt["expiry"].astype(str) == expiry]
    both = sorted(set(sub.loc[sub.option_type == "CE", "strike"]) & set(sub.loc[sub.option_type == "PE", "strike"]))
    if not both:
        return None
    atm = min(both, key=lambda k: abs(k - open_spot))
    ce1 = sub[(sub.option_type == "CE") & (sub.strike == atm)].set_index("ts")["close"].sort_index()
    pe1 = sub[(sub.option_type == "PE") & (sub.strike == atm)].set_index("ts")["close"].sort_index()
    comb1 = (ce1 + pe1).dropna()
    if len(comb1) < 20:
        return None
    rule = bt.TF_RULE[tf]
    def ohlc(s):
        g = s.resample(rule, closed="left", label="left").agg(["first", "max", "min", "last"]).dropna()
        g.columns = ["open", "high", "low", "close"]; return g
    comb = ohlc(comb1); ce_tf = ohlc(ce1); pe_tf = ohlc(pe1)
    spot1 = spot.set_index("ts")["close"].sort_index()
    dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - datetime.strptime(day_str, "%Y%m%d").date()).days
    return dict(comb=comb, ce_tf=ce_tf, pe_tf=pe_tf, ce1=ce1, pe1=pe1, comb1=comb1,
                spot1=spot1, dte=dte, spot_move=spot_move)


def direction(spot1, t, mode="mom15", dir_min=0.0):
    """+1 buy CE, -1 buy PE, 0 = no clear direction (skip). dir_min = min |spot
    move %| vs the reference required to call a direction."""
    s_now = spot1[spot1.index <= t]
    if s_now.empty:
        return 0
    now = s_now.iloc[-1]
    if mode == "open":
        ref = spot1.iloc[0]
    elif mode == "since1230":
        r = spot1[spot1.index.map(_hm) <= 750]
        ref = r.iloc[-1] if len(r) else spot1.iloc[0]
    else:  # mom15 — momentum over the last 15 minutes
        r = spot1[spot1.index <= t - timedelta(minutes=15)]
        ref = r.iloc[-1] if len(r) else spot1.iloc[0]
    move = (now - ref) / ref * 100 if ref else 0
    if abs(move) < dir_min:
        return 0
    return 1 if move >= 0 else -1


def sim_leg(path_after, entry, sl, trail=None):
    vals = path_after.to_numpy()
    if len(vals) == 0:
        return 0.0
    rh = entry
    for v in vals:
        if trail is not None:
            rh = max(rh, v); sl = max(sl, rh - trail)
        if v <= sl:
            return sl - entry
    return float(vals[-1]) - entry


def stats(p):
    p = np.asarray(p, float); w = p[p > 0]
    return (len(p), len(w) / len(p) * 100 if len(p) else 0, p.mean() if len(p) else 0,
            p.sum(), w.mean() if len(w) else 0, p[p <= 0].mean() if len(p[p <= 0]) else 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tf", default="5m")
    ap.add_argument("--ema", type=int, default=21); ap.add_argument("--after", type=int, default=750)
    ap.add_argument("--spot-thr", type=float, default=0.60)
    ap.add_argument("--trail", type=float, default=None)
    ap.add_argument("--sl-pct", type=float, default=30.0, help="single-leg stop, %% of leg entry")
    ap.add_argument("--dir-min", type=float, default=0.0, help="min |spot move %%| to call a direction")
    args = ap.parse_args()
    days = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(bt.OPT_DIR, "*.parquet")))
    tfd = pd.Timedelta(bt.TF_RULE[args.tf])

    res = {m: {"str": [], "leg": [], "right": [], "d0str": [], "d0leg": []}
           for m in ["mom15", "since1230", "open"]}
    for d in days:
        L = load_day(d, args.tf)
        if L is None:
            continue
        if args.spot_thr is not None and not (L["spot_move"] < args.spot_thr):
            continue
        fe = gs.find_entry(L["comb"], args.ema, args.after)
        if fe is None:
            continue
        ei, level, slow = fe
        t = L["comb"].index[ei]
        # straddle PnL (both legs) — the reference
        cpath = L["comb1"][L["comb1"].index >= t + tfd]
        str_pnl = sim_leg(cpath, level, max(slow, level - 25), args.trail)
        # which leg would actually have been the winner (hindsight) for accuracy
        for mode in res:
            dirn = direction(L["spot1"], t, mode, args.dir_min)
            if dirn == 0:      # no clear direction -> skip the single-leg trade
                continue
            leg_tf = L["ce_tf"] if dirn > 0 else L["pe_tf"]
            leg1 = L["ce1"] if dirn > 0 else L["pe1"]
            if t not in leg_tf.index:
                continue
            l_entry = float(leg_tf.loc[t, "close"])
            l_sl = l_entry * (1 - args.sl_pct / 100.0)
            lpath = leg1[leg1.index >= t + tfd]
            leg_pnl = sim_leg(lpath, l_entry, l_sl, args.trail)
            # was the chosen direction the better leg? compare final CE vs PE move
            ce_end = float(L["ce1"].iloc[-1]) - float(L["ce_tf"].loc[t, "close"]) if t in L["ce_tf"].index else 0
            pe_end = float(L["pe1"].iloc[-1]) - float(L["pe_tf"].loc[t, "close"]) if t in L["pe_tf"].index else 0
            right = (dirn >= 0 and ce_end >= pe_end) or (dirn < 0 and pe_end > ce_end)
            res[mode]["str"].append(str_pnl); res[mode]["leg"].append(leg_pnl)
            res[mode]["right"].append(1 if right else 0)
            if L["dte"] == 0:
                res[mode]["d0str"].append(str_pnl); res[mode]["d0leg"].append(leg_pnl)

    print(f"TF={args.tf} EMA{args.ema} after={args.after//60}:{args.after%60:02d} "
          f"spot<{args.spot_thr} trail={args.trail} leg-SL={args.sl_pct}%\n")
    for mode in ["mom15", "since1230", "open"]:
        r = res[mode]
        n, sw, sa, st_, _, _ = stats(r["str"])
        _, lw, la, lt, law, lal = stats(r["leg"])
        acc = np.mean(r["right"]) * 100 if r["right"] else 0
        d0 = stats(r["d0leg"]); d0s = stats(r["d0str"])
        print(f"direction = {mode}:")
        print(f"  STRADDLE (both) : n={n:>4} win={sw:>4.1f}% avg={sa:>+5.2f} total={st_:>+6.0f}   DTE0 avg={d0s[2]:>+5.2f}")
        print(f"  SINGLE LEG (dir): n={n:>4} win={lw:>4.1f}% avg={la:>+5.2f} total={lt:>+6.0f}   DTE0 avg={d0[2]:>+5.2f}  | dir-correct {acc:.0f}%  avgWin={law:.1f} avgLoss={lal:.1f}")
        print()


if __name__ == "__main__":
    main()
