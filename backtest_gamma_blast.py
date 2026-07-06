"""Backtest the Gamma Blast long-straddle strategy on the historical option feed.

Rules (same as the /straddle live signal):
  R1 entry only after 12:30 IST
  R2 NIFTY spot net move < 0.60% from open to 12:30
  R3 on the ATM combined-premium chart, a candle closes above EMA21 then the
     next candle breaks its high -> LONG straddle entry (buy CE+PE at ATM)
  R4 exit on stop-loss or take-profit (both swept here to FIND the best)

For each trading day we reconstruct the nearest-expiry ATM straddle combined
premium (resample each leg to the TF, sum CE+PE), find the entry, then simulate
exits over a grid of SL modes x TP% to report win-rate / PnL / expectancy.
Long straddle: PnL(points) = exit_premium - entry_premium; Rs = points x lot.

Run (VM):
    python backtest_gamma_blast.py --tf 5m
    python backtest_gamma_blast.py --tf 5m --start 2024-01-01
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import date, datetime

import numpy as np
import pandas as pd

ROOT = os.environ.get("INVESTEQ_DATA",
                      "/home/ajay/investeq_ajs/DATA" if os.name != "nt"
                      else r"C:\Users\User\Desktop\investeq_ajs\DATA")
OPT_DIR = os.path.join(ROOT, "options")
SPOT_DIR = os.path.join(ROOT, "spot")
LOT = 75
TF_RULE = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}


def _ema(closes: np.ndarray, n: int = 21) -> np.ndarray:
    """EMA, SMA-seeded (TradingView convention). NaN before warmup."""
    out = np.full(len(closes), np.nan)
    if len(closes) < n:
        return out
    k = 2.0 / (n + 1)
    ema = closes[:n].mean()
    out[n - 1] = ema
    for i in range(n, len(closes)):
        ema = closes[i] * k + ema * (1 - k)
        out[i] = ema
    return out


def _resample_leg(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    g = (df.set_index("ts").resample(TF_RULE[tf], closed="left", label="left")
           .agg(open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last")).dropna(subset=["open", "close"]))
    return g


def _hm(ts) -> int:
    return ts.hour * 60 + ts.minute


def combined_straddle(day_str: str, tf: str):
    """Return (comb_tf, path_1m, dte, atm, expiry, spot_move_pct) or None.

    comb_tf  : TF candles of the TRADEABLE combined premium (CE.close+PE.close at
               1-minute, then OHLC-resampled) — used for entry detection.
    path_1m  : the per-minute tradeable combined price (Series) — used for exit
               fills so SL/TP trigger only at prices that actually traded (the
               chart's summed-leg high/low aren't simultaneous, so they're not
               tradeable and would overstate hits)."""
    op = os.path.join(OPT_DIR, f"{day_str}.parquet")
    sp = os.path.join(SPOT_DIR, f"{day_str}.parquet")
    if not (os.path.exists(op) and os.path.exists(sp)):
        return None
    opt = pd.read_parquet(op)
    spot = pd.read_parquet(sp)
    if opt.empty or spot.empty:
        return None
    opt["ts"] = pd.to_datetime(opt["timestamp"])
    spot["ts"] = pd.to_datetime(spot["timestamp"])
    spot = spot.sort_values("ts")

    open_spot = float(spot.iloc[0]["open"])
    till = spot[spot["ts"].apply(_hm) <= 750]
    at1230 = float(till.iloc[-1]["close"]) if len(till) else open_spot
    spot_move = abs(at1230 - open_spot) / open_spot * 100 if open_spot else np.nan

    exps = sorted(opt["expiry"].astype(str).unique())
    if not exps:
        return None
    expiry = exps[0]
    sub = opt[opt["expiry"].astype(str) == expiry]
    ce_ks = set(sub.loc[sub.option_type == "CE", "strike"])
    pe_ks = set(sub.loc[sub.option_type == "PE", "strike"])
    both = sorted(ce_ks & pe_ks)
    if not both:
        return None
    atm = min(both, key=lambda k: abs(k - open_spot))

    # tradeable per-minute combined premium = CE.close + PE.close (simultaneous)
    ce1 = (sub[(sub.option_type == "CE") & (sub.strike == atm)]
           .set_index("ts")["close"].sort_index())
    pe1 = (sub[(sub.option_type == "PE") & (sub.strike == atm)]
           .set_index("ts")["close"].sort_index())
    path = (ce1 + pe1).dropna()
    if len(path) < 20:
        return None
    # TF candles from the tradeable path (real OHLC of a price that traded)
    comb = (path.resample(TF_RULE[tf], closed="left", label="left")
                .agg(["first", "max", "min", "last"]).dropna())
    comb.columns = ["open", "high", "low", "close"]

    d = datetime.strptime(day_str, "%Y%m%d").date()
    ed = datetime.strptime(expiry, "%Y-%m-%d").date()
    dte = (ed - d).days
    return comb, path, int(atm), expiry, spot_move


def find_entry(comb: pd.DataFrame):
    """Return (entry_idx, level, setup_low) per R1+R3, or None."""
    closes = comb["close"].to_numpy()
    ema = _ema(closes, 21)
    times = comb.index
    highs = comb["high"].to_numpy()
    lows = comb["low"].to_numpy()
    for i in range(1, len(comb)):
        if _hm(times[i]) < 750:
            continue
        if np.isnan(ema[i - 1]):
            continue
        if closes[i - 1] > ema[i - 1] and highs[i] > highs[i - 1]:
            return i, float(highs[i - 1]), float(lows[i - 1])
    return None


def simulate(path_after, level, setup_low, sl_mode, tp_pct):
    """Simulate a long straddle. `path_after` = per-minute tradeable combined
    price AFTER the breakout candle. Fills at real prices. (pnl_points, outcome)."""
    if sl_mode == "note":       # candle low, capped at 25 pts of risk
        sl = max(setup_low, level - 25)
    elif sl_mode.startswith("pt"):
        sl = level - float(sl_mode[2:])
    elif sl_mode.startswith("pct"):
        sl = level * (1 - float(sl_mode[3:]) / 100.0)
    else:
        sl = setup_low
    tp = level * (1 + tp_pct / 100.0) if tp_pct else None

    vals = path_after.to_numpy()
    if len(vals) == 0:
        return 0.0, "EOD"
    for v in vals:
        hit_sl = v <= sl
        hit_tp = tp is not None and v >= tp
        if hit_sl and hit_tp:
            return sl - level, "SL"        # conservative
        if hit_sl:
            return sl - level, "SL"
        if hit_tp:
            return tp - level, "TP"
    return float(vals[-1]) - level, "EOD"  # exit at last traded price


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="5m", choices=list(TF_RULE))
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2100-01-01")
    ap.add_argument("--require-spot-filter", action="store_true", default=True)
    ap.add_argument("--dump", default="", help="write per-trade CSV for the eval notebook")
    args = ap.parse_args()

    days = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(OPT_DIR, "*.parquet")))
    s = args.start.replace("-", ""); e = args.end.replace("-", "")
    days = [d for d in days if s <= d <= e]

    SL_MODES = ["note", "pt20", "pt25", "pt30", "pt40", "pct10", "pct15", "pct20"]
    TP_PCTS = [10, 15, 20, 25, 30, 40, 50, 75, 100, 0]   # 0 = no TP (EOD exit)

    tf_delta = pd.Timedelta(TF_RULE[args.tf])
    # gather entries once: (day, dte, path_after, level, setup_low)
    entries = []
    signal_days = set()
    control = {}   # day -> pnl of an UNCONDITIONAL 12:30 entry (same stop/EOD exit)
    trade_recs = []  # rich per-trade rows for --dump / the eval notebook
    scanned = filtered_spot = 0
    for d in days:
        r = combined_straddle(d, args.tf)
        if r is None:
            continue
        comb, path, atm, expiry, spot_move = r
        dte = (datetime.strptime(expiry, "%Y-%m-%d").date()
               - datetime.strptime(d, "%Y%m%d").date()).days
        scanned += 1
        if args.require_spot_filter and not (spot_move < 0.60):
            filtered_spot += 1
            continue

        # CONTROL: buy the straddle at the first candle >= 12:30 (market), same
        # note stop (candle low / 25pt) + EOD exit. Tests whether the EMA21
        # breakout adds anything over "just buy at 12:30 on a quiet day".
        times = comb.index
        ci = next((k for k in range(len(comb)) if _hm(times[k]) >= 750), None)
        if ci is not None:
            clvl = float(comb["close"].iloc[ci]); clow = float(comb["low"].iloc[ci])
            cpath = path[path.index >= times[ci] + tf_delta]
            control[d] = (simulate(cpath, clvl, clow, "note", 0)[0], dte)

        fe = find_entry(comb)
        if fe is None:
            continue
        entry_idx, level, setup_low = fe
        brk_end = comb.index[entry_idx] + tf_delta          # monitor real fills after the breakout candle
        path_after = path[path.index >= brk_end]
        entries.append((d, dte, path_after, level, setup_low))
        signal_days.add(d)
        pnl, outcome = simulate(path_after, level, setup_low, "note", 0)
        trade_recs.append(dict(
            date=f"{d[:4]}-{d[4:6]}-{d[6:]}", dte=dte, atm=atm, expiry=expiry,
            entry_time=comb.index[entry_idx].strftime("%H:%M"), spot_move=round(spot_move, 3),
            level=round(level, 2), sl=round(max(setup_low, level - 25), 2),
            pnl=round(pnl, 2), rupees=round(pnl * LOT, 0), outcome=outcome,
            control_1230=round(control.get(d, (np.nan,))[0], 2)))

    print(f"days scanned (with data): {scanned}")
    print(f"days rejected by R2 spot<0.60% filter: {filtered_spot}")
    print(f"days with a valid entry: {len(entries)}  (TF={args.tf})\n")
    if not entries:
        return

    if args.dump:
        pd.DataFrame(trade_recs).to_csv(args.dump, index=False)
        print(f"[dump] wrote {len(trade_recs)} trades -> {args.dump}\n")

    def stats(pnls):
        pnls = np.array(pnls, float)
        wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
        return dict(n=len(pnls), win=len(wins) / len(pnls) * 100,
                    avg=pnls.mean(), total=pnls.sum(),
                    avg_win=wins.mean() if len(wins) else 0,
                    avg_loss=losses.mean() if len(losses) else 0,
                    rupees=pnls.mean() * LOT)

    # sweep
    rows = []
    for sl in SL_MODES:
        for tp in TP_PCTS:
            pnls = [simulate(pa, lv, slw, sl, tp)[0] for (_, _, pa, lv, slw) in entries]
            st = stats(pnls)
            rows.append((sl, tp, st))

    rows.sort(key=lambda r: -r[2]["total"])
    print("=== TOP 12 SL x TP by total PnL (points; long straddle) ===")
    print(f"{'SL':>6} {'TP%':>5} {'n':>4} {'win%':>6} {'avgP':>7} {'totalP':>9} {'avgWin':>7} {'avgLoss':>8} {'avg Rs':>8}")
    for sl, tp, st in rows[:12]:
        tpl = "EOD" if tp == 0 else f"{tp}%"
        print(f"{sl:>6} {tpl:>5} {st['n']:>4} {st['win']:>6.1f} {st['avg']:>7.1f} {st['total']:>9.0f} "
              f"{st['avg_win']:>7.1f} {st['avg_loss']:>8.1f} {st['rupees']:>8.0f}")

    # the exact note rule (SL=note, exit EOD) as baseline
    base = [r for r in rows if r[0] == "note" and r[1] == 0][0][2]
    print(f"\n=== Note's raw rule (SL=candle-low/25pt, no TP, exit EOD) ===")
    print(f"LONG straddle : trades={base['n']}  win%={base['win']:.1f}  avgPnL={base['avg']:.1f} pts "
          f"(Rs {base['rupees']:.0f})  total={base['total']:.0f} pts  "
          f"avgWin={base['avg_win']:.1f}  avgLoss={base['avg_loss']:.1f}")
    # Same signal, inverse position (SHORT straddle) — mirror PnL, for reference.
    inv_pnls = [-simulate(pa, lv, slw, "note", 0)[0] for (_, _, pa, lv, slw) in entries]
    inv = stats(inv_pnls)
    print(f"SHORT (mirror): trades={inv['n']}  win%={inv['win']:.1f}  avgPnL={inv['avg']:.1f} pts "
          f"(Rs {inv['rupees']:.0f})  total={inv['total']:.0f} pts  [shorting the breakout, no SL]")

    # Cost sensitivity — a long straddle is 4 option legs (2 in, 2 out). Net edge
    # per trade = gross avg - round-trip cost (points).
    base_pnls = [simulate(pa, lv, slw, "note", 0)[0] for (_, _, pa, lv, slw) in entries]
    n = len(base_pnls); gross = np.mean(base_pnls)
    print("\n=== cost sensitivity (LONG, note rule) — round-trip pts across 4 legs ===")
    for cost in (0, 2, 3, 4, 6):
        net = gross - cost
        print(f"  cost {cost:>2} pts -> net {net:+.2f} pts/trade (Rs {net*LOT:+.0f})  total {net*n:+.0f} pts")

    # Expiry-day-only (DTE<=1) — the classic gamma-blast window
    exp_pnls = [simulate(pa, lv, slw, "note", 0)[0] for (_, dte, pa, lv, slw) in entries if dte <= 1]
    if exp_pnls:
        es = stats(exp_pnls)
        print(f"\n=== Expiry window only (DTE<=1), LONG note rule ===")
        print(f"trades={es['n']}  win%={es['win']:.1f}  avgPnL={es['avg']:.2f} pts (Rs {es['rupees']:.0f})  "
              f"total={es['total']:.0f} pts  avgWin={es['avg_win']:.1f}  avgLoss={es['avg_loss']:.1f}")

    # CONTROL — does the EMA21 breakout entry beat just buying at 12:30?
    print(f"\n=== CONTROL: EMA21-breakout signal vs unconditional 12:30 buy "
          f"(same SL=candle-low/25pt, EOD exit) ===")
    sig = stats([simulate(pa, lv, slw, "note", 0)[0] for (_, _, pa, lv, slw) in entries])
    ctrl_same = stats([control[d][0] for d in signal_days if d in control])   # same days as the signal
    ctrl_all = stats([v[0] for v in control.values()])                        # every quiet (R2) day
    print(f"  SIGNAL      (breakout)        : n={sig['n']}  win%={sig['win']:.1f}  "
          f"avgPnL={sig['avg']:.2f} pts  total={sig['total']:.0f}")
    print(f"  CONTROL 12:30 (signal days)   : n={ctrl_same['n']}  win%={ctrl_same['win']:.1f}  "
          f"avgPnL={ctrl_same['avg']:.2f} pts  total={ctrl_same['total']:.0f}")
    print(f"  CONTROL 12:30 (all quiet days): n={ctrl_all['n']}  win%={ctrl_all['win']:.1f}  "
          f"avgPnL={ctrl_all['avg']:.2f} pts  total={ctrl_all['total']:.0f}")
    edge = sig['avg'] - ctrl_same['avg']
    print(f"  -> breakout edge over 12:30-buy on the same days: {edge:+.2f} pts/trade "
          f"({'signal adds value' if edge > 0.3 else 'signal adds little' if edge > -0.3 else 'signal WORSE'})")

    # DTE breakdown for the best config
    bsl, btp, _ = rows[0]
    by_dte = {}
    for (d, dte, pa, lv, slw) in entries:
        p = simulate(pa, lv, slw, bsl, btp)[0]
        by_dte.setdefault(dte, []).append(p)
    print(f"\n=== DTE breakdown for best config (SL={bsl}, TP={'EOD' if btp==0 else str(btp)+'%'}) ===")
    print(f"{'DTE':>4} {'n':>4} {'win%':>6} {'avgP':>7} {'totalP':>9}")
    for dte in sorted(by_dte):
        st = stats(by_dte[dte])
        print(f"{dte:>4} {st['n']:>4} {st['win']:>6.1f} {st['avg']:>7.1f} {st['total']:>9.0f}")


if __name__ == "__main__":
    main()
