"""Full A/B/C/D × timeframe grid sweep — efficient shared-measurement design.

One pass per expiry day records the raw inputs of every condition at every
OI tick (distance-to-wall, CE unwind % per lookback, PE buildup % per
lookback, last-closed-candle close per TF). Every threshold combination is
then a cheap scan over those measurements — 324 combos for the cost of ~1.

Validated against the real SignalEngine (same signals for the live config)
before results are trusted.

    python -m expiry_blast.grid_sweep            # run sweep (multi-process)
    python -m expiry_blast.grid_sweep --validate # engine-equivalence check
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from .config import BlastConfig
from .data import DATA_ROOT, HistoricalFeed, expiry_days
from .engine import SignalEngine

OUT = Path(__file__).parent / "sweep" / "grid_results.csv"

# ── Grid ─────────────────────────────────────────────────────────────────────
PROX     = [0.15, 0.30]          # A
UNWIND   = [5.0, 10.0, 15.0]     # B
BUILDUP  = [5.0, 10.0, 20.0]     # C threshold
STRIKES  = [1, 2]                # C strikes required
TFS      = [3, 5, 15]            # D candle TF (min)
LOOKBACK = [10, 15, 30]          # B/C rolling window (min)

# bracket exit for quality scoring
TP, SL = 0.8, 0.4


def measure_day(day_iso: str) -> list[dict]:
    """Per-tick raw measurements for one expiry day (all TFs, all lookbacks)."""
    day = date.fromisoformat(day_iso)
    try:
        feed = HistoricalFeed(day)
    except FileNotFoundError:
        return []
    cfg = BlastConfig()          # window + staleness + strike_step only
    rows = []
    for ts in feed.oi_tick_times():
        t = ts.time()
        if t < cfg.window_start or t > cfg.window_end:
            continue
        chain = feed.chain_at(ts)
        if chain is None:
            continue
        tick_ts, ce, pe = chain
        if (ts - tick_ts).total_seconds() / 60.0 > cfg.stale_oi_max_min:
            continue
        if ce.empty:
            continue
        spot = feed.spot_at(ts)
        if spot is None:
            continue
        wall = int(ce.idxmax())
        rec = {"day": day_iso, "ts": str(ts), "spot": spot, "wall": wall,
               "dist_pct": abs(spot - wall) / wall * 100.0}
        for lb in LOOKBACK:
            base_ts = ts - pd.Timedelta(minutes=lb)
            cur = feed.oi_at(wall, "CE", ts)
            base = feed.oi_at(wall, "CE", base_ts)
            chg = None
            if cur and base and base[0] <= base_ts and base[1] > 0:
                chg = (cur[1] - base[1]) / base[1] * 100.0
            rec[f"ce_chg_{lb}"] = chg
            for j, k in enumerate((wall - cfg.strike_step,
                                   wall - 2 * cfg.strike_step)):
                curp = feed.oi_at(k, "PE", ts)
                basep = feed.oi_at(k, "PE", base_ts)
                chgp = None
                if curp and basep and basep[0] <= base_ts and basep[1] > 0:
                    chgp = (curp[1] - basep[1]) / basep[1] * 100.0
                rec[f"pe{j}_chg_{lb}"] = chgp
        for tf in TFS:
            cdl = feed.last_closed_candle(ts, tf)
            rec[f"close_{tf}"] = cdl["close"] if cdl else None
        rows.append(rec)

    # Premium + bracket outcome are only needed for ticks that could fire —
    # resolve lazily later, but cache the option leg here for cheap reuse.
    return rows


def first_signal(day_rows: list[dict], prox, unwind, buildup,
                 strikes, tf, lb) -> dict | None:
    for r in day_rows:
        if r["dist_pct"] > prox:
            continue
        ce = r.get(f"ce_chg_{lb}")
        if ce is None or ce > -unwind:
            continue
        n_ok = sum(1 for j in (0, 1)
                   if (r.get(f"pe{j}_chg_{lb}") or -1e9) >= buildup)
        if n_ok < strikes:
            continue
        close = r.get(f"close_{tf}")
        if close is None or close <= r["wall"]:
            continue
        return r
    return None


_BRACKET_CACHE: dict = {}


def bracket(day_iso: str, ts: str, spot: float) -> tuple[str, float] | None:
    """(outcome, entry_premium) for buying ATM CE at `ts` with TP/SL bracket."""
    key = (day_iso, ts)
    if key in _BRACKET_CACHE:
        return _BRACKET_CACHE[key]
    day = date.fromisoformat(day_iso)
    p = DATA_ROOT / "options" / f"{day:%Y%m%d}.parquet"
    out = None
    if p.exists():
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        atm = int(round(spot / 50) * 50)
        leg = df[(df["strike"] == atm) & (df["option_type"] == "CE")
                 & (pd.to_datetime(df["expiry"]).dt.date == day)
                 ].sort_values("timestamp")
        ent_t = pd.Timestamp(ts)
        nxt = leg[leg["timestamp"] >= ent_t]
        if len(nxt):
            e = float(nxt["open"].iloc[0])
            walkleg = leg[leg["timestamp"] > ent_t]
            tp_px, sl_px = e * (1 + TP), e * (1 - SL)
            res = None
            for _, b in walkleg.iterrows():
                hit_tp, hit_sl = b["high"] >= tp_px, b["low"] <= sl_px
                if hit_tp and hit_sl:
                    res = "SL"; break
                if hit_tp:
                    res = "TP"; break
                if hit_sl:
                    res = "SL"; break
            if res is None:
                res = "EOD"
            out = (res, e)
    _BRACKET_CACHE[key] = out
    return out


def run_sweep(workers: int):
    days = [d.isoformat() for d in expiry_days()]
    print(f"measuring {len(days)} expiry days with {workers} workers ...")
    all_rows: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for day_iso, rows in zip(days, pool.map(measure_day, days,
                                                chunksize=8)):
            if rows:
                all_rows[day_iso] = rows
    print(f"measured {sum(len(v) for v in all_rows.values()):,} ticks "
          f"on {len(all_rows)} days")

    combos = list(itertools.product(PROX, UNWIND, BUILDUP, STRIKES,
                                    TFS, LOOKBACK))
    print(f"scanning {len(combos)} combos ...")
    results = []
    for prox, unw, bld, stk, tf, lb in combos:
        sigs = []
        for day_iso, rows in all_rows.items():
            s = first_signal(rows, prox, unw, bld, stk, tf, lb)
            if s is not None:
                sigs.append(s)
        n = len(sigs)
        row = {"prox": prox, "unwind": unw, "buildup": bld, "strikes": stk,
               "tf": tf, "lookback": lb, "n": n}
        if n:
            outs = []
            for s in sigs:
                b = bracket(s["day"], s["ts"], s["spot"])
                if b:
                    outs.append(b[0])
            if outs:
                tp_n, sl_n = outs.count("TP"), outs.count("SL")
                m = len(outs)
                row.update(TP=tp_n, SL=sl_n, EOD=outs.count("EOD"),
                           win_pct=round(100 * tp_n / m),
                           exp_R=round((tp_n * (TP / SL) - sl_n) / m, 2))
        results.append(row)
    df = pd.DataFrame(results)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")

    fired = df[df["n"] > 0].copy()
    print(f"\ncombos that fire at least once: {len(fired)}/{len(df)}")
    print("\n=== TOP 15 BY EXPECTANCY (n >= 10) ===")
    f10 = fired[fired["n"] >= 10].sort_values(
        ["exp_R", "n"], ascending=False)
    print(f10.head(15).to_string(index=False))
    print("\n=== TOP 15 BY TRADE COUNT ===")
    print(fired.sort_values(["n", "exp_R"], ascending=False)
          .head(15).to_string(index=False))


def validate():
    """Fast path must reproduce the real engine for the live config."""
    cfg = BlastConfig(spot_proximity_pct=0.30)
    eng = SignalEngine(cfg)
    days = expiry_days()[-100:]   # recent 100 days are plenty for equivalence
    mismatches = 0
    fast_n = eng_n = 0
    for d in days:
        try:
            feed = HistoricalFeed(d)
        except FileNotFoundError:
            continue
        res = eng.run_day(feed)
        eng_sig = res.signals[0]["timestamp"] if res.signals else None
        rows = measure_day(d.isoformat())
        s = first_signal(rows, 0.30, 10.0, 10.0, 1, 5, 15)
        fast_sig = s["ts"] if s else None
        eng_n += eng_sig is not None
        fast_n += fast_sig is not None
        if eng_sig != fast_sig:
            mismatches += 1
            print(f"  MISMATCH {d}: engine={eng_sig} fast={fast_sig}")
    print(f"validate: engine fired {eng_n}, fast path fired {fast_n}, "
          f"mismatching days: {mismatches}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.validate:
        validate()
    else:
        run_sweep(args.workers)
