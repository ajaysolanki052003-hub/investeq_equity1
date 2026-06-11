"""Backtest hook — replay the entry engine over historical expiry days.

Run from the repo root:

    python -m expiry_blast.backtest                       # full history
    python -m expiry_blast.backtest --start 2025-01-01 --end 2026-04-30
    python -m expiry_blast.backtest --proximity 0.2 --unwind 8 --buildup 15
    python -m expiry_blast.backtest --audit-all           # audit even no-signal days

Outputs (under --out, default expiry_blast/output/):
    signals.csv     one row per entry signal, with post-entry premium context
    audit.jsonl     per-evaluation condition log (signal days; --audit-all for all)
    summary.txt     run parameters + headline counts

Entry-only validation: there is no exit logic yet, so the post-entry columns
(premium_eod / premium_max_after) are *context*, not P&L.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from .config import BlastConfig
from .data import HistoricalFeed, expiry_days
from .engine import SignalEngine

OUT_DEFAULT = Path(__file__).parent / "output"


def _post_entry_context(feed: HistoricalFeed, sig: dict) -> dict:
    """Premium of the bought strike at EOD and its post-entry max — context
    for judging entry quality while exits don't exist yet."""
    df = feed._options()
    if df.empty:
        return {}
    leg = df[(df["strike"] == sig["entry_strike"])
             & (df["option_type"] == "CE")
             & (df["timestamp"] > pd.Timestamp(sig["timestamp"]))]
    if leg.empty:
        return {}
    return {"premium_eod": float(leg["close"].iloc[-1]),
            "premium_max_after": float(leg["high"].max())}


def run(cfg: BlastConfig, start: date | None, end: date | None,
        out_dir: Path, audit_all: bool = False) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = SignalEngine(cfg)

    days = expiry_days(start, end)
    print(f"{len(days)} expiry days in range "
          f"({days[0] if days else '-'} .. {days[-1] if days else '-'})")

    rows, n_skipped = [], 0
    audit_f = (out_dir / "audit.jsonl").open("w", encoding="utf-8")
    try:
        for i, d in enumerate(days, 1):
            try:
                feed = HistoricalFeed(d)
            except FileNotFoundError as e:
                n_skipped += 1
                audit_f.write(json.dumps({"day": str(d),
                                          "skipped": str(e)}) + "\n")
                continue
            res = engine.run_day(feed)
            if res.signals or audit_all:
                audit_f.write(json.dumps(res.to_dict()) + "\n")
            for sig in res.signals:
                sig = dict(sig)
                sig.update(_post_entry_context(feed, sig))
                rows.append(sig)
                print(f"  SIGNAL {sig['timestamp']}  wall={sig['max_call_oi_strike']}"
                      f"  CE {sig['call_oi_pct_change']}%  buy {sig['entry_strike']}CE"
                      f" @ {sig['entry_premium']}")
            if i % 100 == 0:
                print(f"  ... {i}/{len(days)} days, {len(rows)} signals")
    finally:
        audit_f.close()

    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(out_dir / "signals.csv", index=False)

    summary = [
        f"run at        : {datetime.now().isoformat(timespec='seconds')}",
        f"config        : {json.dumps(cfg.to_dict())}",
        f"range         : {days[0] if days else '-'} .. {days[-1] if days else '-'}",
        f"expiry days   : {len(days)}  (skipped {n_skipped}: missing data)",
        f"entry signals : {len(sig_df)}",
    ]
    if len(sig_df) and "premium_eod" in sig_df.columns:
        with_ctx = sig_df.dropna(subset=["entry_premium", "premium_eod"])
        if len(with_ctx):
            up = (with_ctx["premium_eod"] > with_ctx["entry_premium"]).mean()
            mfe = (with_ctx["premium_max_after"] / with_ctx["entry_premium"]
                   ).median()
            summary += [
                f"premium higher at EOD : {up:.0%} of signals (no-exit context)",
                f"median max-premium-after / entry : {mfe:.2f}x",
            ]
    text = "\n".join(summary)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nwrote {out_dir / 'signals.csv'}")
    return sig_df


def main():
    ap = argparse.ArgumentParser(description="Expiry Blast entry backtest")
    ap.add_argument("--instrument", default="NIFTY")
    ap.add_argument("--start", type=date.fromisoformat, default=None)
    ap.add_argument("--end", type=date.fromisoformat, default=None)
    ap.add_argument("--proximity", type=float, default=None,
                    help="SPOT_PROXIMITY_PCT")
    ap.add_argument("--unwind", type=float, default=None,
                    help="CALL_OI_UNWIND_PCT")
    ap.add_argument("--buildup", type=float, default=None,
                    help="PUT_OI_BUILDUP_PCT")
    ap.add_argument("--put-strikes", type=int, default=None,
                    help="how many of the 2 below-wall strikes must pass C (1 or 2)")
    ap.add_argument("--lookback", type=int, default=None,
                    help="OI_LOOKBACK_MIN")
    ap.add_argument("--window", default=None,
                    help="entry window, e.g. 10:30-14:45")
    ap.add_argument("--config", default=None, help="JSON config file")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--audit-all", action="store_true",
                    help="write audit rows for no-signal days too")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    cfg = (BlastConfig.from_json(args.config) if args.config
           else BlastConfig(instrument=args.instrument))
    over = {}
    if args.proximity is not None:
        over["spot_proximity_pct"] = args.proximity
    if args.unwind is not None:
        over["call_oi_unwind_pct"] = args.unwind
    if args.buildup is not None:
        over["put_oi_buildup_pct"] = args.buildup
    if args.put_strikes is not None:
        over["put_strikes_required"] = args.put_strikes
    if args.lookback is not None:
        over["oi_lookback_min"] = args.lookback
    if args.window:
        s, e = args.window.split("-")
        over["window_start"] = time(*map(int, s.split(":")))
        over["window_end"] = time(*map(int, e.split(":")))
    if over:
        cfg = BlastConfig.from_dict({**cfg.to_dict(), **{
            k: (v if not isinstance(v, time) else v.strftime("%H:%M"))
            for k, v in over.items()}})

    run(cfg, args.start, args.end, args.out, args.audit_all)


if __name__ == "__main__":
    main()
