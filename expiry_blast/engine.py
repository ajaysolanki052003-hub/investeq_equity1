"""Entry signal engine — Expiry Day Short Covering Blast.

Evaluates, on each OI tick inside the entry window of an expiry day:

    A  Proximity      spot within SPOT_PROXIMITY_PCT of the max-Call-OI strike
    B  Call unwind     wall CE OI down >= CALL_OI_UNWIND_PCT over the lookback
    C  Put writing     PE OI of the 2 strikes below the wall each up
                       >= PUT_OI_BUILDUP_PCT over the same lookback
    D  Price confirm   last CLOSED 5-min spot candle closes ABOVE the wall

All four true simultaneously → BUY 1 lot ATM CE (one signal per day).
Every check — pass, fail or skip — lands in the day's audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta

import pandas as pd

from .config import BlastConfig
from .data import Feed

log = logging.getLogger("expiry_blast")


@dataclass
class EntrySignal:
    timestamp: str               # signal time (IST)
    instrument: str
    expiry: str
    max_call_oi_strike: int      # the resistance wall
    wall_call_oi: float          # wall CE OI at signal time
    call_oi_pct_change: float    # B: % change over lookback (negative = unwind)
    put_oi_pct_changes: dict     # C: {strike: % change} for the 2 strikes below
    candle_close: float          # D: confirming 5-min close
    spot: float
    entry_strike: int            # ATM strike bought
    entry_premium: float | None  # fill proxy (next 1-min bar open); None if no data
    lot_size: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DayResult:
    day: str
    instrument: str
    expiry: str | None
    signals: list = field(default_factory=list)      # list[EntrySignal]
    audit: list = field(default_factory=list)        # one dict per evaluation
    skipped_reason: str | None = None                # day-level skip (not expiry etc.)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signals"] = [s if isinstance(s, dict) else s.to_dict()
                        for s in self.signals]
        return d


def _pct(cur: float, base: float) -> float | None:
    if base is None or cur is None or base <= 0:
        return None
    return (cur - base) / base * 100.0


class SignalEngine:
    def __init__(self, cfg: BlastConfig | None = None):
        self.cfg = cfg or BlastConfig()

    # ── Single evaluation (one OI tick) ─────────────────────────────────
    def evaluate(self, feed: Feed, ts: pd.Timestamp) -> dict:
        """Run conditions A-D as of `ts`. Returns the audit record; if every
        condition passed it carries a 'signal' key with the EntrySignal."""
        cfg = self.cfg
        rec: dict = {"ts": str(ts), "skipped": None, "conditions": {},
                     "all_pass": False}

        chain = feed.chain_at(ts)
        if chain is None:
            rec["skipped"] = "no chain snapshot yet"
            return rec
        tick_ts, ce, pe = chain

        # Guardrail: stale / missing OI feed → skip, don't evaluate.
        age_min = (ts - tick_ts).total_seconds() / 60.0
        if age_min > cfg.stale_oi_max_min:
            rec["skipped"] = f"stale OI feed ({age_min:.1f} min old)"
            return rec
        if ce.empty:
            rec["skipped"] = "empty CE chain"
            return rec

        spot = feed.spot_at(ts)
        if spot is None:
            rec["skipped"] = "no spot price"
            return rec
        rec["spot"] = round(spot, 2)

        # 1. The resistance wall: strike with max total Call OI.
        wall = int(ce.idxmax())
        wall_oi = float(ce.max())
        rec["wall_strike"] = wall
        rec["wall_call_oi"] = wall_oi

        # ── A. Proximity ─────────────────────────────────────────────────
        dist_pct = abs(spot - wall) / wall * 100.0
        a_pass = dist_pct <= cfg.spot_proximity_pct
        rec["conditions"]["A_proximity"] = {
            "pass": a_pass, "spot": round(spot, 2), "wall": wall,
            "dist_pct": round(dist_pct, 4), "max_pct": cfg.spot_proximity_pct}

        # ── B. Call unwinding at the wall ────────────────────────────────
        base_ts = ts - timedelta(minutes=cfg.oi_lookback_min)
        b = {"pass": False, "strike": wall}
        cur = feed.oi_at(wall, "CE", ts)
        base = feed.oi_at(wall, "CE", base_ts)
        ce_chg = None
        if cur is None or base is None:
            b["note"] = "no OI history at lookback start"
        # Baseline must actually be old data, not the same fresh tick.
        elif base[0] > base_ts:
            b["note"] = "baseline tick inside lookback window"
        else:
            ce_chg = _pct(cur[1], base[1])
            if ce_chg is None:
                b["note"] = "zero baseline OI"
            else:
                b.update(oi_now=cur[1], oi_then=base[1],
                         pct_change=round(ce_chg, 2),
                         threshold=-cfg.call_oi_unwind_pct)
                b["pass"] = ce_chg <= -cfg.call_oi_unwind_pct
        rec["conditions"]["B_call_unwind"] = b

        # ── C. Put writing below the wall ────────────────────────────────
        put_strikes = [wall - cfg.strike_step, wall - 2 * cfg.strike_step]
        pe_chgs: dict[int, float | None] = {}
        c_detail, n_ok = [], 0
        for k in put_strikes:
            cur = feed.oi_at(k, "PE", ts)
            base = feed.oi_at(k, "PE", base_ts)
            chg = None
            if cur is not None and base is not None and base[0] <= base_ts:
                chg = _pct(cur[1], base[1])
            ok = chg is not None and chg >= cfg.put_oi_buildup_pct
            n_ok += ok
            pe_chgs[k] = round(chg, 2) if chg is not None else None
            c_detail.append({"strike": k, "pass": ok,
                             "pct_change": pe_chgs[k],
                             "threshold": cfg.put_oi_buildup_pct})
        c_pass = n_ok >= cfg.put_strikes_required
        rec["conditions"]["C_put_writing"] = {
            "pass": c_pass, "passed": n_ok,
            "required": cfg.put_strikes_required, "strikes": c_detail}

        # ── D. Price confirmation ────────────────────────────────────────
        candle = feed.last_closed_candle(ts, cfg.candle_tf_min)
        if candle is None:
            d = {"pass": False, "note": "no closed candle yet"}
        else:
            d = {"pass": candle["close"] > wall,
                 "candle_time": str(candle["time"]),
                 "close": round(candle["close"], 2), "wall": wall}
        rec["conditions"]["D_price_confirm"] = d

        rec["all_pass"] = (a_pass and b["pass"] and c_pass and d["pass"])
        for name, c in rec["conditions"].items():
            log.debug("%s %s %s", ts, name, "PASS" if c["pass"] else "fail")

        if rec["all_pass"]:
            atm = int(round(spot / cfg.strike_step) * cfg.strike_step)
            premium = feed.premium_at(atm, "CE", ts)
            sig = EntrySignal(
                timestamp=str(ts), instrument=cfg.instrument,
                expiry=str(feed.expiry),
                max_call_oi_strike=wall, wall_call_oi=wall_oi,
                call_oi_pct_change=round(ce_chg, 2),
                put_oi_pct_changes=pe_chgs,
                candle_close=round(candle["close"], 2),
                spot=round(spot, 2),
                entry_strike=atm, entry_premium=premium,
                lot_size=cfg.lot_size)
            rec["signal"] = sig.to_dict()
            log.info("ENTRY SIGNAL %s — BUY %d CE @ %s (wall %d)",
                     ts, atm, premium, wall)
        return rec

    # ── Full day ─────────────────────────────────────────────────────────
    def run_day(self, feed: Feed) -> DayResult:
        cfg = self.cfg
        res = DayResult(day=str(feed.day), instrument=cfg.instrument,
                        expiry=str(feed.expiry))

        # EXECUTION_DAY guardrail: only the contract's expiry day.
        if getattr(feed, "is_expiry_day", True) is False:
            res.skipped_reason = (f"not an expiry day "
                                  f"(nearest expiry {feed.expiry})")
            return res

        n_signals = 0
        for ts in feed.oi_tick_times():
            t = ts.time()
            if t < cfg.window_start or t > cfg.window_end:
                continue
            rec = self.evaluate(feed, ts)
            res.audit.append(rec)
            if "signal" in rec:
                res.signals.append(rec["signal"])
                n_signals += 1
                if n_signals >= cfg.max_signals_per_day:
                    break       # one entry per day — stop evaluating
        return res
