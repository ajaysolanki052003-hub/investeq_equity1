"""Data layer for the Expiry Blast engine.

The engine talks to the `Feed` interface only, so the same signal code runs
against historical parquets (backtest) and a live broker API (phase 2).

HistoricalFeed reads the canonical per-day parquets maintained by the repo's
fetchers:

    DATA/oi/YYYYMMDD.parquet       timestamp, expiry, strike, option_type, oi
                                   (~3-min cadence, full near-ATM chain,
                                    2 nearest weekly expiries)
    DATA/spot/YYYYMMDD.parquet     1-min index OHLCV
    DATA/options/YYYYMMDD.parquet  1-min option OHLCV per strike/expiry/type

All timestamps are naive IST, matching the rest of the repo.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))

OI_DIR = DATA_ROOT / "oi"
SPOT_DIR = DATA_ROOT / "spot"
OPT_DIR = DATA_ROOT / "options"


# ─── Feed interface ──────────────────────────────────────────────────────────

class Feed(ABC):
    """Point-in-time market access for one instrument on one trading day.

    Every method takes `ts` (the engine's clock) and must never look ahead:
    answers are built only from data stamped <= ts.
    """

    day: date
    expiry: date          # contract expiring today (the one we trade)

    @abstractmethod
    def oi_tick_times(self) -> list[pd.Timestamp]:
        """Chain snapshot timestamps for the day — the evaluation clock."""

    @abstractmethod
    def chain_at(self, ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series, pd.Series]:
        """(tick_ts, ce_oi, pe_oi) of the expiring contract as of `ts`.
        ce_oi / pe_oi are Series indexed by strike. tick_ts is the snapshot's
        own stamp so the caller can judge staleness."""

    @abstractmethod
    def oi_at(self, strike: int, option_type: str,
              ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
        """Latest (tick_ts, oi) for one leg at or before `ts`, else None."""

    @abstractmethod
    def spot_at(self, ts: pd.Timestamp) -> float | None:
        """Latest spot close at or before `ts`."""

    @abstractmethod
    def last_closed_candle(self, ts: pd.Timestamp, tf_min: int) -> dict | None:
        """Most recent fully-closed `tf_min`-minute spot candle as of `ts`.
        {'time','open','high','low','close'} or None before the first close."""

    @abstractmethod
    def premium_at(self, strike: int, option_type: str,
                   ts: pd.Timestamp) -> float | None:
        """Tradeable premium for the expiring contract's leg at `ts` —
        the next 1-min bar's open (market order fill proxy), falling back
        to the latest close at or before `ts`."""


# ─── Historical (backtest) feed ──────────────────────────────────────────────

class HistoricalFeed(Feed):
    def __init__(self, day: date, data_root: Path = DATA_ROOT):
        self.day = day
        key = day.strftime("%Y%m%d")
        oi_path = data_root / "oi" / f"{key}.parquet"
        if not oi_path.exists():
            raise FileNotFoundError(f"no OI data for {day}: {oi_path}")

        oi = pd.read_parquet(oi_path)
        oi["timestamp"] = pd.to_datetime(oi["timestamp"])
        self.expiry = min(pd.to_datetime(oi["expiry"]).dt.date)
        # Trade only the contract expiring today.
        oi = oi[pd.to_datetime(oi["expiry"]).dt.date == self.expiry]

        # Wide per-side frames: index = tick timestamp, columns = strike.
        self._ce = (oi[oi["option_type"] == "CE"]
                    .pivot_table(index="timestamp", columns="strike",
                                 values="oi", aggfunc="last").sort_index())
        self._pe = (oi[oi["option_type"] == "PE"]
                    .pivot_table(index="timestamp", columns="strike",
                                 values="oi", aggfunc="last").sort_index())
        self._ticks = self._ce.index.union(self._pe.index).sort_values()
        # Snapshot views: per-strike last-known OI carried forward, so a
        # strike that last ticked a minute or two ago stays in the chain
        # instead of flickering out of the max-OI scan.
        self._ce_ff = self._ce.ffill()
        self._pe_ff = self._pe.ffill()

        spot_path = data_root / "spot" / f"{key}.parquet"
        if not spot_path.exists():
            raise FileNotFoundError(f"no spot data for {day}: {spot_path}")
        sp = pd.read_parquet(spot_path)
        sp["timestamp"] = pd.to_datetime(sp["timestamp"])
        self._spot = (sp.sort_values("timestamp")
                        .drop_duplicates("timestamp", keep="last")
                        .reset_index(drop=True))

        self._opt_path = data_root / "options" / f"{key}.parquet"
        self._opt = None  # loaded lazily — only needed once a signal fires

    @property
    def is_expiry_day(self) -> bool:
        return self.expiry == self.day

    # ── OI ───────────────────────────────────────────────────────────────
    def oi_tick_times(self) -> list[pd.Timestamp]:
        return list(self._ticks)

    def chain_at(self, ts):
        ce_hist = self._ce_ff.loc[:ts]
        if ce_hist.empty:
            return None
        tick = ce_hist.index[-1]
        ce = ce_hist.iloc[-1].dropna()
        pe_hist = self._pe_ff.loc[:ts]
        pe = (pe_hist.iloc[-1].dropna() if not pe_hist.empty
              else pd.Series(dtype=float))
        return tick, ce, pe

    def oi_at(self, strike, option_type, ts):
        frame = self._ce if option_type == "CE" else self._pe
        if strike not in frame.columns:
            return None
        col = frame[strike].dropna()
        col = col[col.index <= ts]
        if col.empty:
            return None
        return col.index[-1], float(col.iloc[-1])

    # ── Spot ─────────────────────────────────────────────────────────────
    def spot_at(self, ts):
        df = self._spot[self._spot["timestamp"] <= ts]
        return float(df["close"].iloc[-1]) if len(df) else None

    @lru_cache(maxsize=4)
    def _candles(self, tf_min: int) -> pd.DataFrame:
        out = (self._spot.set_index("timestamp")
               .resample(f"{tf_min}min", closed="left", label="left")
               .agg({"open": "first", "high": "max",
                     "low": "min", "close": "last"})
               .dropna(subset=["open", "close"])
               .reset_index())
        out["close_time"] = out["timestamp"] + pd.Timedelta(minutes=tf_min)
        return out

    def last_closed_candle(self, ts, tf_min):
        c = self._candles(tf_min)
        c = c[c["close_time"] <= ts]
        if c.empty:
            return None
        r = c.iloc[-1]
        return {"time": r["timestamp"], "open": float(r["open"]),
                "high": float(r["high"]), "low": float(r["low"]),
                "close": float(r["close"])}

    # ── Premiums ─────────────────────────────────────────────────────────
    def _options(self) -> pd.DataFrame:
        if self._opt is None:
            if not self._opt_path.exists():
                self._opt = pd.DataFrame()
            else:
                df = pd.read_parquet(self._opt_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df[pd.to_datetime(df["expiry"]).dt.date == self.expiry]
                self._opt = df.sort_values("timestamp")
        return self._opt

    def premium_at(self, strike, option_type, ts):
        df = self._options()
        if df.empty:
            return None
        leg = df[(df["strike"] == strike) & (df["option_type"] == option_type)]
        if leg.empty:
            return None
        nxt = leg[leg["timestamp"] >= ts]
        if len(nxt):
            return float(nxt["open"].iloc[0])
        prv = leg[leg["timestamp"] < ts]
        return float(prv["close"].iloc[-1]) if len(prv) else None


# ─── Live feed (phase 2 stub) ────────────────────────────────────────────────

class LiveFeed(Feed):
    """Live broker-API feed — same interface, real-time data.

    Phase 2: wire to the Groww endpoints already used elsewhere in the repo
    (ema_scanner/groww_client.py for auth, api.groww.in option-chain +
    live-quote endpoints). Each method maps to:
        chain_at / oi_at   → option-chain endpoint (OI per strike)
        spot_at            → index live quote
        last_closed_candle → 5-min historical candles, latest closed
        premium_at         → option live quote (ask, for a market-buy proxy)
    """

    def __init__(self, *a, **kw):
        raise NotImplementedError("LiveFeed lands in phase 2 — backtest with "
                                  "HistoricalFeed for now.")

    def oi_tick_times(self): ...
    def chain_at(self, ts): ...
    def oi_at(self, strike, option_type, ts): ...
    def spot_at(self, ts): ...
    def last_closed_candle(self, ts, tf_min): ...
    def premium_at(self, strike, option_type, ts): ...


# ─── Expiry-day discovery ────────────────────────────────────────────────────

def expiry_days(start: date | None = None, end: date | None = None,
                data_root: Path = DATA_ROOT) -> list[date]:
    """All days in DATA/oi whose nearest contract expires that same day.

    Peeking the expiry column of ~1500 parquets takes a while, so the full
    scan is cached next to the data and invalidated when files are added."""
    import json

    files = sorted((data_root / "oi").glob("*.parquet"))
    cache = data_root / "oi" / "_expiry_days.json"
    all_days: list[date] | None = None
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("n_files") == len(files):
                all_days = [date.fromisoformat(s) for s in c["days"]]
        except Exception:
            pass
    if all_days is None:
        all_days = []
        for p in files:
            try:
                d = datetime.strptime(p.stem, "%Y%m%d").date()
            except ValueError:
                continue
            exp = pd.read_parquet(p, columns=["expiry"])["expiry"]
            if pd.to_datetime(exp).dt.date.min() == d:
                all_days.append(d)
        try:
            cache.write_text(json.dumps(
                {"n_files": len(files),
                 "days": [d.isoformat() for d in all_days]}))
        except OSError:
            pass

    return [d for d in all_days
            if (start is None or d >= start) and (end is None or d <= end)]
