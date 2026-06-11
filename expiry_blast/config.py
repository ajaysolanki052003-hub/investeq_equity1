"""Configuration for the Expiry Day Short Covering Blast entry engine.

Every threshold from the strategy spec lives here — nothing index-specific is
hardcoded in the engine. Override per-run via BlastConfig(**overrides), the
backtest CLI flags, or a JSON file (BlastConfig.from_json).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import time
from pathlib import Path


# Strike spacing + lot size per index. Used as defaults; both overridable.
INSTRUMENT_DEFAULTS = {
    "NIFTY":     {"strike_step": 50,  "lot_size": 75},
    "BANKNIFTY": {"strike_step": 100, "lot_size": 30},
}


@dataclass(frozen=True)
class BlastConfig:
    # ── Instrument ───────────────────────────────────────────────────────
    instrument: str = "NIFTY"            # "NIFTY" | "BANKNIFTY"
    strike_step: int = 0                 # 0 → take from INSTRUMENT_DEFAULTS
    lot_size: int = 0                    # 0 → take from INSTRUMENT_DEFAULTS

    # ── Session gating ───────────────────────────────────────────────────
    # Entries allowed only inside this IST window, only on expiry day.
    window_start: time = time(10, 30)
    window_end: time = time(14, 45)

    # ── Signal thresholds (spec: INPUT PARAMETERS) ───────────────────────
    spot_proximity_pct: float = 0.15     # A: spot within % of the call wall
    call_oi_unwind_pct: float = 10.0     # B: min % DROP in wall CE OI
    put_oi_buildup_pct: float = 20.0     # C: min % RISE in PE OI, 2 strikes below
    # Spec default: BOTH strikes below the wall must pass C. Backtests show
    # the 2nd strike's PE buildup lags the 1st by several minutes, so the
    # strict form never fires — set to 1 to require only the nearest strike.
    put_strikes_required: int = 2
    oi_lookback_min: int = 15            # rolling window for B and C, minutes

    # ── Price confirmation ───────────────────────────────────────────────
    candle_tf_min: int = 5               # D: last CLOSED candle of this TF

    # ── Guardrails ───────────────────────────────────────────────────────
    max_signals_per_day: int = 1
    # OI ticks arrive every ~3 min; if the newest tick is older than this the
    # feed is considered stale and the evaluation is skipped (audited).
    stale_oi_max_min: float = 7.0

    def __post_init__(self):
        inst = self.instrument.upper()
        if inst not in INSTRUMENT_DEFAULTS:
            raise ValueError(f"unknown instrument {self.instrument!r}; "
                             f"expected one of {list(INSTRUMENT_DEFAULTS)}")
        object.__setattr__(self, "instrument", inst)
        if self.strike_step <= 0:
            object.__setattr__(self, "strike_step",
                               INSTRUMENT_DEFAULTS[inst]["strike_step"])
        if self.lot_size <= 0:
            object.__setattr__(self, "lot_size",
                               INSTRUMENT_DEFAULTS[inst]["lot_size"])

    # ── (De)serialization ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["window_start"] = self.window_start.strftime("%H:%M")
        d["window_end"] = self.window_end.strftime("%H:%M")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BlastConfig":
        d = dict(d)
        for k in ("window_start", "window_end"):
            if isinstance(d.get(k), str):
                hh, mm = d[k].split(":")
                d[k] = time(int(hh), int(mm))
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> "BlastConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
