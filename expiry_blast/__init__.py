"""Expiry Day Short Covering Blast — entry signal engine (phase 1: entry only).

Modules:
    config    — BlastConfig: every threshold the spec calls configurable
    data      — Feed abstraction: HistoricalFeed (DATA parquets) + LiveFeed stub
    engine    — SignalEngine: conditions A-D, audit log, one-entry-per-day
    backtest  — CLI runner over historical expiry days
    app       — FastAPI dashboard (mounted at /blast behind nginx)
"""

from .config import BlastConfig
from .engine import SignalEngine, EntrySignal

__all__ = ["BlastConfig", "SignalEngine", "EntrySignal"]
