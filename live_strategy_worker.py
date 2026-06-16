"""Live data worker for the Strategy Suite (/strategy) — NSE market hours only.

Every minute during the session:
  1. Append the latest 1-min NIFTY index bars to DATA/nifty_1m_master.parquet
     (incremental fetch via the historical-candles endpoint — index candles
     are served intraday, unlike option OI which settles overnight).
  2. Read live OI for the 6 ATM±1 option legs (CE/PE × ATM-50/ATM/ATM+50,
     nearest weekly expiry) from /v1/live-data/quote, and append one row
     (timestamp, ce_oi, pe_oi, atm, spot) to DATA/_atm_oi_intraday.parquet.

The FastAPI app's caches are keyed on the parquets' mtime, so each append
shows up on the next /api/merged_trades request (~1s recompute of today only).

Today's live OI rows are TRANSIENT: next trading morning the 09:00 IST
investeq-oi-daily timer fetches the settled per-strike OI and rebuilds the
aggregate, replacing them with canonical data.

Units note: live-quote OI and the settled historical OI may use different
units (contracts vs lots). At startup the worker reconciles by comparing
yesterday's settled per-strike OI with the quote's previous_open_interest
and applies the median ratio to every live reading.

Run (on the VM, via investeq-live-strategy.service):
    python live_strategy_worker.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent / "ema_scanner"))
from groww_client import get_access_token, fetch_candles  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import strategy_telegram  # noqa: E402  — Strategy Suite OI-cross Telegram alerts

ROOT = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
MASTER    = ROOT / "nifty_1m_master.parquet"
OI_INTRA  = ROOT / "_atm_oi_intraday.parquet"
OI_DIR    = ROOT / "oi"
INSTR_CSV = ROOT / "_groww_instruments.csv"
LTP_JSON  = ROOT / "_nifty_ltp.json"   # tick feed for the UI's forming candle
LTP_POLL_SEC = 4

LTP_URL   = "https://api.groww.in/v1/live-data/ltp"
QUOTE_URL = "https://api.groww.in/v1/live-data/quote"
CHAIN_URL = "https://api.groww.in/v1/option-chain/exchange/NSE/underlying/NIFTY"
INSTR_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"

STRIKE_STEP = 50
MONS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SESSION_OPEN  = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
CYCLE_SEC     = 60


def now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def log(msg: str) -> None:
    print(f"[{now_ist():%H:%M:%S}] {msg}", flush=True)


_RULE_CHANGE = date(2025, 9, 1)   # NIFTY weekly expiry moved Thu -> Tue


def next_weekly_expiry(d: date) -> date:
    target_wd = 1 if d >= _RULE_CHANGE else 3
    while d.weekday() != target_wd:
        d += timedelta(days=1)
    return d


def gsym(expiry: date, strike: int, opt: str) -> str:
    return (f"NSE-NIFTY-{expiry.day:02d}{MONS[expiry.month-1]}"
            f"{str(expiry.year)[2:]}-{strike}-{opt}")


# ─── Groww plumbing ──────────────────────────────────────────────────────────

class Groww:
    def __init__(self):
        self.token = None
        self.sess = requests.Session()
        # The LTP tick thread and the per-minute cycle share this client —
        # serialize API calls so the Session isn't used concurrently.
        self.lock = threading.Lock()
        self._auth()

    def _auth(self):
        self.token = get_access_token(os.environ["GROWW_TOTP_JWT"],
                                      os.environ["GROWW_TOTP_SECRET"])
        log("auth OK")

    def _get(self, url: str, params: dict, tries: int = 4):
        backoff = 1.0
        for i in range(tries):
            h = {"Accept": "application/json",
                 "Authorization": f"Bearer {self.token}",
                 "X-API-VERSION": "1.0"}
            try:
                with self.lock:
                    r = self.sess.get(url, headers=h, params=params, timeout=10)
            except requests.RequestException:
                time.sleep(backoff); backoff *= 2; continue
            if r.status_code == 200:
                return r.json().get("payload") or {}
            if r.status_code == 401:
                self._auth(); continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(backoff); backoff *= 2; continue
            log(f"GET {url} -> {r.status_code} {r.text[:120]}")
            return None
        return None

    def spot_ltp(self) -> float | None:
        p = self._get(LTP_URL, {"segment": "CASH",
                                "exchange_symbols": "NSE_NIFTY"})
        v = (p or {}).get("NSE_NIFTY")
        return float(v) if v else None

    def quote_oi(self, trading_symbol: str) -> tuple[float, float] | None:
        """(open_interest, previous_open_interest) for one FNO leg."""
        p = self._get(QUOTE_URL, {"exchange": "NSE", "segment": "FNO",
                                  "trading_symbol": trading_symbol})
        if not p or p.get("open_interest") is None:
            return None
        return float(p["open_interest"]), float(p.get("previous_open_interest") or 0)

    def option_chain(self, expiry: date) -> dict | None:
        """Full NIFTY chain for one expiry — per-strike CE/PE OI + spot LTP
        in a single request (vs 1 LTP + 6 quotes), the cheapest way to feed
        the per-minute OI row. Once a minute, so patient retries are fine."""
        return self._get(CHAIN_URL, {"expiry_date": expiry.isoformat()},
                         tries=6)


# ─── Instruments map: groww_symbol -> trading_symbol ─────────────────────────

def load_symbol_map() -> dict[str, str]:
    """NIFTY FNO rows of the public instruments dump. Re-downloaded when the
    cached copy is older than ~20h (new weeklies appear after each expiry)."""
    fresh = (INSTR_CSV.exists()
             and time.time() - INSTR_CSV.stat().st_mtime < 20 * 3600)
    if not fresh:
        log("downloading instruments dump ...")
        r = requests.get(INSTR_URL, timeout=60)
        r.raise_for_status()
        INSTR_CSV.write_bytes(r.content)
    out = {}
    with INSTR_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gs = row.get("groww_symbol") or ""
            if gs.startswith("NSE-NIFTY-") and row.get("segment") == "FNO":
                out[gs] = row["trading_symbol"]
    log(f"symbol map: {len(out):,} NIFTY FNO instruments")
    return out


# ─── Unit reconciliation ─────────────────────────────────────────────────────

def oi_unit_scale(g: Groww, symmap: dict, expiry: date, atm: int) -> float:
    """Median(settled_yesterday / previous_open_interest) across the ATM±1
    legs — 1.0 when live-quote OI shares units with the historical data."""
    files = sorted(OI_DIR.glob("*.parquet"))
    if not files:
        return 1.0
    last = files[-1]
    try:
        df = pd.read_parquet(last)
    except Exception:
        return 1.0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[pd.to_datetime(df["expiry"]).dt.date == expiry]
    if df.empty:
        return 1.0
    ratios = []
    for strike in (atm - STRIKE_STEP, atm, atm + STRIKE_STEP):
        for opt in ("CE", "PE"):
            leg = df[(df["strike"] == strike) & (df["option_type"] == opt)]
            ts = symmap.get(gsym(expiry, strike, opt))
            if leg.empty or ts is None:
                continue
            q = g.quote_oi(ts)
            if not q or q[1] <= 0:
                continue
            settled = float(leg.sort_values("timestamp")["oi"].iloc[-1])
            if settled > 0:
                ratios.append(settled / q[1])
    if not ratios:
        return 1.0
    scale = float(pd.Series(ratios).median())
    log(f"OI unit scale: {scale:.3f} (from {len(ratios)} legs)")
    # Snap to 1.0 when the ratio is just day-to-day OI drift, not a unit gap.
    return 1.0 if 0.5 < scale < 2.0 else scale


# ─── Parquet appends (atomic) ────────────────────────────────────────────────

def _atomic_save(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def update_master(token: str) -> pd.Timestamp | None:
    """Incremental 1-min index bars up to now. Returns new last ts."""
    if not MASTER.exists():
        log("no master parquet — run fetch_nifty_master.py first")
        return None
    df = pd.read_parquet(MASTER)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    last = df["timestamp"].max()
    start = (last + pd.Timedelta(minutes=1)).to_pydatetime()
    end = now_ist()
    if start >= end:
        return last
    new = fetch_candles("NIFTY", "1m",
                        start.strftime("%Y-%m-%d %H:%M:%S"),
                        end.strftime("%Y-%m-%d %H:%M:%S"),
                        token, exchange="NSE", segment="CASH",
                        delay_s=0, verbose=False)
    if new is None or new.empty:
        return last
    new = new.rename(columns={"datetime": "ts"})
    new["timestamp"] = pd.to_datetime(new["ts"])
    new = new[["timestamp", "open", "high", "low", "close", "volume"]]
    new = new[(new["timestamp"].dt.time >= SESSION_OPEN)
              & (new["timestamp"].dt.time <= SESSION_CLOSE)]
    if new.empty:
        return last
    merged = (pd.concat([df, new], ignore_index=True)
              .sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last")
              .reset_index(drop=True))
    _atomic_save(merged, MASTER)
    return merged["timestamp"].max()


def append_oi_row(ts: pd.Timestamp, ce: float, pe: float,
                  atm: int, spot: float) -> None:
    df = pd.read_parquet(OI_INTRA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    row = pd.DataFrame([{"timestamp": ts, "ce_oi": ce, "pe_oi": pe,
                         "atm": atm, "spot": spot}])
    merged = (pd.concat([df, row], ignore_index=True)
              .sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last")
              .reset_index(drop=True))
    _atomic_save(merged, OI_INTRA)


# ─── LTP tick thread ─────────────────────────────────────────────────────────

def ltp_tick_loop(g: "Groww") -> None:
    """Every few seconds during the session, write the latest NIFTY LTP to a
    tiny JSON the UI polls to animate the forming candle. Atomic replace so
    readers never see a torn write."""
    while True:
        t = now_ist()
        if not in_session(t):
            time.sleep(60)
            continue
        try:
            ltp = g.spot_ltp()
            if ltp:
                tmp = LTP_JSON.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(
                    {"ts": t.strftime("%Y-%m-%dT%H:%M:%S"), "ltp": ltp}))
                tmp.replace(LTP_JSON)
        except Exception as e:
            log(f"ltp tick error: {e!r}")
        time.sleep(LTP_POLL_SEC)


# ─── Main loop ───────────────────────────────────────────────────────────────

def in_session(t: datetime) -> bool:
    return (t.weekday() < 5
            and SESSION_OPEN <= t.time() <= dtime(15, 35))


def seconds_to_next_open(t: datetime) -> float:
    nxt = t.replace(hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute,
                    second=0, microsecond=0)
    while nxt <= t or nxt.weekday() >= 5:
        nxt += timedelta(days=1)
        nxt = nxt.replace(hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute)
    return (nxt - t).total_seconds()


def run() -> None:
    g = Groww()
    strategy_telegram._startup_creds_check(log)
    threading.Thread(target=ltp_tick_loop, args=(g,), daemon=True).start()
    symmap: dict[str, str] = {}
    scale = 1.0
    scale_done_for: date | None = None

    while True:
        t = now_ist()
        if not in_session(t):
            wait = min(seconds_to_next_open(t), 3600)
            log(f"off-hours — sleeping {wait/60:.0f} min")
            time.sleep(wait)
            continue

        cycle_start = time.time()
        try:
            today = t.date()
            expiry = next_weekly_expiry(today)

            if not symmap:
                symmap = load_symbol_map()

            # 1. index bars
            last = update_master(g.token)

            # 2. live OI snapshot — one option-chain call covers spot LTP +
            #    per-strike OI for every leg we need.
            chain = g.option_chain(expiry)
            spot = float(chain.get("underlying_ltp") or 0) if chain else 0.0
            if spot:
                atm = int(round(spot / STRIKE_STEP) * STRIKE_STEP)
                if scale_done_for != today:
                    scale = oi_unit_scale(g, symmap, expiry, atm)
                    scale_done_for = today
                strikes = chain.get("strikes") or {}
                ce_sum = pe_sum = 0.0
                got = 0
                for strike in (atm - STRIKE_STEP, atm, atm + STRIKE_STEP):
                    legs = strikes.get(str(strike)) or {}
                    ce = (legs.get("CE") or {}).get("open_interest")
                    pe = (legs.get("PE") or {}).get("open_interest")
                    if ce is not None and pe is not None:
                        got += 1
                        ce_sum += float(ce) * scale
                        pe_sum += float(pe) * scale
                if got == 3:
                    ts = pd.Timestamp(t.replace(second=0, microsecond=0))
                    append_oi_row(ts, ce_sum, pe_sum, atm, spot)
                    log(f"tick {ts:%H:%M} spot={spot:.1f} atm={atm} "
                        f"ce={ce_sum:,.0f} pe={pe_sum:,.0f} "
                        f"(master->{str(last)[11:16] if last is not None else '?'})")
                    # Fire Telegram alerts for any fresh Strategy Suite signal
                    # the just-appended OI row produced. Non-fatal by design.
                    try:
                        strategy_telegram.maybe_alert(today, log=log)
                    except Exception as e:
                        log(f"telegram alert error: {e!r}")
                else:
                    log(f"skip OI append — only {got}/3 strikes in chain")
            else:
                log("skip cycle — no chain / spot")
        except Exception as e:
            log(f"cycle error: {e!r}")

        time.sleep(max(5.0, CYCLE_SEC - (time.time() - cycle_start)))


if __name__ == "__main__":
    run()
