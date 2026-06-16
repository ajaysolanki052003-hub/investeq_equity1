"""Live Telegram alerts for Strategy Suite (/strategy) OI-crossover signals.

Hook contract:
    maybe_alert(today, log=...) is called once per minute by
    live_strategy_worker.run() right after it appends the latest ATM±1 OI row
    to DATA/_atm_oi_intraday.parquet. It recomputes TODAY's merged 'both'-mode
    entries — exactly what /api/merged_trades shows on the dashboard:
        • ΣOI  — cumulative-since-09:15 OI crossover (min-gap-7 whipsaw filter)
        • Σ·10 — rolling-window OI crossover (3m TF, window 10)
        • the two merged in time order with same-side dedup within 7 minutes
    and posts a Telegram message for every entry that has fired since the last
    alert.

State:
    DATA/_strategy_alerts.json holds the list of today's entry-timestamps that
    have already been alerted, so a worker restart or a same-minute re-run
    never re-sends a signal.

Credentials:
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from env (systemd EnvironmentFile=
    /etc/investeq.env) — the SAME bot/chat the EMA scanner uses. Missing creds
    → silently skipped (logged once at worker startup).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# Same entry generators the FastAPI app uses for /api/merged_trades.
from strategies.sigma_total_oi_entry import compute_entries as _sigma_oi   # noqa: E402
from strategies.sigma5_entry_exit import compute_entries as _sigma_n       # noqa: E402

DATA = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
OI_INTRA = DATA / "_atm_oi_intraday.parquet"
STATE = DATA / "_strategy_alerts.json"

DEDUP_SEC = 7 * 60   # same-side dedup window — matches /api/merged_trades


def _fresh_oi_today(today: date) -> pd.DataFrame | None:
    """Read the OI parquet FRESH (the strategy modules' lru_cache would
    otherwise serve a stale frame inside this long-lived worker) and return
    only today's rows — both strategies reset their accumulators per-day, so
    today's slice yields identical entries to the full series."""
    try:
        df = pd.read_parquet(OI_INTRA)
    except Exception:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"].dt.date == today].sort_values("timestamp")
    return df.reset_index(drop=True)


def _merged_entries(frame: pd.DataFrame, today: date) -> list[dict]:
    """Replicate /api/merged_trades 'both' mode: ΣOI + Σ·10 entries merged in
    time order with same-side dedup within 7 minutes (first signal wins)."""
    sigot, _ = _sigma_oi(min_gap_bars=7, df=frame)
    sig10, _ = _sigma_n(tf="3m", window=10, df=frame)
    merged = ([{**e, "_source": "ΣOI"} for e in sigot] +
              [{**e, "_source": "Σ·10"} for e in sig10])
    merged = [e for e in merged if e["day"] == str(today)]
    merged.sort(key=lambda e: (str(e["day"]), str(e["entry_time"])))
    kept, last = [], {}
    for e in merged:
        key = (e["day"], e["side"])
        ts = pd.Timestamp(e["entry_time"])
        prev = last.get(key)
        if prev is not None and (ts - prev).total_seconds() < DEDUP_SEC:
            continue
        kept.append(e)
        last[key] = ts
    return kept


def _load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE)


def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception as e:
        print(f"[telegram] send failed: {e}", flush=True)
        return False


def _format(e: dict) -> str:
    head = "🟢 NIFTY *LONG*" if e["side"] == "LONG" else "🔴 NIFTY *SHORT*"
    t = str(e["entry_time"])[11:16]
    return (f"{head} — Strategy Suite\n"
            f"{e['_source']} OI crossover · {t} IST\n"
            f"Spot {float(e['entry_spot']):,.1f} · ATM {int(e['entry_atm'])}")


def maybe_alert(today: date, log=print) -> None:
    """Post Telegram alerts for any merged 'both'-mode entry that fired today
    and hasn't been alerted yet. Safe to call every minute; the caller also
    wraps it so a failure never disturbs the OI-append loop.

    Seed-on-first-run: the FIRST cycle of a new trading day (or the first
    cycle after alerts come online — e.g. creds added + a mid-session
    restart) records whatever signals have already fired WITHOUT sending,
    so a restart never dumps a burst of stale morning signals. Only crosses
    that fire from that point on are alerted."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return  # creds missing — _startup_creds_check already logged it once

    frame = _fresh_oi_today(today)
    if frame is None or len(frame) < 3:
        return

    entries = _merged_entries(frame, today)
    daykey = str(today)
    state = _load_state()

    if state.get("day") != daykey:
        # New day / first run with alerts active — seed, do not send.
        seed = [str(e["entry_time"]) for e in entries]
        _save_state({"day": daykey, "alerted": seed})
        if seed:
            log(f"[telegram] seeded {len(seed)} pre-existing signal(s) for "
                f"{daykey} (not sent — restart catch-up suppressed)")
        return

    if not entries:
        return
    alerted = set(state.get("alerted", []))
    fresh = [e for e in entries if str(e["entry_time"]) not in alerted]
    if not fresh:
        return

    for e in fresh:
        if _send_telegram(token, chat_id, _format(e)):
            alerted.add(str(e["entry_time"]))
            log(f"[telegram] sent {e['side']} {e['_source']} "
                f"@ {str(e['entry_time'])[11:16]}")
        else:
            log(f"[telegram] send failed for {e['side']} "
                f"@ {str(e['entry_time'])[11:16]} — will retry next cycle")
    _save_state({"day": daykey, "alerted": sorted(alerted)})


def _startup_creds_check(log=print) -> None:
    if not (os.environ.get("TELEGRAM_BOT_TOKEN")
            and os.environ.get("TELEGRAM_CHAT_ID")):
        log("[telegram] creds missing (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) "
            "— Strategy Suite alerts DISABLED")
    else:
        log("[telegram] Strategy Suite alerts enabled")


if __name__ == "__main__":
    # Manual check: python strategy_telegram.py [YYYY-MM-DD]
    # Prints the merged entries it WOULD alert (no send unless creds present).
    d = (pd.Timestamp(sys.argv[1]).date() if len(sys.argv) > 1
         else (pd.Timestamp.utcnow() + pd.Timedelta(hours=5, minutes=30)).date())
    print(f"computing merged entries for {d} ...")
    fr = _fresh_oi_today(d)
    if fr is None or fr.empty:
        print("no OI rows for that day"); sys.exit(0)
    ents = _merged_entries(fr, d)
    print(f"{len(ents)} merged entries:")
    for e in ents:
        print("  " + _format(e).replace("\n", " | "))
