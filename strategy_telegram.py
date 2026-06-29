"""Live Telegram alerts for Strategy Suite (/strategy) OI-crossover signals.

Hook contract:
    maybe_alert(today, log=...) is called once per minute by
    live_strategy_worker.run() right after it appends the latest ATM±1 OI row
    to DATA/_atm_oi_intraday.parquet.

Single source of truth:
    Earlier this module RE-IMPLEMENTED the merge+dedup that /api/merged_trades
    does, but only the first two steps of it — so it alerted on raw crossovers
    that the dashboard actually SUPPRESSES (a signal firing while a prior trade
    is still open). That drift is the whole bug class ("telegram shows an extra
    signal that isn't on the chart").

    So we no longer compute anything here. We ask the SAME FastAPI endpoint the
    chart's VIEW panel calls — /api/merged_trades with the chart's exact locked
    params (mode=both, position_mode=single, and the endpoint's default
    SL/TGT/trail/time-stop) — and alert on precisely the trades it returns.
    Whatever the chart shows, Telegram sends. Future tuning of the entry/exit/
    overlap rules flows through automatically; the alert can never go stale.

    The app has no internal auth (the login gate is nginx-only), so the worker
    reaches it directly on the loopback. Base URL is STRATEGY_API_BASE
    (default http://127.0.0.1:8705 — the live uvicorn port on the VM).

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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA = Path(os.environ.get(
    "INVESTEQ_DATA",
    r"C:\Users\User\Desktop\investeq_ajs\DATA"
    if os.name == "nt" else "/home/ajay/investeq_ajs/DATA"))
STATE = DATA / "_strategy_alerts.json"

# The chart's VIEW panel locks every param except the Opp-OK/Single toggle,
# whose default button is `single` (oi_options_trading/app.py). Everything else
# (SL/TGT/trail/trail_pct/time-stop) matches the endpoint's own defaults, so we
# only need to pin mode + position_mode to reproduce the chart exactly.
API_BASE = os.environ.get("STRATEGY_API_BASE", "http://127.0.0.1:8705").rstrip("/")
TRADES_URL = f"{API_BASE}/api/merged_trades?mode=both&position_mode=single"


def _chart_trades(today: date, log=print) -> list[dict] | None:
    """Today's trades EXACTLY as the dashboard VIEW panel shows them — fetched
    from the live /api/merged_trades endpoint. Returns None on any transport/
    server error (caller treats None as 'skip this cycle, retry next minute');
    returns [] when the endpoint is healthy but has no trades for today."""
    try:
        with urllib.request.urlopen(TRADES_URL, timeout=10) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        log(f"[telegram] /api/merged_trades fetch failed: {e!r}")
        return None
    if not isinstance(payload, dict) or payload.get("error"):
        log(f"[telegram] /api/merged_trades error: "
            f"{payload.get('error') if isinstance(payload, dict) else payload!r}")
        return None
    daykey = str(today)
    trades = [t for t in (payload.get("trades") or []) if str(t.get("day")) == daykey]
    trades.sort(key=lambda t: str(t.get("entry_time")))
    return trades


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


_SRC_LABEL = {"sigma_oi": "ΣOI", "sigma10": "Σ·10"}


def _format(t: dict) -> str:
    head = "🟢 NIFTY *LONG*" if t["side"] == "LONG" else "🔴 NIFTY *SHORT*"
    src = _SRC_LABEL.get(t.get("source"), t.get("source") or "OI")
    hm = str(t.get("entry_time"))[11:16]
    atm = t.get("entry_atm")
    atm_str = f" · ATM {int(atm)}" if atm is not None else ""
    return (f"{head} — Strategy Suite\n"
            f"{src} OI crossover · {hm} IST\n"
            f"Spot {float(t['entry_spot']):,.1f}{atm_str}")


def maybe_alert(today: date, log=print) -> None:
    """Post Telegram alerts for any VIEW-panel trade that fired today and hasn't
    been alerted yet. Safe to call every minute; the caller also wraps it so a
    failure never disturbs the OI-append loop.

    Seed-on-first-run: the FIRST cycle of a new trading day (or the first cycle
    after alerts come online — e.g. creds added + a mid-session restart) records
    whatever signals have already fired WITHOUT sending, so a restart never
    dumps a burst of stale morning signals. Only crosses that fire from that
    point on are alerted."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return  # creds missing — _startup_creds_check already logged it once

    trades = _chart_trades(today, log=log)
    if trades is None:
        return  # endpoint not reachable this cycle — retry next minute

    daykey = str(today)
    state = _load_state()

    if state.get("day") != daykey:
        # New day / first run with alerts active — seed, do not send.
        seed = [str(t["entry_time"]) for t in trades]
        _save_state({"day": daykey, "alerted": seed})
        if seed:
            log(f"[telegram] seeded {len(seed)} pre-existing signal(s) for "
                f"{daykey} (not sent — restart catch-up suppressed)")
        return

    if not trades:
        return
    alerted = set(state.get("alerted", []))
    fresh = [t for t in trades if str(t["entry_time"]) not in alerted]
    if not fresh:
        return

    for t in fresh:
        if _send_telegram(token, chat_id, _format(t)):
            alerted.add(str(t["entry_time"]))
            log(f"[telegram] sent {t['side']} {t.get('source')} "
                f"@ {str(t['entry_time'])[11:16]}")
        else:
            log(f"[telegram] send failed for {t['side']} "
                f"@ {str(t['entry_time'])[11:16]} — will retry next cycle")
    _save_state({"day": daykey, "alerted": sorted(alerted)})


def _startup_creds_check(log=print) -> None:
    if not (os.environ.get("TELEGRAM_BOT_TOKEN")
            and os.environ.get("TELEGRAM_CHAT_ID")):
        log("[telegram] creds missing (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) "
            "— Strategy Suite alerts DISABLED")
    else:
        log(f"[telegram] Strategy Suite alerts enabled (source: {TRADES_URL})")


if __name__ == "__main__":
    # Manual check: python strategy_telegram.py [YYYY-MM-DD]
    # Prints the VIEW-panel trades it WOULD alert for that day — hits the live
    # endpoint, so the Strategy Suite app must be running (STRATEGY_API_BASE).
    try:                       # emoji-safe printing on a Windows cp1252 console
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    d = (date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1
         else date.today())
    print(f"fetching VIEW-panel trades for {d} from {TRADES_URL} ...")
    ts = _chart_trades(d)
    if ts is None:
        print("endpoint not reachable / returned an error"); sys.exit(1)
    print(f"{len(ts)} trades:")
    for t in ts:
        print("  " + _format(t).replace("\n", " | "))
