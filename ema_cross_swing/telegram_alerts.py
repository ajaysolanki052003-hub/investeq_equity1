"""Telegram alert posting for live 1h EMA21/50 swing-touch signals.

Hook contract:
    maybe_alert_1h(df1h, target_day) is called by scan_universe.py after the
    per-date parquet is written. It applies the live-only / latest-closed-bar
    / NO-TRADE / pierce-both / dedupe filter chain and posts a single
    Telegram message if anything survives.

State:
    A tiny text file at data/scan_cache/.last_alerted_bar.txt stores the
    timestamp of the most-recent bar we've alerted on. Same bar (or older)
    never re-alerts — handles re-runs, manual fires, and the EOD scan that
    chains off the 1d-candle fetcher for the same 15:15 bar.

Credentials:
    Read from env vars TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
    Loaded from /etc/investeq.env via systemd EnvironmentFile=.
"""
from __future__ import annotations
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# State files live next to the per-date parquets. Hidden (leading dot) so
# `glob("scan_*.parquet")` doesn't accidentally pick them up.
HERE = Path(__file__).resolve().parent
_DATA_ROOT = HERE.parent / "ema_scanner" / "data"
_STATE_FILE = _DATA_ROOT / "scan_cache" / ".last_alerted_bar.txt"
_BRIEF_STATE_FILE = _DATA_ROOT / "scan_cache" / ".last_morning_brief.txt"


def _now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _most_recent_closed_1h_bar(now: datetime | None = None) -> pd.Timestamp | None:
    """Return the start-timestamp of the most recently CLOSED 1h NSE bar.

    NSE 1h bars are stamped at :15 past the hour and represent the period
    [HH:15, HH+1:15). The bar starting at 09:15 closes at 10:15.

    Trading bars: 09:15, 10:15, 11:15, 12:15, 13:15, 14:15, 15:15.
    The 15:15 bar closes at 15:30 (NSE close), broker may stamp differently.

    Returns None when no 1h bar has closed yet today (e.g. before 10:15 IST)
    or when called outside NSE hours."""
    n = now or _now_ist()
    # Most recent :15 boundary on the wall clock that is <= now
    close_t = n.replace(minute=15, second=0, microsecond=0)
    if close_t > n:
        close_t -= timedelta(hours=1)
    bar_start = close_t - timedelta(hours=1)

    # Bar must be a real NSE trading-hour bar (09:15 through 15:15)
    if bar_start.hour < 9 or bar_start.hour > 15:
        return None
    if bar_start.hour == 9 and bar_start.minute < 15:
        return None
    return pd.Timestamp(bar_start.replace(microsecond=0))


def _load_last_alerted_bar() -> pd.Timestamp | None:
    if not _STATE_FILE.exists():
        return None
    try:
        s = _STATE_FILE.read_text().strip()
        return pd.Timestamp(s) if s else None
    except Exception:
        return None


def _save_last_alerted_bar(ts: pd.Timestamp) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(ts.isoformat())


def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=data,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            ok = r.status == 200
            if not ok:
                print(f"[telegram] HTTP {r.status}", flush=True)
            return ok
    except Exception as e:
        print(f"[telegram] send failed: {e}", flush=True)
        return False


def _compose_message(df: pd.DataFrame, target_day, bar_ts: pd.Timestamp) -> str:
    bar_label = bar_ts.strftime("%H:%M")
    lines = [
        f"*1h — {target_day} {bar_label}*",
        "```",
        f"{'Sym':<10} {'Side':<4} {'Entry':>9} {'SL':>9} {'Risk%':>6}",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{str(r['Symbol'])[:10]:<10} {r['Side']:<4} "
            f"{float(r['Entry']):>9.2f} {float(r['SL']):>9.2f} "
            f"{float(r['Risk %']):>6.2f}"
        )
    lines.append("```")
    return "\n".join(lines)


def maybe_alert_1h(df1h: pd.DataFrame, target_day) -> None:
    """Apply the alert filter chain and post to Telegram if anything passes.

    Filter chain (all must pass):
      1. Live run only      — target_day == today IST
      2. There IS a closed 1h bar today
      3. Bar not already alerted (dedupe via state file)
      4. Signal Touch Time == that bar's timestamp
      5. Status != NO TRADE
      6. Pierce-both: BUY  → LowBelowBoth True
                      SELL → HighAboveBoth True
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        print("[telegram] creds missing (TELEGRAM_BOT_TOKEN / "
              "TELEGRAM_CHAT_ID) — skipping alert", flush=True)
        return

    if target_day != _now_ist().date():
        return  # backfill / past date — never alert

    bar_ts = _most_recent_closed_1h_bar()
    if bar_ts is None:
        print("[telegram] no closed 1h bar yet — skip", flush=True)
        return

    last_alerted = _load_last_alerted_bar()
    if last_alerted is not None and bar_ts <= last_alerted:
        print(f"[telegram] bar {bar_ts} already alerted "
              f"(last={last_alerted}) — skip", flush=True)
        return

    if df1h is None or df1h.empty or "Touch Time" not in df1h.columns:
        # Mark as alerted (empty bar is still "handled" — avoid re-checking)
        _save_last_alerted_bar(bar_ts)
        return

    df = df1h.copy()
    df["Touch Time"] = pd.to_datetime(df["Touch Time"])
    # Compare timestamps (the bar_ts is stamped at HH:15:00 with no tz)
    target_ts = pd.Timestamp(target_day) + pd.Timedelta(
        hours=bar_ts.hour, minutes=bar_ts.minute)
    df = df[df["Touch Time"] == target_ts]

    if df.empty:
        _save_last_alerted_bar(bar_ts)
        return

    df = df[df["Status"] != "NO TRADE"]

    if "LowBelowBoth" in df.columns and "HighAboveBoth" in df.columns:
        is_buy = df["Side"].astype(str).str.upper() == "BUY"
        is_sell = df["Side"].astype(str).str.upper() == "SELL"
        mask = (is_buy & df["LowBelowBoth"].astype(bool)) | \
               (is_sell & df["HighAboveBoth"].astype(bool))
        df = df[mask]

    if df.empty:
        _save_last_alerted_bar(bar_ts)
        print(f"[telegram] bar {bar_ts}: 0 signals passed pierce-both filter",
              flush=True)
        return

    msg = _compose_message(df, target_day, bar_ts)
    if _send_telegram(token, chat_id, msg):
        _save_last_alerted_bar(bar_ts)
        print(f"[telegram] sent {len(df)} signals for bar {bar_ts}", flush=True)
    else:
        print(f"[telegram] send failed for bar {bar_ts} — will retry next run",
              flush=True)


# ───────────────────── Morning brief (09:00 IST Mon..Fri) ─────────────────────

def _find_last_trading_day_parquet(today=None) -> tuple | None:
    """Return (date, Path) of the most recent scan_<date>.parquet strictly
    before `today`, or None. Used by the morning brief to find yesterday's
    EOD parquet (Friday's parquet on Monday morning, etc.)."""
    today = today or _now_ist().date()
    cache_dir = _DATA_ROOT / "scan_cache"
    if not cache_dir.exists():
        return None
    found = []
    for p in cache_dir.glob("scan_*.parquet"):
        try:
            d = pd.to_datetime(p.stem.replace("scan_", "")).date()
            if d < today:
                found.append((d, p))
        except Exception:
            continue
    if not found:
        return None
    found.sort(reverse=True)
    return found[0]


def _load_last_brief_date():
    if not _BRIEF_STATE_FILE.exists():
        return None
    try:
        return pd.to_datetime(_BRIEF_STATE_FILE.read_text().strip()).date()
    except Exception:
        return None


def _save_brief_date(d) -> None:
    _BRIEF_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BRIEF_STATE_FILE.write_text(str(d))


def _compose_brief_message(df: pd.DataFrame, last_date) -> str:
    lines = [
        f"*1h — {last_date} 15:15*",
        "```",
        f"{'Sym':<10} {'Side':<4} {'Entry':>9} {'SL':>9} {'Risk%':>6}",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{str(r['Symbol'])[:10]:<10} {r['Side']:<4} "
            f"{float(r['Entry']):>9.2f} {float(r['SL']):>9.2f} "
            f"{float(r['Risk %']):>6.2f}"
        )
    lines.append("```")
    return "\n".join(lines)


def send_morning_brief() -> None:
    """At 09:00 IST: read last trading day's parquet, filter df1h to the
    15:15 bar (the final bar covering 15:15 → 15:30), apply NO-TRADE and
    pierce-both filters, then post one Telegram message. Dedupe via
    .last_morning_brief.txt so a same-day re-run is a no-op."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        print("[brief] creds missing — skip", flush=True)
        return

    today = _now_ist().date()
    last_brief = _load_last_brief_date()
    if last_brief == today:
        print(f"[brief] already sent today ({today}) — skip", flush=True)
        return

    found = _find_last_trading_day_parquet(today)
    if found is None:
        print("[brief] no prior parquet found — skip", flush=True)
        return
    last_date, parquet_path = found
    print(f"[brief] sourcing from {parquet_path.name} (last trading day "
          f"{last_date})", flush=True)

    try:
        d = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[brief] parquet read failed: {e}", flush=True)
        return

    df = d[d["_split"] == "1h"] if "_split" in d.columns else pd.DataFrame()
    if df.empty:
        print(f"[brief] no 1h signals on {last_date}", flush=True)
        _save_brief_date(today)
        return

    df = df.copy()
    df["Touch Time"] = pd.to_datetime(df["Touch Time"])
    target_ts = pd.Timestamp(last_date) + pd.Timedelta(hours=15, minutes=15)
    df = df[df["Touch Time"] == target_ts]
    if df.empty:
        print(f"[brief] no signals on {last_date} 15:15 bar", flush=True)
        _save_brief_date(today)
        return

    df = df[df["Status"] != "NO TRADE"]
    if "LowBelowBoth" in df.columns and "HighAboveBoth" in df.columns:
        is_buy = df["Side"].astype(str).str.upper() == "BUY"
        is_sell = df["Side"].astype(str).str.upper() == "SELL"
        mask = (is_buy & df["LowBelowBoth"].astype(bool)) | \
               (is_sell & df["HighAboveBoth"].astype(bool))
        df = df[mask]

    if df.empty:
        print(f"[brief] 0 signals passed filters on {last_date} 15:15 bar",
              flush=True)
        _save_brief_date(today)
        return

    msg = _compose_brief_message(df, last_date)
    if _send_telegram(token, chat_id, msg):
        _save_brief_date(today)
        print(f"[brief] sent {len(df)} signals from {last_date} 15:15",
              flush=True)
    else:
        print("[brief] send failed — will retry on next timer fire", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "morning-brief":
        send_morning_brief()
    else:
        print("Usage: python telegram_alerts.py morning-brief")
        sys.exit(2)
