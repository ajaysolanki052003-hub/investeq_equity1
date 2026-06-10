"""
Swing-Low Daily Scanner — Streamlit UI.

Scans every stock in ../ema_scanner/data/{1h,1d}/ for swing-low project
EMA21/50 touch-entries that fired on a chosen day (default 2026-05-14).
Reports three tables:
   1d entries on the day
   1h entries on the day
   STARRED — stocks present in both timeframes (highest conviction)

Click any row to load a chart for that symbol (last 120 bars) with the
touch marker highlighted.

Run:  streamlit run scan_app.py --server.port 8532
"""
from __future__ import annotations
import os
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


# ── Live-data integration (LTP + 1m overlay on the active chart) ──────────
_SCANNER_ROOT    = Path(__file__).resolve().parent.parent / "ema_scanner"
_LIVE_DIR        = _SCANNER_ROOT / "data" / "live"
_ONE_MIN_DIR     = _SCANNER_ROOT / "data" / "1m"
_LIVE_LTP_PATH   = _LIVE_DIR / "ltp.parquet"
_WATCHLIST_PATH  = _LIVE_DIR / "watchlist.txt"
_SCAN_CACHE_DIR  = _SCANNER_ROOT / "data" / "scan_cache"


def _scan_cache_path(target_day) -> Path:
    """Per-date cache file. The backend cron writes one of these every market
    day at 15:40 IST; the UI reads them for instant date-pick results."""
    return _SCAN_CACHE_DIR / f"scan_{target_day}.parquet"


def _save_scan_cache(target_day, max_sl_pct, df1d, df1h) -> None:
    """Persist a scan result keyed on its target date. The shape is two stacked
    DataFrames (1d, 1h) in one parquet, identified by an _split column."""
    try:
        _SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        d = pd.concat([df1d.assign(_split="1d"),
                       df1h.assign(_split="1h")], ignore_index=True)
        d.attrs["target_day"] = str(target_day)
        d.attrs["max_sl_pct"] = max_sl_pct
        d.attrs["saved_at"]   = _now_ist().isoformat(timespec="seconds")
        d.to_parquet(_scan_cache_path(target_day), index=False)
    except Exception:
        pass


def _refresh_metrics_from_history(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Rewrite Current / P&L % / Peak % / Status using the LATEST OHLC bars.

    The parquet freezes these columns at scan time, so loading a past date
    shows stale prices. This walks each row forward from Touch Time using
    today's CSV — open trades reflect current price, SL HITs are detected
    if SL was crossed any time since the touch, Peak % is the max favorable
    move up to either SL hit or the most recent bar.

    NO TRADE rows stay NO TRADE (tradability decision was final at scan
    time), but their dynamic columns still update so the user sees how the
    untradable signal evolved."""
    if df is None or df.empty:
        return df
    out = df.copy()
    cache: dict = {}
    for i, r in out.iterrows():
        sym = str(r.get("Symbol", ""))
        if not sym:
            continue
        try:
            entry = float(r["Entry"])
        except (TypeError, ValueError):
            continue
        side = str(r.get("Side", ""))
        try:
            ttime = pd.Timestamp(r["Touch Time"])
        except Exception:
            continue
        if sym not in cache:
            try:
                cache[sym] = load_ohlc(sym, tf, DATA_ROOT_DEFAULT)
            except Exception:
                cache[sym] = None
        full = cache[sym]
        if full is None or full.empty or ttime not in full.index:
            continue

        current_close = float(full["Close"].iloc[-1])
        sl       = float(r["SL"])     if pd.notna(r["SL"])     else None
        risk_pct = float(r["Risk %"]) if pd.notna(r["Risk %"]) else None
        was_no_trade = (str(r.get("Status")) == "NO TRADE")

        sl_hit_at = None
        peak_pct  = None
        post = full.loc[ttime:].iloc[1:]
        if sl is not None and not post.empty:
            hit = (post["Low"] <= sl) if side == "BUY" else (post["High"] >= sl)
            if hit.any():
                sl_hit_at = hit[hit].index[0]
            eval_post = post.loc[:sl_hit_at] if sl_hit_at is not None else post
            if not eval_post.empty:
                if side == "BUY":
                    peak_pct = (float(eval_post["High"].max()) - entry) / entry * 100
                else:
                    peak_pct = (entry - float(eval_post["Low"].min())) / entry * 100

        if was_no_trade:
            status = "NO TRADE"
            pnl_pct = ((current_close - entry) if side == "BUY"
                       else (entry - current_close)) / entry * 100
        elif sl_hit_at is not None and risk_pct is not None:
            status = "SL HIT"
            pnl_pct = -risk_pct
        else:
            status = "OPEN"
            pnl_pct = ((current_close - entry) if side == "BUY"
                       else (entry - current_close)) / entry * 100

        out.at[i, "Current"] = round(current_close, 2)
        out.at[i, "P&L %"]   = round(pnl_pct, 2)
        out.at[i, "Peak %"]  = round(peak_pct, 2) if peak_pct is not None else None
        out.at[i, "Status"]  = status
    return out


def _load_scan_cache_for_day(target_day):
    """Returns (df1d, df1h, saved_at_str) for the given date if cached, else
    (None, None, None). Loads in <50ms — perfect for instant date-pick UX."""
    p = _scan_cache_path(target_day)
    if not p.exists():
        return None, None, None
    try:
        d = pd.read_parquet(p)
        d1 = d[d["_split"] == "1d"].drop(columns=["_split"]).reset_index(drop=True)
        d2 = d[d["_split"] == "1h"].drop(columns=["_split"]).reset_index(drop=True)
        # Parquet freezes Current/P&L%/Peak%/Status at scan time. Replay
        # each row forward from Touch Time using the latest CSV so the table
        # reflects current prices, fresh SL hits, and updated peaks.
        d1 = _refresh_metrics_from_history(d1, "1d")
        d2 = _refresh_metrics_from_history(d2, "1h")
        # parquet attrs aren't preserved cross-process; use file mtime instead.
        from datetime import datetime as _dt
        saved_at = _dt.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return d1, d2, saved_at
    except Exception:
        return None, None, None


def _list_cached_days(max_n=60):
    """Return list of dates we have pre-computed scans for, newest first."""
    if not _SCAN_CACHE_DIR.exists():
        return []
    dates = []
    for p in _SCAN_CACHE_DIR.glob("scan_*.parquet"):
        try:
            dates.append(pd.to_datetime(p.stem.replace("scan_", "")).date())
        except Exception:
            continue
    return sorted(dates, reverse=True)[:max_n]


@st.cache_data(ttl=300, show_spinner=False)
def _nse_trading_days_set() -> set:
    """Set of every date the NSE traded. Union of:
       (a) the date columns of up to 10 alphabetically-first 1d CSVs — a
           single lagging symbol (e.g. KIRLOSENG missing the last 2 days)
           won't poison the classifier; and
       (b) every date for which a scan cache parquet exists — those are
           definitionally trading days because we already scanned them.
    Cached 5 min."""
    tds: set = set()
    folder = _SCANNER_ROOT / "data" / "1d"
    if folder.is_dir():
        try:
            samples = sorted(p for p in folder.iterdir()
                             if p.name.endswith("_historical.csv"))[:10]
            for s in samples:
                try:
                    df = pd.read_csv(s, usecols=["datetime"])
                    tds.update(pd.to_datetime(df["datetime"]).dt.date)
                except Exception:
                    continue
        except Exception:
            pass
    # Cached scans are by definition trading days — fold them in too so we
    # never flag a date as "future" if we've already produced a scan for it.
    if _SCAN_CACHE_DIR.exists():
        for p in _SCAN_CACHE_DIR.glob("scan_*.parquet"):
            try:
                tds.add(pd.to_datetime(p.stem.replace("scan_", "")).date())
            except Exception:
                continue
    return tds


def _classify_date(d) -> str:
    """Returns one of: 'weekend', 'holiday', 'future', 'trading'.

    Weekend = Sat/Sun. Holiday = weekday NSE didn't trade (Republic Day,
    Holi, Maharashtra Day, ...). Future = beyond the latest available data."""
    if d.weekday() >= 5:
        return "weekend"
    tds = _nse_trading_days_set()
    if not tds:
        return "trading"  # data not available, assume tradable
    if d in tds:
        return "trading"
    if d > max(tds):
        return "future"
    return "holiday"


def _tf_period_start(ts: datetime, tf: str) -> datetime:
    """Start of the candle that `ts` belongs to, in the given TF.
    1d → today's 09:15. 1h → most recent :15 boundary."""
    if tf == "1d":
        return ts.replace(hour=9, minute=15, second=0, microsecond=0)
    if tf == "1h":
        # 09:15 / 10:15 / .../ 15:15 — most recent one <= ts
        h = ts.hour if ts.minute >= 15 else ts.hour - 1
        if h < 9:
            return ts.replace(hour=9, minute=15, second=0, microsecond=0)
        return ts.replace(hour=h, minute=15, second=0, microsecond=0)
    # Default: minute boundary
    return ts.replace(second=0, microsecond=0)


def _now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _market_open_ist() -> bool:
    n = _now_ist()
    if n.weekday() >= 5:
        return False
    return (n.hour, n.minute) >= (9, 15) and (n.hour, n.minute) <= (15, 30)


def _update_watchlist(symbols: list[str]) -> None:
    """Best-effort rewrite of the LTP worker's watchlist. Silent on failure
    (file system might not be writable in some dev setups)."""
    try:
        _LIVE_DIR.mkdir(parents=True, exist_ok=True)
        _WATCHLIST_PATH.write_text("\n".join(sorted(set(symbols))) + "\n")
    except Exception:
        pass


def _register_watch(symbols) -> None:
    """Accumulate symbols whose LTP this page render needs.
    Flushed once at page-bottom via _flush_watchlist()."""
    if "_watch_set" not in st.session_state:
        st.session_state["_watch_set"] = set()
    for s in symbols:
        if s:
            st.session_state["_watch_set"].add(str(s))


def _flush_watchlist() -> None:
    syms = st.session_state.pop("_watch_set", None)
    if syms:
        _update_watchlist(list(syms))


@st.cache_data(ttl=1.0, show_spinner=False)
def _read_ltp_df() -> pd.DataFrame:
    """Cached LTP table (refreshed every 2s)."""
    if not _LIVE_LTP_PATH.exists():
        return pd.DataFrame(columns=["symbol", "ltp", "ts"])
    try:
        return pd.read_parquet(_LIVE_LTP_PATH)
    except Exception:
        return pd.DataFrame(columns=["symbol", "ltp", "ts"])


@st.cache_data(ttl=10.0, show_spinner=False)
def _read_1m_df(symbol: str) -> pd.DataFrame:
    """Cached read of data/1m/<SYM>_historical.csv."""
    path = _ONE_MIN_DIR / f"{symbol}_historical.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    except Exception:
        return pd.DataFrame()


def _overlay_live_ltp(candles: list[dict], symbol: str, tf: str) -> list[dict]:
    """Build a forming candle for the current period:
      1. Find the 1-minute bars that fall inside the current TF period.
      2. Aggregate them to OHLC (open=first, high=max, low=min, close=last).
      3. Bump the close with the latest LTP if it's newer than the last 1m bar.
      4. Either update the last existing candle (if it covers the same period)
         or append the forming candle to the chart.

    Falls back to a pure-LTP nudge of the last candle if no 1m data is present
    yet (e.g. first minute of the session before the worker has filled in)."""
    if not candles:
        return candles

    ltp_df = _read_ltp_df()
    ltp = None
    if not ltp_df.empty:
        row = ltp_df[ltp_df["symbol"] == symbol]
        if not row.empty:
            ltp = float(row.iloc[0]["ltp"])

    one_m = _read_1m_df(symbol)
    ts_now = _now_ist()
    period_start = _tf_period_start(ts_now, tf)

    if not one_m.empty:
        slice_ = one_m[one_m["datetime"] >= period_start].copy()
    else:
        slice_ = pd.DataFrame()

    if slice_.empty and ltp is None:
        return candles   # nothing fresh to show

    if not slice_.empty:
        o = float(slice_.iloc[0]["open"])
        h = float(slice_["high"].max())
        l = float(slice_["low"].min())
        c = float(slice_.iloc[-1]["close"])
        if ltp is not None:
            c = ltp
            if ltp > h: h = ltp
            if ltp < l: l = ltp
    else:
        # No 1m bars yet — nudge last bar with LTP only.
        last = candles[-1]
        o = last.get("open", ltp)
        h = max(last.get("high", ltp), ltp)
        l = min(last.get("low", ltp), ltp)
        c = ltp

    forming = {"time": int(period_start.timestamp()),
               "open": o, "high": h, "low": l, "close": c}

    last_time = candles[-1].get("time")
    if last_time is None or int(last_time) >= forming["time"]:
        # Last bar already represents this period — replace its OHLC.
        candles[-1].update(forming)
    else:
        # Period has advanced — append the new forming bar.
        candles.append(forming)
    return candles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy_core import (
    add_emas, scan_buy_signals, scan_sell_signals,
    compute_swing_lows, compute_swing_highs, find_swing_sl, find_swing_sh,
)


SWING_WINDOW           = 2     # ± bars used to confirm a swing low / high
SL_MAX_RISK_PCT_DEFAULT = 5.0  # default cap on risk-to-SL when no override


def df_to_pdf_bytes(df: pd.DataFrame, title: str) -> bytes:
    """Render a DataFrame to a landscape A4 PDF and return the raw bytes."""
    import io
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                            leftMargin=10*mm, rightMargin=10*mm,
                            topMargin=10*mm, bottomMargin=10*mm,
                            title=title)
    styles = getSampleStyleSheet()
    h_style = ParagraphStyle("h", parent=styles["Heading2"],
                              fontSize=13, textColor=colors.HexColor("#1b1f2b"),
                              spaceAfter=4)
    sub_style = ParagraphStyle("s", parent=styles["Normal"],
                                fontSize=8, textColor=colors.HexColor("#6b7280"),
                                spaceAfter=8)
    story = [Paragraph(title, h_style),
             Paragraph(f"{len(df)} rows · generated {datetime.now():%Y-%m-%d %H:%M}",
                       sub_style)]
    if df.empty:
        story.append(Paragraph("(no rows)", styles["Normal"]))
    else:
        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M")
        data = [list(d.columns)] + d.astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f8fafc")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)
    doc.build(story)
    return buf.getvalue()


def compute_touch_sl(df, side, ttime):
    """Touch-bar-clamped swing SL.

    BUY  -> min(prior swing low,  touch bar Low)
    SELL -> max(prior swing high, touch bar High)

    This guarantees the SL never sits inside the touch bar's range, which would
    otherwise mean the trade was already stopped out the instant it triggered.
    Returns None if the touch time is missing from the frame.
    """
    if ttime not in df.index:
        return None
    idx = df.index.get_loc(ttime)
    if side == "BUY":
        mask = compute_swing_lows(df, SWING_WINDOW)
        swing_sl = float(find_swing_sl(df, idx, mask))
        touch_lo = float(df["Low"].iloc[idx])
        return min(swing_sl, touch_lo)
    mask = compute_swing_highs(df, SWING_WINDOW)
    swing_sh = float(find_swing_sh(df, idx, mask))
    touch_hi = float(df["High"].iloc[idx])
    return max(swing_sh, touch_hi)


def sl_risk_pct(side, entry, sl):
    """Distance from entry to SL as a positive percent of entry."""
    return ((entry - sl) if side == "BUY" else (sl - entry)) / entry * 100.0


# ─── data paths ───
DATA_ROOT_DEFAULT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "ema_scanner", "data"))
TIMEFRAMES = ["1d", "1h"]


# ─── page ───
st.set_page_config(page_title="Swing-Low Daily Scanner",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ─────────────────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(1100px 600px at 12% -10%, rgba(38,166,154,0.10), transparent 60%),
        radial-gradient(900px 500px at 90% 0%, rgba(66,165,245,0.08), transparent 55%),
        linear-gradient(180deg, #0a0b11 0%, #07080d 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #d8def0;
}
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Hero header ──────────────────────────────────────────────── */
.hero {
    margin: 0 0 1.3rem 0;
    padding: 1.5rem 1.7rem;
    border-radius: 18px;
    background:
        linear-gradient(135deg, rgba(38,166,154,0.10), rgba(66,165,245,0.06) 55%, rgba(250,204,21,0.04)),
        rgba(16,19,28,0.55);
    border: 1px solid rgba(120,144,180,0.14);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(6px);
}
.hero h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.15;
    background: linear-gradient(90deg, #4dd0c0 0%, #60a5fa 55%, #facc15 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.025em;
}
.hero .sub {
    margin-top: 8px;
    color: #9aa3b8;
    font-size: 0.95rem;
    line-height: 1.5;
}
.hero .chips { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
.hero .chip {
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    border: 1px solid rgba(120,144,180,0.22);
    background: rgba(255,255,255,0.03);
    color: #cbd2e0;
}
.hero .chip.accent { color: #4dd0c0; border-color: rgba(38,166,154,0.45); background: rgba(38,166,154,0.10); }
.hero .chip.blue   { color: #93c5fd; border-color: rgba(96,165,250,0.40); background: rgba(96,165,250,0.10); }
.hero .chip.gold   { color: #fde68a; border-color: rgba(250,204,21,0.40); background: rgba(250,204,21,0.08); }

/* ── Typography ──────────────────────────────────────────────── */
h1, h2, h3, h4 { color: #e6ebf4; font-family: 'Inter', sans-serif; }
h2 { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em; }
h3 { font-size: 1.05rem; font-weight: 600; letter-spacing: -0.005em; margin-top: 0.6rem !important; }

/* Treat st.subheader-like h3 as a section title with an accent bar */
[data-testid="stMarkdownContainer"] h3 {
    display: flex; align-items: center; gap: 10px;
}
[data-testid="stMarkdownContainer"] h3::before {
    content: ''; width: 3px; height: 18px; border-radius: 2px;
    background: linear-gradient(180deg, #26a69a, #42a5f5);
    flex-shrink: 0;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0f17 0%, #08090f 100%);
    border-right: 1px solid rgba(120,144,180,0.10);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    color: #cbd2e0;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    color: #93c5fd;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    display: inline-block;
    margin-top: 0.4rem;
}
[data-testid="stSidebar"] hr {
    background: linear-gradient(90deg, transparent, rgba(120,144,180,0.30), transparent);
    margin: 1rem 0 !important;
}

/* ── Inputs ──────────────────────────────────────────────────── */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #131724 !important;
    border: 1px solid rgba(120,144,180,0.20) !important;
    color: #d8def0 !important;
    border-radius: 9px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus {
    border-color: #26a69a !important;
    box-shadow: 0 0 0 3px rgba(38,166,154,0.18) !important;
    outline: none !important;
}
.stSelectbox div[data-baseweb="select"]:hover > div { border-color: rgba(120,144,180,0.40) !important; }

/* Slider */
.stSlider [data-baseweb="slider"] > div > div { background: linear-gradient(90deg, #26a69a, #42a5f5) !important; }
.stSlider [role="slider"] {
    background: #ffffff !important;
    box-shadow: 0 0 0 2px #26a69a, 0 2px 6px rgba(0,0,0,0.4) !important;
}

/* Labels */
.stRadio label, .stCheckbox label, .stSelectbox label,
.stNumberInput label, .stTextInput label, .stDateInput label, .stSlider label {
    color: #a6afc4 !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.005em !important;
}

/* Radio pills (horizontal) */
.stRadio div[role="radiogroup"] {
    gap: 14px;
}
.stRadio div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(120,144,180,0.18);
    padding: 6px 16px;
    border-radius: 10px;
    transition: all 0.12s ease;
}
.stRadio div[role="radiogroup"] > label:hover {
    border-color: rgba(120,144,180,0.40);
    background: rgba(255,255,255,0.05);
}

/* View-mode radio gets even more breathing room */
[data-testid="stRadio"]:has(input[id*="view_mode"]) div[role="radiogroup"] {
    gap: 22px;
}
[data-testid="stRadio"]:has(input[id*="view_mode"]) div[role="radiogroup"] > label {
    padding: 8px 20px;
    font-weight: 500;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #34d399 0%, #06b6d4 55%, #6366f1 100%);
    background-size: 200% 200%;
    background-position: 0% 50%;
    border: none;
    color: #ffffff;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-radius: 12px;
    padding: 0.72rem 1.2rem;
    box-shadow:
        0 6px 22px rgba(52,211,153,0.35),
        0 2px 6px rgba(99,102,241,0.25),
        inset 0 1px 0 rgba(255,255,255,0.20);
    transition: background-position 0.45s ease,
                transform 0.10s ease,
                box-shadow 0.22s ease,
                filter 0.22s ease;
}
.stButton button[kind="primary"]:hover {
    background-position: 100% 50%;
    transform: translateY(-1px);
    box-shadow:
        0 12px 28px rgba(52,211,153,0.52),
        0 4px 10px rgba(99,102,241,0.42),
        inset 0 1px 0 rgba(255,255,255,0.28);
    filter: brightness(1.06);
}
.stButton button[kind="primary"]:active {
    transform: translateY(0);
    filter: brightness(0.97);
}
.stButton button[kind="primary"]:focus { outline: none !important; }

.stDownloadButton button, .stButton button:not([kind="primary"]) {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(120,144,180,0.22);
    color: #d8def0;
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stDownloadButton button:hover, .stButton button:not([kind="primary"]):hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(120,144,180,0.40);
    color: #ffffff;
}

/* ── Metric cards ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.030), rgba(255,255,255,0.008));
    padding: 16px 20px;
    border-radius: 14px;
    border: 1px solid rgba(120,144,180,0.14);
    box-shadow: 0 4px 14px rgba(0,0,0,0.22);
    transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.18s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(120,144,180,0.32);
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(0,0,0,0.30);
}
[data-testid="stMetricLabel"] {
    color: #94a0b8 !important;
    text-transform: uppercase;
    font-size: 0.70rem !important;
    letter-spacing: 0.12em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #e9eef9 !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    font-family: 'JetBrains Mono', 'Inter', monospace !important;
}

/* ── DataFrame card ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(120,144,180,0.14);
    border-radius: 12px;
    overflow: hidden;
    background: rgba(255,255,255,0.012);
}

/* ── Progress bar ────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #26a69a, #42a5f5, #facc15) !important;
}

/* ── HR ──────────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(120,144,180,0.28), transparent) !important;
    margin: 1.4rem 0 !important;
}

/* ── Alert callouts ──────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid rgba(120,144,180,0.18);
    background: rgba(255,255,255,0.025);
}

/* ── Selection / scrollbar polish ────────────────────────────── */
::selection { background: rgba(38,166,154,0.35); color: #ffffff; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0a0c12; }
::-webkit-scrollbar-thumb { background: rgba(120,144,180,0.22); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: rgba(120,144,180,0.40); }

/* ── Footer ──────────────────────────────────────────────────── */
.footer-meta {
    margin-top: 2rem; padding: 0.9rem 0;
    border-top: 1px solid rgba(120,144,180,0.10);
    color: #6a7388; font-size: 0.78rem; text-align: center;
    letter-spacing: 0.04em;
}

/* ── Light AgGrid card (white table on dark page) ────────────── */
.ag-light-card {
    padding: 4px;
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(120,144,180,0.16);
    box-shadow: 0 6px 20px rgba(0,0,0,0.28);
    margin: 6px 0 10px 0;
}

/* ── Pre-scan card title ─────────────────────────────────────── */
.scan-card { margin: 4px 0 -2px 0; }
.scan-card-title {
    color: #93c5fd;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
}

/* Streamlit chrome trim */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { right: 1rem; }

/* ── Mobile readability boost (phones / narrow viewports) ── */
@media (max-width: 768px) {
    html, body, .stApp { font-size: 15px !important; }

    h1 { font-size: 1.35rem !important; line-height: 1.25 !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1.05rem !important; }

    /* Captions — e.g. "Click any row to load that stock in the chart above" */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
    .stCaption, .stCaption * {
        font-size: 14px !important;
        line-height: 1.45 !important;
        color: #d8d8d8 !important;
    }

    /* Streamlit dataframe internals — column headers, cells, filter chips
       (ALL/BUY/SELL), search input + its placeholder ("e.g. RELIANCE (CASH)"),
       and the "Showing N of M" row-count footer.                          */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] * {
        font-size: 14px !important;
    }
    [data-testid="stDataFrame"] input,
    [data-testid="stDataFrame"] input::placeholder {
        font-size: 14px !important;
        color: #d8def0 !important;
        opacity: 1 !important;
    }
    [data-testid="stDataFrame"] [role="option"],
    [data-testid="stDataFrame"] [role="menuitem"],
    [data-testid="stDataFrame"] button {
        font-size: 14px !important;
        color: #e6ebf5 !important;
    }

    /* AgGrid (used for the all-stock table on /scan/) — header + cell text */
    .ag-theme-streamlit, .ag-theme-balham-dark, .ag-theme-alpine-dark,
    .ag-theme-streamlit *, .ag-theme-balham-dark *, .ag-theme-alpine-dark * {
        font-size: 13px !important;
    }
    .ag-header-cell-text { font-size: 13px !important; color: #e6ebf5 !important; }
    .ag-cell, .ag-row { line-height: 1.35 !important; }
    .ag-paging-panel, .ag-paging-panel * {
        font-size: 13px !important; color: #d8def0 !important;
    }

    /* Selectboxes, number inputs, sliders, checkboxes, text inputs */
    .stSelectbox, .stSelectbox *,
    .stNumberInput, .stNumberInput *,
    .stTextInput,  .stTextInput *,
    .stSlider,     .stSlider *,
    .stCheckbox,   .stCheckbox * {
        font-size: 14px !important;
    }

    /* Metrics */
    [data-testid="stMetricLabel"]  { font-size: 13px !important; color: #cfd6e6 !important; }
    [data-testid="stMetricValue"]  { font-size: 22px !important; }

    /* Small / muted spans + progress text */
    small, .small { font-size: 13px !important; color: #c8d0e2 !important; }
    [data-testid="stProgress"] * { font-size: 13px !important; color: #d8def0 !important; }

    /* Chart legend (── Fast trend  ▲ BUY entry  …) */
    .chart-legend, .chart-legend * {
        font-size: 14px !important;
        color: #e6ebf5 !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Swing-Low Daily Scanner</h1>
  <div class="sub">Find tradeable equity setups firing on a chosen day — daily and intraday signals side by side, with ⭐ marking high-conviction overlaps.</div>
  <div class="chips">
    <span class="chip accent">●&nbsp; LIVE scanner</span>
    <span class="chip blue">Daily + intraday</span>
    <span class="chip gold">Swing-based SL</span>
    <span class="chip">click any row → load chart</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── helpers ───
@st.cache_data(ttl=600, show_spinner=False)
def _load_1m_opens_cached(symbol: str, data_root: str, mtime: float):
    """Cached (datetime, open) table for a symbol's 1m CSV. Streamlit-cached
    so successive load_ohlc() calls within a scan reuse the parsed dataframe.
    `mtime` is the file mtime — invalidates the cache when the file changes."""
    path = os.path.join(data_root, "1m", f"{symbol}_historical.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=["datetime", "open"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.dropna(subset=["open"]).sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


def _open_from_first_1m(symbol: str, data_root: str,
                        period_start: pd.Timestamp,
                        period_end: pd.Timestamp):
    """Return the Open of the first 1m candle in [period_start, period_end).
    For a 1d bar this is the 09:15 IST 1-minute candle; for a 1h bar it's
    the hour boundary's 1-minute candle."""
    path = os.path.join(data_root, "1m", f"{symbol}_historical.csv")
    if not os.path.exists(path):
        return None
    df = _load_1m_opens_cached(symbol, data_root, os.path.getmtime(path))
    if df is None or df.empty:
        return None
    sub = df[(df["datetime"] >= period_start) & (df["datetime"] < period_end)]
    if sub.empty:
        return None
    return float(sub.iloc[0]["open"])


@st.cache_data(ttl=600, show_spinner=False)
def load_ohlc(symbol: str, tf: str, data_root: str) -> pd.DataFrame:
    path = os.path.join(data_root, tf, f"{symbol}_historical.csv")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    if df["Open"].isna().any() and tf in ("1d", "1h"):
        period_delta = pd.Timedelta(days=1) if tf == "1d" else pd.Timedelta(hours=1)
        for idx in df.index[df["Open"].isna()]:
            ps = (idx.normalize() + pd.Timedelta(hours=9, minutes=15)
                  if tf == "1d" else idx)
            v = _open_from_first_1m(symbol, data_root, ps, ps + period_delta)
            if v is not None:
                df.at[idx, "Open"] = v
    if df["Open"].isna().any():
        df["Open"] = df["Open"].fillna(df["Close"].shift(1))
    return df.dropna(subset=["High", "Low", "Close"])


def _synth_today_1d_from_1h(symbol: str, data_root: str,
                            target_day: date) -> pd.DataFrame | None:
    """Build a forming 1d candle for `target_day` from today's already-present
    1h bars in data/1h/<sym>_historical.csv.

    The 1d-refresh timer only writes after 15:35 IST, so on intraday scans
    today's 1d bar is missing. This synthesizes one (open = first 1h open,
    high = max, low = min, close = last 1h close, volume = sum) so the 1d
    scan can include today.
    Returns None if no 1h data exists for today.
    """
    try:
        h = load_ohlc(symbol, "1h", data_root)
    except Exception:
        return None
    today_h = h[h.index.normalize() == pd.Timestamp(target_day)]
    if today_h.empty:
        return None
    row = pd.DataFrame({
        "Open":   [float(today_h["Open"].iloc[0])],
        "High":   [float(today_h["High"].max())],
        "Low":    [float(today_h["Low"].min())],
        "Close":  [float(today_h["Close"].iloc[-1])],
        "Volume": [float(today_h["Volume"].sum()) if "Volume" in today_h.columns else 0.0],
    }, index=[pd.Timestamp(target_day)])
    return row


def signals_on_day(symbol: str, tf: str, data_root: str, target_day: date,
                   fast: int, slow: int, compulsory: int,
                   max_sl_pct: float | None = None) -> list[dict]:
    """Return list of touches that fired on `target_day` for one symbol,
    each enriched with SL, current price, P&L%, peak move and tradability."""
    try:
        full = load_ohlc(symbol, tf, data_root)
    except Exception:
        return []
    # If scanning 1d for today before market close, the 1d CSV won't have
    # today's bar yet (the 1d timer fires at 15:35). Synthesize it from the
    # already-fetched 1h bars so today is scannable.
    if tf == "1d":
        _today_ist = _now_ist().date()
        if target_day == _today_ist and pd.Timestamp(target_day) not in full.index:
            synth = _synth_today_1d_from_1h(symbol, data_root, target_day)
            if synth is not None:
                full = pd.concat([full, synth]).sort_index()
    if len(full) < slow + 5:
        return []
    # Scan signals on bars up to and including the target day...
    cutoff = pd.Timestamp(target_day) + pd.Timedelta(days=1)
    df = full[full.index < cutoff]
    if len(df) < slow + 5:
        return []
    df = add_emas(df, fast, slow)
    # ...but use ALL data (including bars after the scan day) to evaluate
    # outcomes — current price, peak, SL hit, etc.
    full_emas = add_emas(full, fast, slow)
    swing_low_mask  = compute_swing_lows(full_emas, SWING_WINDOW)
    swing_high_mask = compute_swing_highs(full_emas, SWING_WINDOW)
    current_close   = float(full["Close"].iloc[-1])

    out = []
    for side, fn in (("BUY", scan_buy_signals), ("SELL", scan_sell_signals)):
        sigs = fn(df, fast, slow, compulsory=compulsory)
        if sigs.empty:
            continue
        sigs = sigs[sigs["Touch Time"].dt.date == target_day]
        for _, r in sigs.iterrows():
            ttime = r["Touch Time"]
            entry = round(float(r["Touch Close"]), 2)
            # Raw swing-based SL using the FULL frame so swings to the right
            # of the touch don't shift its value.
            if ttime in full.index:
                idx = full.index.get_loc(ttime)
                if side == "BUY":
                    swing_sl = float(find_swing_sl(full_emas, idx, swing_low_mask))
                    sl = min(swing_sl, float(full["Low"].iloc[idx]))
                else:
                    swing_sh = float(find_swing_sh(full_emas, idx, swing_high_mask))
                    sl = max(swing_sh, float(full["High"].iloc[idx]))
            else:
                sl = None
            risk_pct = sl_risk_pct(side, entry, sl) if sl is not None else None
            tradable = (max_sl_pct is None) or (risk_pct is not None and risk_pct <= max_sl_pct)
            # SL hit & peak — computed for ALL rows so the table never has
            # blank P&L / Peak cells (NO-TRADE rows show what would have
            # happened too).
            sl_hit_at = None
            peak_pct  = None
            if sl is not None and ttime in full.index:
                post = full.loc[ttime:].iloc[1:]
                if not post.empty:
                    hit = (post["Low"] <= sl) if side == "BUY" else (post["High"] >= sl)
                    if hit.any():
                        sl_hit_at = hit[hit].index[0]
                    eval_post = post.loc[:sl_hit_at] if sl_hit_at is not None else post
                    if not eval_post.empty:
                        if side == "BUY":
                            peak_pct = (float(eval_post["High"].max()) - entry) / entry * 100
                        else:
                            peak_pct = (entry - float(eval_post["Low"].min())) / entry * 100
            # P&L%
            if sl_hit_at is not None and tradable and risk_pct is not None:
                pnl_pct = -risk_pct
            else:
                pnl_pct = ((current_close - entry) if side == "BUY"
                           else (entry - current_close)) / entry * 100
            out.append({
                "Symbol":     symbol,
                "Side":       side,
                "TF":         tf,
                "Touch Time": ttime,
                "Entry":      entry,
                "SL":         round(sl, 2) if sl is not None else None,
                "Risk %":     round(risk_pct, 2) if risk_pct is not None else None,
                "Current":    round(current_close, 2),
                "P&L %":      round(pnl_pct, 2) if pnl_pct is not None else None,
                "Peak %":     round(peak_pct, 2) if peak_pct is not None else None,
                "Status":     ("NO TRADE" if not tradable
                               else ("SL HIT" if sl_hit_at is not None else "OPEN")),
            })
    return out


@st.cache_data(ttl=300)
def latest_available_day(data_root: str = DATA_ROOT_DEFAULT) -> date:
    """Most recent trading date present in the project's 1h CSVs (falls back
    to 1d, then to a sensible hard-coded date). Reads only the file tail so
    it's cheap to call on every rerun."""
    for tf in ("1h", "1d"):
        folder = os.path.join(data_root, tf)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.endswith("_historical.csv")]
        if not files:
            continue
        last = None
        for fname in files[:8]:
            try:
                p = os.path.join(folder, fname)
                with open(p, "rb") as f:
                    f.seek(0, 2); size = f.tell()
                    f.seek(max(0, size - 256))
                    tail = f.read().decode("utf-8", errors="ignore").strip()
                dt = pd.to_datetime(tail.split("\n")[-1].split(",")[-1].strip()).date()
                if last is None or dt > last:
                    last = dt
            except Exception:
                continue
        if last is not None:
            return last
    return date(2026, 5, 14)


def list_symbols(data_root: str, tf: str) -> list[str]:
    folder = os.path.join(data_root, tf)
    if not os.path.isdir(folder):
        return []
    return sorted(f[:-len("_historical.csv")]
                  for f in os.listdir(folder) if f.endswith("_historical.csv"))


def to_ts(t) -> int:
    pt = pd.Timestamp(t)
    if pt.tz is not None:
        pt = pt.tz_localize(None)
    return int(pt.value // 10**9)


# ─── pending jump processor (pre-widget) ───
_pending = st.session_state.pop("_pending_jump", None)
if _pending:
    st.session_state["sel_symbol"] = _pending["symbol"]
    st.session_state["sel_tf"]     = _pending["tf"]
    st.session_state["jumped_to"]  = _pending["symbol"]
    st.session_state["_scan_v"]    = st.session_state.get("_scan_v", 0) + 1
    st.session_state["_chart_v"]   = st.session_state.get("_chart_v", 0) + 1


# ─── fixed defaults (no longer user-tunable) ───
data_root    = DATA_ROOT_DEFAULT
EMA_PARAMS   = {
    "1d": {"fast": 21, "slow": 50, "compulsory": 2},
    "1h": {"fast": 21, "slow": 50, "compulsory": 2},
}
chart_bars   = 120
bar_spacing  = 12
chart_height = 720
show_volume  = True


# ─── pre-scan card ─────────────────────────────────────────────
st.markdown("""
<div class='scan-card'>
  <div class='scan-card-title'>Scan setup</div>
</div>
""", unsafe_allow_html=True)

cc1, cc2, cc3 = st.columns([1.4, 1.6, 1.4])
target_day = cc1.date_input(
    "Entry day", value=latest_available_day(),
    help="Defaults to the latest trading day present in the data. Pick any "
         "earlier date to scan that day; the 1h hour list updates accordingly.",
)
# Persistent classifier badge under the date picker so the user immediately
# sees if they've picked a non-trading day, even before pressing Run scan.
_kind_badge = _classify_date(target_day)
if _kind_badge == "holiday":
    cc1.error(f"🏖️  **NSE Holiday** — markets were closed on {target_day}.")
elif _kind_badge == "weekend":
    cc1.error(f"📅  **{target_day.strftime('%A')}** — markets don't trade on weekends.")
elif _kind_badge == "future":
    cc1.warning(f"🚧  **{target_day}** — no data fetched yet (auto-populates after 15:35 IST).")
max_sl_pct = cc2.number_input(
    "Max SL tolerance (%)",
    min_value=0.5, max_value=20.0,
    value=None, step=0.5,
    placeholder="e.g. 5.0",
    key="max_sl_pct",
    help="Leave empty to use the raw swing low/high. Set a value and any "
         "trade whose SL would exceed it is flagged 'NO TRADE'.",
)
cc3.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
# Button label flips to "Re-run scan" once results exist for the selected day,
# so the user can tell apart "first scan" vs "re-execute".
_has_results_for_day = (st.session_state.get("scan_day") == target_day
                        and st.session_state.get("scan_1d") is not None)
_btn_label = "🔁 Re-run scan" if _has_results_for_day else "⚡ Run scan"
run = cc3.button(_btn_label, type="primary", use_container_width=True,
                 key="run_scan_btn",
                 help="Scan the selected date — instantly served if it was "
                      "already computed by the nightly job, otherwise ~4 min.")

# If the user just picked a different date, clear stale results so they don't
# see yesterday's table under today's heading. Wait for them to press the
# button before showing anything for the new date.
if (st.session_state.get("scan_day") is not None
        and st.session_state.get("scan_day") != target_day
        and not run):
    for k in ("scan_1d", "scan_1h", "scan_day", "scan_max_sl_pct",
              "_cached_loaded_day", "_cached_loaded_at"):
        st.session_state.pop(k, None)

# When the button is pressed AND a precomputed result for that date exists,
# load it instantly under the hood — no spinner, no scan, no banner. The
# user just sees results appear.
#
# Today's cache is refreshed every hour by investeq-scan-cache.timer
# (10:20, 11:20, ... 15:20 IST + final EOD pass at 15:40 IST). So today
# is served from cache too — at worst it lags the current 1h bar by a few
# minutes. Press Re-run scan to force a fresh live compute if needed.
if run:
    _cd1, _cd1h, _saved_at = _load_scan_cache_for_day(target_day)
    if _cd1 is not None and _cd1h is not None:
        st.session_state["scan_1d"] = _cd1
        st.session_state["scan_1h"] = _cd1h
        st.session_state["scan_day"] = target_day
        st.session_state["scan_params"] = EMA_PARAMS
        st.session_state["scan_max_sl_pct"] = (float(max_sl_pct)
                                               if max_sl_pct is not None else None)
        st.session_state["_precomputed_hit"] = True
        run = False  # short-circuit: skip the full scan below

_SCAN_STATE_KEYS = ("_scan_thread", "_scan_progress", "_scan_stop_event",
                    "_scan_target_day", "_scan_max_sl_pct")


def _reset_scan_state():
    """Wipe every key tied to either in-flight progress or finished results,
    so the page returns to the empty "Press Run scan" stage."""
    for k in _SCAN_STATE_KEYS + ("scan_1d", "scan_1h", "scan_day",
                                  "scan_params", "scan_max_sl_pct",
                                  "_cached_loaded_day", "_cached_loaded_at",
                                  "_precomputed_hit"):
        st.session_state.pop(k, None)


def _scan_in_background(jobs, target_day, data_root_local, max_sl_pct_local,
                        progress, stop_event):
    """Run the full universe scan off-thread, polling `stop_event` between
    completed futures so a user-press of Stop aborts quickly."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = {tf: [] for tf in TIMEFRAMES}
    progress["total"] = len(jobs)
    progress["done"]  = 0
    progress["n_err"] = 0
    progress["last_err"] = None
    try:
        with ThreadPoolExecutor(max_workers=16) as ex:
            future_to_tf = {
                ex.submit(signals_on_day, sym, tf, data_root_local, target_day,
                          p["fast"], p["slow"], p["compulsory"],
                          max_sl_pct=max_sl_pct_local): (tf, sym)
                for tf, sym, p in jobs
            }
            for fut in as_completed(future_to_tf):
                if stop_event.is_set():
                    for f in future_to_tf:
                        f.cancel()
                    progress["stopped"] = True
                    return
                tf, sym = future_to_tf[fut]
                try:
                    rows[tf].extend(fut.result())
                except Exception as e:
                    progress["n_err"] += 1
                    progress["last_err"] = f"{tf}/{sym}: {type(e).__name__}: {e}"
                progress["done"] += 1
                progress["current"] = f"{tf}  {sym}"
    except Exception as e:
        progress["fatal"] = f"{type(e).__name__}: {e}"
        return
    progress["rows_1d"] = rows["1d"]
    progress["rows_1h"] = rows["1h"]
    progress["finished"] = True


# ── (A) kick off a background scan when Run scan is pressed ───────────────
if run:
    if not os.path.isdir(data_root):
        st.error(f"Data root not found: {data_root}")
        st.stop()
    # Build job list on the main thread (cheap), then hand off to the worker.
    jobs = []
    for tf in TIMEFRAMES:
        p = EMA_PARAMS[tf]
        for sym in list_symbols(data_root, tf):
            jobs.append((tf, sym, p))
    _msl = float(max_sl_pct) if max_sl_pct is not None else None
    progress_dict: dict = {}
    stop_event = threading.Event()
    th = threading.Thread(
        target=_scan_in_background,
        args=(jobs, target_day, data_root, _msl, progress_dict, stop_event),
        daemon=True,
    )
    th.start()
    st.session_state["_scan_thread"]      = th
    st.session_state["_scan_progress"]    = progress_dict
    st.session_state["_scan_stop_event"]  = stop_event
    st.session_state["_scan_target_day"]  = target_day
    st.session_state["_scan_max_sl_pct"]  = _msl
    st.rerun()

# ── (B) while the thread is alive: show progress + Stop button ────────────
_th = st.session_state.get("_scan_thread")
if _th is not None and _th.is_alive():
    progress_dict = st.session_state.get("_scan_progress", {})
    done = int(progress_dict.get("done", 0))
    total = int(progress_dict.get("total", 1)) or 1
    cur = progress_dict.get("current", "Starting...")
    sp_col, btn_col = st.columns([6, 1])
    sp_col.progress(min(done / total, 1.0),
                    text=f"[{done}/{total}] {cur}")
    if btn_col.button("🛑 Stop", type="secondary", use_container_width=True,
                       key="stop_scan_btn"):
        st.session_state["_scan_stop_event"].set()
        _reset_scan_state()
        st.rerun()
    # Drive a 1-second rerun cadence so progress updates and the Stop click
    # is picked up promptly.
    if st_autorefresh is not None:
        st_autorefresh(interval=1000, key="scan_progress_tick")
    st.stop()

# ── (C) thread finished: collect results, save, clean up ──────────────────
if _th is not None and not _th.is_alive():
    progress_dict = st.session_state.get("_scan_progress", {})
    if progress_dict.get("stopped"):
        _reset_scan_state()
        st.rerun()
    elif progress_dict.get("fatal"):
        st.error(f"Scan failed: {progress_dict['fatal']}")
        _reset_scan_state()
    elif progress_dict.get("finished"):
        _scan_cols = ["Symbol", "Side", "TF", "Touch Time",
                      "Entry", "SL", "Risk %", "Current", "P&L %", "Peak %", "Status"]
        _df1d = pd.DataFrame(progress_dict.get("rows_1d", []), columns=_scan_cols)
        _df1h = pd.DataFrame(progress_dict.get("rows_1h", []), columns=_scan_cols)
        _td   = st.session_state.get("_scan_target_day")
        _mpl  = st.session_state.get("_scan_max_sl_pct")
        st.session_state["scan_1d"] = _df1d
        st.session_state["scan_1h"] = _df1h
        st.session_state["scan_day"] = _td
        st.session_state["scan_params"] = EMA_PARAMS
        st.session_state["scan_max_sl_pct"] = _mpl
        # Persist to disk so a WS reconnect / browser refresh / process
        # restart doesn't blow away the user's scan results.
        _save_scan_cache(_td, _mpl, _df1d, _df1h)
        if progress_dict.get("n_err"):
            st.warning(f"{progress_dict['n_err']} symbols errored during scan. "
                       f"Last error → {progress_dict.get('last_err')}")
        # Done — clear only the thread-state keys; keep scan_1d/scan_1h.
        for k in _SCAN_STATE_KEYS:
            st.session_state.pop(k, None)


# ─── results ───
df1d = st.session_state.get("scan_1d")
df1h = st.session_state.get("scan_1h")

if df1d is None or df1h is None:
    # Nothing loaded yet — explain *why* (holiday / weekend / future) if the
    # date is unscannable, otherwise just prompt the user to press the button.
    _kind = _classify_date(target_day)

    if _kind == "holiday":
        st.error(
            f"🏖️ **{target_day} — NSE HOLIDAY.** Markets were closed on this "
            f"date, so there's nothing to scan. Pick a trading day."
        )
    elif _kind == "weekend":
        st.error(
            f"📅 **{target_day} — Weekend ({target_day.strftime('%A')}).** "
            f"NSE doesn't trade on weekends. Pick a weekday."
        )
    elif _kind == "future":
        st.error(
            f"🚧 **{target_day} — No data yet.** The daily candle fetcher "
            f"hasn't reached this date. It'll auto-populate after market "
            f"close at 15:35 IST."
        )
    else:
        st.info(f"Press **⚡ Run scan** to scan **{target_day}**.")
    st.stop()

scan_day = st.session_state.get("scan_day", target_day)
m1, m2, m3 = st.columns(3)
m1.metric("1d entries", len(df1d))
m2.metric("1h entries", len(df1h))

# Intersection by (Symbol, Side). Pull the per-stock 1d outcome stats so
# the Both-view table is just as colourful as the per-tf tables.
if not df1d.empty and not df1h.empty:
    base_1d = (df1d.sort_values("Touch Time")
                   .drop_duplicates(subset=["Symbol", "Side"], keep="last"))
    keys_1h = df1h[["Symbol", "Side"]].drop_duplicates()
    inter = (base_1d.merge(keys_1h, on=["Symbol", "Side"])
                    .sort_values(["Side", "Symbol"])
                    .reset_index(drop=True))
else:
    inter = pd.DataFrame(columns=df1d.columns if "Symbol" in df1d.columns
                         else ["Symbol", "Side"])
m3.metric("⭐ Both timeframes", len(inter))

# ─── view selector ─────────────────────────────────────────────
VIEW_OPTIONS = ["📅 1d only", "⏰ 1h", "⭐ 1d + 1h (both)"]
if st.session_state.get("view_mode") not in VIEW_OPTIONS:
    st.session_state.pop("view_mode", None)
view = st.radio(
    "View",
    options=VIEW_OPTIONS,
    index=2,
    horizontal=True,
    key="view_mode",
    label_visibility="collapsed",
)
show_1d_only = view == VIEW_OPTIONS[0]
show_1h      = view == VIEW_OPTIONS[1]
show_both    = view == VIEW_OPTIONS[2]

# 1h candle picker (only when the 1h view is active). The dropdown starts
# with "1h (all)" — equivalent to the old "1h only" view — and then lists
# every 1h candle that's closed on the scan day: all 7 for a past day,
# only the ones whose effective close has already passed for today (15:30
# caps the last bucket), and none for a future day.
selected_hour  = None     # None = "1h (all)"
show_1h_all    = False
show_1h_hourly = False
if show_1h:
    _ALL_HOURS_1H = [f"{h:02d}:15" for h in range(9, 16)]
    _today = pd.Timestamp.now().date()
    if scan_day > _today:
        _hour_views = []
    elif scan_day < _today:
        _hour_views = _ALL_HOURS_1H[:]
    else:
        _now = pd.Timestamp.now()
        _session_end = pd.Timestamp(f"{scan_day} 15:30:00")
        _hour_views = [h for h in _ALL_HOURS_1H
                       if min(pd.Timestamp(f"{scan_day} {h}") + pd.Timedelta(hours=1),
                              _session_end) <= _now]
    _ALL_LABEL = "1h (all)"
    _hour_options = [_ALL_LABEL] + _hour_views
    if st.session_state.get("hour_pick") not in _hour_options:
        st.session_state["hour_pick"] = _ALL_LABEL
    _hc1, _hc2 = st.columns([1.5, 4.5])
    pick = _hc1.selectbox(
        "1h candle",
        options=_hour_options,
        key="hour_pick",
        help=f"'1h (all)' keeps every 1h touch on {scan_day}.  "
             f"Pick a specific 09:15-aligned candle to filter.",
    )
    if pick == _ALL_LABEL:
        show_1h_all = True
    else:
        show_1h_hourly = True
        selected_hour = pick

# ─── chart (placed BEFORE the per-tf tables so it's immediately visible) ───
st.markdown("---")
st.subheader("Stock chart")

# Pick a sensible default symbol if none yet:
#   prefer a starred stock (high conviction) → then 1d → then 1h.
if "sel_symbol" not in st.session_state:
    if not inter.empty:
        st.session_state["sel_symbol"] = inter.iloc[0]["Symbol"]
        st.session_state["sel_tf"] = "1d"
    elif not df1d.empty:
        st.session_state["sel_symbol"] = df1d.iloc[0]["Symbol"]
        st.session_state["sel_tf"] = "1d"
    elif not df1h.empty:
        st.session_state["sel_symbol"] = df1h.iloc[0]["Symbol"]
        st.session_state["sel_tf"] = "1h"
    else:
        st.info("No stocks qualified — nothing to chart.")
        st.stop()

# Symbol & timeframe pickers — scoped to the active view so the chart
# only offers stocks from the same list that's shown in the table above.
_syms_1d = set(df1d["Symbol"]) if "Symbol" in df1d.columns else set()
_syms_1h = set(df1h["Symbol"]) if "Symbol" in df1h.columns else set()

if show_1d_only:
    view_syms = sorted(_syms_1d)
    view_default_tf = "1d"
elif show_1h_all:
    view_syms = sorted(_syms_1h)
    view_default_tf = "1h"
elif show_1h_hourly and selected_hour is not None and not df1h.empty:
    _hour_mask = pd.to_datetime(df1h["Touch Time"]).dt.strftime("%H:%M") == selected_hour
    view_syms = sorted(set(df1h[_hour_mask]["Symbol"]))
    view_default_tf = "1h"
elif show_both:
    view_syms = sorted(set(inter["Symbol"])) if not inter.empty else []
    view_default_tf = "1d"
else:
    view_syms = []
    view_default_tf = "1d"

if not view_syms:
    st.info("No qualifying symbols for this view.")
    st.stop()

# Honour a row click (it may target a symbol outside the current view — keep it).
_pending_sym = st.session_state.get("sel_symbol")
if _pending_sym not in view_syms:
    # Pending jumps from a row click are already applied to sel_symbol earlier;
    # if it's not in the active view's list, fall back to that view's first.
    st.session_state["sel_symbol"] = view_syms[0]

# Snap the chart timeframe to the view's natural one when the user
# switches views, but still allow them to override via the picker.
_last_view = st.session_state.get("_last_view")
if _last_view != view:
    st.session_state["sel_tf"] = view_default_tf
    st.session_state["_last_view"] = view

c1, c2, c3 = st.columns([3, 1, 1])
symbol = c1.selectbox("Symbol", view_syms, key="sel_symbol")
chart_tf = c2.selectbox("Timeframe", TIMEFRAMES, key="sel_tf")
# Live mode: when off the page renders once at scan-time values (fast).
# When on, the chart for the currently-selected symbol re-fetches every
# 1 minute. Tables stay at scan-time values (no per-symbol LTP polling).
live_mode = c3.checkbox(
    "🔴 Live", value=False, key="live_mode",
    help="When on, the chart auto-refreshes every 1 minute and the LTP "
         "worker fetches fresh 1m bars only for the symbol shown above. "
         "Tables stay static — leave off for a pure historical view.",
)

if st.session_state.get("jumped_to") == symbol:
    st.success(f"🎯 Jumped from table — {symbol} ({chart_tf})")
    st.session_state.pop("jumped_to", None)

# Load + crop
try:
    df = load_ohlc(symbol, chart_tf, data_root)
except Exception as e:
    st.error(f"Could not load {symbol} on {chart_tf}: {e}")
    st.stop()
if df.empty:
    st.warning("No bars available.")
    st.stop()
# Pull the per-TF params used during the scan so the chart matches exactly
_params = st.session_state.get("scan_params", {
    "1d": {"fast": 21, "slow": 50, "compulsory": 2},
    "1h": {"fast": 21, "slow": 50, "compulsory": 2},
})
_p = _params.get(chart_tf, {"fast": 21, "slow": 50, "compulsory": 2})
fast_l, slow_l, compu_l = _p["fast"], _p["slow"], _p["compulsory"]

df = add_emas(df, fast_l, slow_l)

# Find touches that fired on scan_day for THIS stock+tf
touches_on_day = []
for side, fn in (("BUY", scan_buy_signals), ("SELL", scan_sell_signals)):
    sigs = fn(df, fast_l, slow_l, compulsory=compu_l)
    if sigs.empty:
        continue
    s_day = sigs[sigs["Touch Time"].dt.date == scan_day]
    for _, r in s_day.iterrows():
        touches_on_day.append((side, r["Touch Time"], float(r["Touch Close"])))

# Anchor chart at the first touch on the scan day (showing `chart_bars`
# bars BEFORE it + every bar from then up to the latest available data).
# When the scan day is in the past, this lets the user see how the trade
# played out through "now". When scan_day is today, the touch is near the
# right edge and the behaviour collapses to the old "last N bars" view.
touch_positions = []
for side, ttime, _ in touches_on_day:
    if ttime in df.index:
        touch_positions.append(df.index.get_loc(ttime))
if touch_positions:
    first_touch_pos = min(touch_positions)
    start_pos = max(0, first_touch_pos - int(chart_bars))
    df_chart = df.iloc[start_pos:]
else:
    df_chart = df.iloc[-int(chart_bars):]
candles, ef_data, es_data, vol_data, t_arr = [], [], [], [], []
ef_col = f"EMA{fast_l}"; es_col = f"EMA{slow_l}"
for i in range(len(df_chart)):
    o, h, l, c = (df_chart["Open"].iloc[i], df_chart["High"].iloc[i],
                  df_chart["Low"].iloc[i], df_chart["Close"].iloc[i])
    ef, es = df_chart[ef_col].iloc[i], df_chart[es_col].iloc[i]
    if any(pd.isna(v) for v in (o, h, l, c, ef, es)):
        continue
    t = to_ts(df_chart.index[i])
    t_arr.append(t)
    candles.append({"time": t, "open": float(o), "high": float(h),
                    "low": float(l), "close": float(c)})
    ef_data.append({"time": t, "value": float(ef)})
    es_data.append({"time": t, "value": float(es)})
    if show_volume and "Volume" in df_chart.columns:
        v = df_chart["Volume"].iloc[i]
        if not pd.isna(v):
            vol_data.append({
                "time": t, "value": float(v),
                "color": ("rgba(38,166,154,0.55)" if c >= o
                          else "rgba(239,83,80,0.55)"),
            })
t_min = t_arr[0] if t_arr else None
t_max = t_arr[-1] if t_arr else None

# Markers for the touches on the scan day
markers = []
for side, ttime, close in touches_on_day:
    mt = to_ts(ttime)
    if t_min is None or not (t_min <= mt <= t_max):
        continue
    is_short = side == "SELL"
    markers.append({
        "time": mt,
        "position": "aboveBar" if is_short else "belowBar",
        "color": "#ff1744" if is_short else "#00e676",
        "shape": "arrowDown" if is_short else "arrowUp",
        "text": f"⭐ {side} {close:.2f}",
        "size": 3,
    })

# Per-touch stop-loss: last swing low (BUY) / swing high (SELL) at or
# before the touch, clamped so risk-to-SL stays within SL_MAX_RISK_PCT.
# Also detect whether the SL was hit later (Low <= sl for BUY,
# High >= sl for SELL) — the line stops being drawn at the hit bar.
_max_sl_pct_for_chart = st.session_state.get("scan_max_sl_pct", None)
# sl_levels rows: (side, ttime, sl_price, sl_hit_at_or_None, tradable, risk_pct)
# Non-tradable rows are those whose raw swing SL exceeds the user-set
# Max SL tolerance — the chart skips their SL line and the banner shows
# a "no trade" notice instead of the usual P&L info.
sl_levels = []
for side, ttime, close in touches_on_day:
    sl = compute_touch_sl(df, side, ttime)
    if sl is None:
        continue
    risk = sl_risk_pct(side, close, sl)
    tradable = (_max_sl_pct_for_chart is None) or (risk <= _max_sl_pct_for_chart)
    hit_at = None
    if tradable:
        # Skip the touch bar itself — SL is *set* at that bar, so prices
        # already printed inside it can't count as "hitting" it.
        post = df.loc[ttime:].iloc[1:] if ttime in df.index else df.iloc[0:0]
        if not post.empty:
            hit_mask = (post["Low"] <= sl) if side == "BUY" else (post["High"] >= sl)
            if hit_mask.any():
                hit_at = hit_mask[hit_mask].index[0]
    sl_levels.append((side, ttime, sl, hit_at, tradable, risk))

sl_series_list = []
for side, ttime, sl, hit_at, tradable, _risk in sl_levels:
    if not tradable:
        continue   # SL > tolerance → no trade, nothing to draw
    touch_ts = to_ts(ttime)
    end_ts   = to_ts(hit_at) if hit_at is not None else (t_arr[-1] if t_arr else None)
    if end_ts is None:
        continue
    sl_data = [{"time": t, "value": sl} for t in t_arr if touch_ts <= t <= end_ts]
    if not sl_data:
        continue
    sl_series_list.append({
        "type": "Line", "data": sl_data,
        "options": {
            "color": "#ff5252",
            "lineWidth": 1,
            "lineStyle": 2,   # dashed
            "title": f"SL · {side}" + (" · HIT" if hit_at is not None else ""),
            "priceLineVisible": False,
            "lastValueVisible": True,
            "crosshairMarkerVisible": False,
        },
    })

# Add a red ✖ marker at the SL-hit bar so the chart shows the exit point.
for side, ttime, sl, hit_at, tradable, _risk in sl_levels:
    if not tradable or hit_at is None:
        continue
    mt = to_ts(hit_at)
    if t_min is None or not (t_min <= mt <= t_max):
        continue
    markers.append({
        "time": mt,
        "position": "aboveBar" if side == "BUY" else "belowBar",
        "color": "#ff5252",
        "shape": "circle",
        "text": f"SL HIT {sl:.2f}",
        "size": 2,
    })

# Header line
n_t = len(touches_on_day)
if n_t == 0:
    st.warning(f"{symbol} on {chart_tf}: no touch fired on {scan_day}. "
               "(It may be in the other timeframe's list only.)")
else:
    summary = ", ".join(f"{s} @ {t:%H:%M} = {c:.2f}" if chart_tf == "1h"
                        else f"{s} @ {c:.2f}" for s, t, c in touches_on_day)
    st.caption(f"**{symbol} {chart_tf}** — {n_t} touch{'es' if n_t > 1 else ''} on {scan_day}: {summary}")

# Corner P&L banner — entry vs current, SL, % in favour / against.
# Anchored to the top of the chart card so it visually sits at the chart's
# top-left corner. Uses the latest touch when multiple fired on the day.
if touches_on_day:
    _latest = max(touches_on_day, key=lambda t: t[1])
    _side, _ttime, _entry = _latest
    _current = float(df["Close"].iloc[-1])
    _now_ts  = df.index[-1]
    # Pull the matching sl_levels row (it has sl, hit_at, tradable, risk).
    _row = next(((s, t, sv, hit, ok, rk) for (s, t, sv, hit, ok, rk) in sl_levels
                 if s == _side and t == _ttime), None)
    if _row is not None:
        _sl, _sl_hit_at, _tradable, _risk_pct = _row[2], _row[3], _row[4], _row[5]
    else:
        _sl = compute_touch_sl(df, _side, _ttime)
        _sl_hit_at, _tradable = None, True
        _risk_pct = sl_risk_pct(_side, _entry, _sl) if _sl is not None else None

    _ttime_lbl = (_ttime.strftime("%Y-%m-%d %H:%M") if chart_tf == "1h"
                  else _ttime.strftime("%Y-%m-%d"))

    if not _tradable:
        st.markdown(f"""
        <div style='display:flex; flex-wrap:wrap; gap:14px; align-items:center;
                    padding:8px 14px; margin:-2px 0 6px 0;
                    background: rgba(255,82,82,0.07);
                    border: 1px solid rgba(255,82,82,0.45);
                    border-radius: 10px; font-size: 0.85rem;'>
          <span><b style='color:#ff5252;'>{_side}</b>
                @ <b>{_entry:.2f}</b>
                <span style='color:#8a93a8;'>({_ttime_lbl})</span></span>
          <span>SL: <b>{_sl:.2f}</b>
                <span style='color:#ff5252;'>({_risk_pct:.2f}% &gt; {_max_sl_pct_for_chart}% tolerance)</span></span>
          <span style='color:#ff5252; font-weight:700;'>NO TRADE</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        _post = df.loc[_ttime:] if _ttime in df.index else df.iloc[0:0]
        _post_for_peak = _post.loc[:_sl_hit_at] if _sl_hit_at is not None else _post
        if not _post_for_peak.empty:
            if _side == "BUY":
                _peak_price = float(_post_for_peak["High"].max())
                _peak_at    = _post_for_peak["High"].idxmax()
                _peak_pct   = (_peak_price - _entry) / _entry * 100
            else:
                _peak_price = float(_post_for_peak["Low"].min())
                _peak_at    = _post_for_peak["Low"].idxmin()
                _peak_pct   = (_entry - _peak_price) / _entry * 100
        else:
            _peak_price = _peak_at = _peak_pct = None

        if _sl_hit_at is not None and _risk_pct is not None:
            _pnl_pct = -_risk_pct
            _label   = "SL HIT"
            _accent  = "#ff5252"
            _sign    = ""
        else:
            if _side == "BUY":
                _pnl_pct = (_current - _entry) / _entry * 100
            else:
                _pnl_pct = (_entry - _current) / _entry * 100
            _accent = "#4dd0c0" if _pnl_pct >= 0 else "#ff5252"
            _sign   = "+" if _pnl_pct >= 0 else ""
            _label  = "in favour" if _pnl_pct >= 0 else "against"

        _peak_at_lbl = (_peak_at.strftime("%Y-%m-%d %H:%M") if (chart_tf == "1h" and _peak_at is not None)
                        else _peak_at.strftime("%Y-%m-%d") if _peak_at is not None else "")
        _peak_chunk = (f"<span>Peak in favour: <b style='color:#4dd0c0;'>{_peak_price:.2f}</b> "
                       f"<span style='color:#4dd0c0;'>(+{_peak_pct:.2f}% MFE)</span> "
                       f"<span style='color:#8a93a8;'>@ {_peak_at_lbl}</span></span>"
                       if _peak_price is not None else "")
        _sl_chunk = (f"<span>SL: <b style='color:#ff5252;'>{_sl:.2f}</b> "
                     f"<span style='color:#8a93a8;'>({_risk_pct:.2f}% risk)</span></span>"
                     if _sl is not None else "")

        if _sl_hit_at is not None:
            _hit_lbl = (_sl_hit_at.strftime("%Y-%m-%d %H:%M") if chart_tf == "1h"
                        else _sl_hit_at.strftime("%Y-%m-%d"))
            _current_chunk = (f"<span><b style='color:#ff5252;'>SL HIT</b> "
                              f"@ <b>{_sl:.2f}</b> "
                              f"<span style='color:#8a93a8;'>({_hit_lbl})</span></span>")
        else:
            _current_chunk = (f"<span>Current: <b>{_current:.2f}</b> "
                              f"<span style='color:#8a93a8;'>({_now_ts:%Y-%m-%d %H:%M})</span></span>")

        st.markdown(f"""
        <div style='display:flex; flex-wrap:wrap; gap:14px;
                    padding:8px 14px; margin:-2px 0 6px 0;
                    background: rgba(255,255,255,0.025);
                    border: 1px solid rgba(120,144,180,0.18);
                    border-radius: 10px; font-size: 0.85rem;'>
          <span><b style='color:{_accent};'>{_side}</b>
                entry @ <b>{_entry:.2f}</b>
                <span style='color:#8a93a8;'>({_ttime_lbl})</span></span>
          {_sl_chunk}
          {_current_chunk}
          <span style='color:{_accent}; font-weight:600;'>
                {_sign}{_pnl_pct:.2f}% {_label}</span>
          {_peak_chunk}
        </div>
        """, unsafe_allow_html=True)

# Build chart
chart_options = {
    "height": int(chart_height),
    "layout": {"background": {"type": "solid", "color": "#0e0e10"},
               "textColor": "#d1d4dc"},
    "grid": {"vertLines": {"color": "rgba(42,46,57,0.45)"},
             "horzLines": {"color": "rgba(42,46,57,0.45)"}},
    "crosshair": {"mode": 0},
    "rightPriceScale": {
        "borderColor": "rgba(197,203,206,0.3)",
        "scaleMargins": ({"top": 0.05, "bottom": 0.28} if vol_data
                         else {"top": 0.05, "bottom": 0.08}),
    },
    "timeScale": {
        "borderColor": "rgba(197,203,206,0.3)",
        "timeVisible": chart_tf == "1h",
        "secondsVisible": False,
        "rightOffset": 12, "barSpacing": int(bar_spacing),
    },
    "watermark": {
        "visible": True, "fontSize": 44,
        "color": "rgba(180,180,180,0.07)",
        "text": f"{symbol} · {chart_tf}  ·  EMA{fast_l}/{slow_l}",
        "horzAlign": "center", "vertAlign": "center",
    },
}

series = [
    {"type": "Candlestick", "data": candles,
     "options": {"upColor": "#26a69a", "downColor": "#ef5350",
                 "borderVisible": False,
                 "wickUpColor": "#26a69a", "wickDownColor": "#ef5350"},
     "markers": markers},
    {"type": "Line", "data": ef_data,
     "options": {"color": "#ff9800", "lineWidth": 2, "title": f"EMA{fast_l}",
                 "priceLineVisible": False, "lastValueVisible": True,
                 "crosshairMarkerVisible": False}},
    {"type": "Line", "data": es_data,
     "options": {"color": "#42a5f5", "lineWidth": 2, "title": f"EMA{slow_l}",
                 "priceLineVisible": False, "lastValueVisible": True,
                 "crosshairMarkerVisible": False}},
]
for _sl_series in sl_series_list:
    series.append(_sl_series)

if vol_data:
    series.append({
        "type": "Histogram", "data": vol_data,
        "options": {"priceFormat": {"type": "volume"}, "priceScaleId": "",
                    "priceLineVisible": False, "lastValueVisible": False},
        "priceScale": {"scaleMargins": {"top": 0.78, "bottom": 0.0}},
    })

_chart_v = st.session_state.get("_chart_v", 0)

# Live mode: during market hours, focus the LTP worker exclusively on the
# chart symbol and refresh the page once a minute so a freshly-fetched 1m
# bar is picked up as soon as the worker writes it. _overlay_live_ltp also
# rebuilds the forming bar from the in-flight 1m slice + latest LTP so the
# rightmost candle reflects right-now even between worker fetches.
if live_mode and _market_open_ist():
    _register_watch([symbol])
    candles = _overlay_live_ltp(candles, symbol, chart_tf)
    if st_autorefresh is not None:
        # 60s = the natural 1m bar cadence. The worker fetches a new 1m
        # bar for `symbol` every ONE_MIN_INTERVAL_SEC (=60s) and the UI
        # picks it up on its next rerun.
        st_autorefresh(interval=60_000, key=f"chart_tick_{symbol}_{chart_tf}")

renderLightweightCharts(
    [{"chart": chart_options, "series": series}],
    key=(f"chart_{symbol}_{chart_tf}_{int(chart_bars)}_{int(bar_spacing)}"
         f"_{len(markers)}_{len(sl_series_list)}_{len(candles)}_v{_chart_v}"),
)

st.markdown(
    "<div class='chart-legend' style='display:flex; gap:18px; flex-wrap:wrap; "
    "font-size:13px; color:#d0d6e6; padding:4px 2px 10px 2px;'>"
    f"<span><span style='color:#ff9800;'>━</span> Fast trend line</span>"
    f"<span><span style='color:#42a5f5;'>━</span> Slow trend line</span>"
    "<span><span style='color:#ff5252;'>┄</span> Stop loss</span>"
    "<span><span style='color:#00e676;'>▲</span> BUY entry</span>"
    "<span><span style='color:#ff1744;'>▼</span> SELL entry</span>"
    "</div>", unsafe_allow_html=True,
)


# ─── per-tf tables (clickable) — placed BELOW the chart ───
st.markdown("---")
if show_1d_only or show_1h_all or show_1h_hourly or show_both:
    st.caption("👉 Click any row to load that stock in the chart above.")

_AG_SIDE_STYLE = JsCode("""
function(p){
  if(p.value==='BUY')  return {color:'#15803d', fontWeight:'700', backgroundColor:'rgba(22,163,74,0.10)'};
  if(p.value==='SELL') return {color:'#b91c1c', fontWeight:'700', backgroundColor:'rgba(220,38,38,0.10)'};
  return null;
}""")
_AG_PNL_STYLE = JsCode("""
function(p){
  if(p.value > 0) return {color:'#15803d', fontWeight:'700', backgroundColor:'rgba(22,163,74,0.12)'};
  if(p.value < 0) return {color:'#b91c1c', fontWeight:'700', backgroundColor:'rgba(220,38,38,0.10)'};
  return null;
}""")
_AG_STATUS_STYLE = JsCode("""
function(p){
  if(p.value==='NO TRADE') return {color:'#ffffff', fontWeight:'700', backgroundColor:'#dc2626'};
  if(p.value==='SL HIT')   return {color:'#92400e', fontWeight:'700', backgroundColor:'#fde68a'};
  if(p.value==='OPEN')     return {color:'#1d4ed8', fontWeight:'600', backgroundColor:'rgba(37,99,235,0.10)'};
  return null;
}""")
_AG_PEAK_STYLE = JsCode("""
function(p){
  if(p.value > 0) return {color:'#15803d', fontWeight:'600'};
  return null;
}""")
_AG_RISK_STYLE = JsCode("""
function(p){
  if(p.value !== null && p.value !== undefined) return {color:'#374151'};
  return null;
}""")
_AG_TICK_STYLE = JsCode("""
function(p){
  if(p.value === '✓') return {color:'#ca8a04', fontWeight:'700', textAlign:'center', backgroundColor:'rgba(250,204,21,0.20)'};
  return {textAlign:'center'};
}""")


def aggrid_pick(df: pd.DataFrame, key: str, jump_tf: str, height: int = 380):
    """Render `df` as a colourful AgGrid with full-row click selection.
    On selection, sets `_pending_jump` and reruns so the chart loads."""
    if df.empty:
        st.info("No rows match these filters.")
        return
    # Hide internal alert-gating flags from the UI grid.
    df = df.drop(columns=[c for c in ("LowBelowBoth", "HighAboveBoth")
                          if c in df.columns], errors="ignore")
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, sortable=True, filterable=False,
                                cellStyle={"textAlign": "right"})
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_grid_options(rowSelection="single",
                              suppressRowClickSelection=False,
                              animateRows=True,
                              rowHeight=30,
                              headerHeight=34)
    # Left-aligned text columns
    for c in ("Symbol", "Side", "TF", "Status", "Touch Time"):
        if c in df.columns:
            gb.configure_column(c, cellStyle={"textAlign": "left"})
    if "Symbol" in df.columns:
        gb.configure_column("Symbol", pinned="left", width=130,
                            cellStyle={"textAlign": "left", "fontWeight": "700",
                                       "color": "#0f172a"})
    if "Side" in df.columns:
        gb.configure_column("Side", cellStyle=_AG_SIDE_STYLE, width=80)
    if "TF" in df.columns:
        gb.configure_column("TF", width=70)
    if "Touch Time" in df.columns:
        gb.configure_column("Touch Time", width=160,
                            cellStyle={"textAlign": "left", "color": "#64748b"})
    if "Entry" in df.columns:
        gb.configure_column("Entry", type=["numericColumn"], width=95,
                            valueFormatter="value && value.toFixed(2)",
                            cellStyle={"color": "#111827", "fontWeight": "600",
                                       "textAlign": "right"})
    if "SL" in df.columns:
        gb.configure_column("SL", type=["numericColumn"], width=95,
                            valueFormatter="value && value.toFixed(2)",
                            cellStyle={"color": "#b91c1c", "textAlign": "right"})
    if "Risk %" in df.columns:
        gb.configure_column("Risk %", type=["numericColumn"], width=85,
                            valueFormatter="value && value.toFixed(2)",
                            cellStyle=_AG_RISK_STYLE)
    if "Current" in df.columns:
        gb.configure_column("Current", type=["numericColumn"], width=95,
                            valueFormatter="value && value.toFixed(2)")
    if "P&L %" in df.columns:
        gb.configure_column("P&L %", type=["numericColumn"], width=95,
                            valueFormatter="value && value.toFixed(2)",
                            cellStyle=_AG_PNL_STYLE)
    if "Peak %" in df.columns:
        gb.configure_column("Peak %", type=["numericColumn"], width=95,
                            valueFormatter="value && value.toFixed(2)",
                            cellStyle=_AG_PEAK_STYLE)
    if "Status" in df.columns:
        gb.configure_column("Status", cellStyle=_AG_STATUS_STYLE, width=110)
    if "1d?" in df.columns:
        gb.configure_column("1d?", width=70, cellStyle=_AG_TICK_STYLE)

    grid_options = gb.build()
    # Wrap the (light) grid in a tinted card so it sits cleanly inside the
    # dark page background.
    st.markdown(
        "<div class='ag-light-card'>",
        unsafe_allow_html=True,
    )
    res = AgGrid(
        df, gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        theme="alpine",
        fit_columns_on_grid_load=False,
        height=height,
        key=key,
        reload_data=False,
        custom_css={
            ".ag-root-wrapper":       {"border-radius": "12px",
                                       "border": "1px solid #e5e7eb",
                                       "box-shadow": "0 4px 14px rgba(0,0,0,0.25)"},
            ".ag-header":             {"background-color": "#f8fafc",
                                       "border-bottom": "1px solid #e5e7eb"},
            ".ag-header-cell":        {"color": "#374151",
                                       "font-weight": "600",
                                       "letter-spacing": "0.02em"},
            ".ag-row":                {"background-color": "#ffffff",
                                       "color": "#111827"},
            ".ag-row-odd":            {"background-color": "#fafbfd"},
            ".ag-row-hover":          {"background-color": "#eef2ff !important"},
            ".ag-row-selected":       {"background-color": "#dbeafe !important",
                                       "border-left": "3px solid #2563eb"},
            ".ag-cell":               {"border-right": "1px solid #f1f5f9"},
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)
    sel = res.get("selected_rows")
    if sel is None:
        return
    if hasattr(sel, "to_dict"):       # pandas DataFrame return
        if sel.empty:
            return
        sym = str(sel.iloc[0].get("Symbol", ""))
    elif isinstance(sel, list):
        if not sel:
            return
        sym = str(sel[0].get("Symbol", ""))
    else:
        return
    if not sym:
        return
    if st.session_state.get("sel_symbol") == sym:
        return    # avoid infinite-rerun loop
    st.session_state["_pending_jump"] = {"symbol": sym, "tf": jump_tf}
    st.rerun()


def _live_refresh_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised patch of Current / P&L % / Peak % / Status from the LTP
    feed. Risk % stays static. No-op when Live toggle is off, outside
    market hours, or when ltp.parquet has no rows.

    Performance-tuned: O(n) numpy/pandas operations across the whole frame,
    no per-row iteration, no per-row disk reads. Symbols not yet in
    ltp.parquet keep their scan-time values until the LTP worker fills in."""
    if df.empty or not st.session_state.get("live_mode") or not _market_open_ist():
        return df

    # NOTE: we deliberately do NOT register table symbols with the LTP worker
    # any more — the chart symbol alone is registered (see the chart-render
    # block above). Tables show whatever the LTP feed happens to have from a
    # prior watchlist; missing rows keep their scan-time values. This keeps
    # the worker's per-minute 1m fetch budget focused on the one symbol the
    # user is actually charting.
    ltp_df = _read_ltp_df()
    if ltp_df.empty:
        return df
    ltp_map = dict(zip(ltp_df["symbol"].astype(str), ltp_df["ltp"].astype(float)))
    if not ltp_map:
        return df

    out = df.copy()
    out["_live"] = out["Symbol"].astype(str).map(ltp_map)
    mask = out["_live"].notna() & out["Entry"].notna()
    if not mask.any():
        return out.drop(columns="_live")

    sign = np.where(out["Side"] == "BUY", 1.0, -1.0)
    entry = out["Entry"].astype(float)
    live  = out["_live"].astype(float)
    live_move = sign * (live - entry) / entry * 100.0

    # Current — overwrite where live is available
    out.loc[mask, "Current"] = live[mask].round(2)

    # Status — flip OPEN → SL HIT if live crossed SL
    sl_vals = out["SL"].astype(float)
    cross_buy  = (out["Side"] == "BUY")  & (live <= sl_vals)
    cross_sell = (out["Side"] == "SELL") & (live >= sl_vals)
    crossed = (out["Status"] == "OPEN") & sl_vals.notna() & (cross_buy | cross_sell) & mask
    out.loc[crossed, "Status"] = "SL HIT"

    # P&L % — SL HIT freezes at -risk_pct, else recomputed from live
    risk = out["Risk %"].astype(float)
    is_sl_hit = (out["Status"] == "SL HIT") & risk.notna() & mask
    pnl = pd.Series(np.where(is_sl_hit, -risk, live_move), index=out.index)
    out.loc[mask, "P&L %"] = pnl[mask].round(2)

    # Peak % — max of stored peak and current live move
    prev_peak = out["Peak %"].astype(float)
    new_peak = np.where(prev_peak.isna(), live_move,
                        np.maximum(prev_peak.fillna(-np.inf), live_move))
    out.loc[mask, "Peak %"] = pd.Series(new_peak, index=out.index)[mask].round(2)

    return out.drop(columns="_live")


def show_tf(label: str, tf: str, df: pd.DataFrame):
    df = _live_refresh_metrics(df)
    st.subheader(f"{label} entries on {scan_day}  ({len(df)} rows)")
    if df.empty:
        st.caption("(none)")
        return

    # ── Per-TF filters ──────────────────────────────────────────────────
    if tf == "1h":
        f1, f2, fH, f3 = st.columns([1.3, 1.8, 1.3, 1.0])
    else:
        f1, f2, f3 = st.columns([1.4, 2.0, 1.0])
    side_choice = f1.radio(
        f"Side · {tf}", ["All", "BUY", "SELL"], horizontal=True,
        key=f"flt_side_{tf}",
    )
    q = f2.text_input(
        f"Symbol contains · {tf}", value="", key=f"flt_q_{tf}",
        placeholder="e.g. RELIANCE  (case-insensitive substring)",
    ).strip().upper()

    hour_choice = "All"
    if tf == "1h":
        hours_in_df = sorted(pd.to_datetime(df["Touch Time"]).dt.strftime("%H:%M").unique().tolist())
        hour_choice = fH.selectbox(
            f"Hour · {tf}",
            ["All"] + hours_in_df,
            key=f"flt_hour_{tf}",
            help="Only show entries that fired in this 1h candle (NSE 1h "
                 "candles open at 09:15, 10:15, ..., 15:15).",
        )

    sort_choice = f3.selectbox(
        f"Sort · {tf}",
        ["P&L % ↓", "P&L % ↑", "Peak % ↓", "Side · Symbol", "Symbol", "Touch Time"],
        key=f"flt_sort_{tf}",
    )

    filtered = df.copy()
    if side_choice != "All":
        filtered = filtered[filtered["Side"] == side_choice]
    if q:
        filtered = filtered[filtered["Symbol"].str.upper().str.contains(q, na=False)]
    if tf == "1h" and hour_choice != "All":
        filtered = filtered[pd.to_datetime(filtered["Touch Time"]).dt.strftime("%H:%M") == hour_choice]
    if sort_choice == "P&L % ↓":
        filtered = filtered.sort_values("P&L %", ascending=False, na_position="last")
    elif sort_choice == "P&L % ↑":
        filtered = filtered.sort_values("P&L %", ascending=True, na_position="last")
    elif sort_choice == "Peak % ↓":
        filtered = filtered.sort_values("Peak %", ascending=False, na_position="last")
    elif sort_choice == "Side · Symbol":
        filtered = filtered.sort_values(["Side", "Symbol", "Touch Time"])
    elif sort_choice == "Symbol":
        filtered = filtered.sort_values(["Symbol", "Touch Time"])
    elif sort_choice == "Touch Time":
        filtered = filtered.sort_values("Touch Time")
    filtered = filtered.reset_index(drop=True)

    st.caption(f"Showing **{len(filtered)}** of {len(df)} rows after filter  ·  "
               "click any row to load the chart.")
    aggrid_pick(filtered,
                key=f"tbl_{tf}_v{st.session_state.get('_scan_v', 0)}",
                jump_tf=tf)

if show_1d_only:
    show_tf("1d", "1d", df1d)
elif show_1h_all:
    show_tf("1h", "1h", df1h)
elif show_1h_hourly:
    st.subheader(f"🕐 1h candle {selected_hour} entries — {scan_day}")
    if df1h.empty:
        st.info("No 1h entries on this day — nothing to surface.")
    else:
        st.caption(
            f"1h candle **{selected_hour}** on {scan_day}.  "
            f"Rows tagged ✓ also have a 1d entry on this day "
            f"(treated as the same hour because the day is still open)."
        )
        _df1h_live = _live_refresh_metrics(df1h)
        last = _df1h_live[pd.to_datetime(_df1h_live["Touch Time"]).dt.strftime("%H:%M") == selected_hour].copy()
        if not df1d.empty:
            d1d_keys = set(map(tuple, df1d[["Symbol", "Side"]].drop_duplicates().itertuples(index=False)))
            last["1d?"] = last.apply(
                lambda r: "✓" if (r["Symbol"], r["Side"]) in d1d_keys else "",
                axis=1,
            )
        else:
            last["1d?"] = ""
        last = last.sort_values(by=["1d?", "Side", "Symbol"],
                                ascending=[False, True, True]).reset_index(drop=True)

        lf1, lf2, lf3 = st.columns([1.4, 2.0, 1.4])
        side_choice = lf1.radio(
            f"Side · {selected_hour}", ["All", "BUY", "SELL"], horizontal=True,
            key=f"flt_side_h{selected_hour}",
        )
        q = lf2.text_input(
            f"Symbol contains · {selected_hour}", value="",
            key=f"flt_q_h{selected_hour}",
            placeholder="e.g. RELIANCE  (case-insensitive substring)",
        ).strip().upper()
        only_combined = lf3.checkbox(
            "Show only 1d + 1h combined", value=False,
            key=f"flt_combined_h{selected_hour}",
            help="Only stocks that also have a 1d entry on this day.",
        )

        filtered = last.copy()
        if side_choice != "All":
            filtered = filtered[filtered["Side"] == side_choice]
        if q:
            filtered = filtered[filtered["Symbol"].str.upper().str.contains(q, na=False)]
        if only_combined:
            filtered = filtered[filtered["1d?"] == "✓"]
        filtered = filtered.reset_index(drop=True)

        combined_n = int((last["1d?"] == "✓").sum())
        st.caption(
            f"Showing **{len(filtered)}** of {len(last)} entries at {selected_hour}  ·  "
            f"**{combined_n}** with a matching 1d entry."
        )
        aggrid_pick(filtered,
                    key=f"tbl_h{selected_hour}_v{st.session_state.get('_scan_v', 0)}",
                    jump_tf="1h")
elif show_both:
    st.subheader(f"⭐ Stocks with entry on BOTH 1d AND 1h — {scan_day}  ({len(inter)} rows)")
    if inter.empty:
        st.info("No stock has an entry on both timeframes on this day.")
    else:
        bf1, bf2, bf3 = st.columns([1.4, 2.0, 1.0])
        side_choice = bf1.radio(
            "Side · both", ["All", "BUY", "SELL"], horizontal=True,
            key="flt_side_both",
        )
        q = bf2.text_input(
            "Symbol contains · both", value="",
            key="flt_q_both",
            placeholder="e.g. RELIANCE  (case-insensitive substring)",
        ).strip().upper()
        sort_choice = bf3.selectbox(
            "Sort · both",
            ["Side · Symbol", "Symbol"],
            key="flt_sort_both",
        )

        filtered = _live_refresh_metrics(inter)
        if side_choice != "All":
            filtered = filtered[filtered["Side"] == side_choice]
        if q:
            filtered = filtered[filtered["Symbol"].str.upper().str.contains(q, na=False)]
        if sort_choice == "Side · Symbol":
            filtered = filtered.sort_values(["Side", "Symbol"])
        else:
            filtered = filtered.sort_values("Symbol")
        filtered = filtered.reset_index(drop=True)

        st.caption(f"Showing **{len(filtered)}** of {len(inter)} starred picks  ·  "
                   "click any row to load the chart.")
        aggrid_pick(filtered,
                    key=f"tbl_both_v{st.session_state.get('_scan_v', 0)}",
                    jump_tf="1d")

dl1, dl2 = st.columns(2)
dl1.download_button("Download 1d PDF",
                    data=df_to_pdf_bytes(df1d, f"1d entries — {scan_day}"),
                    file_name=f"entries_{scan_day}_1d.pdf",
                    mime="application/pdf")
dl2.download_button("Download 1h PDF",
                    data=df_to_pdf_bytes(df1h, f"1h entries — {scan_day}"),
                    file_name=f"entries_{scan_day}_1h.pdf",
                    mime="application/pdf")

st.markdown(
    "<div class='footer-meta'>investeq · swing-low daily scanner &nbsp;·&nbsp; "
    "daily + intraday entries · ⭐ starred high-conviction picks</div>",
    unsafe_allow_html=True,
)

# Flush the accumulated watchlist (just the chart symbol now — tables no
# longer contribute) so the LTP worker knows what single symbol to poll
# LTPs for and fetch fresh 1m bars for every 60s. The chart's own
# st_autorefresh at 60s drives the rerun cadence — no extra timer needed.
# the table.
_flush_watchlist()
