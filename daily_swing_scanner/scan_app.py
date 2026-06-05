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
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy_core import add_emas, scan_buy_signals, scan_sell_signals


# ─── data paths ───
DATA_ROOT_DEFAULT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "ema_scanner", "data"))
TIMEFRAMES = ["1d", "1h"]


# ─── page ───
st.set_page_config(page_title="Swing-Low Daily Scanner",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.stApp { background-color: #0e0e10; }
h1, h2, h3 { color: #e0e0e0; }

/* Make st.caption() readable everywhere (not muted grey) */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.stCaption, .stCaption p {
    color: #d0d0d0 !important;
}

/* ── Mobile readability boost (phones / narrow viewports) ── */
@media (max-width: 768px) {
    html, body, .stApp { font-size: 15px !important; }

    h1 { font-size: 1.35rem !important; line-height: 1.25 !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1.05rem !important; }

    /* Captions: "Click any row to load …", chart legend below */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
    .stCaption, .stCaption * {
        font-size: 14px !important;
        line-height: 1.45 !important;
        color: #d8d8d8 !important;
    }

    /* Streamlit dataframe – header, cells, filter chips (ALL/BUY/SELL),
       footer ("Showing N of M"), search input + placeholder            */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] * {
        font-size: 14px !important;
    }
    [data-testid="stDataFrame"] input,
    [data-testid="stDataFrame"] input::placeholder {
        font-size: 14px !important;
        color: #d0d0d0 !important;
        opacity: 1 !important;
    }
    [data-testid="stDataFrame"] [role="option"],
    [data-testid="stDataFrame"] [role="menuitem"],
    [data-testid="stDataFrame"] button {
        font-size: 14px !important;
        color: #e0e0e0 !important;
    }

    /* Selectbox label/option, number inputs, sliders */
    .stSelectbox, .stSelectbox *,
    .stNumberInput, .stNumberInput *,
    .stTextInput,  .stTextInput *,
    .stSlider,     .stSlider *,
    .stCheckbox,   .stCheckbox * {
        font-size: 14px !important;
    }

    /* Metric labels + numbers */
    [data-testid="stMetricLabel"]  { font-size: 13px !important; color: #cfcfcf !important; }
    [data-testid="stMetricValue"]  { font-size: 22px !important; }

    /* Small / muted spans (generic) */
    small, .small { font-size: 13px !important; color: #c8c8c8 !important; }

    /* Progress bar text */
    [data-testid="stProgress"] * { font-size: 13px !important; color: #d0d0d0 !important; }

    /* Chart legend (── EMA21  ▲ BUY touch  …) */
    .chart-legend, .chart-legend * {
        font-size: 14px !important;
        color: #e0e0e0 !important;
    }
}
</style>
""", unsafe_allow_html=True)
st.title("Swing-Low Daily Scanner — EMA21/50 touch entries on a single day")


# ─── helpers ───
def load_ohlc(symbol: str, tf: str, data_root: str) -> pd.DataFrame:
    path = os.path.join(data_root, tf, f"{symbol}_historical.csv")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    if df["Open"].isna().any():
        df["Open"] = df["Open"].fillna(df["Close"].shift(1))
    return df.dropna(subset=["High", "Low", "Close"])


def signals_on_day(symbol: str, tf: str, data_root: str, target_day: date,
                   fast: int, slow: int, compulsory: int) -> list[dict]:
    """Return list of touches that fired on `target_day` for one symbol."""
    try:
        df = load_ohlc(symbol, tf, data_root)
    except Exception:
        return []
    if len(df) < slow + 5:
        return []
    cutoff = pd.Timestamp(target_day) + pd.Timedelta(days=1)
    df = df[df.index < cutoff]
    if len(df) < slow + 5:
        return []
    df = add_emas(df, fast, slow)

    out = []
    for side, fn in (("BUY", scan_buy_signals), ("SELL", scan_sell_signals)):
        sigs = fn(df, fast, slow, compulsory=compulsory)
        if sigs.empty:
            continue
        sigs = sigs[sigs["Touch Time"].dt.date == target_day]
        for _, r in sigs.iterrows():
            out.append({
                "Symbol":     symbol,
                "Side":       side,
                "TF":         tf,
                "Touch Time": r["Touch Time"],
                "Close":      round(float(r["Touch Close"]), 2),
            })
    return out


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


# ─── sidebar ───
with st.sidebar:
    st.header("Scan")
    data_root = st.text_input("Data root", value=DATA_ROOT_DEFAULT,
                              help="Must contain `1d/` and `1h/` sub-folders "
                                   "of `<SYM>_historical.csv`.")
    target_day = st.date_input("Entry day (the day the touch must fire on)",
                               value=date(2026, 5, 14),
                               help="Default = 2026-05-14 (the day the data ends).")
    col1, col2 = st.columns(2)
    fast = col1.number_input("Fast EMA", 5, 200, 21, 1)
    slow = col2.number_input("Slow EMA", 5, 200, 50, 1)
    compulsory = st.number_input("Compulsory touches", 1, 5, 2, 1,
                                 help="First N touches after a cross are always "
                                      "taken; from N+1 onwards, a new HH (for BUY) "
                                      "or LH (for SELL) since the previous touch "
                                      "is required.")
    st.markdown("---")
    st.header("Chart")
    chart_bars   = st.number_input("Chart bars (most recent)", 30, 1000, 120, 10)
    bar_spacing  = st.slider("Bar spacing (px)", 4, 24, 12, 1)
    chart_height = st.slider("Chart height (px)", 400, 1200, 720, 20)
    show_volume  = st.checkbox("Volume pane", value=True)


# ─── scan button ───
run_col, info_col = st.columns([1, 3])
run = run_col.button("🔍 Scan all stocks", type="primary", use_container_width=True)
info_col.markdown(
    f"<div style='padding-top:0.5rem; color:#aaa;'>"
    f"Looking for EMA{fast}/{slow} swing-low touches that fired on "
    f"<b>{target_day}</b>, across <b>1h</b> and <b>1d</b>."
    f"</div>", unsafe_allow_html=True)

if run:
    if not os.path.isdir(data_root):
        st.error(f"Data root not found: {data_root}")
        st.stop()
    prog = st.progress(0.0, text="Starting...")
    rows_per_tf: dict[str, list[dict]] = {tf: [] for tf in TIMEFRAMES}
    total_steps = sum(len(list_symbols(data_root, tf)) for tf in TIMEFRAMES)
    step = 0
    for tf in TIMEFRAMES:
        symbols = list_symbols(data_root, tf)
        for sym in symbols:
            step += 1
            if step % 10 == 0 or step == total_steps:
                prog.progress(step / total_steps,
                              text=f"[{step}/{total_steps}] {tf}  {sym}")
            rows_per_tf[tf].extend(
                signals_on_day(sym, tf, data_root, target_day,
                               int(fast), int(slow), int(compulsory))
            )
    prog.empty()
    st.session_state["scan_1d"] = pd.DataFrame(rows_per_tf["1d"])
    st.session_state["scan_1h"] = pd.DataFrame(rows_per_tf["1h"])
    st.session_state["scan_day"]   = target_day
    st.session_state["scan_fast"]  = int(fast)
    st.session_state["scan_slow"]  = int(slow)
    st.session_state["scan_compu"] = int(compulsory)


# ─── results ───
df1d = st.session_state.get("scan_1d")
df1h = st.session_state.get("scan_1h")

if df1d is None or df1h is None:
    st.info("Hit **🔍 Scan all stocks** to run the scan.")
    st.stop()

scan_day = st.session_state.get("scan_day", target_day)
m1, m2, m3 = st.columns(3)
m1.metric("1d entries", len(df1d))
m2.metric("1h entries", len(df1h))

# intersection by (Symbol, Side)
if not df1d.empty and not df1h.empty:
    d1d = df1d[["Symbol", "Side"]].drop_duplicates()
    d1h = df1h[["Symbol", "Side"]].drop_duplicates()
    inter = d1d.merge(d1h, on=["Symbol", "Side"]).sort_values(["Side", "Symbol"]).reset_index(drop=True)
else:
    inter = pd.DataFrame(columns=["Symbol", "Side"])
m3.metric("⭐ Both timeframes", len(inter))

# ─── starred table ───
st.subheader(f"🌟 Stocks with entry on BOTH 1d AND 1h — {scan_day}")
if inter.empty:
    st.info("No stock has an entry on both timeframes on this day.")
else:
    st.dataframe(inter, use_container_width=True, hide_index=True)


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

# Symbol & timeframe pickers
all_syms = sorted(set(df1d["Symbol"]) | set(df1h["Symbol"]))
if not all_syms:
    st.info("No qualifying symbols.")
    st.stop()

# Guard: if a previously-selected symbol is no longer in the scan results
# (e.g. user changed conditions and re-scanned), reset to the default.
if st.session_state.get("sel_symbol") not in all_syms:
    st.session_state["sel_symbol"] = all_syms[0]

c1, c2 = st.columns([3, 1])
symbol = c1.selectbox("Symbol", all_syms, key="sel_symbol")
chart_tf = c2.selectbox("Timeframe", TIMEFRAMES, key="sel_tf")

if st.session_state.get("jumped_to") == symbol:
    st.success(f"🎯 Jumped from table — {symbol} ({chart_tf})")
    st.session_state.pop("jumped_to", None)

# Load + crop
try:
    df = load_ohlc(symbol, chart_tf, data_root)
except Exception as e:
    st.error(f"Could not load {symbol} on {chart_tf}: {e}")
    st.stop()
cutoff = pd.Timestamp(scan_day) + pd.Timedelta(days=1)
df = df[df.index < cutoff]
if df.empty:
    st.warning(f"No bars before {cutoff:%Y-%m-%d}.")
    st.stop()
df = add_emas(df, st.session_state.get("scan_fast", 21),
                  st.session_state.get("scan_slow", 50))

# Find touches that fired on scan_day for THIS stock+tf
touches_on_day = []
fast_l, slow_l = st.session_state.get("scan_fast", 21), st.session_state.get("scan_slow", 50)
compu_l = st.session_state.get("scan_compu", 2)
for side, fn in (("BUY", scan_buy_signals), ("SELL", scan_sell_signals)):
    sigs = fn(df, fast_l, slow_l, compulsory=compu_l)
    if sigs.empty:
        continue
    s_day = sigs[sigs["Touch Time"].dt.date == scan_day]
    for _, r in s_day.iterrows():
        touches_on_day.append((side, r["Touch Time"], float(r["Touch Close"])))

# Crop to last N bars
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

# Header line
n_t = len(touches_on_day)
if n_t == 0:
    st.warning(f"{symbol} on {chart_tf}: no touch fired on {scan_day}. "
               "(It may be in the other timeframe's list only.)")
else:
    summary = ", ".join(f"{s} @ {t:%H:%M} = {c:.2f}" if chart_tf == "1h"
                        else f"{s} @ {c:.2f}" for s, t, c in touches_on_day)
    st.caption(f"**{symbol} {chart_tf}** — {n_t} touch{'es' if n_t > 1 else ''} on {scan_day}: {summary}")

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
if vol_data:
    series.append({
        "type": "Histogram", "data": vol_data,
        "options": {"priceFormat": {"type": "volume"}, "priceScaleId": "",
                    "priceLineVisible": False, "lastValueVisible": False},
        "priceScale": {"scaleMargins": {"top": 0.78, "bottom": 0.0}},
    })

_chart_v = st.session_state.get("_chart_v", 0)
renderLightweightCharts(
    [{"chart": chart_options, "series": series}],
    key=(f"chart_{symbol}_{chart_tf}_{int(chart_bars)}_{int(bar_spacing)}"
         f"_{len(markers)}_{len(candles)}_v{_chart_v}"),
)

st.markdown(
    "<div class='chart-legend' style='display:flex; gap:18px; flex-wrap:wrap; "
    "font-size:13px; color:#d0d0d0; padding:4px 2px 10px 2px;'>"
    f"<span><span style='color:#ff9800;'>━</span> EMA{fast_l}</span>"
    f"<span><span style='color:#42a5f5;'>━</span> EMA{slow_l}</span>"
    "<span><span style='color:#00e676;'>▲</span> BUY touch ⭐</span>"
    "<span><span style='color:#ff1744;'>▼</span> SELL touch ⭐</span>"
    f"<span>Touch must satisfy: 2 compulsory + higher swing-high gate</span>"
    "</div>", unsafe_allow_html=True,
)


# ─── per-tf tables (clickable) — placed BELOW the chart ───
st.markdown("---")
st.caption("👉 Click any row to load that stock in the chart above.")

def show_tf(label: str, tf: str, df: pd.DataFrame):
    st.subheader(f"{label} entries on {scan_day}  ({len(df)} rows)")
    if df.empty:
        st.caption("(none)")
        return
    df = df.copy()
    df = df.sort_values(["Side", "Symbol", "Touch Time"]).reset_index(drop=True)
    key = f"tbl_{tf}_v{st.session_state.get('_scan_v', 0)}"
    event = st.dataframe(df, use_container_width=True, hide_index=True,
                         on_select="rerun", selection_mode="single-row", key=key)
    sel = (event.selection.rows if hasattr(event, "selection")
           else event.get("selection", {}).get("rows", []))
    if sel:
        row = df.iloc[sel[0]]
        st.session_state["_pending_jump"] = {"symbol": str(row["Symbol"]),
                                             "tf":     tf}
        st.rerun()

show_tf("1d", "1d", df1d)
show_tf("1h", "1h", df1h)

dl1, dl2 = st.columns(2)
dl1.download_button("Download 1d CSV",
                    data=df1d.to_csv(index=False).encode(),
                    file_name=f"entries_{scan_day}_1d.csv", mime="text/csv")
dl2.download_button("Download 1h CSV",
                    data=df1h.to_csv(index=False).encode(),
                    file_name=f"entries_{scan_day}_1h.csv", mime="text/csv")
