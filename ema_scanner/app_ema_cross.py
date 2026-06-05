"""
EMA21/50 Zone-Filter Watchlist Scanner — TradingView Lightweight-Charts UI.

Reads OHLC CSVs from `<this-folder>/data/{1h,1d}/`.
The EMA pair is fixed at 21 / 50 for both timeframes.

What it does
------------
Scans every stock and reports only those whose LATEST CLOSED bar
(strictly before today — i.e. yesterday's daily bar, or yesterday's last 1h
bar) satisfies an EMA21/50 zone-filter. Stocks whose data hasn't been
updated to yesterday (stale CSVs) are silently dropped.

  Preconditions (per side):
    BUY  → EMA21 > EMA50 at the latest bar (uptrend regime, a bullish cross
            has already happened)
    SELL → EMA21 < EMA50 at the latest bar

  Trigger conditions (ANY of the enabled rules fires the signal):
    A · BUY:  low ≤ EMA21 and close > EMA21
         SELL: high ≥ EMA21 and close < EMA21
    B · BUY:  EMA50 < close < EMA21
         SELL: EMA21 < close < EMA50
    C · BUY:  close < EMA50 and high > EMA50
         SELL: close > EMA50 and low < EMA50

Output: a clickable watchlist table — click a row to open the chart with
the qualifying bar highlighted, so you can plan tomorrow's moves.

Run:  streamlit run app_ema_cross.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts


DATA_ROOT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TIMEFRAMES = ["1h", "1d"]
FAST_EMA   = 21
SLOW_EMA   = 50


# ───────────────────────── page ─────────────────────────
st.set_page_config(page_title="EMA21/50 Zone Scanner",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.stApp { background-color: #131722; }
h1, h2, h3 { color: #d1d4dc; }
div[data-testid="stMetricValue"] { color: #d1d4dc; }
</style>
""", unsafe_allow_html=True)
st.title("EMA21/50 Zone Scanner — tomorrow's watchlist")


# ───────────────────────── data IO ─────────────────────────
def tf_dir(tf: str) -> str:
    return os.path.join(DATA_ROOT, tf)


@st.cache_data(show_spinner=False)
def list_symbols(tf: str) -> list[str]:
    d = tf_dir(tf)
    if not os.path.isdir(d):
        return []
    return sorted(fn[:-len("_historical.csv")]
                  for fn in os.listdir(d) if fn.endswith("_historical.csv"))


@st.cache_data(show_spinner=False)
def load_ohlc(symbol: str, tf: str) -> pd.DataFrame:
    path = os.path.join(tf_dir(tf), f"{symbol}_historical.csv")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].dropna()


# ───────────────────────── EMA + qualifier ─────────────────────────
def add_emas(df: pd.DataFrame, fast: int = FAST_EMA, slow: int = SLOW_EMA) -> pd.DataFrame:
    df = df.copy()
    df[f"EMA{fast}"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df[f"EMA{slow}"] = df["Close"].ewm(span=slow, adjust=False).mean()
    return df


def _qualify_long(o, h, l, c, ef, es, *, use_A, use_B, use_C):
    """BUY trigger (ef > es assumed). Returns tag or None."""
    if use_A and l <= ef and c > ef:    return "A · touched 21, close > 21"
    if use_B and es < c < ef:           return "B · close between 21 & 50"
    if use_C and c < es and h > es:     return "C · below both, high > 50"
    return None


def _qualify_short(o, h, l, c, ef, es, *, use_A, use_B, use_C):
    """SELL trigger (ef < es assumed). Returns tag or None."""
    if use_A and h >= ef and c < ef:    return "A · touched 21, close < 21"
    if use_B and ef < c < es:           return "B · close between 21 & 50"
    if use_C and c > es and l < es:     return "C · above both, low < 50"
    return None


def scan_latest(df: pd.DataFrame, side: str,
                use_A: bool, use_B: bool, use_C: bool,
                *, cutoff: pd.Timestamp, max_stale_days: int) -> dict | None:
    """Check whether the latest CLOSED bar (strictly before `cutoff`) passes
    the zone filter on `side`. Stocks whose newest bar is older than
    `max_stale_days` (e.g. CSV last updated in 2025) are rejected.
    Returns a metadata dict or None if it doesn't qualify."""
    df = df[df.index < cutoff]
    if len(df) < SLOW_EMA + 10:
        return None
    last_dt = df.index[-1]
    age_days = (cutoff - last_dt).total_seconds() / 86400.0
    if age_days > max_stale_days:
        return None
    df = add_emas(df)
    last = df.iloc[-1]
    o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
    ef = float(last[f"EMA{FAST_EMA}"])
    es = float(last[f"EMA{SLOW_EMA}"])

    # EMA-regime gate
    if side == "BUY"  and not (ef > es): return None
    if side == "SELL" and not (ef < es): return None

    # Trigger gate
    trig = (_qualify_long(o, h, l, c, ef, es, use_A=use_A, use_B=use_B, use_C=use_C)
            if side == "BUY"
            else _qualify_short(o, h, l, c, ef, es, use_A=use_A, use_B=use_B, use_C=use_C))
    if trig is None:
        return None

    # bars since the most recent cross of this direction (for context)
    if side == "BUY":
        above = (df[f"EMA{FAST_EMA}"] > df[f"EMA{SLOW_EMA}"]).values
        cross = above & ~np.concatenate([[False], above[:-1]])
    else:
        below = (df[f"EMA{FAST_EMA}"] < df[f"EMA{SLOW_EMA}"]).values
        cross = below & ~np.concatenate([[False], below[:-1]])
    cross_idxs = np.where(cross)[0]
    cross_idx = int(cross_idxs[-1]) if len(cross_idxs) else None
    bars_since_cross = (len(df) - 1 - cross_idx) if cross_idx is not None else None
    cross_time = df.index[cross_idx] if cross_idx is not None else None

    return {
        "Side":   side,
        "Trigger": trig,
        "Last bar": df.index[-1],
        "Close":  round(c, 2),
        f"EMA{FAST_EMA}": round(ef, 2),
        f"EMA{SLOW_EMA}": round(es, 2),
        "Δ vs 21 %": round((c - ef) / ef * 100, 2),
        "Δ vs 50 %": round((c - es) / es * 100, 2),
        "Bars since cross": bars_since_cross,
        "Last cross":       cross_time,
    }


def to_ts(t) -> int:
    pt = pd.Timestamp(t)
    if pt.tz is not None:
        pt = pt.tz_localize(None)
    return int(pt.value // 10**9)


# ───────────────────────── pending-jump processor (pre-widget) ─────────────────────────
_pending = st.session_state.pop("_pending_jump", None)
if _pending:
    st.session_state["sel_symbol"] = _pending["symbol"]
    st.session_state["jumped_to"]  = _pending["symbol"]
    st.session_state["_scan_v"]  = st.session_state.get("_scan_v",  0) + 1
    st.session_state["_chart_v"] = st.session_state.get("_chart_v", 0) + 1


# ───────────────────────── sidebar ─────────────────────────
with st.sidebar:
    st.header("Scanner")
    st.caption(f"📂 `{DATA_ROOT}`")
    tf = st.selectbox("Timeframe", TIMEFRAMES, index=1,
                       help="EMA pair is fixed 21/50 on both 1h and 1d.")
    side_mode = st.radio("Side", ["BUY", "SELL", "Both"], horizontal=True, index=2)

    st.markdown("**Trigger conditions** (any enabled fires the signal)")
    cond_A = st.checkbox("A · touched 21 EMA & close past 21", value=True)
    cond_B = st.checkbox("B · close between 21 & 50 EMAs",      value=True)
    cond_C = st.checkbox("C · close past 50 EMA & wick reached 50", value=True)

    st.markdown("**Freshness**")
    max_stale = st.number_input(
        "Reject if last bar is older than (days)",
        min_value=1, max_value=400, value=10, step=1,
        help="Anything beyond this many days behind today is treated as a "
             "stale CSV and dropped. Default 10 covers the current data lag "
             "(latest bar in the CSVs is 2026-05-08). For strict 'yesterday "
             "only' set to 1 — but you'll need to refresh the data first.",
    )

    st.markdown("---")
    st.header("Chart")
    chart_height = st.slider("Chart height (px)", 400, 1200, 720, 20)
    show_volume   = st.checkbox("Volume pane",         value=True)
    show_history  = st.checkbox("Mark ALL historic triggers", value=False,
                                help="Mark every past bar in the same regime "
                                     "that also passed the enabled conditions.")
    chart_bars    = st.number_input("Chart bars (most recent)", 30, 5000, 120, 10,
                                    help="Fewer bars = wider candles. "
                                         "120 ≈ 6 months on 1d, ~3 weeks on 1h.")
    bar_spacing   = st.slider("Bar spacing (px)", 4, 24, 12, 1,
                              help="Width of each candle in pixels. "
                                   "Increase for a more zoomed-in view.")
    show_crosses  = st.checkbox("Mark EMA21/50 crossovers", value=True,
                                help="Place a marker at every bar where EMA21 "
                                     "crosses EMA50 (bullish = gold ▲, "
                                     "bearish = red ▼).")


# ───────────────────────── header ─────────────────────────
symbols = list_symbols(tf)
if not symbols:
    st.error(f"No CSVs in {tf_dir(tf)}"); st.stop()
if not (cond_A or cond_B or cond_C):
    st.warning("Enable at least one trigger condition in the sidebar.")
    st.stop()


# ───────────────────────── scan button ─────────────────────────
run_col, cnt_col = st.columns([1, 3])
run = run_col.button("🔍 Scan all stocks", type="primary", use_container_width=True)
_cutoff_disp = (pd.Timestamp.now().normalize() - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d")
cnt_col.markdown(
    f"<div style='padding-top:0.5rem; color:#aaa;'>"
    f"Scanning <b>{len(symbols)}</b> symbols · timeframe <b>{tf}</b> · "
    f"side <b>{side_mode}</b> · "
    f"conditions {''.join(c for c, ok in [('A',cond_A),('B',cond_B),('C',cond_C)] if ok) or '—'} · "
    f"qualifying bar must be on/before <b>{_cutoff_disp}</b> "
    f"(within <b>{int(max_stale)}</b> day(s))"
    f"</div>", unsafe_allow_html=True)

if run:
    cutoff = pd.Timestamp.now().normalize()   # today 00:00 → strictly excludes today's bars
    prog = st.progress(0.0, text="Starting...")
    rows = []
    for k, sym in enumerate(symbols):
        prog.progress((k + 1) / len(symbols), text=f"[{k+1}/{len(symbols)}] {sym}")
        try:
            df_s = load_ohlc(sym, tf)
        except Exception:
            continue
        for side in (["BUY", "SELL"] if side_mode == "Both" else [side_mode]):
            meta = scan_latest(df_s, side, cond_A, cond_B, cond_C,
                               cutoff=cutoff, max_stale_days=int(max_stale))
            if meta is None:
                continue
            rows.append({"Symbol": sym, **meta})
    prog.empty()
    df_results = pd.DataFrame(rows)
    if not df_results.empty:
        df_results = df_results.sort_values(["Side", "Symbol"]).reset_index(drop=True)
    st.session_state["scan_results"] = df_results
    st.session_state["scan_tf"] = tf
    st.session_state["scan_cutoff"] = cutoff


# ───────────────────────── results ─────────────────────────
df_res = st.session_state.get("scan_results")
res_tf = st.session_state.get("scan_tf", tf)

if df_res is not None and not df_res.empty:
    n_buy  = int((df_res["Side"] == "BUY").sum())
    n_sell = int((df_res["Side"] == "SELL").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Watchlist size", len(df_res))
    c2.metric("BUY candidates",  n_buy)
    c3.metric("SELL candidates", n_sell)

    st.caption("👉 **Click a row** to open the chart for that symbol with the "
               "qualifying bar highlighted.")
    scan_key = f"scan_v{st.session_state.get('_scan_v', 0)}"
    event = st.dataframe(
        df_res, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key=scan_key,
    )
    sel_rows = (event.selection.rows if hasattr(event, "selection")
                else event.get("selection", {}).get("rows", []))
    if sel_rows:
        row = df_res.iloc[sel_rows[0]]
        st.session_state["_pending_jump"] = {"symbol": str(row["Symbol"])}
        st.rerun()

    st.download_button(
        "Download watchlist CSV",
        data=df_res.to_csv(index=False).encode(),
        file_name=f"watchlist_{res_tf}_{side_mode}.csv",
        mime="text/csv",
    )
elif df_res is not None:
    st.info("No symbols matched. Try toggling more conditions on, or switch side.")
else:
    st.info("Hit **Scan all stocks** to build the watchlist.")


# ───────────────────────── chart (single symbol) ─────────────────────────
st.markdown("---")
st.subheader("Stock chart")

# symbol selector (also jumped-to from scan)
if "sel_symbol" not in st.session_state or st.session_state["sel_symbol"] not in symbols:
    st.session_state["sel_symbol"] = "RELIANCE" if "RELIANCE" in symbols else symbols[0]

symbol = st.selectbox("Symbol", symbols, key="sel_symbol")
if st.session_state.get("jumped_to") == symbol:
    st.success(f"🎯 Jumped from scan — {symbol}")
    st.session_state.pop("jumped_to", None)

df = load_ohlc(symbol, tf)
if df.empty:
    st.warning("No data."); st.stop()

# Mirror scanner's cutoff: drop today's partial bar so the chart's last bar
# is the same bar the scanner qualified on (yesterday's close on 1d).
chart_cutoff = pd.Timestamp.now().normalize()
df = df[df.index < chart_cutoff]
if df.empty:
    st.warning(f"No bars before {chart_cutoff:%Y-%m-%d} — data is empty after the today-cutoff."); st.stop()
df = add_emas(df)

# evaluate triggers on every bar (for marker placement)
ef_col, es_col = f"EMA{FAST_EMA}", f"EMA{SLOW_EMA}"
trig_map: dict[int, tuple[str, str]] = {}   # bar_idx -> (side, tag)
for i in range(len(df)):
    o, h, l, c = (float(df["Open"].iloc[i]),  float(df["High"].iloc[i]),
                  float(df["Low"].iloc[i]),    float(df["Close"].iloc[i]))
    ef = float(df[ef_col].iloc[i]); es = float(df[es_col].iloc[i])
    if pd.isna(ef) or pd.isna(es):
        continue
    if ef > es:
        t = _qualify_long(o, h, l, c, ef, es, use_A=cond_A, use_B=cond_B, use_C=cond_C)
        if t and side_mode in ("BUY", "Both"):
            trig_map[i] = ("BUY", t)
    elif ef < es:
        t = _qualify_short(o, h, l, c, ef, es, use_A=cond_A, use_B=cond_B, use_C=cond_C)
        if t and side_mode in ("SELL", "Both"):
            trig_map[i] = ("SELL", t)

# what the latest bar shows
last_i = len(df) - 1
latest_trig = trig_map.get(last_i)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bars", f"{len(df):,}")
m2.metric("Last bar", f"{df.index[-1]:%Y-%m-%d %H:%M}")
m3.metric("Close", f"{df['Close'].iloc[-1]:.2f}")
if latest_trig:
    side, tag = latest_trig
    m4.metric("Latest signal", f"{side}", tag)
else:
    m4.metric("Latest signal", "—")

# crop chart to most recent N bars
df_chart = df.iloc[-int(chart_bars):]
candles, ef_data, es_data, vol_data, time_array = [], [], [], [], []
for i in range(len(df_chart)):
    o, h, l, c = (df_chart["Open"].iloc[i], df_chart["High"].iloc[i],
                  df_chart["Low"].iloc[i],  df_chart["Close"].iloc[i])
    ef, es = df_chart[ef_col].iloc[i], df_chart[es_col].iloc[i]
    if any(pd.isna(v) for v in (o, h, l, c, ef, es)):
        continue
    t = to_ts(df_chart.index[i])
    time_array.append(t)
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

t_min = time_array[0] if time_array else None
t_max = time_array[-1] if time_array else None
chart_start_idx = max(0, len(df) - int(chart_bars))

# markers
markers = []

def _add_marker(i, side, tag, *, big=False):
    if i < 0 or i >= len(df):
        return
    mt = to_ts(df.index[i])
    if t_min is None or not (t_min <= mt <= t_max):
        return
    is_short = side == "SELL"
    markers.append({
        "time": mt,
        "position": "aboveBar" if is_short else "belowBar",
        "color": "#ff1744" if is_short else "#00e676",
        "shape": "arrowDown" if is_short else "arrowUp",
        "text": (f"⭐ {tag}" if big else tag.split(" · ")[0]),
        "size": 3 if big else 1,
    })

if show_history:
    for i, (side, tag) in trig_map.items():
        if i == last_i:
            continue
        _add_marker(i, side, tag, big=False)

# always star-mark the latest bar if it triggered
if latest_trig:
    _add_marker(last_i, latest_trig[0], latest_trig[1], big=True)

# EMA21/50 crossover markers (gold ▲ bullish, red ▼ bearish) — drawn inBar so
# they don't fight trigger arrows that sit above/below the candle.
if show_crosses:
    ef_arr = df[ef_col].values
    es_arr = df[es_col].values
    above_prev = ef_arr[chart_start_idx - 1] > es_arr[chart_start_idx - 1] \
                 if chart_start_idx >= 1 else None
    for i in range(chart_start_idx, len(df)):
        if pd.isna(ef_arr[i]) or pd.isna(es_arr[i]):
            continue
        above_now = ef_arr[i] > es_arr[i]
        if above_prev is None:
            above_prev = above_now
            continue
        if above_now != above_prev:
            mt = to_ts(df.index[i])
            if t_min is not None and t_min <= mt <= t_max:
                markers.append({
                    "time": mt,
                    "position": "inBar",
                    "color": "#ffd54f" if above_now else "#ff5252",
                    "shape": "circle",
                    "text": ("🟢 21↑50" if above_now else "🔴 21↓50"),
                    "size": 1,
                })
        above_prev = above_now

# dedupe markers by (time, position) so trigger ▲▼ + crossover ●  can coexist
seen, deduped = set(), []
for m in sorted(markers, key=lambda m: (m["time"], m.get("position", ""))):
    key = (m["time"], m.get("position", ""))
    if key in seen:
        continue
    seen.add(key); deduped.append(m)
markers = deduped


# ───────────────────────── render chart ─────────────────────────
chart_options = {
    "height": chart_height,
    "layout": {"background": {"type": "solid", "color": "#131722"},
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
        "timeVisible": tf == "1h",
        "secondsVisible": False,
        "rightOffset": 12, "barSpacing": int(bar_spacing),
    },
    "watermark": {
        "visible": True, "fontSize": 44,
        "color": "rgba(180,180,180,0.07)",
        "text": f"{symbol} · {tf}  ·  EMA{FAST_EMA}/{SLOW_EMA}",
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
     "options": {"color": "#ff9800", "lineWidth": 2,
                 "title": f"EMA{FAST_EMA}",
                 "priceLineVisible": False, "lastValueVisible": True,
                 "crosshairMarkerVisible": False}},
    {"type": "Line", "data": es_data,
     "options": {"color": "#42a5f5", "lineWidth": 2,
                 "title": f"EMA{SLOW_EMA}",
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
    key=(f"chart_{symbol}_{tf}_{int(chart_bars)}_{int(bar_spacing)}"
         f"_{show_history}_{show_crosses}_{len(markers)}_{len(candles)}_v{_chart_v}"),
)

st.markdown(
    "<div style='display:flex; gap:18px; flex-wrap:wrap; font-size:12px; "
    "color:#bbb; padding:4px 2px 10px 2px;'>"
    f"<span><span style='color:#ff9800;'>━</span> EMA{FAST_EMA}</span>"
    f"<span><span style='color:#42a5f5;'>━</span> EMA{SLOW_EMA}</span>"
    "<span><span style='color:#ffd54f;'>●</span> Bullish 21/50 cross</span>"
    "<span><span style='color:#ff5252;'>●</span> Bearish 21/50 cross</span>"
    "<span><span style='color:#00e676;'>▲</span> BUY trigger</span>"
    "<span><span style='color:#ff1744;'>▼</span> SELL trigger</span>"
    "<span>⭐ = latest-bar trigger (today's setup)</span>"
    "</div>",
    unsafe_allow_html=True,
)
