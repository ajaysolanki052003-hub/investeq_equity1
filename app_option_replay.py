"""
NIFTY Option-Chain Replay  —  advanced visualization.

Run:
    streamlit run app_option_replay.py

Features beyond the existing http://34.100.162.71:8765/ UI:
  * Expiry dropdown auto-populated from selected day's chain
  * Multi-strike multi-leg overlay (compare CE/PE across strikes on one chart)
  * Trade markers per minute (volume bubbles + signed volume hints)
  * IV pane computed on the fly via Black-76 inversion
  * Independent timeframe selectors for the price pane and the IV/OI panes
  * Spot price overlay on the candle pane
  * Option-chain snapshot table at the cursor time
"""

from __future__ import annotations
import os, glob
from datetime import datetime, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq
from scipy.stats import norm


# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(r"C:\Users\User\Desktop\investeq_ajs\DATA")
OPT_DIR   = ROOT / "options"
OI_DIR    = ROOT / "oi"
SPOT_DIR  = ROOT / "spot"
INDEX_DIR = ROOT / "index"

RISK_FREE_RATE         = 0.065
TRADING_SECONDS_PER_YEAR = 252 * 6.25 * 3600
SESSION_OPEN  = time(9, 15)
SESSION_CLOSE = time(15, 30)


# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY Option Chain Replay",
    page_icon="⚡",
    layout="wide",
)

PRIMARY = "#60a5fa"     # blue
GREEN   = "#26a69a"
RED     = "#ef5350"
AMBER   = "#facc15"
PURPLE  = "#c084fc"
INK     = "#0d0f1a"
PANEL   = "#13151f"


# ─── Black-Scholes / IV ───────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(market_price, S, K, T, r=RISK_FREE_RATE, option_type="CE"):
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if option_type == "CE" else max(K - S, 0)
    if market_price <= intrinsic + 0.01:
        return np.nan
    try:
        return brentq(
            lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price,
            0.001, 5.0, xtol=1e-6, maxiter=100,
        )
    except (ValueError, RuntimeError):
        return np.nan


def years_to_expiry(ts: pd.Timestamp, expiry_str: str) -> float:
    expiry_dt = pd.to_datetime(expiry_str).replace(hour=15, minute=30)
    diff = (expiry_dt - ts).total_seconds()
    return max(diff / TRADING_SECONDS_PER_YEAR, 1e-8)


# ─── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def list_trading_days() -> list[str]:
    files = sorted(SPOT_DIR.glob("*.parquet"))
    return [f.stem for f in files]


@st.cache_data(show_spinner=False)
def load_spot(date_str: str) -> pd.DataFrame:
    path = SPOT_DIR / f"{date_str}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_options(date_str: str) -> pd.DataFrame:
    path = OPT_DIR / f"{date_str}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_oi(date_str: str) -> pd.DataFrame:
    path = OI_DIR / f"{date_str}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def list_expiries_on(date_str: str) -> list[str]:
    df = load_options(date_str)
    if df.empty:
        return []
    return sorted(df["expiry"].unique().tolist())


@st.cache_data(show_spinner=False)
def list_strikes(date_str: str, expiry: str) -> list[int]:
    df = load_options(date_str)
    if df.empty:
        return []
    return sorted(df.loc[df["expiry"] == expiry, "strike"].unique().tolist())


# ─── Resampling ───────────────────────────────────────────────────────────────
TIMEFRAMES = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
              "30m": "30min", "1h": "60min"}


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df.empty:
        return df
    rule = TIMEFRAMES[tf]
    g = (df.set_index("timestamp")
           .resample(rule, closed="left", label="left")
           .agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"})
           .dropna(subset=["open", "close"]))
    return g.reset_index()


def resample_series(df: pd.DataFrame, col: str, tf: str, agg: str = "last") -> pd.DataFrame:
    if df.empty:
        return df
    rule = TIMEFRAMES[tf]
    return (df.set_index("timestamp")[col]
              .resample(rule, closed="left", label="left").agg(agg)
              .dropna()
              .reset_index())


# ─── Build chart ──────────────────────────────────────────────────────────────
def build_chart(
    date_str: str,
    expiry: str,
    legs: list[tuple[int, str]],
    tf_price: str,
    tf_aux: str,
    show_iv: bool,
    show_oi: bool,
    show_trades: bool,
    show_spot: bool,
    t_from: time,
    t_to: time,
) -> go.Figure:
    spot_full = load_spot(date_str)
    opt_full  = load_options(date_str)
    oi_full   = load_oi(date_str) if show_oi else pd.DataFrame()

    if opt_full.empty:
        return go.Figure(layout=dict(template="plotly_dark",
                                     title="No options data for this day"))

    # Time-range filter
    day = pd.to_datetime(date_str, format="%Y%m%d").date()
    t0 = pd.Timestamp.combine(day, t_from)
    t1 = pd.Timestamp.combine(day, t_to)
    opt_full  = opt_full[(opt_full["timestamp"] >= t0) & (opt_full["timestamp"] <= t1)]
    spot_full = spot_full[(spot_full["timestamp"] >= t0) & (spot_full["timestamp"] <= t1)]
    if not oi_full.empty:
        oi_full = oi_full[(oi_full["timestamp"] >= t0) & (oi_full["timestamp"] <= t1)]

    # Subplot layout
    n_rows = 1
    row_heights = [0.62]
    titles = [f"Price · {tf_price}"]
    if show_trades:
        n_rows += 1; row_heights.append(0.12); titles.append(f"Trade volume · {tf_price}")
    if show_iv:
        n_rows += 1; row_heights.append(0.16); titles.append(f"Implied Vol · {tf_aux}")
    if show_oi:
        n_rows += 1; row_heights.append(0.16); titles.append(f"Open Interest · {tf_aux}")

    # Normalise row heights
    rh = np.array(row_heights, dtype=float)
    row_heights = (rh / rh.sum()).tolist()

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.035, row_heights=row_heights,
        subplot_titles=titles,
    )

    # ── Row 1 — candles per leg, plus optional spot overlay ──
    for strike, opt_type in legs:
        leg_df = opt_full[(opt_full["expiry"] == expiry) &
                          (opt_full["strike"] == strike) &
                          (opt_full["option_type"] == opt_type)]
        if leg_df.empty:
            continue
        bars = resample_ohlcv(leg_df[["timestamp", "open", "high", "low", "close", "volume"]],
                              tf_price)
        if bars.empty:
            continue
        name = f"{strike} {opt_type}"
        up = GREEN if opt_type == "CE" else RED
        dn = RED   if opt_type == "CE" else GREEN
        fig.add_trace(go.Candlestick(
            x=bars["timestamp"], open=bars["open"], high=bars["high"],
            low=bars["low"], close=bars["close"],
            name=name, increasing_line_color=up, decreasing_line_color=dn,
            increasing_fillcolor=up, decreasing_fillcolor=dn,
        ), row=1, col=1)

        # Trade markers on top of candles — radius scaled by volume
        if show_trades:
            vmax = max(bars["volume"].max(), 1)
            sizes = 4 + 22 * np.sqrt(bars["volume"] / vmax)
            fig.add_trace(go.Scatter(
                x=bars["timestamp"], y=bars["close"],
                mode="markers",
                name=f"{name} trades",
                marker=dict(size=sizes, color=up, opacity=0.35, line=dict(width=0)),
                hovertemplate=("t=%{x|%H:%M}<br>"
                               "px=%{y:.2f}<br>"
                               "vol=%{customdata:,.0f}<extra></extra>"),
                customdata=bars["volume"].values,
                showlegend=False,
            ), row=1, col=1)

            # Volume bars in dedicated pane
            vol_row = 2
            fig.add_trace(go.Bar(
                x=bars["timestamp"], y=bars["volume"],
                name=f"{name} vol", marker_color=up, opacity=0.75,
                hovertemplate="t=%{x|%H:%M}<br>vol=%{y:,.0f}<extra></extra>",
                showlegend=False,
            ), row=vol_row, col=1)

    # Spot overlay
    if show_spot and not spot_full.empty:
        spot_bars = resample_ohlcv(spot_full[["timestamp","open","high","low","close","volume"]],
                                   tf_price)
        fig.add_trace(go.Scatter(
            x=spot_bars["timestamp"], y=spot_bars["close"],
            name="NIFTY spot", mode="lines",
            line=dict(color=AMBER, width=1.2),
            yaxis="y2",
            hovertemplate="spot=%{y:.1f}<extra></extra>",
        ), row=1, col=1)

    # ── Row 3 (or wherever) — IV per leg ──
    if show_iv and not spot_full.empty:
        iv_row = 2 + int(show_trades)
        for strike, opt_type in legs:
            leg_df = opt_full[(opt_full["expiry"] == expiry) &
                              (opt_full["strike"] == strike) &
                              (opt_full["option_type"] == opt_type)].copy()
            if leg_df.empty:
                continue
            leg_aux = resample_ohlcv(leg_df[["timestamp","open","high","low","close","volume"]],
                                     tf_aux)
            spot_aux = resample_ohlcv(spot_full[["timestamp","open","high","low","close","volume"]],
                                      tf_aux)
            if leg_aux.empty or spot_aux.empty:
                continue
            merged = pd.merge_asof(
                leg_aux[["timestamp","close"]].rename(columns={"close":"opt"}),
                spot_aux[["timestamp","close"]].rename(columns={"close":"spot"}),
                on="timestamp", direction="backward",
            )
            ivs = [
                implied_vol(row["opt"], row["spot"], strike,
                            years_to_expiry(row["timestamp"], expiry), option_type=opt_type)
                for _, row in merged.iterrows()
            ]
            merged["iv"] = ivs
            merged = merged.dropna(subset=["iv"])
            if merged.empty:
                continue
            color = PRIMARY if opt_type == "CE" else PURPLE
            fig.add_trace(go.Scatter(
                x=merged["timestamp"], y=merged["iv"] * 100,
                name=f"{strike} {opt_type} IV", mode="lines",
                line=dict(color=color, width=1.4),
                hovertemplate=(f"{strike} {opt_type}<br>"
                               "t=%{x|%H:%M}<br>"
                               "IV=%{y:.2f}%<extra></extra>"),
            ), row=iv_row, col=1)

    # ── OI pane ──
    if show_oi and not oi_full.empty:
        oi_row = 2 + int(show_trades) + int(show_iv)
        for strike, opt_type in legs:
            leg_oi = oi_full[(oi_full["expiry"] == expiry) &
                             (oi_full["strike"] == strike) &
                             (oi_full["option_type"] == opt_type)]
            if leg_oi.empty:
                continue
            oi_series = resample_series(leg_oi, "oi", tf_aux, agg="last")
            color = PRIMARY if opt_type == "CE" else PURPLE
            fig.add_trace(go.Scatter(
                x=oi_series["timestamp"], y=oi_series["oi"],
                name=f"{strike} {opt_type} OI", mode="lines",
                line=dict(color=color, width=1.2, dash="dot"),
                hovertemplate=f"{strike} {opt_type}<br>OI=%{{y:,.0f}}<extra></extra>",
            ), row=oi_row, col=1)

    # ── Layout polish ──
    fig.update_layout(
        template="plotly_dark",
        height=820,
        paper_bgcolor=INK, plot_bgcolor=INK,
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        # spot overlay
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    title=dict(text="NIFTY spot", font=dict(color=AMBER)),
                    tickfont=dict(color=AMBER)),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["15:30", "09:15"])])

    # Hide candle rangesliders on every row
    for i in range(1, n_rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor="#1e2030", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#1e2030", row=i, col=1)

    return fig


# ─── Snapshot table ───────────────────────────────────────────────────────────
def chain_snapshot(date_str: str, expiry: str, cursor: pd.Timestamp,
                   atm_band: int = 8) -> pd.DataFrame:
    opt = load_options(date_str)
    oi  = load_oi(date_str)
    spot = load_spot(date_str)
    if opt.empty or spot.empty:
        return pd.DataFrame()

    spot_at = spot[spot["timestamp"] <= cursor]
    if spot_at.empty:
        return pd.DataFrame()
    spot_px = float(spot_at.iloc[-1]["close"])

    snap = (opt[(opt["expiry"] == expiry) &
                (opt["timestamp"] >= cursor - pd.Timedelta("3min")) &
                (opt["timestamp"] <= cursor + pd.Timedelta("3min"))]
            .sort_values("timestamp")
            .groupby(["strike", "option_type"])
            .agg(close=("close", "last"),
                 volume=("volume", "sum"))
            .reset_index())

    if snap.empty:
        return pd.DataFrame()

    oi_snap = (oi[(oi["expiry"] == expiry) &
                  (oi["timestamp"] >= cursor - pd.Timedelta("3min")) &
                  (oi["timestamp"] <= cursor + pd.Timedelta("3min"))]
               .sort_values("timestamp")
               .groupby(["strike", "option_type"])
               .agg(oi=("oi", "last"))
               .reset_index()) if not oi.empty else pd.DataFrame()
    if not oi_snap.empty:
        snap = snap.merge(oi_snap, on=["strike", "option_type"], how="left")

    # Restrict to ATM ± band
    strikes = sorted(snap["strike"].unique())
    atm = min(strikes, key=lambda k: abs(k - spot_px))
    step = (strikes[1] - strikes[0]) if len(strikes) > 1 else 50
    lo, hi = atm - atm_band * step, atm + atm_band * step
    snap = snap[(snap["strike"] >= lo) & (snap["strike"] <= hi)]

    # Pivot to CE | strike | PE
    ce = snap[snap["option_type"] == "CE"].set_index("strike")
    pe = snap[snap["option_type"] == "PE"].set_index("strike")
    all_strikes = sorted(set(ce.index).union(pe.index))
    rows = []
    for k in all_strikes:
        rows.append({
            "CE OI"  : ce.loc[k, "oi"]     if k in ce.index and "oi" in ce else None,
            "CE Vol" : ce.loc[k, "volume"] if k in ce.index else None,
            "CE LTP" : ce.loc[k, "close"]  if k in ce.index else None,
            "Strike" : k,
            "PE LTP" : pe.loc[k, "close"]  if k in pe.index else None,
            "PE Vol" : pe.loc[k, "volume"] if k in pe.index else None,
            "PE OI"  : pe.loc[k, "oi"]     if k in pe.index and "oi" in pe else None,
        })
    return pd.DataFrame(rows)


# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='background:{PANEL};padding:10px 18px;border-bottom:1px solid #2a2d3a;"
    f"display:flex;gap:18px;align-items:center;'>"
    f"<span style='font-size:18px;font-weight:700;color:#fff;'>⚡ NIFTY Option Chain Replay</span>"
    f"<span style='color:#6b7280;font-size:12px;'>"
    f"advanced visualization · multi-leg overlay · live IV · independent timeframes"
    f"</span></div>",
    unsafe_allow_html=True,
)

trading_days = list_trading_days()
if not trading_days:
    st.error(f"No spot data found at {SPOT_DIR}")
    st.stop()

# Default to a known day with data (or last available)
default_day = "20240321" if "20240321" in trading_days else trading_days[-1]

with st.sidebar:
    st.markdown("### Day")
    date_str = st.selectbox(
        "Trading day",
        trading_days,
        index=trading_days.index(default_day),
        format_func=lambda s: f"{s[:4]}-{s[4:6]}-{s[6:]}",
    )

    st.markdown("### Expiry")
    expiries = list_expiries_on(date_str)
    if not expiries:
        st.error("No options data for this day.")
        st.stop()
    expiry = st.selectbox("Expiry", expiries, index=0)

    st.markdown("### Strikes")
    strikes = list_strikes(date_str, expiry)
    if not strikes:
        st.error("No strikes for this expiry.")
        st.stop()

    spot_now = load_spot(date_str)
    atm_default = strikes[len(strikes) // 2]
    if not spot_now.empty:
        atm_default = min(strikes, key=lambda k: abs(k - float(spot_now.iloc[-1]["close"])))

    # default 3 strikes centered on ATM
    idx0 = max(strikes.index(atm_default) - 1, 0)
    default_strikes = strikes[idx0: idx0 + 3]
    pick_strikes = st.multiselect("Pick strikes", strikes, default=default_strikes)
    pick_types   = st.multiselect("Side", ["CE", "PE"], default=["CE", "PE"])

    legs = [(k, t) for k in pick_strikes for t in pick_types]
    if not legs:
        st.warning("Select at least one strike and side.")
        st.stop()

    st.markdown("### Timeframes")
    tf_price = st.radio("Price pane", list(TIMEFRAMES), index=2, horizontal=True)
    tf_aux   = st.radio("IV / OI pane", list(TIMEFRAMES), index=3, horizontal=True)

    st.markdown("### Panes")
    show_trades = st.checkbox("Trade markers + volume",  value=True)
    show_iv     = st.checkbox("Implied vol",              value=True)
    show_oi     = st.checkbox("Open interest",            value=True)
    show_spot   = st.checkbox("Spot overlay",             value=True)

    st.markdown("### Window")
    t_from = st.time_input("From", value=SESSION_OPEN, step=300)
    t_to   = st.time_input("To",   value=SESSION_CLOSE, step=300)


# ─── Main: chart ──────────────────────────────────────────────────────────────
with st.spinner("Rendering…"):
    fig = build_chart(
        date_str=date_str, expiry=expiry, legs=legs,
        tf_price=tf_price, tf_aux=tf_aux,
        show_iv=show_iv, show_oi=show_oi,
        show_trades=show_trades, show_spot=show_spot,
        t_from=t_from, t_to=t_to,
    )
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# ─── Footer: snapshot table at last cursor time ──────────────────────────────
st.markdown("### Chain snapshot")
day = pd.to_datetime(date_str, format="%Y%m%d").date()
cursor = pd.Timestamp.combine(day, t_to)
snap = chain_snapshot(date_str, expiry, cursor)
if snap.empty:
    st.caption("No chain snapshot available at cursor time.")
else:
    fmt = {
        "CE OI" : "{:,.0f}", "CE Vol": "{:,.0f}", "CE LTP": "{:.2f}",
        "Strike": "{:.0f}",
        "PE LTP": "{:.2f}",  "PE Vol": "{:,.0f}", "PE OI" : "{:,.0f}",
    }
    spot_at = spot_now[spot_now["timestamp"] <= cursor]
    spot_px = float(spot_at.iloc[-1]["close"]) if not spot_at.empty else None
    if spot_px is not None:
        atm = min(snap["Strike"], key=lambda k: abs(k - spot_px))
        st.caption(f"Cursor: **{cursor:%H:%M}** · spot **{spot_px:.1f}** · ATM **{int(atm)}**")
        styled = (snap.style
                       .format(fmt, na_rep="—")
                       .apply(lambda r: ["background-color:#1e3a1e" if r["Strike"] == atm else ""
                                          for _ in r], axis=1))
    else:
        styled = snap.style.format(fmt, na_rep="—")
    st.dataframe(styled, use_container_width=True, hide_index=True)
