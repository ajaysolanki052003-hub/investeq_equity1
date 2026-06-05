"""
EMA Crossover + Swing-Low SL — TradingView-style Visualizer
============================================================
Uses streamlit-lightweight-charts (the official TradingView Lightweight Charts library)
for a true TradingView look-and-feel.

Run with:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_lightweight_charts import renderLightweightCharts

# ───────────────────────── PAGE CONFIG ─────────────────────────
st.set_page_config(
    page_title='EMA Swing-Low Strategy',
    layout='wide',
    initial_sidebar_state='expanded',
)

# inject a tiny bit of CSS so the chart container is dark all the way to the edges
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stApp { background-color: #0e0e10; }
    h1, h2, h3, h4, h5 { color: #e0e0e0; }
    div[data-testid="stMetricValue"] { color: #e0e0e0; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title('EMA Crossover + Swing-Low SL — TradingView-style Chart')

# ───────────────────────── SYMBOLS ─────────────────────────
@st.cache_data
def load_symbols():
    path = 'nifty500_symbols.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        col = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
        return sorted(df[col].dropna().astype(str).str.strip().str.upper().unique().tolist())
    return ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']

# ───────────────────────── STRATEGY ─────────────────────────
def add_emas(df, fast, slow):
    df = df.copy()
    df[f'EMA{fast}'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df[f'EMA{slow}'] = df['Close'].ewm(span=slow, adjust=False).mean()
    return df

def bullish_cross_indices(df, fast, slow):
    above = df[f'EMA{fast}'] > df[f'EMA{slow}']
    cross = above & ~above.shift(1, fill_value=False)
    cross.iloc[0] = False
    return list(np.where(cross.values)[0])

def scan_signals(symbol, df, fast, slow, compulsory_touches=2):
    """Scan EMA-crossover touches with a higher-swing-high uptrend gate.

    Per bullish cross:
      - First `compulsory_touches` qualifying touches are ALWAYS taken.
      - From the (compulsory+1)-th touch onwards, the touch is taken only if the
        highest high between the last taken touch and this candidate exceeds the
        highest high between the two prior taken touches — i.e. price made a new
        higher high since the last touch (Dow-theory uptrend).
      - Cross window ends when fast EMA crosses back below slow EMA.
    """
    df = add_emas(df, fast, slow)
    rows = []
    highs = df['High'].values
    for cidx in bullish_cross_indices(df, fast, slow):
        cross_ts = df.index[cidx]
        taken_idx = []  # bar indices of touches taken so far in this cross
        for i in range(cidx + 1, len(df)):
            low   = float(df['Low'].iloc[i]);   close = float(df['Close'].iloc[i])
            open_ = float(df['Open'].iloc[i]);  high  = float(df['High'].iloc[i])
            ef = float(df[f'EMA{fast}'].iloc[i]); es = float(df[f'EMA{slow}'].iloc[i])
            if ef <= es:
                break
            if not (low <= ef and close > ef and close > open_):
                continue

            # Uptrend gate (only after the compulsory count is met) ─────────────
            if len(taken_idx) >= compulsory_touches:
                last_t = taken_idx[-1]
                prev_t = taken_idx[-2]
                hi_recent = highs[last_t + 1 : i + 1].max() if i > last_t else -np.inf
                hi_prior  = highs[prev_t + 1 : last_t + 1].max() if last_t > prev_t else -np.inf
                if not (hi_recent > hi_prior):
                    continue  # uptrend not confirmed — skip but keep scanning

            rows.append({
                'Stock': symbol,
                'Cross Time': cross_ts,
                'Touch Time': df.index[i],
                'Touch Idx':  i,
                'Touch Number': len(taken_idx) + 1,
                'Touch Open': open_, 'Touch High': high,
                'Touch Low':  low,   'Touch Close': close,
                f'EMA{fast}': ef,    f'EMA{slow}': es,
            })
            taken_idx.append(i)
    return pd.DataFrame(rows), df

def compute_atr(df, period=14):
    """Wilder's ATR — TR smoothed with EMA(alpha=1/period). Returns a Series
    aligned with df.index (first `period-1` values are NaN)."""
    h = df['High'].astype(float); l = df['Low'].astype(float); c = df['Close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def compute_swing_lows(df, window):
    n = len(df)
    lows = df['Low'].values
    swing = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        left  = lows[i-window:i].min()
        right = lows[i+1:i+window+1].min()
        if lows[i] < left and lows[i] < right:
            swing[i] = True
    return swing

def find_swing_sl(df, touch_idx, swing_mask):
    if swing_mask[touch_idx]:
        return float(df['Low'].iloc[touch_idx]), 'touch is swing low'
    prior = np.where(swing_mask[:touch_idx])[0]
    if len(prior) > 0:
        s = prior[-1]
        return float(df['Low'].iloc[s]), f'prior swing low @ {df.index[s].strftime("%Y-%m-%d %H:%M")}'
    return float(df['Low'].iloc[touch_idx]), 'fallback to touch low'

def evaluate_trade(df, idx, entry, sl_price, target_price):
    """LONG trade evaluation. Target above entry, SL below."""
    target_hit = sl_hit = False
    target_idx = sl_idx = None
    max_high = float('-inf')
    for j in range(idx + 1, len(df)):
        hi = float(df['High'].iloc[j]); lo = float(df['Low'].iloc[j])
        max_high = max(max_high, hi)
        if hi >= target_price and not target_hit:
            target_hit = True; target_idx = j
        if lo <= sl_price and not sl_hit:
            sl_hit = True; sl_idx = j
        if target_hit or sl_hit:
            break
    if target_hit and sl_hit:
        if target_idx < sl_idx:    outcome, exit_idx = 'TARGET HIT', target_idx
        elif sl_idx < target_idx:  outcome, exit_idx = 'SL HIT', sl_idx
        else:                      outcome, exit_idx = 'AMBIGUOUS', min(target_idx, sl_idx)
    elif target_hit: outcome, exit_idx = 'TARGET HIT', target_idx
    elif sl_hit:     outcome, exit_idx = 'SL HIT', sl_idx
    else:            outcome, exit_idx = 'OPEN', len(df) - 1
    return outcome, exit_idx, max_high

# ───────── SELL-SIDE (SHORT) — mirror of the long logic ─────────
def bearish_cross_indices(df, fast, slow):
    below = df[f'EMA{fast}'] < df[f'EMA{slow}']
    cross = below & ~below.shift(1, fill_value=False)
    cross.iloc[0] = False
    return list(np.where(cross.values)[0])

def scan_sell_signals(symbol, df_ema, fast, slow, compulsory_touches=2):
    """Bearish-cross touches with a lower-swing-low downtrend gate from touch
    #(compulsory+1) onwards. `df_ema` must already have EMA columns."""
    rows = []
    lows = df_ema['Low'].values
    for cidx in bearish_cross_indices(df_ema, fast, slow):
        cross_ts = df_ema.index[cidx]
        taken_idx = []
        for i in range(cidx + 1, len(df_ema)):
            low   = float(df_ema['Low'].iloc[i]);   close = float(df_ema['Close'].iloc[i])
            open_ = float(df_ema['Open'].iloc[i]);  high  = float(df_ema['High'].iloc[i])
            ef = float(df_ema[f'EMA{fast}'].iloc[i]); es = float(df_ema[f'EMA{slow}'].iloc[i])
            if ef >= es:                          # EMAs flipped back bullish — close window
                break
            if not (high >= ef and close < ef and close < open_):
                continue
            if len(taken_idx) >= compulsory_touches:
                last_t = taken_idx[-1]; prev_t = taken_idx[-2]
                lo_recent = lows[last_t + 1 : i + 1].min() if i > last_t else np.inf
                lo_prior  = lows[prev_t + 1 : last_t + 1].min() if last_t > prev_t else np.inf
                if not (lo_recent < lo_prior):    # downtrend not confirmed
                    continue
            rows.append({
                'Stock': symbol,
                'Cross Time': cross_ts,
                'Touch Time': df_ema.index[i],
                'Touch Idx':  i,
                'Touch Number': len(taken_idx) + 1,
                'Touch Open': open_, 'Touch High': high,
                'Touch Low':  low,   'Touch Close': close,
                f'EMA{fast}': ef,    f'EMA{slow}': es,
            })
            taken_idx.append(i)
    return pd.DataFrame(rows)

def compute_swing_highs(df, window):
    n = len(df)
    highs = df['High'].values
    swing = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        left  = highs[i-window:i].max()
        right = highs[i+1:i+window+1].max()
        if highs[i] > left and highs[i] > right:
            swing[i] = True
    return swing

def find_swing_sh(df, touch_idx, swing_mask):
    """SL for shorts: nearest prior swing high (or touch's own high if touch is one)."""
    if swing_mask[touch_idx]:
        return float(df['High'].iloc[touch_idx]), 'touch is swing high'
    prior = np.where(swing_mask[:touch_idx])[0]
    if len(prior) > 0:
        s = prior[-1]
        return float(df['High'].iloc[s]), f'prior swing high @ {df.index[s].strftime("%Y-%m-%d %H:%M")}'
    return float(df['High'].iloc[touch_idx]), 'fallback to touch high'

def evaluate_trade_short(df, idx, entry, sl_price, target_price):
    """SHORT trade evaluation. Target BELOW entry, SL ABOVE."""
    target_hit = sl_hit = False
    target_idx = sl_idx = None
    min_low = float('inf')
    for j in range(idx + 1, len(df)):
        hi = float(df['High'].iloc[j]); lo = float(df['Low'].iloc[j])
        min_low = min(min_low, lo)
        if lo <= target_price and not target_hit:
            target_hit = True; target_idx = j
        if hi >= sl_price and not sl_hit:
            sl_hit = True; sl_idx = j
        if target_hit or sl_hit:
            break
    if target_hit and sl_hit:
        if target_idx < sl_idx:    outcome, exit_idx = 'TARGET HIT', target_idx
        elif sl_idx < target_idx:  outcome, exit_idx = 'SL HIT', sl_idx
        else:                      outcome, exit_idx = 'AMBIGUOUS', min(target_idx, sl_idx)
    elif target_hit: outcome, exit_idx = 'TARGET HIT', target_idx
    elif sl_hit:     outcome, exit_idx = 'SL HIT', sl_idx
    else:            outcome, exit_idx = 'OPEN', len(df) - 1
    return outcome, exit_idx, min_low

# ───────────────────────── DATA FETCH ─────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol, interval, period):
    ticker = symbol if symbol.endswith('.NS') else f'{symbol}.NS'
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=False)
    except Exception as e:
        return None, str(e)
    if df is None or df.empty:
        return None, 'no data returned'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[keep].dropna()
    return df, None

# ───────────────────────── HELPERS ─────────────────────────
def to_ts(t):
    """pandas Timestamp -> int UNIX seconds, treating naive as local-displayed UTC."""
    pt = pd.Timestamp(t)
    if pt.tz is not None:
        pt = pt.tz_localize(None)
    return int(pt.value // 10**9)

# ───────────────────────── PER-STOCK RECOMMENDATIONS ─────────────────────────
@st.cache_data
def load_recommendations():
    """Load stock_method_map.csv produced by analyze_stock_methods.py.
    Returns DataFrame or None if file missing."""
    path = 'stock_method_map.csv'
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_oos_check():
    path = 'stock_oos_check.csv'
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_validated_symbols():
    """Return sorted list of NSE symbols that survived OOS validation."""
    path = 'stock_method_map_validated.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return sorted(df.Symbol.dropna().astype(str).str.upper().unique().tolist())

METHOD_TO_LABEL = {'target_pct': 'Target %', 'R': 'R-multiple', 'ATR': 'ATR'}

def get_rec(recs_df, symbol, side):
    if recs_df is None or recs_df.empty:
        return None
    sub = recs_df[(recs_df.Symbol == symbol) & (recs_df.Side == side)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()

# ───────────────────────── SIDEBAR ─────────────────────────
symbols = load_symbols()
recs_df = load_recommendations()
oos_df  = load_oos_check()
validated_symbols = load_validated_symbols()

with st.sidebar:
    st.header('Settings')

    mode_opts = ['Dropdown', 'Type symbol']
    if validated_symbols:
        mode_opts.append('OOS-validated only')
    pick_mode = st.radio('Symbol input', mode_opts, horizontal=True)
    if pick_mode == 'Dropdown':
        default_idx = symbols.index('RELIANCE') if 'RELIANCE' in symbols else 0
        symbol = st.selectbox('Stock (NSE)', symbols, index=default_idx)
    elif pick_mode == 'Type symbol':
        symbol = st.text_input('NSE Symbol (no .NS suffix)', value='RELIANCE').strip().upper()
    else:  # OOS-validated only
        st.caption(f'🏆 {len(validated_symbols)} symbols that passed both in-sample edge '
                   'AND out-of-sample survival checks.')
        default_idx = (validated_symbols.index('RELIANCE')
                       if 'RELIANCE' in validated_symbols else 0)
        symbol = st.selectbox('OOS-validated NSE stocks', validated_symbols, index=default_idx)

    interval = st.selectbox('Timeframe', ['1d', '1h', '15m'], index=0, key='interval')

    # Build period options for this interval
    if interval == '1h':
        period_opts = ['30d', '60d', '90d', '180d', '360d', '720d']
        default_oos = '720d'    # longest available
        default_reg = '180d'
    elif interval == '15m':
        period_opts = ['7d', '15d', '30d', '60d']
        default_oos = '60d'
        default_reg = '30d'
    else:  # 1d
        period_opts = ['6mo', '1y', '2y', '3y', '5y', '10y', 'max']
        default_oos = '5y'      # covers full 2020-2026 analysis window
        default_reg = '2y'

    # Auto-bump history when symbol just changed AND that symbol is OOS-validated.
    # This makes the chart visualize the SAME time-period the analysis used.
    is_validated_now = (validated_symbols is not None
                        and symbol in validated_symbols)
    period_apply_key = (symbol, interval)
    if st.session_state.get('_last_period_apply') != period_apply_key:
        if is_validated_now:
            st.session_state['history_period'] = default_oos
        elif st.session_state.get('history_period') not in period_opts:
            # interval changed and old period is no longer valid for this interval
            st.session_state['history_period'] = default_reg
        st.session_state['_last_period_apply'] = period_apply_key

    # Final safety: if session_state has a value not in current options, fall back
    if st.session_state.get('history_period') not in period_opts:
        st.session_state['history_period'] = default_reg

    period = st.selectbox('History', period_opts, key='history_period')
    if is_validated_now:
        st.caption(f'📐 History auto-set to **{period}** to match the analysis '
                   'window for this OOS-validated stock.')

    st.markdown('---')
    st.subheader('Strategy params')

    # ─── Initialize every strategy-param key in session_state with a default ───
    # so the apply logic below never collides with widget-default arguments.
    _DEFAULTS = {
        'side_mode':    'Both',
        'fast_ema':     21,
        'slow_ema':     50,
        'swing_window': 2,
        'target_mode':  'Target %',
        'target_pct':   5.0,
        'R_mult':       2.0,
        'atr_period':   14,
        'atr_mult':     2.0,
        'auto_apply':   True,
    }
    for _k, _v in _DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ─── Read CURRENT side_mode / auto_apply directly from session_state so the
    # apply logic can run BEFORE any controlled widget renders. ───
    side_mode_now  = st.session_state['side_mode']
    auto_apply_now = st.session_state['auto_apply']

    # ─── Per-stock recommendation banner ───
    rec_buy  = get_rec(recs_df, symbol, 'BUY')
    rec_sell = get_rec(recs_df, symbol, 'SELL')
    if recs_df is None:
        st.caption('ℹ️ `stock_method_map.csv` not found — recommendations unavailable. '
                   'Run `analyze_stock_methods.py` to generate it.')
    else:
        def fmt_rec(rec, side_label, emoji):
            if rec is None:
                return f'{emoji} **{side_label}** — no stable recommendation for this stock.'
            survives = ''
            if oos_df is not None:
                o = oos_df[(oos_df.Symbol == symbol) &
                           (oos_df.Side == ('BUY' if side_label == 'BUY' else 'SELL'))]
                if not o.empty:
                    survives = '  ·  ✅ OOS-validated' if bool(o.iloc[0].oos_survives) \
                               else '  ·  ⚠️ OOS decay'
            return (f"{emoji} **{side_label}** — {METHOD_TO_LABEL.get(rec['Method'], rec['Method'])} "
                    f"`{rec['Param']:g}`  ·  preset *{rec['Preset']}*  ·  "
                    f"{rec['Trades']} trades  ·  exp **{rec['Expectancy_IS_OOS_mixed']:+.2f}%/trade**  "
                    f"·  stability {rec['Stability']}{survives}")
        st.markdown(fmt_rec(rec_buy,  'BUY',  '📈'))
        st.markdown(fmt_rec(rec_sell, 'SELL', '📉'))

    # ─── Apply recommendation BEFORE rendering controlled widgets ───
    # Choose which side's rec to apply (BUY > SELL > none for "Both")
    rec_for_apply = rec_buy if side_mode_now != 'Sell only' else rec_sell
    if rec_for_apply is None and side_mode_now == 'Both':
        rec_for_apply = rec_sell
    current_apply_key = (symbol, side_mode_now, auto_apply_now)
    last_apply_key    = st.session_state.get('_last_auto_applied')
    applied_this_run  = False
    if auto_apply_now and rec_for_apply is not None and current_apply_key != last_apply_key:
        method_label = METHOD_TO_LABEL.get(rec_for_apply['Method'])
        if method_label:
            st.session_state['target_mode'] = method_label
        param = float(rec_for_apply['Param'])
        if rec_for_apply['Method'] == 'target_pct':
            st.session_state['target_pct'] = round(min(max(param, 0.5), 25.0), 2)
        elif rec_for_apply['Method'] == 'R':
            st.session_state['R_mult'] = round(min(max(param, 0.5), 10.0), 2)
        elif rec_for_apply['Method'] == 'ATR':
            st.session_state['atr_mult'] = round(min(max(param, 0.25), 10.0), 2)
        st.session_state['_last_auto_applied'] = current_apply_key
        applied_this_run = True

    if applied_this_run:
        st.success(
            f"✅ Applied recommendation for {symbol} "
            f"({rec_for_apply.get('Side', '?')}): "
            f"**{METHOD_TO_LABEL.get(rec_for_apply['Method'], rec_for_apply['Method'])}** "
            f"= `{rec_for_apply['Param']:g}`",
            icon='✅')

    # ─── Render side / auto-apply widgets — pure session_state-driven ───
    side_mode = st.radio('Side', ['Buy only', 'Sell only', 'Both'],
                         horizontal=True, key='side_mode')
    auto_apply = st.checkbox('🤖 Auto-apply recommended target mode + param',
                             key='auto_apply',
                             help='When a new symbol is picked, the target-mode/param widgets '
                                  'below are auto-set to that stock’s recommendation from '
                                  'stock_method_map.csv. Uncheck to set them manually.')

    c1, c2 = st.columns(2)
    fast = c1.number_input('Fast EMA', min_value=2, max_value=200, key='fast_ema')
    slow = c2.number_input('Slow EMA', min_value=3, max_value=400, key='slow_ema')

    swing_window = st.number_input('Swing window (±)', min_value=1, max_value=10,
                                   key='swing_window')
    target_mode  = st.radio('Target mode', ['Target %', 'R-multiple', 'ATR'],
                            horizontal=True, key='target_mode',
                            help='Target % = fixed percentage off entry. '
                                 'R-multiple = target distance = SL distance × R. '
                                 'ATR = target distance = ATR(period) × multiplier '
                                 '(volatility-scaled, independent of SL).')
    target_pct   = st.slider('Target %', min_value=0.5, max_value=25.0, step=0.25,
                             disabled=target_mode != 'Target %', key='target_pct')
    R_mult       = st.slider('R-multiple', min_value=0.5, max_value=10.0, step=0.25,
                             disabled=target_mode != 'R-multiple', key='R_mult',
                             help='Target = Entry ± (SL distance × R)')
    cA, cB = st.columns(2)
    atr_period = cA.number_input('ATR period', min_value=2, max_value=100,
                                 disabled=target_mode != 'ATR', key='atr_period')
    atr_mult   = cB.number_input('ATR multiplier', min_value=0.25, max_value=10.0,
                                 step=0.25, disabled=target_mode != 'ATR', key='atr_mult',
                                 help='Target = Entry ± ATR × multiplier')

    st.markdown('---')
    st.subheader('Display')
    show_only_decided = st.checkbox('Hide OPEN trades', value=False)
    show_sl_target    = st.checkbox('Show SL / Target lines', value=True)
    show_swing_lows   = st.checkbox('Mark swing lows', value=False)
    show_volume       = st.checkbox('Show volume pane', value=True)
    show_markers      = st.checkbox('Show entry markers', value=True)
    chart_height      = st.slider('Chart height (px)', 400, 1000, 620, 20)

# ───────────────────────── MAIN ─────────────────────────
if not symbol:
    st.info('Enter a symbol to begin.')
    st.stop()
if fast >= slow:
    st.error('Fast EMA must be < Slow EMA.')
    st.stop()

with st.spinner(f'Fetching {symbol} {interval}/{period}...'):
    df, err = fetch_data(symbol, interval, period)

if df is None:
    st.error(f'No data for `{symbol}.NS` ({err}). Check symbol or try a longer period.')
    st.stop()
if len(df) < slow + 10:
    st.warning(f'Only {len(df)} candles; need more history for EMA{slow}.')

signals, df_ema = scan_signals(symbol, df, fast, slow, compulsory_touches=2)
swing_mask  = compute_swing_lows(df_ema, swing_window)
swing_mask_h = compute_swing_highs(df_ema, swing_window)
sell_signals = (scan_sell_signals(symbol, df_ema, fast, slow, compulsory_touches=2)
                if side_mode in ('Sell only', 'Both') else pd.DataFrame())
atr_series   = compute_atr(df_ema, int(atr_period)) if target_mode == 'ATR' else None

trades = []
skipped = 0

# ── LONG (BUY) trades ──
if side_mode in ('Buy only', 'Both'):
  for _, sig in signals.iterrows():
    idx   = int(sig['Touch Idx'])
    entry = float(sig['Touch Close'])
    sl_price, sl_source = find_swing_sl(df_ema, idx, swing_mask)
    if sl_price >= entry:
        skipped += 1
        continue
    sl_pct       = (entry - sl_price) / entry * 100
    if target_mode == 'R-multiple':
        sl_dist          = entry - sl_price
        target_price     = entry + sl_dist * R_mult
        target_pct_eff   = sl_pct * R_mult
    elif target_mode == 'ATR':
        atr_v = float(atr_series.iloc[idx]) if atr_series is not None else float('nan')
        if not np.isfinite(atr_v) or atr_v <= 0:
            skipped += 1
            continue
        target_price     = entry + atr_v * atr_mult
        target_pct_eff   = (target_price - entry) / entry * 100
    else:
        target_price     = entry * (1 + target_pct / 100)
        target_pct_eff   = target_pct
    outcome, exit_idx, max_high = evaluate_trade(df_ema, idx, entry, sl_price, target_price)
    pnl_pct = (target_pct_eff if outcome == 'TARGET HIT'
               else -sl_pct if outcome == 'SL HIT'
               else (float(df_ema['Close'].iloc[-1]) - entry) / entry * 100)
    trades.append({
        'Side':         'BUY',
        'Touch Time':   sig['Touch Time'],
        'Cross Time':   sig['Cross Time'],
        'Entry':        round(entry, 2),
        'SL Price':     round(sl_price, 2),
        'SL %':         round(sl_pct, 2),
        'SL Source':    sl_source,
        'Target Price': round(target_price, 2),
        'R:R':          round(target_pct_eff / sl_pct, 2) if sl_pct > 0 else None,
        'Outcome':      outcome,
        'Exit Time':    df_ema.index[exit_idx],
        'Best Price':   round(max_high, 2) if max_high != float('-inf') else None,
        'PnL %':        round(pnl_pct, 2),
    })

# ── SHORT (SELL) trades ──
if side_mode in ('Sell only', 'Both'):
  for _, sig in sell_signals.iterrows():
    idx   = int(sig['Touch Idx'])
    entry = float(sig['Touch Close'])
    sl_price, sl_source = find_swing_sh(df_ema, idx, swing_mask_h)
    if sl_price <= entry:                                  # no room above
        skipped += 1
        continue
    sl_pct       = (sl_price - entry) / entry * 100        # positive
    if target_mode == 'R-multiple':
        sl_dist          = sl_price - entry
        target_price     = entry - sl_dist * R_mult
        target_pct_eff   = sl_pct * R_mult
    elif target_mode == 'ATR':
        atr_v = float(atr_series.iloc[idx]) if atr_series is not None else float('nan')
        if not np.isfinite(atr_v) or atr_v <= 0:
            skipped += 1
            continue
        target_price     = entry - atr_v * atr_mult
        target_pct_eff   = (entry - target_price) / entry * 100
    else:
        target_price     = entry * (1 - target_pct / 100)
        target_pct_eff   = target_pct
    outcome, exit_idx, min_low = evaluate_trade_short(df_ema, idx, entry, sl_price, target_price)
    pnl_pct = (target_pct_eff if outcome == 'TARGET HIT'
               else -sl_pct if outcome == 'SL HIT'
               else (entry - float(df_ema['Close'].iloc[-1])) / entry * 100)
    trades.append({
        'Side':         'SELL',
        'Touch Time':   sig['Touch Time'],
        'Cross Time':   sig['Cross Time'],
        'Entry':        round(entry, 2),
        'SL Price':     round(sl_price, 2),
        'SL %':         round(sl_pct, 2),
        'SL Source':    sl_source,
        'Target Price': round(target_price, 2),
        'R:R':          round(target_pct_eff / sl_pct, 2) if sl_pct > 0 else None,
        'Outcome':      outcome,
        'Exit Time':    df_ema.index[exit_idx],
        'Best Price':   round(min_low, 2) if min_low != float('inf') else None,
        'PnL %':        round(pnl_pct, 2),
    })

trades_df = pd.DataFrame(trades)
if not trades_df.empty:
    trades_df = trades_df.sort_values('Touch Time').reset_index(drop=True)

# ───────────────────────── METRICS ─────────────────────────
if not trades_df.empty:
    total = len(trades_df)
    n_buy  = (trades_df['Side'] == 'BUY').sum()
    n_sell = (trades_df['Side'] == 'SELL').sum()
    tgt   = (trades_df['Outcome'] == 'TARGET HIT').sum()
    sl_c  = (trades_df['Outcome'] == 'SL HIT').sum()
    op    = (trades_df['Outcome'] == 'OPEN').sum()
    decided = tgt + sl_c
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric('Signals', f'{total}', f'-{skipped} skipped' if skipped else None)
    c2.metric('Buy / Sell', f'{int(n_buy)} / {int(n_sell)}')
    c3.metric('Target Hits', int(tgt))
    c4.metric('SL Hits', int(sl_c))
    c5.metric('Win Rate', f'{tgt/decided*100:.1f}%' if decided else 'n/a')
    c6.metric('Avg PnL', f'{trades_df["PnL %"].mean():.2f}%')
    c7.metric('Total PnL', f'{trades_df["PnL %"].sum():.2f}%')

# ───────────────────────── FOCUS-ON-TRADE NAV ─────────────────────────
focused_idx = None
if not trades_df.empty:
    opts = ['Show all trades']
    for i, r in trades_df.reset_index(drop=True).iterrows():
        tt = pd.Timestamp(r['Touch Time'])
        opts.append(
            f"#{i+1:>3}  ·  [{r['Side']:<4}]  {tt.strftime('%Y-%m-%d %H:%M')}  ·  "
            f"{r['Outcome']:<10}  ·  Entry {r['Entry']}  SL {r['SL Price']} "
            f"({r['SL %']}%)  ·  PnL {r['PnL %']:+.2f}%"
        )
    pick = st.selectbox(
        '🎯  Focus chart on a single trade  —  pick to zoom & isolate that SL',
        opts, index=0, key='focus_pick',
    )
    if pick != 'Show all trades':
        focused_idx = opts.index(pick) - 1

# ───────────────────────── TRADINGVIEW-STYLE CHART ─────────────────────────
# De-duplicate index (yfinance sometimes returns duplicates around market open)
df_ema = df_ema[~df_ema.index.duplicated(keep='last')].sort_index()

# When focused, zoom the candle dataset to that trade's [cross, exit] window
# (plus padding) and isolate the SL/Target/Entry lines to just that trade.
if focused_idx is None:
    df_chart = df_ema
    trades_to_plot = trades
    chart_mask = None
else:
    sel = trades_df.iloc[focused_idx]
    cross_ts = pd.Timestamp(sel['Cross Time'])
    exit_ts  = pd.Timestamp(sel['Exit Time'])
    span = exit_ts - cross_ts
    if pd.isna(span) or span.total_seconds() <= 0:
        span = pd.Timedelta(hours=2) if interval != '1d' else pd.Timedelta(days=3)
    pad = max(
        span * 0.30,
        pd.Timedelta(days=3) if interval == '1d' else pd.Timedelta(hours=8),
    )
    t_from = cross_ts - pad
    t_to   = exit_ts + pad
    chart_mask = (df_ema.index >= t_from) & (df_ema.index <= t_to)
    df_chart = df_ema[chart_mask]
    trades_to_plot = [trades[focused_idx]]

# Build datasets — drop any rows with NaN OHLC, keep pure-Python types ---------
candles_data, ema_fast_data, ema_slow_data, time_array = [], [], [], []
for i in range(len(df_chart)):
    o = df_chart['Open'].iloc[i]; h = df_chart['High'].iloc[i]
    l = df_chart['Low'].iloc[i];  c = df_chart['Close'].iloc[i]
    ef = df_chart[f'EMA{fast}'].iloc[i]; es = df_chart[f'EMA{slow}'].iloc[i]
    if any(pd.isna(v) for v in (o, h, l, c, ef, es)):
        continue
    t = to_ts(df_chart.index[i])
    time_array.append(t)
    candles_data.append({'time': t, 'open': float(o), 'high': float(h),
                         'low': float(l), 'close': float(c)})
    ema_fast_data.append({'time': t, 'value': float(ef)})
    ema_slow_data.append({'time': t, 'value': float(es)})

volume_data = []
if show_volume and 'Volume' in df_chart.columns:
    for i in range(len(df_chart)):
        v = df_chart['Volume'].iloc[i]
        c_close = df_chart['Close'].iloc[i]; c_open = df_chart['Open'].iloc[i]
        if pd.isna(v) or pd.isna(c_close) or pd.isna(c_open):
            continue
        volume_data.append({
            'time':  to_ts(df_chart.index[i]),
            'value': float(v),
            'color': 'rgba(38, 166, 154, 0.55)' if c_close >= c_open
                     else 'rgba(239, 83, 80, 0.55)',
        })

# Markers (entries + optional swing lows) on candlestick series
marker_color = {
    'TARGET HIT': '#00e676',
    'SL HIT':     '#ff1744',
    'OPEN':       '#9e9e9e',
    'AMBIGUOUS':  '#ffea00',
}
markers = []
chart_t_min = time_array[0]  if time_array else None
chart_t_max = time_array[-1] if time_array else None
if show_markers:
    for t in trades_to_plot:
        if show_only_decided and t['Outcome'] == 'OPEN':
            continue
        mt = to_ts(t['Touch Time'])
        if chart_t_min is not None and not (chart_t_min <= mt <= chart_t_max):
            continue
        is_short = t.get('Side') == 'SELL'
        markers.append({
            'time':     mt,
            'position': 'aboveBar' if is_short else 'belowBar',
            'color':    marker_color.get(t['Outcome'], '#888'),
            'shape':    'arrowDown' if is_short else 'arrowUp',
            'text':     f"{'S' if is_short else 'L'}·{t['Outcome'][0]} {t['PnL %']:+.1f}%",
            'size':     2,
        })

    if show_swing_lows:
        sw_idx = np.where(swing_mask)[0]
        for s in sw_idx:
            st_t = to_ts(df_ema.index[s])
            if chart_t_min is not None and not (chart_t_min <= st_t <= chart_t_max):
                continue
            markers.append({
                'time':     st_t,
                'position': 'belowBar',
                'color':    'rgba(236, 64, 122, 0.95)',
                'shape':    'circle',
                'text':     'sw',
                'size':     1,
            })

# Markers must be strictly ascending in time with no duplicates
seen_times = set()
deduped = []
for m in sorted(markers, key=lambda m: m['time']):
    if m['time'] in seen_times:
        continue
    seen_times.add(m['time'])
    deduped.append(m)
markers = deduped

# SL / Target horizontal segments — one short line series per trade ------------
sl_target_series = []
if show_sl_target:
    for t in trades_to_plot:
        if show_only_decided and t['Outcome'] == 'OPEN':
            continue
        x0 = to_ts(t['Touch Time']); x1 = to_ts(t['Exit Time'])
        if x1 <= x0: continue
        if chart_t_min is not None:
            x0 = max(x0, chart_t_min)
            x1 = min(x1, chart_t_max)
            if x1 <= x0: continue
        # SL segment (red dashed)
        sl_target_series.append({
            'type': 'Line',
            'data': [{'time': x0, 'value': float(t['SL Price'])},
                     {'time': x1, 'value': float(t['SL Price'])}],
            'options': {
                'color':      'rgba(255, 23, 68, 0.85)',
                'lineWidth':  1,
                'lineStyle':  2,  # dashed
                'priceLineVisible':   False,
                'lastValueVisible':   False,
                'crosshairMarkerVisible': False,
            },
        })
        # Target segment (green dashed)
        sl_target_series.append({
            'type': 'Line',
            'data': [{'time': x0, 'value': float(t['Target Price'])},
                     {'time': x1, 'value': float(t['Target Price'])}],
            'options': {
                'color':      'rgba(0, 230, 118, 0.85)',
                'lineWidth':  1,
                'lineStyle':  2,
                'priceLineVisible':   False,
                'lastValueVisible':   False,
                'crosshairMarkerVisible': False,
            },
        })
        # Entry segment (subtle dotted)
        sl_target_series.append({
            'type': 'Line',
            'data': [{'time': x0, 'value': float(t['Entry'])},
                     {'time': x1, 'value': float(t['Entry'])}],
            'options': {
                'color':      'rgba(255, 255, 255, 0.35)',
                'lineWidth':  1,
                'lineStyle':  1,  # dotted
                'priceLineVisible':   False,
                'lastValueVisible':   False,
                'crosshairMarkerVisible': False,
            },
        })

# Chart options — TradingView dark theme ----------------------------------------
chart_options = {
    'height': chart_height,
    'layout': {
        'background': {'type': 'solid', 'color': '#131722'},
        'textColor':  '#d1d4dc',
    },
    'grid': {
        'vertLines': {'color': 'rgba(42, 46, 57, 0.45)'},
        'horzLines': {'color': 'rgba(42, 46, 57, 0.45)'},
    },
    'crosshair': {'mode': 0},
    'rightPriceScale': {
        'borderColor':  'rgba(197, 203, 206, 0.3)',
        'scaleMargins': {'top': 0.08, 'bottom': 0.28},
    },
    'timeScale': {
        'borderColor':    'rgba(197, 203, 206, 0.3)',
        'timeVisible':    interval in ('1h', '15m'),
        'secondsVisible': False,
        'rightOffset':    8,
        'barSpacing':     8,
    },
    'watermark': {
        'visible':   True,
        'fontSize':  44,
        'color':     'rgba(180, 180, 180, 0.07)',
        'text':      f'{symbol}.NS · {interval}',
        'horzAlign': 'center',
        'vertAlign': 'center',
    },
}

candlestick_series = {
    'type': 'Candlestick',
    'data': candles_data,
    'options': {
        'upColor':       '#26a69a',
        'downColor':     '#ef5350',
        'borderVisible': False,
        'wickUpColor':   '#26a69a',
        'wickDownColor': '#ef5350',
    },
    'markers': markers,
}

ema_fast_series = {
    'type': 'Line',
    'data': ema_fast_data,
    'options': {
        'color':      '#ff9800',
        'lineWidth':  2,
        'title':      f'EMA{fast}',
        'priceLineVisible': False,
        'lastValueVisible': True,
        'crosshairMarkerVisible': False,
    },
}
ema_slow_series = {
    'type': 'Line',
    'data': ema_slow_data,
    'options': {
        'color':      '#42a5f5',
        'lineWidth':  2,
        'title':      f'EMA{slow}',
        'priceLineVisible': False,
        'lastValueVisible': True,
        'crosshairMarkerVisible': False,
    },
}

volume_series = {
    'type': 'Histogram',
    'data': volume_data,
    'options': {
        'priceFormat':  {'type': 'volume'},
        'priceScaleId': '',     # overlay on its own scale
    },
    'priceScale': {
        'scaleMargins': {'top': 0.78, 'bottom': 0},
    },
}

all_series = [candlestick_series, ema_fast_series, ema_slow_series] \
             + sl_target_series \
             + ([volume_series] if volume_data else [])

renderLightweightCharts(
    [{'chart': chart_options, 'series': all_series}],
    key='main_chart',
)

# Tiny legend below chart (chart only shows series titles for line series)
legend_html = (
    "<div style='display:flex; gap:18px; flex-wrap:wrap; "
    "font-family:-apple-system,Segoe UI,Roboto; font-size:12px; color:#bbb; "
    "padding:4px 2px 10px 2px;'>"
    f"<span><span style='color:#26a69a;'>▲</span> bullish candle</span>"
    f"<span><span style='color:#ef5350;'>▼</span> bearish candle</span>"
    f"<span><span style='color:#ff9800;'>━</span> EMA{fast}</span>"
    f"<span><span style='color:#42a5f5;'>━</span> EMA{slow}</span>"
    f"<span><span style='color:#00e676;'>▲</span> Long Target</span>"
    f"<span><span style='color:#ff1744;'>▲</span> Long SL</span>"
    f"<span><span style='color:#00e676;'>▼</span> Short Target</span>"
    f"<span><span style='color:#ff1744;'>▼</span> Short SL</span>"
    f"<span><span style='color:#9e9e9e;'>▲▼</span> Open</span>"
    f"<span><span style='color:rgba(255,23,68,0.85);'>┄</span> SL line</span>"
    f"<span><span style='color:rgba(0,230,118,0.85);'>┄</span> Target line</span>"
    "</div>"
)
st.markdown(legend_html, unsafe_allow_html=True)

# ───────────────────────── TRADES TABLES (Buy | Sell) ─────────────────────────
st.subheader('Trades')
if trades_df.empty:
    st.info('No valid trades for these settings. Try adjusting EMA/target/swing window or a longer history.')
else:
    def color_outcome(v):
        if v == 'TARGET HIT': return 'background-color: #1b5e20; color: #d4ffd4'
        if v == 'SL HIT':     return 'background-color: #b71c1c; color: #ffd4d4'
        if v == 'OPEN':       return 'background-color: #424242; color: #e0e0e0'
        return ''
    def render_table(sub_df, title):
        st.markdown(f"**{title}** · {len(sub_df)} trades · "
                    f"Σ PnL **{sub_df['PnL %'].sum():.2f}%**")
        if sub_df.empty:
            st.caption('— no trades on this side —')
            return
        show = sub_df.copy()
        show['Touch Time'] = show['Touch Time'].astype(str)
        show['Cross Time'] = show['Cross Time'].astype(str)
        show['Exit Time']  = show['Exit Time'].astype(str)
        try:
            styled = show.style.applymap(color_outcome, subset=['Outcome'])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(show, use_container_width=True, hide_index=True)

    col_buy, col_sell = st.columns(2)
    with col_buy:
        render_table(trades_df[trades_df['Side'] == 'BUY'], '📈 LONG (Buy)')
    with col_sell:
        render_table(trades_df[trades_df['Side'] == 'SELL'], '📉 SHORT (Sell)')

    csv_bytes = trades_df.copy().astype(
        {'Touch Time': str, 'Cross Time': str, 'Exit Time': str}
    ).to_csv(index=False).encode()
    mode_tag = (f'r{R_mult}'                       if target_mode == 'R-multiple' else
                f'atr{atr_period}x{atr_mult}'      if target_mode == 'ATR'        else
                f'tp{target_pct}')
    st.download_button('Download all trades CSV', csv_bytes,
                       file_name=f'trades_{symbol}_{interval}_{mode_tag}.csv',
                       mime='text/csv')

# ───────────────────────── SL SUMMARY CHART ─────────────────────────
if not trades_df.empty:
    st.subheader('SL distance per trade')
    st.caption(
        'One bar per trade — bar height = SL % below entry. '
        'Green = target hit, red = SL hit, grey = still open. '
        'The currently-focused trade (selectbox above) is outlined in white.'
    )

    sl_summary_data = []
    sl_focus_outline = []
    for i, r in trades_df.reset_index(drop=True).iterrows():
        outcome = r['Outcome']
        base = ('rgba(0, 230, 118, 0.85)'  if outcome == 'TARGET HIT' else
                'rgba(255, 23, 68, 0.85)'  if outcome == 'SL HIT'     else
                'rgba(158, 158, 158, 0.75)')
        if focused_idx is not None and i == focused_idx:
            # brighter, fully opaque to mark the focused trade
            base = ('rgba(0, 230, 118, 1.0)'  if outcome == 'TARGET HIT' else
                    'rgba(255, 23, 68, 1.0)'  if outcome == 'SL HIT'     else
                    'rgba(220, 220, 220, 1.0)')
        sl_summary_data.append({
            'time':  to_ts(r['Touch Time']),
            'value': float(r['SL %']),
            'color': base,
        })
        if focused_idx is not None and i == focused_idx:
            # white horizontal "marker line" so it stands out even at low SL%
            sl_focus_outline.append({'time': to_ts(r['Touch Time']),
                                     'value': float(r['SL %'])})

    # Strictly ascending, deduped
    sl_summary_data.sort(key=lambda d: d['time'])
    seen_ts = set()
    dedup = []
    for d in sl_summary_data:
        if d['time'] in seen_ts: continue
        seen_ts.add(d['time']); dedup.append(d)
    sl_summary_data = dedup

    sl_summary_series = [{
        'type': 'Histogram',
        'data': sl_summary_data,
        'options': {
            'priceFormat':  {'type': 'price', 'precision': 2, 'minMove': 0.01},
            'priceLineVisible': False,
            'lastValueVisible': False,
        },
    }]
    if sl_focus_outline:
        sl_summary_series.append({
            'type': 'Line',
            'data': sl_focus_outline,
            'options': {
                'color':      'rgba(255, 255, 255, 1.0)',
                'lineWidth':  3,
                'pointMarkersVisible': True,
                'pointMarkersRadius':  6,
                'priceLineVisible':    False,
                'lastValueVisible':    False,
                'crosshairMarkerVisible': False,
            },
        })

    sl_chart_options = {
        'height': 240,
        'layout': {'background': {'type': 'solid', 'color': '#131722'},
                   'textColor':  '#d1d4dc'},
        'grid':   {'vertLines': {'color': 'rgba(42, 46, 57, 0.45)'},
                   'horzLines': {'color': 'rgba(42, 46, 57, 0.45)'}},
        'timeScale': {'timeVisible':    interval in ('1h', '15m'),
                      'secondsVisible': False,
                      'borderColor':    'rgba(197, 203, 206, 0.3)',
                      'rightOffset':    4,
                      'barSpacing':     max(4, int(140 / max(1, len(trades_df))))},
        'rightPriceScale': {'borderColor': 'rgba(197, 203, 206, 0.3)',
                            'scaleMargins': {'top': 0.08, 'bottom': 0.08}},
        'crosshair': {'mode': 0},
    }
    renderLightweightCharts(
        [{'chart': sl_chart_options, 'series': sl_summary_series}],
        key='sl_summary_chart',
    )
    st.caption(
        f'Min SL: {trades_df["SL %"].min():.2f}%   ·   '
        f'Median SL: {trades_df["SL %"].median():.2f}%   ·   '
        f'Max SL: {trades_df["SL %"].max():.2f}%   ·   '
        f'Mean SL: {trades_df["SL %"].mean():.2f}%   ·   '
        f'Use the “Focus chart on a single trade” box above to zoom the main chart to any trade.'
    )

with st.expander('How SL is determined'):
    st.markdown(
        '''
        **LONG (Buy) side**
        - A **swing low** = a candle whose `Low` is strictly lower than the `Low`s of the
          previous *N* and next *N* candles (`Swing window` in sidebar).
        - For each entry (touch candle after a bullish EMA cross):
          - If the touch candle itself is a swing low → `SL = Low(touch candle)`.
          - Otherwise → `SL = Low(most recent prior swing low)`.
          - Fallback to touch candle's low if none exists.
          - Trade is **skipped** if `SL ≥ Entry`.
        - **Target** depends on the selected mode:
          - *Target %*: `Target = Entry × (1 + Target %)`
          - *R-multiple*: `Target = Entry + (Entry − SL) × R`
          - *ATR*: `Target = Entry + ATR(period) × multiplier`

        **SHORT (Sell) side** *(mirror of the above)*
        - A **swing high** = a candle whose `High` is strictly higher than the `High`s of
          the previous *N* and next *N* candles.
        - For each entry (touch candle after a bearish EMA cross — fast EMA crosses below
          slow EMA, then a bearish pullback that pokes EMA_fast from below):
          - If the touch candle itself is a swing high → `SL = High(touch candle)`.
          - Otherwise → `SL = High(most recent prior swing high)`.
          - Fallback to touch candle's high if none exists.
          - Trade is **skipped** if `SL ≤ Entry`.
        - **Target** depends on the selected mode:
          - *Target %*: `Target = Entry × (1 − Target %)`
          - *R-multiple*: `Target = Entry − (SL − Entry) × R`
          - *ATR*: `Target = Entry − ATR(period) × multiplier`
        - Touch condition: `High ≥ EMA_fast` AND `Close < EMA_fast` AND `Close < Open`.
        - First 2 touches per bearish cross are compulsory; from touch #3 onwards the
          downtrend gate requires a **lower swing low** since the previous touch.
        '''
    )
