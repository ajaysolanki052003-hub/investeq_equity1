"""Shared strategy primitives — extracted from app.py so the optimizer and the
Streamlit app stay in sync. EMA crossover + swing-based SL + uptrend gate for
LONG, mirrored for SHORT. ATR computed via Wilder's smoothing."""

import numpy as np
import pandas as pd


# ─────────────────────────── INDICATORS ───────────────────────────
def add_emas(df, fast, slow):
    df = df.copy()
    df[f'EMA{fast}'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df[f'EMA{slow}'] = df['Close'].ewm(span=slow, adjust=False).mean()
    return df


def compute_atr(df, period=14):
    h = df['High'].astype(float); l = df['Low'].astype(float); c = df['Close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_swing_lows(df, window):
    """Vectorised: a bar is a swing low when its Low is strictly less than
    the min of the `window` bars before AND after it."""
    s = df['Low'].astype(float)
    left  = s.shift(1).rolling(window).min()
    right = s.shift(-window).rolling(window).min()
    return ((s < left) & (s < right)).fillna(False).values


def compute_swing_highs(df, window):
    s = df['High'].astype(float)
    left  = s.shift(1).rolling(window).max()
    right = s.shift(-window).rolling(window).max()
    return ((s > left) & (s > right)).fillna(False).values


# ─────────────────────────── CROSSES + TOUCHES ───────────────────────────
def bullish_cross_indices(df, fast, slow):
    above = df[f'EMA{fast}'] > df[f'EMA{slow}']
    cross = above & ~above.shift(1, fill_value=False)
    cross.iloc[0] = False
    return list(np.where(cross.values)[0])


def bearish_cross_indices(df, fast, slow):
    below = df[f'EMA{fast}'] < df[f'EMA{slow}']
    cross = below & ~below.shift(1, fill_value=False)
    cross.iloc[0] = False
    return list(np.where(cross.values)[0])


def scan_buy_signals(df, fast, slow, compulsory=2):
    """First 2 touches per cross compulsory; from #3 onwards require new HH since
    previous touch (Dow-theory uptrend)."""
    rows = []
    highs = df['High'].values
    for cidx in bullish_cross_indices(df, fast, slow):
        cross_ts = df.index[cidx]
        taken = []
        for i in range(cidx + 1, len(df)):
            o = float(df['Open'].iloc[i]); h = float(df['High'].iloc[i])
            l = float(df['Low'].iloc[i]);  c = float(df['Close'].iloc[i])
            ef = float(df[f'EMA{fast}'].iloc[i]); es = float(df[f'EMA{slow}'].iloc[i])
            if ef <= es:
                break
            if not (l <= ef and c > ef and c > o):
                continue
            if len(taken) >= compulsory:
                last_t = taken[-1]; prev_t = taken[-2]
                hi_recent = highs[last_t+1:i+1].max() if i > last_t else -np.inf
                hi_prior  = highs[prev_t+1:last_t+1].max() if last_t > prev_t else -np.inf
                if not (hi_recent > hi_prior):
                    continue
            rows.append({
                'Cross Time': cross_ts, 'Touch Time': df.index[i],
                'Touch Idx': i, 'Touch Close': c,
            })
            taken.append(i)
    return pd.DataFrame(rows)


def scan_sell_signals(df, fast, slow, compulsory=2):
    rows = []
    lows = df['Low'].values
    for cidx in bearish_cross_indices(df, fast, slow):
        cross_ts = df.index[cidx]
        taken = []
        for i in range(cidx + 1, len(df)):
            o = float(df['Open'].iloc[i]); h = float(df['High'].iloc[i])
            l = float(df['Low'].iloc[i]);  c = float(df['Close'].iloc[i])
            ef = float(df[f'EMA{fast}'].iloc[i]); es = float(df[f'EMA{slow}'].iloc[i])
            if ef >= es:
                break
            if not (h >= ef and c < ef and c < o):
                continue
            if len(taken) >= compulsory:
                last_t = taken[-1]; prev_t = taken[-2]
                lo_recent = lows[last_t+1:i+1].min() if i > last_t else np.inf
                lo_prior  = lows[prev_t+1:last_t+1].min() if last_t > prev_t else np.inf
                if not (lo_recent < lo_prior):
                    continue
            rows.append({
                'Cross Time': cross_ts, 'Touch Time': df.index[i],
                'Touch Idx': i, 'Touch Close': c,
            })
            taken.append(i)
    return pd.DataFrame(rows)


# ─────────────────────────── SL FINDERS ───────────────────────────
def find_swing_sl(df, touch_idx, swing_mask):
    if swing_mask[touch_idx]:
        return float(df['Low'].iloc[touch_idx])
    prior = np.where(swing_mask[:touch_idx])[0]
    if len(prior) > 0:
        return float(df['Low'].iloc[prior[-1]])
    return float(df['Low'].iloc[touch_idx])


def find_swing_sh(df, touch_idx, swing_mask):
    if swing_mask[touch_idx]:
        return float(df['High'].iloc[touch_idx])
    prior = np.where(swing_mask[:touch_idx])[0]
    if len(prior) > 0:
        return float(df['High'].iloc[prior[-1]])
    return float(df['High'].iloc[touch_idx])


# ─────────────────────────── TRADE EVAL ───────────────────────────
def eval_long(highs, lows, idx, sl_price, target_price):
    for j in range(idx + 1, len(highs)):
        hi = highs[j]; lo = lows[j]
        tgt = hi >= target_price
        sl  = lo <= sl_price
        if tgt and sl:
            return 'AMBIGUOUS', j
        if tgt: return 'TARGET HIT', j
        if sl:  return 'SL HIT', j
    return 'OPEN', len(highs) - 1


def eval_short(highs, lows, idx, sl_price, target_price):
    for j in range(idx + 1, len(highs)):
        hi = highs[j]; lo = lows[j]
        tgt = lo <= target_price
        sl  = hi >= sl_price
        if tgt and sl:
            return 'AMBIGUOUS', j
        if tgt: return 'TARGET HIT', j
        if sl:  return 'SL HIT', j
    return 'OPEN', len(highs) - 1


# ─────────────────────────── BUILD TRADES (parametric) ───────────────────────────
def build_long_trades(df, signals, swing_low_mask, atr_series,
                      method, target_pct, R_mult, atr_mult):
    """Returns list of trade dicts. method in {'target_pct','R','ATR'}."""
    highs = df['High'].values
    lows  = df['Low'].values
    closes = df['Close'].values
    out = []
    for _, sig in signals.iterrows():
        idx   = int(sig['Touch Idx'])
        entry = float(sig['Touch Close'])
        sl_price = find_swing_sl(df, idx, swing_low_mask)
        if sl_price >= entry:
            continue
        sl_pct = (entry - sl_price) / entry * 100
        if method == 'target_pct':
            target_price = entry * (1 + target_pct / 100)
            tgt_pct_eff  = target_pct
        elif method == 'R':
            target_price = entry + (entry - sl_price) * R_mult
            tgt_pct_eff  = sl_pct * R_mult
        elif method == 'ATR':
            atr_v = atr_series.iloc[idx] if atr_series is not None else float('nan')
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            target_price = entry + atr_v * atr_mult
            tgt_pct_eff  = (target_price - entry) / entry * 100
        else:
            raise ValueError(method)
        outcome, exit_idx = eval_long(highs, lows, idx, sl_price, target_price)
        if outcome == 'TARGET HIT':
            pnl = tgt_pct_eff
        elif outcome == 'SL HIT':
            pnl = -sl_pct
        elif outcome == 'AMBIGUOUS':
            pnl = -sl_pct      # conservative
        else:
            pnl = (closes[-1] - entry) / entry * 100
        out.append({'entry_idx': idx, 'exit_idx': exit_idx,
                    'entry': entry, 'sl': sl_price, 'target': target_price,
                    'sl_pct': sl_pct, 'tgt_pct': tgt_pct_eff,
                    'outcome': outcome, 'pnl_pct': pnl})
    return out


def build_short_trades(df, signals, swing_high_mask, atr_series,
                       method, target_pct, R_mult, atr_mult):
    highs = df['High'].values
    lows  = df['Low'].values
    closes = df['Close'].values
    out = []
    for _, sig in signals.iterrows():
        idx   = int(sig['Touch Idx'])
        entry = float(sig['Touch Close'])
        sl_price = find_swing_sh(df, idx, swing_high_mask)
        if sl_price <= entry:
            continue
        sl_pct = (sl_price - entry) / entry * 100
        if method == 'target_pct':
            target_price = entry * (1 - target_pct / 100)
            tgt_pct_eff  = target_pct
        elif method == 'R':
            target_price = entry - (sl_price - entry) * R_mult
            tgt_pct_eff  = sl_pct * R_mult
        elif method == 'ATR':
            atr_v = atr_series.iloc[idx] if atr_series is not None else float('nan')
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            target_price = entry - atr_v * atr_mult
            tgt_pct_eff  = (entry - target_price) / entry * 100
        else:
            raise ValueError(method)
        outcome, exit_idx = eval_short(highs, lows, idx, sl_price, target_price)
        if outcome == 'TARGET HIT':
            pnl = tgt_pct_eff
        elif outcome == 'SL HIT':
            pnl = -sl_pct
        elif outcome == 'AMBIGUOUS':
            pnl = -sl_pct
        else:
            pnl = (entry - closes[-1]) / entry * 100
        out.append({'entry_idx': idx, 'exit_idx': exit_idx,
                    'entry': entry, 'sl': sl_price, 'target': target_price,
                    'sl_pct': sl_pct, 'tgt_pct': tgt_pct_eff,
                    'outcome': outcome, 'pnl_pct': pnl})
    return out


# ─────────────────────────── METRICS ───────────────────────────
def metrics_of_trades(trades):
    """Returns dict of summary stats. Assumes 'pnl_pct' and 'outcome' keys."""
    if not trades:
        return dict(n_trades=0, n_target=0, n_sl=0, n_open=0,
                    win_rate=np.nan, mean_pnl=np.nan, median_pnl=np.nan,
                    total_pnl=0.0, expectancy=np.nan, max_dd=np.nan,
                    avg_win=np.nan, avg_loss=np.nan, profit_factor=np.nan)
    pnls = np.array([t['pnl_pct'] for t in trades], dtype=float)
    n = len(pnls)
    outs = [t['outcome'] for t in trades]
    n_tgt = sum(o == 'TARGET HIT' for o in outs)
    n_sl  = sum(o == 'SL HIT' for o in outs)
    n_op  = sum(o == 'OPEN' for o in outs)
    decided = n_tgt + n_sl
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    avg_win  = wins.mean()   if len(wins)   else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    win_rate = (len(wins) / n) if n else np.nan
    expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if n else np.nan
    # Equity curve & max drawdown in %-PnL units
    eq = np.cumsum(pnls)
    peaks = np.maximum.accumulate(eq)
    drawdowns = peaks - eq
    max_dd = drawdowns.max() if n else np.nan
    gross_win  = wins.sum()             if len(wins)   else 0.0
    gross_loss = -losses.sum()          if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else np.inf
    return dict(
        n_trades=n, n_target=int(n_tgt), n_sl=int(n_sl), n_open=int(n_op),
        win_rate=float(win_rate),
        mean_pnl=float(pnls.mean()), median_pnl=float(np.median(pnls)),
        total_pnl=float(pnls.sum()),
        expectancy=float(expectancy),
        max_dd=float(max_dd),
        avg_win=float(avg_win), avg_loss=float(avg_loss),
        profit_factor=float(pf),
    )
