"""Per-stock × per-preset × per-method × per-side backtest grid.

Runs the EMA(21,50) + swing SL + uptrend gate strategy from strategy_core.py on
every CSV in data/1d/ for 6 preset target triples × 3 methods × 2 sides × 2
sample windows (in-sample / out-of-sample).

Output: stock_method_preset_metrics.csv  — one row per
(Symbol, Side, Method, Preset, Sample) with all summary metrics.

Run:
    python optimize_per_stock.py
"""
import argparse
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from strategy_core import (
    add_emas, compute_atr, compute_swing_lows, compute_swing_highs,
    scan_buy_signals, scan_sell_signals,
    build_long_trades, build_short_trades,
    metrics_of_trades,
)

# ─────────── CONFIG ───────────
DATA_DIR    = 'data/1d'
OUT_CSV     = 'stock_method_preset_metrics.csv'
FAST, SLOW  = 21, 50
SWING_W     = 2
ATR_PERIOD  = 14

IS_END      = pd.Timestamp('2024-12-31 23:59:59')   # in-sample cut-off
OOS_START   = pd.Timestamp('2025-01-01 00:00:00')   # out-of-sample start

PRESETS = [
    # name        target_pct  R_mult  atr_mult
    ('Tight',         3.0,    1.5,    1.5),
    ('Normal',        5.0,    2.0,    2.0),
    ('Loose',         8.0,    3.0,    3.0),
    ('Wide',         12.0,    4.0,    4.0),
    ('Mixed-A',       5.0,    2.5,    1.5),
    ('Mixed-B',       8.0,    1.5,    3.0),
]
METHODS = ['target_pct', 'R', 'ATR']
SIDES   = ['BUY', 'SELL']


def load_one(path):
    """Read a candle CSV, returns DataFrame with DatetimeIndex and OHLCV columns."""
    df = pd.read_csv(path)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif 'timestamp' in df.columns:
        df.index = pd.to_datetime(df['timestamp'], unit='s')
        df = df.drop(columns=['timestamp'])
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # normalize column names
    rename_map = {c: c.title() for c in ['open', 'high', 'low', 'close', 'volume']
                  if c in df.columns}
    df = df.rename(columns=rename_map)
    keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    return df[keep].dropna()


def median_atr_pct(df, atr_series):
    """Median(ATR / Close) as a percentage — a stock-volatility feature."""
    ratio = (atr_series / df['Close']) * 100
    return float(ratio.dropna().median())


def filter_trades_by_window(trades, df_index, sample_window):
    """sample_window in {'IS', 'OOS', 'ALL'}. Filter trades whose entry_idx
    timestamp falls in that window."""
    if sample_window == 'ALL':
        return trades
    out = []
    for t in trades:
        ts = df_index[t['entry_idx']]
        if sample_window == 'IS' and ts <= IS_END:
            out.append(t)
        elif sample_window == 'OOS' and ts >= OOS_START:
            out.append(t)
    return out


def process_one(symbol, df):
    """Returns list of metric-rows for this symbol."""
    if len(df) < SLOW + 30:
        return []
    df_ema = add_emas(df, FAST, SLOW)
    atr_series = compute_atr(df_ema, ATR_PERIOD)
    swing_low  = compute_swing_lows(df_ema, SWING_W)
    swing_high = compute_swing_highs(df_ema, SWING_W)
    buy_sigs   = scan_buy_signals(df_ema, FAST, SLOW)
    sell_sigs  = scan_sell_signals(df_ema, FAST, SLOW)

    feat_atr_pct = median_atr_pct(df_ema, atr_series)
    feat_avg_price = float(df_ema['Close'].mean())
    feat_listing_age = (df_ema.index[-1] - df_ema.index[0]).days / 365.25

    rows = []
    for preset_name, tp, R, am in PRESETS:
        for method in METHODS:
            # Build trades once for the entire history (per side × method × preset)
            long_trades  = build_long_trades(
                df_ema, buy_sigs, swing_low, atr_series,
                method=method, target_pct=tp, R_mult=R, atr_mult=am)
            short_trades = build_short_trades(
                df_ema, sell_sigs, swing_high, atr_series,
                method=method, target_pct=tp, R_mult=R, atr_mult=am)
            for side, trades in (('BUY', long_trades), ('SELL', short_trades)):
                for sample in ('IS', 'OOS', 'ALL'):
                    sub = filter_trades_by_window(trades, df_ema.index, sample)
                    m   = metrics_of_trades(sub)
                    rows.append({
                        'Symbol':        symbol,
                        'Side':          side,
                        'Method':        method,
                        'Preset':        preset_name,
                        'Sample':        sample,
                        'target_pct':    tp,
                        'R_mult':        R,
                        'atr_mult':      am,
                        'atr_pct_med':   feat_atr_pct,
                        'avg_price':     feat_avg_price,
                        'listing_age_y': feat_listing_age,
                        **m,
                    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None,
                    help='process only first N CSVs (smoke test)')
    ap.add_argument('--data-dir', default=DATA_DIR)
    ap.add_argument('--out', default=OUT_CSV)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, '*_historical.csv')))
    if args.limit:
        paths = paths[:args.limit]
    print(f'[plan] {len(paths)} stocks  ·  '
          f'{len(PRESETS)} presets  ·  {len(METHODS)} methods  ·  '
          f'2 sides  ·  3 sample windows  '
          f'=> ~{len(paths) * len(PRESETS) * len(METHODS) * 2 * 3} rows', flush=True)

    t0 = time.time()
    all_rows = []
    ok = err = skip = 0
    for i, path in enumerate(paths, start=1):
        symbol = os.path.basename(path).replace('_historical.csv', '')
        try:
            df = load_one(path)
        except Exception as e:
            err += 1
            print(f'  [{i:4d}] {symbol:<14}  load FAILED: {e}', flush=True)
            continue
        try:
            rows = process_one(symbol, df)
        except Exception as e:
            err += 1
            print(f'  [{i:4d}] {symbol:<14}  process FAILED: {e}', flush=True)
            continue
        if not rows:
            skip += 1
        else:
            all_rows.extend(rows)
            ok += 1
        if i % 50 == 0 or i == len(paths):
            el = time.time() - t0
            print(f'  [{i:4d}/{len(paths)}]  ok={ok}  skip={skip}  err={err}  ({el:.0f}s)',
                  flush=True)

    out = pd.DataFrame(all_rows)
    out.to_csv(args.out, index=False)
    el = time.time() - t0
    print(f'\n[done] {len(out)} rows  ·  {el:.0f}s elapsed  ·  saved -> {args.out}',
          flush=True)


if __name__ == '__main__':
    main()
