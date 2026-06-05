# EMA Cross + Swing-Low SL — Project Folder

TradingView-style visualizer + per-stock optimizer for an EMA-crossover
entry strategy with swing-low stop-loss management.

## Layout

```
ema_cross_swing/
├── app.py                          ← Streamlit visualizer (entry point)
├── strategy_core.py                ← signal-scan + backtest core
├── optimize_per_stock.py           ← per-stock parameter optimizer
├── analyze_stock_methods.py        ← method-mapping analysis
├── EMAcode.py                      ← original EMA reference impl
├── nifty500_symbols.csv            ← scan universe (~500 NSE symbols)
├── requirements.txt                ← pip deps
├── run.bat                         ← double-click to launch on :8531
│
├── notebooks/                      ← exploratory notebooks
│   ├── ema_strategy_full.ipynb
│   ├── ema_strategy_sl_target_input.ipynb
│   ├── ema_strategy_swing_low_sl.ipynb       ← canonical strategy notebook
│   └── ema_strategy_v2_final.ipynb
│
├── cached_csv/                     ← per-symbol OHLC cache (5 stocks × 1d/1h)
│   ├── HDFCBANK_historical.csv      / _1h.csv
│   ├── RELIANCE_historical.csv      / _1h.csv
│   ├── INFY_historical.csv          / _1h.csv
│   ├── ICICIBANK_historical.csv     / _1h.csv
│   └── TCS_historical.csv           / _1h.csv
│
├── stock_methods/                  ← per-stock chosen-method maps
│   ├── stock_method_map.csv
│   ├── stock_method_map_validated.csv
│   └── stock_method_preset_metrics.csv
│
├── signals/                        ← backtest output CSVs (~3 MB)
│   └── pair_*, opt_*, signals_*, trades_*  for several EMA pairs / SL/TP combos
│
└── signals_swing/                  ← swing-SL backtest outputs (~0.5 MB)
    └── opt_grid_*, per_stock_best_*, top_*, trades_*
```

## Strategy summary

Entry: bullish EMA-fast / EMA-slow cross — then take "touches" of the
fast EMA from above as long-entry candidates. The first
`compulsory_touches` (default 2) touches are always taken; later touches
require a higher swing-high than the previous taken-touch window
(Dow-theory uptrend gate). Cross-window ends when the fast EMA crosses
back below the slow EMA.

Stop loss: most-recent swing low (configurable lookback).
Targets: fixed % or RR-multiple of stop distance.

## Run the visualizer

```
pip install -r requirements.txt
streamlit run app.py --server.port 8531
```

or just double-click `run.bat`.

The app pulls live OHLC via **yfinance** (no local data needed). The
files in `cached_csv/` are kept as a fallback / dev cache.

## Run the per-stock optimizer

```
python optimize_per_stock.py --data-dir <path-to-bulk-1d-CSVs>
```

A bulk CSV set already lives in this repo at
`../ema_scanner/data/1d/`, so you can call:

```
python optimize_per_stock.py --data-dir ..\ema_scanner\data\1d
```

Output → `stock_method_preset_metrics.csv` in the cwd.
