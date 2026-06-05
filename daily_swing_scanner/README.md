# Daily Swing Scanner

Scans every stock in the universe for a **swing-low project EMA touch entry
that fires on one specific day**, on both 1h and 1d. Highlights stocks that
qualified on **both** timeframes (highest conviction).

Built on the same `scan_buy_signals` / `scan_sell_signals` logic as the
swing-low backtester — no future-data leakage, same Dow-theory uptrend gate.

## Layout

```
daily_swing_scanner/
├── scan_app.py                 ← Streamlit UI (entry point)
├── scan_cli.py                 ← headless version that dumps CSVs to disk
├── strategy_core.py            ← shared scan_buy_signals / scan_sell_signals
├── entries_2026-05-14_1d.csv   ← last scan output (1d)
├── entries_2026-05-14_1h.csv   ← last scan output (1h)
├── requirements.txt
├── run.bat                     ← double-click to launch on :8532
└── README.md
```

The scanner reads OHLC CSVs from the sibling folder
`../ema_scanner/data/{1h,1d}/`. Move this folder anywhere that keeps that
sibling relationship and it still works (or override the **Data root** field
in the sidebar).

## Run

```
pip install -r requirements.txt
streamlit run scan_app.py --server.port 8532
```

or double-click `run.bat`.

## Signal logic

For each stock:

**BUY (long)** — for every bullish EMA21/EMA50 cross, walk forward bar-by-bar
inside the window (until EMA21 ≤ EMA50 closes it). A bar is a qualifying
touch iff:

- `low ≤ EMA21`  (price wicked into the fast EMA)
- `close > EMA21` (closed back above)
- `close > open` (green candle)

The first 2 touches per window are always taken (compulsory).
From the 3rd touch onwards, the touch is taken only if the highest high
between the previous taken touch and now exceeds the highest high between
the two prior taken touches — i.e. price made a **new higher high since
the last taken touch** (Dow-theory uptrend gate).

**SELL (short)** — mirrored:

- `high ≥ EMA21`
- `close < EMA21`
- `close < open` (red candle)

3rd-touch gate requires a **new lower low** since the previous taken touch.

The app filters the resulting touch list to only those whose `Touch Time.date()`
equals the scan day (default **2026-05-14**).

## Output

- **🌟 Starred table** — stocks with the same-side entry on BOTH 1d and 1h
  on the same day. Highest conviction.
- **1d table** — every 1d touch on the day, clickable to load chart.
- **1h table** — every 1h touch on the day (a stock may fire multiple
  times if multiple windows or multiple intraday touches), clickable to
  load chart.
- **Chart** — last 120 bars (configurable), EMA21/50 lines, qualifying
  touch marked with a ⭐ green ▲ (BUY) or red ▼ (SELL).
- **CSV downloads** for both timeframes.
