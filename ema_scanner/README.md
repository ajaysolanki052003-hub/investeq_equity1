# EMA21/50 Zone Scanner

Self-contained watchlist scanner for tomorrow's moves.
Reads OHLC CSVs from `./data/{1h,1d}/` and reports stocks whose latest
closed bar (strictly before today) sits in a tradable EMA21/50 zone.

## Layout

```
ema_scanner/
├── app_ema_cross.py        ← Streamlit app (entry point)
├── _quick_scan.py          ← headless preview (prints top 20 BUY / SELL)
├── _data_freshness.py      ← inspect last-bar dates per CSV
├── requirements.txt
├── run.bat                 ← double-click to launch on :8530
└── data/
    ├── 1h/  *_historical.csv  (524 symbols)
    └── 1d/  *_historical.csv  (524 symbols)
```

`DATA_ROOT` is resolved relative to `app_ema_cross.py`, so the folder is
fully portable — move `ema_scanner/` anywhere and it still works.

## Run

```
pip install -r requirements.txt
streamlit run app_ema_cross.py --server.port 8530
```

or just double-click `run.bat`.

## Filter rules

For each side (BUY / SELL), the **EMA regime** must already be set
(EMA21 above EMA50 for BUY, below for SELL). Then any of these
**triggers** on the latest closed bar fires the signal:

| Tag | BUY                                | SELL                               |
|-----|------------------------------------|------------------------------------|
| A   | low ≤ EMA21 and close > EMA21      | high ≥ EMA21 and close < EMA21     |
| B   | EMA50 < close < EMA21              | EMA21 < close < EMA50              |
| C   | close < EMA50 and high > EMA50     | close > EMA50 and low < EMA50      |

A **freshness gate** (sidebar: "Reject if last bar is older than (days)")
silently drops stocks whose newest CSV bar is too far behind today —
default 10 days covers the current data lag.

## Chart

- Click any row in the watchlist → chart opens for that symbol.
- Chart cuts at today 00:00, so the last candle is the qualifying bar.
- Latest-bar trigger = ⭐ marker; historic triggers (toggle in sidebar)
  use small ▲ (BUY) / ▼ (SELL).
- EMA21/50 crossovers shown as 🟢 / 🔴 circles inside the candle.
