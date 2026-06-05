# investeq_ajs

Personal research-and-trading workspace built around a 6-year NSE NIFTY index-options dataset and a stock-OHLC corpus for swing strategies. The repo is organised as **one project per folder** plus a handful of single-file Streamlit/FastAPI apps at the root. Every app reads from a parquet/CSV store on disk — no network calls during runtime.

> ⚠ **Sensitive content reminder.** `DATA/README.md` contains a Groww API token at the very top of the file. Don't commit it to a public repo; if it leaks, rotate the token at the broker.

---

## Repo map

```
investeq_ajs/
├── app_option_replay_tv.py    # FastAPI + TradingView lightweight-charts UI  (port 8703)
├── app_option_replay.py       # Earlier Streamlit + Plotly replay UI         (port 8702)
├── app_ema_cross.py           # EMA21/50 zone-filter watchlist scanner       (port 8501)
├── nifty_options_ml.ipynb     # ML/DL on the options dataset (mirror of ML_DL/notebooks/...)
│
├── DATA/                      # NIFTY index options + spot + OI parquets (2020-01 → 2026-04)
│   ├── options/YYYYMMDD.parquet  · 1526 daily files, OHLCV per option contract
│   ├── oi/YYYYMMDD.parquet       · 1526 daily files, OI per contract
│   ├── spot/YYYYMMDD.parquet     · 1548 daily files, 1-min NIFTY index OHLCV
│   ├── index/expiries.parquet    · all option-expiry dates (weekly + monthly)
│   ├── straddle_pnl.parquet      · 294 weekly straddle events with P&L (notebook output)
│   └── README.md                 · full schema + gotchas + column types
│
├── ML_DL/                     # ML / deep-learning workstream on the options data
│   ├── README.md                 · full spec: tasks, features, labels, splits, metrics, references
│   ├── notebooks/                · canonical nifty_options_ml.ipynb (executed)
│   ├── features/                 · cached engineered features per run
│   ├── models/                   · pickled trained models per run
│   ├── outputs/                  · metrics JSON per run
│   └── plots/                    · equity curves, attribution, etc.
│
├── OPTION_STRATEGY/           # Rule-based option-strategy engine
│
├── daily_swing_scanner/       # Swing-low EMA scanner (refactor of ema_cross_swing)
│   ├── scan_app.py               · Streamlit UI                                 (port 8532)
│   ├── scan_cli.py               · headless CLI version of the same scan
│   └── strategy_core.py          · pure-logic library
│
├── ema_cross_swing/           # Original EMA-cross swing project + notebooks
│   ├── app.py                    · TradingView-style visualiser                  (port 8534)
│   ├── scan_app.py               · scanner UI                                    (port 8533)
│   ├── EMAcode.py / analyze_stock_methods.py / optimize_per_stock.py
│   └── notebooks/                · strategy iterations
│
├── ema_scanner/               · Latest EMA scanner + Groww data client
│   ├── app_ema_cross.py          · scanner UI (newer build of root app_ema_cross) (port 8535)
│   ├── groww_client.py           · broker REST client
│   ├── incremental_fetch.py      · keeps data/{1h,1d}/*.csv up to date
│   └── _quick_scan.py / _data_freshness.py
│
└── NEW_FOLDER/                · scratch / experiments
```

---

## Apps at a glance

| # | App | Path | Stack | Port | What it does |
|---|-----|------|-------|------|--------------|
| 1 | **Option Chain Replay (TV)** | `app_option_replay_tv.py` | FastAPI + lightweight-charts | **8703** | The main UI built in this repo. Split CE/PE candle panes, IV + RV (close-to-close or Parkinson) + OI + vol-smile panes, live crosshair-driven stats chips, and SELL/BUY markers from the notebook's straddle events (with LGBM probabilities). |
| 2 | **Option Chain Replay (Plotly)** | `app_option_replay.py` | Streamlit + Plotly | 8702 | Earlier Plotly-based replay — kept for reference. Multi-leg overlay in one figure, independent TF for IV/OI panes. |
| 3 | **EMA21/50 Watchlist Scanner** | `app_ema_cross.py` | Streamlit + lightweight-charts | 8501 | Scans `ajs/data/{1h,1d}/*.csv` and lists stocks whose latest **closed** bar passes an EMA21/50 zone filter (BUY/SELL flavours). Click a row → chart with the qualifying bar marked. |
| 4 | **Daily Swing Scanner (UI)** | `daily_swing_scanner/scan_app.py` | Streamlit + lightweight-charts | 8532 | EMA21/50 touch entries on a chosen day. Shows 1d-only, 1h-only, and ★STARRED (both timeframes) tables. |
| 5 | **Daily Swing Scanner (CLI)** | `daily_swing_scanner/scan_cli.py` | Pure Python | — | Headless version of #4. Writes `entries_<DATE>_1d.csv` and `entries_<DATE>_1h.csv`, prints star-list to stdout. Edit `TARGET_DAY` in the file to change scan date. |
| 6 | **EMA Swing-Low Visualiser** | `ema_cross_swing/app.py` | Streamlit + lightweight-charts | 8534 | TradingView-style chart for a single ticker with entry/SL markers from yfinance data. |
| 7 | **EMA Swing Scanner (alt build)** | `ema_cross_swing/scan_app.py` | Streamlit + lightweight-charts | 8533 | Same scan logic as #4, older folder. |
| 8 | **EMA Scanner (newest)** | `ema_scanner/app_ema_cross.py` | Streamlit + lightweight-charts | 8535 | Newest scanner. Reads from `ema_scanner/data/{1h,1d}` (refreshed by `incremental_fetch.py`). |
| 9 | **NIFTY Options ML Notebook** | `nifty_options_ml.ipynb` (root) and `ML_DL/notebooks/...` | Jupyter | — | 47-cell pipeline: data audit → Black-Scholes IV → straddle P&L → features (v1 + v2 with PCR / max-pain / IV-term-slope / DTE / DOW) → logistic / LightGBM / MLP walk-forward → backtest with probability threshold → artifact persistence under `ML_DL/`. Generates `DATA/straddle_pnl.parquet` and `ML_DL/{features,models,outputs,plots}/straddle_lgbm_*.parquet`. |

The TV UI (#1) is the one this thread iterated on heavily — it's the most featureful and the one to start with.

---

## One-time install

```powershell
cd C:\Users\User\Desktop\investeq_ajs
python -m pip install -q `
  streamlit plotly scipy scikit-learn pyarrow lightgbm joblib `
  fastapi "uvicorn[standard]" jupyter nbconvert ipykernel `
  streamlit-lightweight-charts yfinance
```

Add `shap` if you want SHAP attribution in the ML notebook (`pip install shap`).

---

## Run lines

### Option-chain replay UIs

```powershell
# 1. TradingView-style (recommended) — http://localhost:8703
python -m uvicorn app_option_replay_tv:app --host 0.0.0.0 --port 8703

# 2. Plotly version — http://localhost:8702
streamlit run app_option_replay.py --server.port 8702 --server.headless true
```

### EMA / swing scanners

```powershell
# 3. EMA21/50 watchlist (root)           — http://localhost:8501
streamlit run app_ema_cross.py --server.port 8501 --server.headless true

# 4. Daily Swing Scanner (UI)            — http://localhost:8532
streamlit run daily_swing_scanner\scan_app.py --server.port 8532 --server.headless true

# 5. Daily Swing Scanner (CLI, no UI)
python daily_swing_scanner\scan_cli.py

# 6. EMA Swing-Low Visualiser            — http://localhost:8534
streamlit run ema_cross_swing\app.py --server.port 8534 --server.headless true

# 7. EMA Swing Scanner (alt build)       — http://localhost:8533
streamlit run ema_cross_swing\scan_app.py --server.port 8533 --server.headless true

# 8. EMA Scanner (newest)                — http://localhost:8535
streamlit run ema_scanner\app_ema_cross.py --server.port 8535 --server.headless true
```

### ML notebook

```powershell
# 9a. Headless full re-run (~25–45 min for the first time; ~2 min with cache)
$env:MPLBACKEND="Agg"
python -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 `
  ML_DL\notebooks\nifty_options_ml.ipynb

# 9b. Interactive
jupyter notebook ML_DL\notebooks\nifty_options_ml.ipynb
# or:
jupyter lab ML_DL\notebooks\nifty_options_ml.ipynb
```

### Bash / Git-Bash equivalents

```bash
# Streamlit
MPLBACKEND=Agg streamlit run app_option_replay.py --server.port 8702 --server.headless true
# FastAPI
python -m uvicorn app_option_replay_tv:app --host 0.0.0.0 --port 8703
# Notebook
MPLBACKEND=Agg python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
  ML_DL/notebooks/nifty_options_ml.ipynb
```

---

## TV-UI feature cheat-sheet (`app_option_replay_tv.py`)

Top toolbar (left → right):

| Control | Effect |
|---|---|
| Date | Pick any NSE trading day (auto-discovered from `DATA/spot/*.parquet`) |
| Lookback | 1 / 2 / 3 / 5 / 10 days — how many days of context to load (window ends at Date) |
| **Strategy event** | All 294 weekly straddle events from `DATA/straddle_pnl.parquet`, labelled `WIN/LOSS  ±pts  p=<LGBM>`. Pick one → auto-fills date=expiry, expiry=event-expiry, strike=event-ATM, side=BOTH, lookback=3, and plots SELL/BUY markers. |
| Signals | Toggle the SELL/BUY arrow markers on/off without losing the active event |
| Expiry | All expiries available on the selected date |
| Strike | All strikes for the chosen expiry — ★ tags ATM |
| Side | `BOTH (CE \| PE)` (default) — splits the price row into two synced panes. `CE` or `PE` to focus on one side |
| TF (price) | 1m / 3m / 5m / 15m / 30m / 1h — candle granularity |
| TF (IV / OI) | Independent timeframe for the IV/RV/OI panes |
| Show | `IV` / `RV` / `OI` / `Spot` toggles |
| RV estimator | close-to-close (default) or Parkinson (high-low, variance-efficient) |
| RV window | Rolling lookback in bars (default 30) |

Stats chips (live, follow the crosshair):

- **Cursor / Now** — hovered time and date
- **O / H / L / C** — current bar
- **IV** — option-implied vol with the day's range
- **RV (spot)** — rolling realised vol with the day's range
- **IV − RV** — variance risk premium at the cursor
- **Spot** — NIFTY level with the day's range
- **ΔOI from open** — OI at cursor minus OI at session-open of the cursor's day
- **Δ / Γ** and **Θ / ν** — full Greeks per bar (Black-Scholes, calendar-year T)
- **premium Δ today** — close at cursor minus open of the cursor's day

Panes (stacked vertically):

1. **CE candles** | **PE candles** — synced 2-up split, each with its own volume histogram and spot overlay
2. **IV / RV** — CE IV (blue), PE IV (purple), RV (orange dashed)
3. **OI** — CE OI (blue), PE OI (purple)
4. **Vol smile** — IV across ATM±12 strikes at the current cursor time; CE solid, PE dashed, ATM strike marked yellow

All panes share a time axis and crosshair; pan/zoom one and the others follow.

### Notable backend fixes baked in

| Bug | Fix |
|---|---|
| `years_to_expiry` in the original notebook mixed calendar seconds with a trading-second denominator → IV deflated ~5× (showed 2-3% instead of 13-15%) | Now uses calendar-year (`(expiry − ts) / (365·24·3600)`) consistently |
| `rolling().apply(lambda x: x[-1] - ...)` failed on a pandas Series (label lookup) | Added `raw=True` so the callback gets a numpy array |
| MLP `early_stopping=True` blew up on small TimeSeriesSplit folds where one class had <2 members | Disabled `early_stopping` (small dataset, `max_iter` handles convergence) |
| `/api/smile` was double-shifting the timestamp by IST offset | Now interprets the chart-time int directly (matches `to_unix()`'s naive-as-UTC convention) |
| Chart displayed IST timestamps shifted by +5:30 (15:30 → 21:00) in IST browsers | Custom `timeFormatter`/`tickMarkFormatter` use `getUTC*` so the chart always shows NSE wall-clock |
| ΔOI chip stuck to last-day's value as you swept the crosshair | Built per-day stats client-side; chips now look up `dayStats[dayKey(cursor)]` |

---

## API endpoints (`app_option_replay_tv.py`)

| Method | Path | Returns |
|---|---|---|
| GET | `/api/days` | List of trading days (`YYYYMMDD`) |
| GET | `/api/expiries?date=YYYYMMDD` | Expiries that trade on that day |
| GET | `/api/strikes?date=&expiry=` | `{strikes:[…], atm:int}` |
| GET | `/api/leg?date=&expiry=&strike=&type=CE\|PE&tf=5m&lookback_days=3&...` | `{candles, volume, iv, rv, oi, spot, greeks, stats, days}` |
| GET | `/api/smile?date=&expiry=&time=<unix>&band=12` | `{strikes, ce_iv, pe_iv, atm, spot, cursor}` |
| GET | `/api/chain?date=&expiry=&band=8` | Latest CE\|strike\|PE snapshot of the chain |
| GET | `/api/events` | All 294 straddle events from `DATA/straddle_pnl.parquet`, joined with the latest LGBM out-of-fold probabilities |

`/api/leg` also accepts `with_iv`, `with_oi`, `with_spot`, `with_rv`, `rv_window`, `rv_estimator` (`close` or `parkinson`).

---

## ML pipeline summary

The notebook builds a short-straddle strategy:

- **Entry**: sell ATM CE+PE at the trading day **before expiry** at 15:20 IST
- **Exit**: buy back at 09:20 IST on expiry day — same strike that was sold (no re-ATM)
- 296 weekly events from 2020-01 to 2026-03 → 147 with full v2 features (52-week IV-rank warmup drops the first year)

Baseline (blind, no filter):

- Win rate **73.0%**, mean P&L **+3.07 pts/trade**, total **+909.70 pts**
- Worst trade **−523.35** (2026-02-03, 25100 ATM, +764 pt overnight gap), best +115.60
- By year (win %): 71.7 / 78.8 / 71.7 / 70.5 / 65.1 / 77.1 / 80.0  (2020 → 2026)

> **Earlier headline numbers (95.2% win, +8,190 pts, worst −125)** were the result of a buy-back bug: the notebook re-picked ATM on expiry-day spot at exit instead of buying back the strike that was sold. Fixed 2026-05-21 — see commit history and the `_regen_straddle_pnl.py` helper.

ML results on the v2 feature set (PCR, max-pain distance, IV term-slope, intraday/last-hour returns, DTE/DOW, plus the v1 IV-rank/RV/IV-RV-spread features):

| Model | CV mean AUC | Holdout AUC |
|---|---|---|
| Logistic (v1) | 0.492 | 0.458 |
| LightGBM (v2) | 0.521 | 0.473 |
| MLP (v2)      | 0.385 | — |

With realistic labels (73% wins instead of 95%) LGBM's OOF probabilities now spread out — min 0.065, median 0.886, max 0.997 — so threshold-based filtering actually moves the equity curve. At threshold **0.84** the filter takes 69/147 trades for **+568.8 pts** (max DD −276.8). The Logistic + LGBM holdout AUCs are now slightly below 0.5, so the model's edge on out-of-sample data is marginal at best — the v2 features capture in-sample CV signal that doesn't generalise. Real progress will need either richer features, probability calibration, or switching to expected-P&L regression rather than win/loss classification.

Each notebook run writes:

- `ML_DL/features/straddle_lgbm_<YYYYMMDD_HHMM>.parquet` (146 events × all features + OOF probabilities)
- `ML_DL/models/straddle_lgbm_<…>.pkl` (LGBM + features list + best threshold)
- `ML_DL/outputs/straddle_lgbm_<…>_metrics.json`
- `ML_DL/plots/straddle_lgbm_<…>_equity.png`

See `ML_DL/README.md` for the full spec (task catalogue, feature catalogue, labelling protocols, validation rules, references).

---

## Data layout

Detailed schema lives at `DATA/README.md`. TL;DR:

| Source | Purpose | Join key |
|---|---|---|
| `DATA/spot/YYYYMMDD.parquet` | 1-min NIFTY spot OHLCV | `timestamp` |
| `DATA/options/YYYYMMDD.parquet` | 1-min OHLCV per option contract | `timestamp, expiry, strike, option_type` |
| `DATA/oi/YYYYMMDD.parquet` | 1-min OI per contract | same as options |
| `DATA/index/expiries.parquet` | full expiry list | — |
| `DATA/straddle_pnl.parquet` | notebook output: 294 weekly straddle events | `expiry` |

Gotchas (from `DATA/README.md`):

- Timestamps are **strings, no tz**, IST — cast and localise.
- Contract minutes are **sparse** — reindex against the session grid before sequence modelling.
- **Lot size** changed over the window (75 → 50 → 25 → 75) — load from a date-keyed table, never hardcode.
- **No survivorship adjustment needed** — every traded contract is present, including those that expired worthless.
- Derive the trading calendar from `spot/` filenames, not `trading_days.parquet`.

The EMA scanners read CSVs from a separate location (`ajs/data/{1h,1d}/` or `ema_scanner/data/{1h,1d}/`) refreshed by `ema_scanner/incremental_fetch.py`.

---

## Stop / clean-up

Windows — kill whatever's listening on a port:

```powershell
$pids = (Get-NetTCPConnection -LocalPort 8703 -ErrorAction SilentlyContinue).OwningProcess | Sort-Object -Unique
$pids | ForEach-Object { Stop-Process -Id $_ -Force }
```

Bash:

```bash
lsof -ti :8703 | xargs -r kill -9    # POSIX
netstat -ano | grep ':8703'          # Windows, then taskkill /PID <pid> /F
```

---

## Reference UI (external)

`http://34.100.162.71:8765/` — teammate's pre-existing option-chain-replay site, hosted on the GCP VM that originally stored the dataset. Not in this repo; used as design inspiration for `app_option_replay_tv.py`.
