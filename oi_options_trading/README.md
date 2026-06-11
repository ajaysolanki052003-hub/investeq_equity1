# NIFTY OI Strategy Suite

Live trading-signal app for **NIFTY index intraday**, built around
**Open-Interest (OI) cross signals on ATM±1 strikes**. Combines two
complementary methods into a single managed-trade stream, walks every
trade through SL / Target / Trail / EOD exit logic, and renders the
full multi-year trade history + cumulative-PnL curve on a single page.

> **Live demo (private VM):** `http://34.93.70.239/strategy/`

---

## 1 · What it does

For every minute of NIFTY index trading from 2020 to today:

1. Read CE/PE OI at the live ATM and its two flanking strikes (ATM-50,
   ATM, ATM+50) on the nearest expiry. ATM is recomputed at every tick.
2. Run two independent **entry rules** (described below) to flag
   LONG / SHORT signals throughout the day.
3. **Merge** the two streams with same-side-within-7-minutes dedup, and
   apply a single-position-at-a-time filter.
4. For each surviving signal, **walk the 1-minute index price forward**
   from `entry_time` and apply (SL, Target, Multi-step Trail, EOD)
   exit rules to determine the realised P&L.
5. Plot every trade on the candle chart with entry+exit markers, and
   render the running cumulative P&L below as a 6-year area curve.

The result is a fully-deterministic, leak-free, auditable trade history
with **PF ≈ 1.42**, **win-rate ~58 %**, and **+9,500 NIFTY pts** of
realised P&L over six years on the locked-in defaults.

---

## 2 · The two methods

Both look at the same two series:

- `ce_oi(t)` = sum of CE OI at the live `ATM-50`, `ATM`, `ATM+50` strikes
- `pe_oi(t)` = sum of PE OI at the same three strikes

They differ in *how* they collapse those into a cross signal.

### 2.1 · Σ-10 (rolling-window cross)

> "Smooth OI with a rolling 10-bucket sum; fire on the smoothed cross."

- **Compute** (per-day reset):
  ```
  Σ_10_CE(t) = sum( ce_oi(t-9) … ce_oi(t) )
  Σ_10_PE(t) = sum( pe_oi(t-9) … pe_oi(t) )
  diff(t)    = Σ_10_PE(t) − Σ_10_CE(t)
  ```
- **Trigger** (per-day, gated to **09:45 – 14:45 IST**):
  - **LONG**  = `diff` crosses up **AND** raw `pe_oi > ce_oi` at the same bucket
  - **SHORT** = mirror cross-down with raw `ce_oi > pe_oi`
- **One entry per day.** First qualifying cross only.
- TF-aware: on 3-minute candles, Σ-10 covers the last 30 minutes.
- Implementation: `strategies/sigma5_entry_exit.py`

### 2.2 · ΣOI (per-day cumulative cross since 09:15)

> "Cumulative CE vs PE OI from market open. Cross = the dominant side has flipped."

- **Compute** (per-day reset at 09:15):
  ```
  ce_tot(t) = sum( ce_oi(09:15) … ce_oi(t) )
  pe_tot(t) = sum( pe_oi(09:15) … pe_oi(t) )
  ```
- **Trigger** (per-day, gated to **09:24 – 14:45 IST**):
  - **LONG**  when `pe_tot − ce_tot` crosses up
  - **SHORT** when it crosses down
- **Multi-entry per day allowed** (back-to-back flips as the lines oscillate)
- **Min-gap-7 whipsaw filter baked in:** a new cross is only valid if at
  least 7 minutes have passed since the previously accepted cross on the
  same day. Pass `min_gap_bars=0` to `compute_entries()` to disable.
- Implementation: `strategies/sigma_total_oi_entry.py`

### 2.3 · Why both

Σ-10 catches **smoothed regime shifts** that take several candles to
confirm. ΣOI catches **sustained, day-cumulative imbalances** that
survive the daily noise floor. Run together, the combined trade stream
beats either method alone on PF and total PnL.

---

## 3 · From signals to trades

`/api/merged_trades` applies four layers of trade management:

### 3.1 · Same-side 7-minute dedup *(mode=both only)*

If Σ-10 and ΣOI fire the same side within 7 minutes on the same day,
only the first signal is kept. Filters out the trivial case where the
two methods agree near-simultaneously.

### 3.2 · Next-candle entry

A cross detected on candle `C` does **not** fill on `C`. It fills on the
**open of candle `C+1`**. Leak-free (no peek at the cross-candle's
future), matches how a live trader executes.

### 3.3 · Position-overlap (`single` vs `opposite_ok`)

| Mode | Rule |
|---|---|
| **`single`** *(default)* | Block ALL new signals while any trade is open |
| `opposite_ok` | Block only same-side re-entries; opposite signals allowed |

Backtest comparison (both methods, MULTI trail, current data):

| Position mode | Trades | WR% | PF | Total PnL | Avg/trade |
|---|---:|---:|---:|---:|---:|
| **`single`** *(default)* | 1636 | 58.4 % | **1.42** | +9527 | +5.82 |
| `opposite_ok` | 1845 | 58.3 % | 1.39 | +10115 | +5.48 |

`single` wins on per-trade edge and PF; `opposite_ok` gets slightly more
PnL via more trades. **`single` is the live default** for cleaner
execution.

### 3.4 · Exit rules

Walk the 1-min NIFTY close forward from `entry_time`:

| Exit | Trigger |
|------|---------|
| **TGT**   | Close reaches `entry × (1 + tgt_pct)` for LONG (mirror SHORT) |
| **SL**    | Close reaches `entry × (1 − sl_pct)` BEFORE the trail activates |
| **TRAIL** | After trail activation, close reaches the trailed stop |
| **EOD**   | Day's last 1-min bar (~15:29) — square-off if no other hit |

The trail is **MULTI-STEP staircase**: once the close moves into
profit by `trail_pct` points, the SL ratchets to
`close − trail_distance`. Every subsequent advance ratchets it
further; the SL **never** moves against the trade.

### 3.5 · Live locked defaults

| Knob | Value | Approx at 22 000 spot |
|---|---|---|
| `mode` | `both` | – |
| `position_mode` | `single` | – |
| `sl_pct` | 0.18 % | ~40 pts |
| `tgt_pct` | 0.50 % | ~110 pts |
| `trail_pct` | 0.135 % | ~30 pts |
| `trail` | ON (MULTI) | – |

These were picked from the standalone trail-variant analysis (see §6) —
**30-pt MULTI** was the win-rate winner.

### 3.6 · Live performance (current data)

```
1636 trades · 1.41 / day
WR 58.4 %  ·  PF 1.42  ·  Total PnL +9527 pts
SL / TGT / EOD / TRAIL = 649 / 129 / 106 / 752
```

---

## 4 · Reading the live UI

```
─ Top bar ──────────────────────────────────────────────────────────────
◆ STRATEGY SUITE   TF[3m]   FIT ALL   PnL   [LAST 24123.45]   bars/days/range
─ TRADES panel ────────────────────────────────────────────────────────
▶ TRADES   Both · Single                                       <live stats>
─ 3-minute candle chart ───────────────────────────────────────────────
  trade arrows + exit dots · hover for SL/Target/Exit/P&L
─ Cumulative-PnL pane ─────────────────────────────────────────────────
  CUM PnL · <final> · max <peak> · min <trough> · DD <max DD>
  area chart spanning the full 6-year trade history, year/month x-axis
```

### Marker colors

| Element | Color | Meaning |
|---|---|---|
| Entry arrow ▲▼ | 🟢 green | Trade closed in profit (TGT or trailed-into-profit) |
| Entry arrow ▲▼ | 🔴 red | Trade closed at SL (initial stop hit) |
| Entry arrow ▲▼ | 🟡 amber | Trade closed via TRAIL at exact breakeven |
| Exit dot ● | 🟢 green | TGT hit |
| Exit dot ● | 🔴 dark-red | SL hit |
| Exit dot ● | 🟡 amber | TRAIL exit |
| Exit dot ● | ⚫ grey | EOD square-off |

### Hover tooltip

Floating rectangle next to the cursor showing **side · day · entry ·
SL · target · exit · reason · P&L** (in pts and %). Auto-flips position
to stay inside the chart bounds.

### Top-bar `PnL` button

Show/hide the entire cumulative-PnL pane. Lit blue when visible.

---

## 4.5 · LIVE mode

The suite runs **live during NSE hours** — today's entries appear on the
chart within ~1 minute of the cross.

### Data side — `live_strategy_worker.py` (`investeq-live-strategy.service`)

Every 60 s during the session (Mon–Fri 09:15–15:35 IST) the worker:

1. Appends the latest **1-min NIFTY bars** to `DATA/nifty_1m_master.parquet`
   (historical-candles endpoint serves index bars intraday).
2. Reads **live OI** for the 6 ATM±1 legs (CE/PE × ATM−50/ATM/ATM+50,
   nearest weekly expiry) via `/v1/live-data/quote` — this endpoint carries
   `open_interest` intraday, unlike the historical archive which settles
   OI overnight — and appends one `(timestamp, ce_oi, pe_oi, atm, spot)`
   row to `DATA/_atm_oi_intraday.parquet`.

Today's live OI rows are **transient**: next morning the 09:00 IST
`investeq-oi-daily` timer rebuilds the aggregate from settled per-strike
data, replacing them with the canonical series. At startup the worker
reconciles live-vs-settled OI units (median settled/previous-OI ratio
across the 6 legs) and scales if they diverge beyond 2×.

### App side — mtime-keyed caches + historical/today split

Every cache in `app.py` is keyed on the parquet's mtime, so each worker
append busts it on the next request. Entry computation is split: both
strategies reset per-day, so **historical days are computed once per data
version** (~40 s, re-warmed only after the morning rebuild or a restart)
and **only today's slice is recomputed per request** (~1 s warm).

### UI — `LIVE` button

Top-bar `LIVE` toggle (default ON): refetches candles + merged trades
every 60 s during market hours. Today's trade pops onto the chart with
its entry arrow ~1 min after the cross candle closes.

---

## 5 · Backtest notebook

`notebooks/oi_strategy_backtest.ipynb` — runnable end-to-end.

- Loads `DATA/_atm_oi_intraday.parquet` + `DATA/nifty_1m_master.parquet`
- Generates entries for **Σ-N** (windows 5/10/15/20) and **ΣOI** (with min-gap-7)
- Walks the vectorised SL/TGT exit on every (strategy, window, SL, TGT) combo
- Grid: SL `0.10 – 0.25 %` × TGT `0.20 – 0.50 %` (the realistic intraday band)
- Outputs:
  - Per-config rows in a single dataframe
  - SL × TGT heatmaps for total PnL and win-rate
  - Σ-N window sensitivity plot
  - Per-strategy equity curve
  - Single-best-config per-trade breakdown

The notebook is locked to **3-minute TF only** so the numbers match the
live app exactly.

---

## 6 · Trail-SL variant analyser

`../analyze_trail_variants.py` — stand-alone CLI.

Compares 5 trail-SL variants on the merged Σ-10 + ΣOI stream:

| Variant | Description |
|---|---|
| OFF | No trail — first SL or TGT hit wins |
| 40-pt ONE-SHOT | When in profit by 40 pts, SL → entry. Stays there. |
| 30-pt ONE-SHOT | When in profit by 30 pts, SL → entry. Stays there. |
| 40-pt MULTI    | Staircase: SL = max(SL, close − 40) once activated |
| 30-pt MULTI    | Staircase: SL = max(SL, close − 30) once activated |

Sample output (current data, SL=0.18 % / TGT=0.50 %, position=single):

```
Variant            n    /day   WR%   TotPnL   PF    SL/TGT/EOD/TRL
OFF             1441   1.29  37.8%  +8216   1.29   839/306/296/  0
40pt ONE-SHOT   1472   1.31  32.5%  +8732   1.36   734/286/233/219
30pt ONE-SHOT   1490   1.33  29.7%  +9260   1.42   666/270/202/352
40pt MULTI      1505   1.34  47.6%  +7713   1.31   747/225/178/355
30pt MULTI      1554   1.39  53.2%  +7798   1.34   694/172/115/573
```

**Decision rule** to pick the live default: highest win-rate at
acceptable PF → **30-pt MULTI**.

---

## 7 · REST API surface

The FastAPI app at `app.py` serves both the HTML UI and these endpoints:

| Method · Path | Returns |
|---|---|
| `GET /` | Strategy-suite HTML |
| `GET /api/nifty?tf=3m` | OHLC candles at the requested TF |
| `GET /api/nifty_oi?tf=3m` | CE/PE OI line series for the OI sub-pane |
| `GET /api/strategy/sigma_total_entries?min_gap=7` | Raw ΣOI entry signals |
| `GET /api/strategy/sigma5_entries?tf=3m` | Raw Σ-N entry signals (TF-aware) |
| `GET /api/merged_trades` | **Headline endpoint** — merged trades + stats |
| `GET /api/range` | First / last / row-count metadata |

### `/api/merged_trades` query parameters

| Param | Default | Notes |
|---|---|---|
| `mode` | `both` | `sigma10` · `sigma_oi` · `both` (merged + dedup) |
| `sl_pct` | `0.18` | Initial stop, % of entry spot |
| `tgt_pct` | `0.50` | Target, % of entry spot |
| `trail` | `true` | Multi-step trail SL on / off |
| `trail_pct` | `0.135` | Trail step, % — decoupled from `sl_pct` |
| `position_mode` | `opposite_ok` * | UI sets `single` by default |

Response carries `trades[]` (each with `source: "sigma10" | "sigma_oi"`)
and a `stats{}` block including `n`, `win_rate`, `pf`, `total_pnl`,
`sl_hits / tgt_hits / eod_exits / trail_hits`, `avg_per_day`,
`sigma10_n`, `sigma_oi_n`.

---

## 8 · Running locally

### Prereqs

- Python 3.11+
- The two parquet inputs in `../DATA/`:
  - `nifty_1m_master.parquet` (OHLC for 1-min NIFTY)
  - `_atm_oi_intraday.parquet` (per-tick ATM±1 OI aggregate)
- Deps: `fastapi uvicorn pandas numpy matplotlib` (+ `jupyter` for the notebook)

### Start the strategy suite

```bash
# from the project root:
python -m uvicorn oi_options_trading.app:app --host 127.0.0.1 --port 8705
# → open http://localhost:8705/ in a browser
```

First load of `/api/merged_trades` cold-warms the strategy caches
(~30 – 60 s on first hit). Every subsequent request is < 1 s — only the
exit walk re-runs.

### Run the backtest notebook

```bash
jupyter notebook oi_options_trading/notebooks/oi_strategy_backtest.ipynb
```

### Run the trail-variant analyser

```bash
python analyze_trail_variants.py
```

---

## 9 · Deploying

`scp` the changed files to the VM and bounce the systemd unit:

```bash
scp -i ~/.ssh/gcp_ajay oi_options_trading/app.py \
  ajay@34.93.70.239:/home/ajay/investeq_ajs/oi_options_trading/app.py
ssh -i ~/.ssh/gcp_ajay ajay@34.93.70.239 \
  "sudo systemctl restart investeq-strategy"
```

Or use the full deployer for big bundles:

```bash
bash deploy/deploy.sh
```

Service runs uvicorn on `127.0.0.1:8705` behind nginx at `/strategy/`.

---

## 10 · Operating notes

- **OI cadence is ~3 minutes**, not every minute. The 1-min OI parquet
  repeats the last observed value between updates. ΣOI's
  `pe_tot − ce_tot` stays flat between ticks, so false crosses don't
  fire mid-gap. Verified empirically (61 % of consecutive 1-min ticks
  have identical OI ≈ the 2/3 expected for a 3-min cadence).
- **TRAIL exits are not all breakeven.** With MULTI staircase trail, a
  TRAIL exit can happen at any SL above entry once the close has
  advanced. Most TRAIL exits in the live config are wins.
- **Win-rate counting**: `pnl_pts > 0` = win. A TRAIL that exited
  exactly at breakeven counts as a loss; one that ratcheted up first
  counts as a win.
- **Why no Avg PnL in the UI?** The honest `total_pnl / n` is +5.82, but
  the API also exposes an `avg_pnl` that excludes TRAIL — which is
  misleading on MULTI mode (most trades resolve as TRAIL, leaving only
  SL hits in the subset). UI shows only `total_pnl` and `pf` to avoid
  the confusion.

---

## 11 · File layout

```
oi_options_trading/
├── app.py                            FastAPI strategy suite
├── README.md                         this file
└── notebooks/
    └── oi_strategy_backtest.ipynb    3-min backtest grid + heatmaps

strategies/
├── sigma5_entry_exit.py              Σ-N rolling cross
├── sigma_total_oi_entry.py           ΣOI cumulative cross (min-gap-7)
└── multi_strike_oi_crossover.py      legacy baseline + crossover

analyze_trail_variants.py             trail-variant comparison CLI
build_atm_oi_intraday.py              rebuilds DATA/_atm_oi_intraday.parquet
```
