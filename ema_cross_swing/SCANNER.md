# Investeq EMA21/50 Swing-Low Scanner — System Reference

> Production deployment: `http://34.93.70.239/scan/`
> VM: GCP `asia-south1-c`, 2-core, `ajay@34.93.70.239`
> Code root: `/home/ajay/investeq_ajs/ema_cross_swing/`
> Data root: `/home/ajay/investeq_ajs/ema_scanner/data/`

This document captures the runtime architecture, caching strategy, data
synthesis rules, and operational schedule. It complements (does not replace)
the source files: `scan_app.py`, `scan_lib.py`, `scan_universe.py`,
`strategy_core.py`.

---

## 1. What the scanner does

For a chosen `target_day` and across the full NSE universe (~523 symbols on
1h, ~1047 1d+1h jobs combined), find every EMA21/50 **swing-low touch** (BUY
side) and **swing-high touch** (SELL side) that fired on that day. For each
hit, compute:

| Column | Meaning |
|---|---|
| Symbol / Side / TF | NSE symbol, BUY or SELL, "1d" or "1h" |
| Touch Time | Timestamp of the touch candle (IST) |
| Entry | Close of the touch candle |
| SL | Swing-based stop (low for BUY, high for SELL), capped to candle extreme |
| Risk % | `abs(entry - SL) / entry × 100` |
| Current | Most recent Close for the symbol |
| P&L % | If SL hit post-touch → `-Risk%`; else `(current - entry) / entry × 100` |
| Peak % | Max favourable excursion between touch and SL-hit (or "now") |
| Status | OPEN / SL HIT / NO TRADE (if Risk% > user-configured max) |

The exact gate logic lives in `strategy_core.py` (`scan_buy_signals`,
`scan_sell_signals`, `compute_swing_lows`, `find_swing_sl`). Defaults:
`fast=21`, `slow=50`, `compulsory=2`, `SWING_WINDOW=2`.

---

## 2. Repository layout

```
ema_cross_swing/
├── scan_app.py          # Streamlit UI (mounted at /scan)
├── scan_lib.py          # Pure-Python scan library (no Streamlit dep)
├── scan_universe.py     # Headless cron entry-point
├── strategy_core.py     # EMA / swing / touch primitives
└── SCANNER.md           # this document

ema_scanner/data/
├── 1d/  <SYM>_historical.csv    # daily OHLCV, end-of-day updated
├── 1h/  <SYM>_historical.csv    # hourly OHLCV, refreshed hourly
├── 1m/  <SYM>_historical.csv    # 1-minute OHLCV (~124 symbols only)
└── scan_cache/
    └── scan_<YYYY-MM-DD>.parquet   # one per scanned day
```

---

## 3. Data flow (single chart)

```
            +-----------------------+
            |  NSE / broker feed    |
            +-----------+-----------+
                        |
            +-----------v-----------+
            | live_workers.py       |     systemd timers (Mon..Fri, IST)
            |  candles --interval=  |        09:16, 10:16, ... 15:16, 15:31  (1h)
            +-----------+-----------+        15:35                            (1d)
                        |
                        v
            ema_scanner/data/<tf>/<SYM>_historical.csv
                        |
                        v
            +-----------+-----------+
            | scan_lib.load_ohlc()  |  ← repairs missing Open from 1m bar
            +-----------+-----------+
                        |
                        v
            scan_lib.signals_on_day()  ← synthesises today's 1d from 1h
                        |
        +---------------+---------------+
        |                               |
+-------v---------+        +------------v------------+
| scan_universe   |        | scan_app.py (UI)        |
|  (cron)         |        |  Threaded background    |
|  writes parquet |        |  scan + Stop button     |
+--------+--------+        +------------+------------+
         |                              |
         v                              v
ema_scanner/data/scan_cache/        st.session_state
  scan_<date>.parquet  ←─────  loaded on Run scan button
```

---

## 4. The `load_ohlc` repair pipeline

`scan_lib.load_ohlc` reads `<SYM>_historical.csv` and returns a clean
DateTime-indexed OHLCV frame. It applies two repairs in order:

### 4.1  Open recovery from 1m

For 1d and 1h timeframes, if the `Open` column has any NaN:

1. For each missing bar at index `idx`:
   - **1d**: `period_start = idx.normalize() + 09:15`, `period_end = + 1 day`
   - **1h**: `period_start = idx`, `period_end = idx + 1 hour`
2. Load the symbol's 1m CSV (LRU-cached by mtime → cache busts when the live
   fetcher updates the file).
3. Find the **first 1m candle whose `datetime` falls inside
   `[period_start, period_end)`** and use its `open`.
4. Cache the value on `df.at[idx, "Open"]`.

For 1d this means the recovered Open is the **09:15 IST 1m candle's open** —
i.e., the actual opening tick of the session.

Code: `scan_lib.py:35-67` (`_load_1m_opens`, `_recover_open_from_1m`)
and `scan_lib.py:84-94` (the repair loop).

### 4.2  Final fallback

Any Open still NaN after step 4.1 is filled with the **previous bar's Close**
(`df["Open"].fillna(df["Close"].shift(1))`). This protects symbols that
aren't in the 1m universe (~400 of 523 symbols don't have a 1m CSV).

### 4.3  Drop rule

Rows where High/Low/Close are NaN are dropped (`df.dropna(subset=[...])`).

---

## 5. Today's forming 1d bar — synthesis

The 1d candle for *today* is not written to the CSV until **15:35 IST** (when
`investeq-candles-1d.timer` fires). For scans run earlier in the day, today's
1d bar is **synthesised on the fly** from today's available 1h bars:

```
synth_1d_open    = first  1h bar's Open   (= 09:15 1h bar Open = day's open)
synth_1d_high    = max    of today's 1h Highs
synth_1d_low     = min    of today's 1h Lows
synth_1d_close   = last   1h bar's Close  (most-recently-closed hour)
synth_1d_volume  = sum    of today's 1h Volumes
```

Code: `scan_lib.py:100-117` (`_synth_today_1d_from_1h`), invoked at
`scan_lib.py:144-149` inside `signals_on_day`. The synth bar is concatenated
onto the loaded 1d series before EMAs are computed.

### Caveats

- **Open is exact.**
- **Close lags by up to 1h** — it's the last *completed* hour's close.
- **High/Low are conservative** — only across completed hours. Intra-hour
  extremes don't show until that hour finalises at the next `:15`.
- For symbols that don't have today's 09:15 1h bar yet (~38% of universe
  during the early morning), the synth returns `None` → no 1d signal for
  that symbol today.

### Through-day timeline of the synth

| IST | 1h bars available | Synth 1d represents |
|---|---|---|
| 10:16 | 09:15 only | 09:15 → 10:15 |
| 11:16 | + 10:15 | 09:15 → 11:15 |
| 12:16 | + 11:15 | 09:15 → 12:15 |
| 13:16 | + 12:15 | 09:15 → 13:15 |
| 14:16 | + 13:15 | 09:15 → 14:15 |
| 15:16 | + 14:15 | 09:15 → 15:15 |
| 15:31 | + 15:15 | full session |
| 15:35 | real 1d in CSV | synth no longer used |

---

## 6. Per-date parquet cache

### 6.1  Format

`ema_scanner/data/scan_cache/scan_<YYYY-MM-DD>.parquet` — one row per signal,
columns = `SCAN_COLS + ["_split"]` where `_split ∈ {"1d","1h"}`. Saved with
`pd.to_parquet(..., index=False)`. Attribute `saved_at` is written on load.

### 6.2  Write path

`scan_universe._save_scan_cache` is the only writer — invoked from:

- `scan_universe.scan_for_day` (cron / backfill)
- `scan_app.py` after a UI-triggered scan completes (so a manual scan also
  populates the cache for future fast loads)

The parquet at `scan_<today>.parquet` is **overwritten** on each run — no
hourly suffix files, no cleanup needed.

### 6.3  Read path

`scan_app._load_scan_cache_for_day(target_day)` reads the parquet in <50ms
and returns `(df1d, df1h, saved_at_iso)`. On Run scan press, the UI
short-circuits to the cache if a parquet exists for the target day — for
today's date too, since the hourly refresh keeps it fresh (see §7).

To force a live recompute regardless of cache: press the **Re-run scan**
button (it bypasses the short-circuit when the user explicitly wants fresh
data, e.g. immediately after a new 1h bar closes but before the next :20
cache fire).

### 6.4  History-replay refresh on load

The parquet freezes `Current` / `P&L %` / `Peak %` / `Status` at scan
time. Loading a past-date parquet a day or a week later would otherwise
show the trade exactly as it looked at scan time — stale. To fix this,
`_load_scan_cache_for_day` runs each split frame through
`_refresh_metrics_from_history(df, tf)` before returning.

For every row:

1. Reload the symbol's OHLC for that TF (today's CSV, cached by
   `@st.cache_data` on `load_ohlc`).
2. Take the latest Close → write to `Current`.
3. Walk `[Touch Time + 1 bar … latest]` for an SL hit:
   - **BUY**:  `Low ≤ SL` anywhere in that window
   - **SELL**: `High ≥ SL` anywhere in that window
   - First hit (if any) freezes `Status = SL HIT` and `P&L % = -Risk%`
4. Compute `Peak %` as the max favorable move from Entry across
   `[Touch Time + 1 bar … (SL hit or latest)]`.
5. Open trades (no SL hit) set `P&L % = (latest − entry)/entry`
   (sign-flipped for SELL).

**NO TRADE rows stay NO TRADE** — the tradability decision (risk% vs the
scan's `max_sl_pct`) was final at scan time. Their dynamic columns still
update so the user can see how the un-tradable signal evolved.

Cost: ~1–2 s for a typical day's ~100 signals, paid once per Run press
(session state holds the refreshed frames after that). Per-symbol OHLC
reads ride the existing `@st.cache_data(ttl=600)` on `load_ohlc`, so
viewing a past date after viewing today is near-free.

Today's intraday LTP layer (`_live_refresh_metrics`) still runs on top
during market hours when the Live toggle is on — it overlays live ticks
on whatever this history pass produced.

---

## 7. Scheduling — systemd timers

All timers run `Mon..Fri` only. NSE session: 09:15 → 15:30 IST.

### 7.1  `investeq-candles-1h.timer`

Fires at **:16 IST** past every market hour + once at **15:31**. Triggers
`live_workers.py candles --interval 1h`, which writes/appends today's just-
closed 1h bar to each `1h/<SYM>_historical.csv`.

Schedule: `09:16, 10:16, 11:16, 12:16, 13:16, 14:16, 15:16, 15:31` IST.

### 7.2  `investeq-candles-1d.timer`

Fires once at **15:35 IST** (10:05 UTC). Writes today's completed 1d bar
to each `1d/<SYM>_historical.csv`.

### 7.3  `investeq-scan-cache.service` — chained off candle fetchers

The scan-cache run is **not** triggered on a fixed clock offset. Instead, it
fires immediately after every successful candle-fetcher run via systemd's
`OnSuccess=` directive (systemd ≥ 249).

```
# /etc/systemd/system/investeq-candles-1h.service  [Unit]
OnSuccess=investeq-scan-cache.service

# /etc/systemd/system/investeq-candles-1d.service  [Unit]
OnSuccess=investeq-scan-cache.service
```

Why this design (option B from operational review):

- **No fixed offset → no race.** A `:20` IST scan could race a slow `:16`
  fetcher; chaining off `OnSuccess` only fires *after* the fetcher's
  oneshot process has exited 0.
- **No wasted clock entries.** No `OnCalendar=Mon..Fri 10:20` /11:20 /...
  list to maintain.
- **Each fetch run gets its own scan**, including the bonus 09:16 IST fire
  (writes nothing useful since markets just opened, but the chained scan is
  harmless — it just overwrites the parquet with the previous day's view).
- **`Type=oneshot` serialises overlap.** If scans take 5 min and a fetch
  fires every hour, no instance can collide with another.

Timeline on a normal trading day (e.g., 2026-06-05):

| IST | Event | Result in `scan_<today>.parquet` |
|---|---|---|
| 09:16 | 1h fetch (no new bar — markets just opened) → chained scan | mostly empty for today |
| 10:16 | 1h fetch (09:15 bar) → chained scan completes ~10:22 | 1h signals on 09:15 bar; synth 1d from 09:15 bar |
| 11:16 | 1h fetch (10:15 bar) → chained scan completes ~11:22 | + 1h signals on 10:15; richer synth 1d |
| 12:16 → 15:16 | each hour, ~6-min cycle | progressively richer cache |
| 15:31 | 1h fetch (15:15 bar) → chained scan | full 1h day; synth 1d from all 7 bars |
| 15:35 | 1d fetch → chained scan completes ~15:41 | **definitive** EOD parquet with real 1d bar |

### 7.4  Catch-up safety net + reboot recovery

The `investeq-scan-cache.timer` is retained as a single-entry safety net:

```
[Timer]
OnCalendar=Mon..Fri 10:15:00 UTC          # 15:45 IST, 10 min after EOD chain
Persistent=true
```

Purpose:

- **Reboot catch-up.** `Persistent=true` causes systemd to fire once on
  boot if the scheduled time was missed while the VM was down — guarantees
  at least one definitive end-of-day pass per trading day.
- **Belt-and-braces.** If a candle service ever exits 0 but the `OnSuccess`
  link silently fails (extremely rare, but covers config-reload edge
  cases), this entry runs the scan anyway.

The 15:45 IST slot is **10 minutes after** the chained EOD scan would have
finished, so on a normal day this fires and is a no-op overwrite of the
already-good parquet.

---

## 8. Date classification (UI badge under the date picker)

`scan_app._classify_date(d)` returns one of `"trading"`, `"holiday"`,
`"weekend"`, `"future"`:

1. **Weekend** — `d.weekday() >= 5`.
2. Build set of NSE trading days from:
   - The union of `datetime` columns of the **first 10 alphabetically-sorted
     1d CSVs** (using `pd.read_csv(..., usecols=["datetime"])` for speed).
   - The set of dates already in `scan_cache/` (cached scans imply the
     date was a trading day).
3. **Trading** — `d` is in the set.
4. **Future** — `d > max(trading_days_set)`.
5. **Holiday** — neither weekend, nor in set, nor future.

Why "10 sorted CSVs" instead of `iterdir`? Filesystem inode order is **not
alphabetical** — early picks like KIRLOSENG were lagging by 2 days and
mis-classified 2026-06-04 as "future". Sorting + union with cached scans is
robust.

Code: `scan_app.py:81-130` (helpers), `scan_app.py:~983` (badge render).

The UI shows one of:
- 🏖️ **NSE Holiday** — markets were closed
- 📅 **<Weekday>** — markets don't trade on weekends
- 🚧 **<date>** — no data fetched yet (auto-populates after 15:35 IST)
- (no badge) when it's a normal trading day

---

## 9. UI flow — `scan_app.py`

### 9.1  Initial render

- Date picker defaults to today (IST).
- Classifier badge under it (see §8).
- **Run scan** button. No cache list, no auto-load — the user must press
  the button to see anything.

### 9.2  Pressing Run scan

```python
if run:
    df1d, df1h, saved_at = _load_scan_cache_for_day(target_day)
    if df1d is not None:
        # cache hit — populate session_state, skip live scan
        ...
        run = False
```

`_load_scan_cache_for_day` also pipes both frames through
`_refresh_metrics_from_history` so `Current` / `P&L %` / `Peak %` /
`Status` reflect the latest OHLC, not the frozen scan-time values (see
§6.4).

If no parquet exists for the target day, a **background thread** is started
that runs the same `signals_on_day` per-symbol via a `ThreadPoolExecutor`
(`max_workers=16`). The Streamlit page polls progress with
`st_autorefresh(interval=1000)`.

### 9.3  While the scan runs

- A progress bar shows `[done/total] current_symbol`.
- A **🛑 Stop** button appears next to the progress bar.
- Pressing Stop:
  1. Sets `threading.Event` → futures cancel.
  2. Calls `_reset_scan_state()` → wipes every key in `_SCAN_STATE_KEYS`
     plus the cached result keys.
  3. `st.rerun()` → returns to the initial page state.

### 9.4  Scan completes

Background thread writes its rows to `progress["rows_1d"]` / `["rows_1h"]`
and sets `progress["finished"] = True`. The main script picks up the
results on the next autorefresh tick, materialises them as DataFrames,
saves a parquet to `scan_cache/`, populates `session_state`, clears the
thread keys, and renders the results table.

### 9.5  After results render

The button label switches to **🔁 Re-run scan**. Selecting a different date
clears the stale session state (so old results don't bleed into a new
date's view).

Key code: `scan_app.py:~1080-1230`.

---

## 9b. Telegram live alerts (1h only)

Live 1h touches are posted to a Telegram group as soon as the chained
scan-cache run completes. The hook lives in `telegram_alerts.py` and is
invoked from `scan_universe.py` *only* after the single-day branch finishes
writing the parquet (never from `--last-n` backfill, never from the UI).

### Filter chain (all 6 must pass)

1. **Live run** — `target_day == today IST`. Past-date scans never alert.
2. **A 1h bar has closed today** — pre-10:15 IST runs are skipped.
3. **Bar not yet alerted** — `data/scan_cache/.last_alerted_bar.txt` stores
   the latest alerted bar timestamp; `bar_ts <= last` → skip. Handles
   re-runs, manual fires, and the EOD scan double-firing for 15:15.
4. **Touch on the just-closed bar** — `Touch Time == bar_ts`. (We don't
   alert on stale signals from earlier in the day.)
5. **Status != NO TRADE** — Risk% must be within the cap.
6. **Pierce-both EMAs** —
   `BUY  →  Low_t  <= EMA21(t)  AND  Low_t  <= EMA50(t)`
   `SELL →  High_t >= EMA21(t)  AND  High_t >= EMA50(t)`
   The booleans are computed in `scan_lib.signals_on_day` and persisted in
   the parquet as `LowBelowBoth` / `HighAboveBoth`.

### Message format

One Markdown message per chained run:

```
*1h EMA21/50 signals — 2026-06-05 11:15 bar*
_3 touch(es) — pierce both EMAs_

```
Sym        Side  Entry        SL  Risk%
TCS        BUY   2246.50  2221.10   1.13
INFY       BUY   1452.30  1428.70   1.62
HDFCBANK   SELL  1648.00  1671.40   1.42
```
```

If zero signals survive the filter, the state file is still updated (so
the next chained run doesn't re-check the same bar) but no message is sent.

### Morning brief (09:00 IST Mon..Fri)

Separate from the live intraday alerts, a once-a-day "morning brief" posts
the 1h signals from the **last trading day's 15:15 bar** (the final bar of
the day, covering 15:15 → 15:30 IST). Same NO-TRADE + pierce-both filters
apply. On Monday morning it picks up Friday's parquet automatically.

- Service: `/etc/systemd/system/investeq-morning-brief.service`
  → `ExecStart=...python telegram_alerts.py morning-brief`
- Timer: `OnCalendar=Mon..Fri 09:00 Asia/Kolkata`, `Persistent=true`
- Dedupe state: `data/scan_cache/.last_morning_brief.txt` (one brief per
  trading day; same-day re-run no-ops)
- Source data: `_find_last_trading_day_parquet()` — most recent
  `scan_<date>.parquet` strictly before today, robust to weekends/holidays

### Credentials

`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are stored in
`/etc/investeq.env` (root:root, mode 0600). systemd reads the env file at
service start; rotating the token requires re-saving the env file (no
daemon-reload, no service restart needed beyond the next chained run).

### Op tips

```bash
# Reset the dedupe state (forces the next chained run to re-alert today's
# latest bar — useful after a code change in the filter)
sudo rm /home/ajay/investeq_ajs/ema_scanner/data/scan_cache/.last_alerted_bar.txt

# Disable alerts temporarily without touching code (token blank → hook
# logs "creds missing" and exits cleanly)
sudo sed -i 's/^TELEGRAM_BOT_TOKEN=.*/TELEGRAM_BOT_TOKEN=/' /etc/investeq.env

# Test alert format without a full scan (Python REPL on VM)
cd /home/ajay/investeq_ajs/ema_cross_swing && \
  set -a; source /etc/investeq.env; set +a; \
  ../.venv/bin/python -c "from telegram_alerts import _send_telegram; \
    import os; \
    _send_telegram(os.environ['TELEGRAM_BOT_TOKEN'], \
                   os.environ['TELEGRAM_CHAT_ID'], '*manual test*')"
```

---

## 10. Operational quick-reference

### Common SSH commands

```bash
# tail the live scan-cache run
sudo journalctl -u investeq-scan-cache.service -f

# next fire time of the intraday timer
systemctl list-timers investeq-scan-cache.timer --all --no-pager

# manually rebuild today's parquet (e.g., after fixing a bug)
sudo systemctl start investeq-scan-cache.service

# backfill last N days, skipping ones already cached
cd /home/ajay/investeq_ajs/ema_cross_swing
/home/ajay/investeq_ajs/.venv/bin/python scan_universe.py \
    --last-n 30 --skip-existing

# restart the Streamlit UI
sudo systemctl restart investeq-scan.service
```

### File paths

| Purpose | Path |
|---|---|
| Streamlit unit | `/etc/systemd/system/investeq-scan.service` |
| Scan-cache service | `/etc/systemd/system/investeq-scan-cache.service` |
| Scan-cache timer | `/etc/systemd/system/investeq-scan-cache.timer` |
| 1h fetcher timer | `/etc/systemd/system/investeq-candles-1h.timer` |
| 1d fetcher timer | `/etc/systemd/system/investeq-candles-1d.timer` |
| Env file | `/etc/investeq.env` |
| Venv | `/home/ajay/investeq_ajs/.venv/` |
| Code | `/home/ajay/investeq_ajs/ema_cross_swing/` |
| Data | `/home/ajay/investeq_ajs/ema_scanner/data/` |
| SSH key (from dev box) | `~/.ssh/gcp_ajay` |

---

## 11. Performance notes

| Quantity | Value |
|---|---|
| Symbols in 1h universe | 523 |
| Symbols in 1d universe | similar, ~520 |
| Symbols in 1m universe | 124 (Open-recovery only helps this subset) |
| Total jobs per scan | ~1047 (1d + 1h combined) |
| VM cores | 2 |
| ThreadPoolExecutor workers | 16 (I/O-bound, oversubscription is fine) |
| Typical full-scan time | 3–6 min |
| Parquet load time | <50 ms |
| Memory ceiling per scan | ~150 MB |

The bottleneck is CSV parsing — most symbols have 2–5 years of history. The
LRU cache on `_load_1m_opens` is the only intra-run cache; everything else
is recomputed per scan (deliberately stateless).

---

## 12. Known limitations & future work

- **No live-tick price in synth Close** — synth 1d Close lags by up to 1h.
  A future enhancement could query `live_ltp` for the most recent tick and
  override `synth_close`.
- **1m universe is small (124/523)** — Open repair only helps that subset.
  For others, missing-Open fallback is "prior bar's Close" which is fine
  for the swing-touch strategy but imperfect.
- **Intraday cache uses both 1d and 1h** — the user can press Re-run scan
  for a strict live recompute. (We considered a "1h-only intraday" mode but
  decided against it on user feedback.)
- **No multi-user concurrency control** — if two users hit Run scan on the
  same date simultaneously, both kick off threads. Streamlit's
  `session_state` is per-session so they don't collide, but VM CPU is shared.
- **The 1h fetcher misses some symbols intermittently** — on 2026-06-05 it
  refreshed 324/523 symbols at 10:16. The 132 symbols still on Jun 4 mtime
  and 11 stuck on May 15 need investigation in the fetcher itself.
