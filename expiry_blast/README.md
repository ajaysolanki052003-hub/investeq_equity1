# Expiry Blast — Expiry Day Short Covering (entry engine, phase 1)

Intraday options-buying entry engine for Indian index options (NIFTY /
BANKNIFTY). On the contract's **expiry day only**, it hunts the moment the
market presses through the biggest Call-OI wall while call writers run for
cover and put writers step in underneath — classic short-covering fuel — and
buys the ATM call.

**Phase 1 = entry only.** No exits, no risk management yet; the backtest's
post-entry premium columns are context, not P&L.

## Entry logic

Evaluated on every OI tick (≈3-min cadence) inside the entry window
(10:30–14:45 IST), expiry day only:

| # | Condition | Default |
|---|-----------|---------|
| — | Wall = strike with max total Call OI (expiring contract) | — |
| A | Spot within `SPOT_PROXIMITY_PCT` of the wall | 0.15 % |
| B | Wall CE OI down ≥ `CALL_OI_UNWIND_PCT` over the lookback | 10 % |
| C | PE OI of the 2 strikes below the wall each up ≥ `PUT_OI_BUILDUP_PCT` | 20 % |
| D | Last closed 5-min spot candle closes **above** the wall | — |

All four true simultaneously → **BUY 1 lot ATM CE** (strike nearest spot,
fill proxy = next 1-min bar open). Guardrails: one entry per day, evaluation
skipped (and audited) when the OI feed is stale (> `stale_oi_max_min`), every
condition check logged pass/fail.

## Layout

```
expiry_blast/
├── config.py     BlastConfig — every threshold, nothing index-specific hardcoded
├── data.py       Feed interface · HistoricalFeed (DATA parquets) · LiveFeed stub
├── engine.py     SignalEngine — conditions A–D, audit trail, EntrySignal
├── backtest.py   CLI replay over historical expiry days
└── app.py        FastAPI dashboard (port 8706, nginx /blast/)
```

Data comes from the repo's canonical per-day parquets: `DATA/oi/` (per-strike
CE/PE OI, ~3-min), `DATA/spot/` (1-min index OHLC), `DATA/options/` (1-min
option premiums). 299 expiry days covered, 2020-01 → 2026-04.

## Backtest

```bash
python -m expiry_blast.backtest                          # full history, spec defaults
python -m expiry_blast.backtest --start 2025-01-01 --end 2026-04-30
python -m expiry_blast.backtest --buildup 10 --put-strikes 1
python -m expiry_blast.backtest --audit-all              # audit no-signal days too
```

Outputs in `expiry_blast/output/`: `signals.csv`, `audit.jsonl` (every
condition check), `summary.txt`.

### Validation findings (full history, NIFTY)

* The engine's plumbing is verified: walls are stable intraday, conditions
  fire individually at sane rates (A ~22 % of evals, B ~2.6 %, C ~6 %,
  D ~0.8 % in 2025–26).
* **Spec defaults produce 0 entries in 6 years.** The binding constraint is
  condition C's strict form: the 2nd strike below the wall lags the 1st by
  several minutes, so "both ≥ 20 % in the same 15-min window" never coincides
  with the breakout (A·B·D aligned 115 times in 2025–26; C was never
  simultaneously true on both strikes).
* Genuine short-covering moments do show the pattern on the *nearest* strike
  (e.g. 2026-04-07 13:35: spot through the 23000 wall, CE −10 %+, first PE
  below +22→30 %). `put_strikes_required: 1` (CLI `--put-strikes 1`) is the
  honest relaxation to study.
* With `--put-strikes 1` (every other threshold at spec): **7 entries in
  299 expiry days** (2022-01-27, 2023-03-29, 2024-02-22, 2024-03-07,
  2024-06-27, 2025-03-06, 2026-04-07). Entry-quality context: 71 % had the
  bought premium higher at EOD; median post-entry max premium = 1.95× entry.
  Small sample — context, not P&L.

## Dashboard

```bash
python -m uvicorn expiry_blast.app:app --host 127.0.0.1 --port 8706
```

Pick any expiry day → 5-min candles with the call wall overlaid, entry marker
if fired, and the full tick-by-tick audit table (click a row for the raw
record). Deployed at `/blast/` behind the portfolio login
(`deploy/investeq-blast.service` + nginx location).

## Phase 2 (later)

* Exits / risk management (the spec's "later phase").
* `LiveFeed` — wire `data.py`'s stub to the Groww endpoints already used by
  `ema_scanner/groww_client.py`, run the engine on the live chain during
  expiry-day sessions.
* BANKNIFTY: config supports it (`strike_step` 100); needs its own OI/spot/
  options history in `DATA/`.
