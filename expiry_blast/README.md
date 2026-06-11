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
* **The original spec values produce 0 entries in 6 years.** The binding
  constraint is condition C's strict form: the 2nd strike below the wall
  lags the 1st by several minutes, so "both ≥ 20 % in the same 15-min
  window" never coincides with the breakout (A·B·D aligned 115 times in
  2025–26; C was never simultaneously true on both strikes).

### Relaxation sweep (8 variants × 299 expiry days)

`analyze_sweep.py` + `bracket_sim.py`. Holding to EOD is a bad metric for
expiry-day longs (theta), so quality is judged with a bracket exit on the
premium — TP +80 % / SL −40 %, whichever 1-min bar prints first
(same-bar → SL, conservative):

| Variant | C rule | B | A | n | /yr | win % | expectancy |
|---|---|---|---|---:|---:|---:|---:|
| spec+1of2 | 1of2 @ 20 % | 10 % | 0.15 % | 7 | 1.1 | 29 % | −0.14 R |
| **b10 (default)** | **1of2 @ 10 %** | 10 % | 0.15 % | **14** | **2.2** | **43 %** | **+0.29 R** |
| b5 | 1of2 @ 5 % | 10 % | 0.15 % | 22 | 3.4 | 41 % | +0.23 R |
| u5 | 1of2 @ 10 % | 5 % | 0.15 % | 22 | 3.4 | 41 % | +0.23 R |
| noC | C disabled | 10 % | 0.15 % | 41 | 6.4 | 37 % | +0.10 R |
| loose | C off | 5 % | 0.25 % | 60 | 9.4 | 30 % | −0.10 R |

Takeaways:

* **C = 1-of-2 @ +10 % is the sweet spot** — now the shipped default
  (~2.2 trades/yr, best per-trade edge). `--buildup 5` or `--unwind 5`
  buys ~3.4/yr at slightly lower edge; dropping C entirely (6.4/yr) keeps a
  thin positive edge; loosening A/B beyond that destroys it.
* Widening proximity (0.25 %) adds **zero** trades once C is relaxed — A is
  not the binding constraint. Keep 0.15 %.
* The bracket shape matters: symmetric ±50 % is breakeven at best — the
  edge only shows with asymmetric cut-losers/let-winners-run (median
  post-entry max is ~1.9× entry across variants). Phase-2 exits should be
  built that way.
* All samples are small (n ≤ 60) — treat expectancies as direction, not
  gospel.

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
