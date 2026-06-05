# NIFTY Option Strategy Backtesting

End-to-end design for backtesting and analysing NIFTY index option strategies on the local dataset at `..\DATA` (≈6 years of 1-minute OHLCV + OI, 2020-01-01 → 2026-04-30). This README is the canonical spec for the project — every strategy, parameter grid, and output convention lives here.

- **Project root:** `C:\Users\User\Desktop\investeq_ajs\OPTION_STRATEGY`
- **Data source:** `C:\Users\User\Desktop\investeq_ajs\DATA` (see `..\DATA\README.md` for schema)
- **Underlying:** NSE NIFTY 50 index options (weekly + monthly expiries)
- **Bar size:** 1-minute (resampleable to 3/5/15/30/60-min)
- **Universe:** every listed strike + expiry for each trading minute
- **Goal:** uniform pipeline to run any single-leg or multi-leg strategy with parameter sweeps and produce comparable P&L / risk reports.

---

## 1. Data layout (cheat-sheet)

Full schemas in `..\DATA\README.md`. The pieces this project consumes:

| Folder / file | What it gives you | Join key |
|---|---|---|
| `..\DATA\options\YYYYMMDD.parquet` | per-minute OHLCV per `(expiry, strike, option_type)` contract | `timestamp, expiry, strike, option_type` |
| `..\DATA\oi\YYYYMMDD.parquet` | per-minute open interest snapshot | same as above |
| `..\DATA\spot\YYYYMMDD.parquet` | per-minute NIFTY index OHLC | `timestamp` |
| `..\DATA\index\expiries.parquet` | full list of expiry dates | — |
| `..\DATA\spot_all.parquet` | concatenated spot bars (2024-01-01+) | `timestamp, date` |

**Identifying a contract:** `(expiry, strike, option_type)` is the unique key. Always join `options/` ← `oi/` with `how="left"` because OI snapshots can be ~6 rows short per day.

**Trading-day calendar:** derive from `spot/` filenames — `trading_days.parquet` is incomplete.

**Timestamps:** strings, IST, no tz suffix. Cast with `pd.to_datetime` and localise to `Asia/Kolkata` only if comparing across tz.

---

## 2. Project layout (proposed)

```
OPTION_STRATEGY/
├── README.md                       # this file
├── config/
│   ├── universe.yaml               # date range, session, capital, slippage, fees
│   └── strategies/                 # one yaml per strategy with parameter grid
│       ├── short_straddle.yaml
│       ├── iron_condor.yaml
│       └── ...
├── src/
│   ├── data/
│   │   ├── loader.py               # parquet readers, caching, lazy windows
│   │   ├── chain.py                # build option chain at a timestamp
│   │   ├── expiry.py               # weekly/monthly/expiry-N resolver
│   │   └── greeks.py               # IV solver + delta/gamma/vega/theta
│   ├── strategies/
│   │   ├── base.py                 # Strategy ABC: entry/exit/sizing/legs
│   │   ├── directional/
│   │   ├── neutral/
│   │   ├── volatility/
│   │   ├── calendar/
│   │   ├── ratio/
│   │   └── event/
│   ├── engine/
│   │   ├── backtest.py             # event loop, P&L mark-to-mid
│   │   ├── portfolio.py            # positions, margin, MTM
│   │   ├── execution.py            # slippage models, lot rounding
│   │   └── risk.py                 # stop-loss, trailing, hedge triggers
│   ├── analytics/
│   │   ├── metrics.py              # CAGR, Sharpe, Sortino, MaxDD, win%, hit-ratio
│   │   ├── attribution.py          # P&L by leg / by greek / by regime
│   │   └── reports.py              # html/markdown + matplotlib plots
│   └── cli.py                      # `python -m option_strategy run <yaml>`
├── notebooks/                      # exploration, ad-hoc charts
├── results/
│   └── <run_id>/                   # one dir per backtest run
│       ├── config.yaml             # frozen copy of the params used
│       ├── trades.parquet
│       ├── equity.parquet
│       ├── greeks.parquet
│       └── report.md
└── tests/
```

A `run_id` is `{strategy}_{params_hash}_{YYYYMMDDHHMM}` so results are reproducible and never overwritten.

---

## 3. Strategy catalogue

Each entry lists: **construction** (legs), **typical bias**, and the **parameter axes** you should sweep on backtest. Lot size for NIFTY changed multiple times in the data window (75 → 50 → 25 → 75); resolve it from the trade date, don't hardcode.

### 3.1 Directional — single leg

| # | Strategy | Legs | View |
|---|---|---|---|
| 1.1 | Long Call | +1 CE | bullish |
| 1.2 | Long Put | +1 PE | bearish |
| 1.3 | Short Call (naked) | -1 CE | bearish / range-bound |
| 1.4 | Short Put (naked) | -1 PE | bullish / range-bound |
| 1.5 | Covered Call (synthetic, long futures + short CE) | +1 FUT, -1 CE | mildly bullish |
| 1.6 | Protective Put | +1 FUT, +1 PE | bullish with floor |

**Parameter axes:** strike offset (ATM, ATM±1, ATM±2 ... ATM±10), expiry bucket (current-week, next-week, current-month, next-month), entry time (9:20, 9:30, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00), exit rule (EOD / on opposite signal / fixed time / trailing SL / target %), stop-loss (% of premium, % of underlying, ATR-based), position size (fixed lots, % capital, vol-targeted).

### 3.2 Directional — vertical spreads

| # | Strategy | Legs | View |
|---|---|---|---|
| 2.1 | Bull Call Spread | +1 CE (lower K), -1 CE (higher K) | bullish, capped |
| 2.2 | Bear Put Spread | +1 PE (higher K), -1 PE (lower K) | bearish, capped |
| 2.3 | Bull Put Spread (credit) | -1 PE (higher K), +1 PE (lower K) | bullish, credit |
| 2.4 | Bear Call Spread (credit) | -1 CE (lower K), +1 CE (higher K) | bearish, credit |
| 2.5 | Diagonal Bull/Bear | long far-month + short near-month, different strikes | trending |

**Parameter axes:** spread width (50 / 100 / 150 / 200 / 300 / 500 pts), distance of short-strike from spot (ITM, ATM, +1σ, +2σ), expiry, roll trigger (Δ-based, % MTM), close-on-expiry-day vs T-1.

### 3.3 Neutral / range-bound — short premium

| # | Strategy | Legs | View |
|---|---|---|---|
| 3.1 | Short Straddle | -1 CE @K, -1 PE @K (same strike) | low vol, pin |
| 3.2 | Short Strangle | -1 CE @K_h, -1 PE @K_l | low vol, wider range |
| 3.3 | Iron Condor | short strangle + long wings | defined-risk neutral |
| 3.4 | Iron Butterfly | short straddle + long wings | tighter defined-risk |
| 3.5 | Jade Lizard | short put + short call spread | mildly bullish, no upside risk |
| 3.6 | Big Lizard | short straddle + long OTM CE | similar with tighter call hedge |

**Parameter axes:**
- **Strike selection:** ATM / by Δ (0.10, 0.15, 0.20, 0.25, 0.30) / by % away from spot (0.5, 1, 1.5, 2, 3) / by 1σ-expected-move
- **Wing width** (for defined-risk): 100 / 200 / 300 / 500 pts
- **Entry day relative to expiry:** T-7, T-5, T-3, T-2, T-1, T-0 (intraday)
- **Entry time of day**
- **Stop-loss:** % of premium collected (25/50/75/100%), spot move (±0.5%, ±1%), Δ breach (e.g. short-leg Δ → 0.40)
- **Take-profit:** 25/50/75% of max premium
- **Adjustment rules:** roll un-tested side, convert to IC, add hedge when spot breaches 1σ
- **Re-entry:** allow / disallow after SL

### 3.4 Volatility — long premium

| # | Strategy | Legs | View |
|---|---|---|---|
| 4.1 | Long Straddle | +1 CE @K, +1 PE @K | big move, either way |
| 4.2 | Long Strangle | +1 CE @K_h, +1 PE @K_l | big move, cheaper |
| 4.3 | Reverse Iron Condor | long strangle + short wings | defined-cost vol play |
| 4.4 | Reverse Iron Butterfly | long straddle + short wings | same, tighter |
| 4.5 | Strap | +2 CE +1 PE | bullish bias on vol |
| 4.6 | Strip | +1 CE +2 PE | bearish bias on vol |

**Parameter axes:** entry around scheduled events (RBI policy, Fed, Budget, Union Election results, monthly expiry, earnings days near index), IV percentile filter (only enter when IV-rank < X), holding period (intraday, overnight, T-to-event+1).

### 3.5 Calendar / time spreads

| # | Strategy | Legs | View |
|---|---|---|---|
| 5.1 | Call Calendar | -1 CE near expiry, +1 CE far expiry, same K | theta + IV rise |
| 5.2 | Put Calendar | -1 PE near expiry, +1 PE far expiry, same K | same |
| 5.3 | Double Calendar | calendars at two strikes | wider neutral |
| 5.4 | Diagonal Calendar | calendar + different strikes | directional + theta |

**Parameter axes:** near-leg expiry (current-week vs current-month), far-leg expiry (next-month, +2 months), strike (ATM / ±1σ), roll trigger.

### 3.6 Ratio / Backspreads

| # | Strategy | Legs | View |
|---|---|---|---|
| 6.1 | Call Ratio Spread | +1 CE, -2 CE (higher K) | mildly bullish, vol crush |
| 6.2 | Put Ratio Spread | +1 PE, -2 PE (lower K) | mildly bearish, vol crush |
| 6.3 | Call Backspread | -1 CE, +2 CE (higher K) | sharp rally hedge |
| 6.4 | Put Backspread | -1 PE, +2 PE (lower K) | sharp crash hedge |
| 6.5 | Broken-Wing Butterfly | asym wings | skew play |

**Parameter axes:** ratio (1:2, 1:3, 2:3), strike distances, expiry.

### 3.7 Event-driven & intraday

| # | Strategy | Trigger |
|---|---|---|
| 7.1 | Expiry-day Short Straddle (intraday) | enter 9:20–10:00 on expiry day, exit 15:15 |
| 7.2 | Expiry-day Iron Fly | same with wings |
| 7.3 | ORB (Opening Range Breakout) — directional CE/PE | first 15-min range break → buy CE/PE |
| 7.4 | Gap-fade / Gap-fill | overnight gap > X% → fade with credit spread |
| 7.5 | Event vol-crush short straddle | enter day-before-event, exit on event day open |
| 7.6 | Pre-event long straddle | enter T-3, exit T-0 morning |
| 7.7 | VIX-spike short premium | enter when India-VIX > Nth percentile |

### 3.8 Hedged / portfolio overlays

| # | Strategy | Notes |
|---|---|---|
| 8.1 | Collar | +FUT + long PE - short CE |
| 8.2 | Tail-risk hedge | rolling deep OTM long PE (1–3% of capital/month) |
| 8.3 | Delta-hedged short straddle | rebalance Δ to 0 at fixed frequency or Δ-threshold |
| 8.4 | Gamma-scalp long straddle | rebalance underlying to lock realised gamma |

### 3.9 Statistical / signal-driven

| # | Strategy | Signal |
|---|---|---|
| 9.1 | IV-Rank mean-reversion | short premium when IV-rank > 80, long when < 20 |
| 9.2 | Skew-trade | long the rich side / short the cheap side of put-call skew |
| 9.3 | Term-structure | front-month vs back-month IV slope |
| 9.4 | OI-shift | enter direction of largest CE/PE OI build-up |
| 9.5 | Max-Pain anchor | bias spot to expiry-day max-pain strike |
| 9.6 | PCR (Put-Call Ratio) bands | contrarian directional |

---

## 4. Cross-cutting parameter axes

Every strategy is parameterised by some subset of the following — keep names consistent across yaml configs:

| Axis | Common values |
|---|---|
| `entry_time` | 09:20, 09:30, 09:45, 10:00, 11:00, 12:00, 13:00, 14:00, 14:30, 15:00 |
| `exit_time` | 15:10, 15:15, 15:20, EOD-1min |
| `expiry_bucket` | `weekly_current`, `weekly_next`, `monthly_current`, `monthly_next` |
| `strike_method` | `atm`, `atm_offset:N`, `delta:0.20`, `pct_otm:1.5`, `sigma:1.0` |
| `wing_method` | `points:200`, `pct:1.0`, `delta:0.10` |
| `stop_loss` | `pct_premium:30`, `pct_spot:0.5`, `delta:0.40`, `mtm:-5000` |
| `take_profit` | `pct_premium:50`, `mtm:5000` |
| `sizing` | `lots:1`, `pct_capital:2`, `vol_target:0.10` |
| `slippage` | `ticks:1`, `bps:5`, `pct:0.5` |
| `fees` | brokerage flat + STT (sell side) + exchange + GST + SEBI |
| `lot_size` | resolved per date (75/50/25/75 historically) |
| `dte_filter` | min/max days-to-expiry |
| `iv_filter` | only trade if IV-rank within `[a, b]` |
| `regime_filter` | trend filter (ADX>X), realised-vol filter, day-of-week, day-of-month |
| `weekday_filter` | Mon, Tue, Wed, Thu (expiry), Fri |
| `event_filter` | exclude / include around RBI / Fed / Budget / Election dates |
| `holiday_buffer` | skip T-1 / T+1 of NSE holiday |
| `roll_rule` | T-1 close, T-2 close, delta-breach, none |

Sweep these on a Cartesian grid (or random/Bayesian search for large spaces). Persist every combination tried.

---

## 5. Backtest engine — minimum spec

1. **Calendar:** trading days from `glob("..\DATA\spot\*.parquet")` filenames.
2. **Per-day loop:** lazy-load `options/`, `oi/`, `spot/` for the date(s) you need (cache LRU).
3. **Chain builder:** at a given `timestamp`, return a DataFrame indexed by `(expiry, strike, option_type)` with `close, oi, mid, bid, ask` (mid = close if no bid/ask; ask/bid synthesised via tick-size if needed).
4. **Strike resolver:** `atm`, `delta(target)`, `pct_otm(x)`, `sigma(n)` — uses IV from a Black-76 solver or a cached IV surface (precompute once and store in `cache/iv/YYYYMMDD.parquet`).
5. **Order book:** fills at next-minute open with slippage, lot-rounded, margin-checked.
6. **MTM:** mark with mid every minute; positions cash-settled on expiry against final-settlement (last-half-hour VWAP of spot).
7. **Risk hooks:** evaluated per minute — SL, TP, Δ-breach, time-stop, adjustment.
8. **P&L:** per-trade (entry, exit, legs, gross, fees, net, MAE, MFE, duration), per-day equity curve, per-strategy roll-up.

---

## 6. Output / reporting

For every `run_id` write to `results/<run_id>/`:

- `config.yaml` — frozen params
- `trades.parquet` — one row per closed trade with all legs
- `legs.parquet` — one row per leg fill (entry + exit)
- `equity.parquet` — daily equity, drawdown, exposure, Δ/Γ/V/Θ
- `greeks.parquet` — minute-level portfolio greeks
- `report.md` — auto-generated with:
  - headline metrics: CAGR, Sharpe, Sortino, Calmar, Max DD, MAR, Win %, Avg win / Avg loss, Expectancy, Profit factor, Payoff ratio, # trades, Avg DTE, Avg hold
  - equity curve, drawdown curve, monthly heatmap, daily-returns histogram
  - per-weekday and per-expiry-week P&L
  - parameter-sensitivity table for the sweep
  - regime split (bull / bear / sideways / high-vol / low-vol)

---

## 7. Reproducibility & hygiene

- **Random seed:** set in `config/universe.yaml` even for deterministic strategies (used by sampling-based sweeps).
- **Walk-forward:** train on 2020–2023, test out-of-sample on 2024–2026 by default. Configurable.
- **No look-ahead:** when constructing chain at `t`, never reference rows with `timestamp > t`. The chain builder must enforce this.
- **Survivor-free by construction:** the parquet files contain every contract that traded, including those that expired worthless. No survivorship adjustment needed.
- **Lot-size table:** maintain `config/lot_size_history.yaml` mapping `(from_date, to_date) → lot_size`.
- **Fees model:** keep `config/fees_<YYYY>.yaml` so historic STT / brokerage rates can be applied accurately.

---

## 8. Quick start — minimal backtest skeleton

```python
import pandas as pd, glob, os
from pathlib import Path

DATA = Path(r"C:\Users\User\Desktop\investeq_ajs\DATA")

def trading_days():
    return sorted(p.stem for p in (DATA / "spot").glob("*.parquet"))

def load_day(date_str):
    opt = pd.read_parquet(DATA / "options" / f"{date_str}.parquet")
    oi  = pd.read_parquet(DATA / "oi"      / f"{date_str}.parquet")
    spot= pd.read_parquet(DATA / "spot"    / f"{date_str}.parquet")
    for df in (opt, oi, spot):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    opt = opt.merge(oi, on=["timestamp","expiry","strike","option_type"], how="left")
    return opt, spot

def atm_strike(spot_close, strikes):
    return int(strikes[(strikes - spot_close).abs().argmin()])

def short_straddle_eod(date_str, entry="09:20", exit_="15:15"):
    opt, spot = load_day(date_str)
    s_at_entry = spot.loc[spot.timestamp.dt.strftime("%H:%M") == entry, "close"].iloc[0]
    strikes = opt["strike"].unique()
    K = atm_strike(s_at_entry, pd.Series(strikes))
    # nearest weekly expiry on/after today:
    expiry = sorted(e for e in opt["expiry"].unique() if e >= date_str[:4]+"-"+date_str[4:6]+"-"+date_str[6:])[0]
    legs = opt[(opt.expiry==expiry) & (opt.strike==K) & (opt.option_type.isin(["CE","PE"]))]
    entry_px = legs[legs.timestamp.dt.strftime("%H:%M")==entry].set_index("option_type")["close"]
    exit_px  = legs[legs.timestamp.dt.strftime("%H:%M")==exit_].set_index("option_type")["close"]
    pnl = (entry_px.sum() - exit_px.sum())  # short → collect premium, pay back at exit
    return {"date": date_str, "K": K, "expiry": expiry, "pnl_per_lot": pnl}
```

Build everything else on this pattern: deterministic, file-driven, no global state.

---

## 9. Open items / next steps

- [ ] Decide between PyArrow dataset, DuckDB, or Polars as the primary loader (lean DuckDB for SQL filters + Polars for in-memory ops).
- [ ] Pre-compute IV surface to `cache/iv/YYYYMMDD.parquet` once — solving Black-76 every backtest is the slow path.
- [ ] Encode NSE holiday calendar (2020–2026) in `config/holidays.yaml`.
- [ ] Decide margin model (SPAN+exposure approximation vs flat per-lot heuristic).
- [ ] Wire `cli.py` so a strategy is one command: `python -m option_strategy run config/strategies/short_straddle.yaml`.
- [ ] Add an `examples/` notebook for each strategy family.

---

## 10. References / prior work

Curated reading list — the canonical sources behind the pricing models, strategies, risk metrics, and Indian-market specifics used in this project. Skim before designing a new strategy; cite when you reuse a method.

### 10.1 Foundational pricing & greeks
- Black, F. & Scholes, M. (1973). **The Pricing of Options and Corporate Liabilities.** *Journal of Political Economy*, 81(3).
- Merton, R. C. (1973). **Theory of Rational Option Pricing.** *Bell Journal of Economics*, 4(1).
- Black, F. (1976). **The Pricing of Commodity Contracts.** *Journal of Financial Economics*, 3 — the Black-76 model used for index options.
- Cox, J., Ross, S., Rubinstein, M. (1979). **Option Pricing: A Simplified Approach.** *JFE*, 7 — binomial tree.
- Heston, S. (1993). **A Closed-Form Solution for Options with Stochastic Volatility.** *RFS*, 6(2).
- Dupire, B. (1994). **Pricing with a Smile.** *Risk Magazine* — local-vol surface.
- Hagan, P., Kumar, D., Lesniewski, A., Woodward, D. (2002). **Managing Smile Risk.** *Wilmott Magazine* — SABR.
- Gatheral, J. (2004). **A Parsimonious Arbitrage-Free Implied Volatility Parameterization.** Global Derivatives — SVI.
- Gatheral, J. (2006). **The Volatility Surface: A Practitioner's Guide.** Wiley.

### 10.2 Textbook references (working bibliography)
- Hull, J. C. **Options, Futures, and Other Derivatives.** Pearson — pricing, greeks, hedging.
- Natenberg, S. **Option Volatility and Pricing.** McGraw-Hill — practitioner playbook for vol trading.
- McMillan, L. G. **Options as a Strategic Investment.** NYIF — strategy catalogue baseline.
- Sinclair, E. **Volatility Trading.** Wiley — short-vol mechanics, risk, sizing.
- Sinclair, E. **Option Trading: Pricing and Volatility Strategies.** Wiley.
- Taleb, N. N. **Dynamic Hedging.** Wiley — tail risk, gamma scalping.

### 10.3 Strategy-specific empirical studies
- Coval, J. & Shumway, T. (2001). **Expected Option Returns.** *Journal of Finance*, 56(3) — why short-vol earns a premium.
- Bondarenko, O. (2014). **Why Are Put Options So Expensive?** *Quarterly Journal of Finance* — variance-risk premium.
- Santa-Clara, P. & Saretto, A. (2009). **Option Strategies: Good Deals and Margin Calls.** *Journal of Financial Markets*.
- Israelov, R. & Nielsen, L. N. (2015). **Covered Call Strategies: One Fact and Eight Myths.** *Financial Analysts Journal*.
- Israelov, R. & Klein, M. (2016). **Risk and Return of Equity Index Collar Strategies.** AQR working paper.
- Goltz, F. & Lai, W. N. (2009). **Empirical Properties of the Volatility Risk Premium in Indian Equity Index Options.** EDHEC.
- Doran, J. & Krieger, K. (2010). **Implications for Asset Returns in the Implied Volatility Skew.** *Financial Analysts Journal*.

### 10.4 Realised vs implied vol, vol risk premium
- Bollerslev, T. (1986). **Generalized Autoregressive Conditional Heteroskedasticity (GARCH).** *Journal of Econometrics*, 31.
- Andersen, T., Bollerslev, T., Diebold, F., Labys, P. (2003). **Modeling and Forecasting Realized Volatility.** *Econometrica*, 71(2).
- Corsi, F. (2009). **A Simple Approximate Long-Memory Model of Realized Volatility (HAR-RV).** *Journal of Financial Econometrics*, 7(2).
- Carr, P. & Wu, L. (2009). **Variance Risk Premiums.** *RFS*, 22(3).

### 10.5 Risk metrics & evaluation
- Sharpe, W. F. (1994). **The Sharpe Ratio.** *Journal of Portfolio Management*.
- Sortino, F. & van der Meer, R. (1991). **Downside Risk.** *JPM*.
- Bailey, D. & López de Prado, M. (2014). **The Deflated Sharpe Ratio.** *JPM* — corrects for multiple-testing in strategy sweeps. Apply when reporting the best of N parameter combos.

### 10.6 NIFTY / Indian-market specific
- NSE India. **NIFTY 50 Index Methodology** and **F&O Product Notes** — official spec for contract size, expiry rules, settlement, lot-size history.
- SEBI circulars on **STT, derivatives margining, and lot-size revisions** (2020-2024) — needed to reconstruct historical fee/margin models accurately.
- Kakati, M. (2006). **An Empirical Analysis of Pricing Indian Index Options.** Indian-market BSM bias study.
- Mishra, B. (2010). **Performance of Option Pricing Models on NIFTY Index Options.** *Decision* (IIM-C).
- Misra, D., Kannan, R., Misra, S. D. (2006). **Implied Volatility Surfaces: A Study of NIFTY Options.** *ICFAI Journal of Derivatives Markets*.
- Tripathi, V. & Gupta, S. (2010). **Effectiveness of the Black-Scholes Model for Pricing NIFTY Index Options.** *Asia-Pacific Journal of Finance and Banking*.

### 10.7 Practitioner / industry notes worth tracking
- AQR Capital — option overlay and volatility-strategy whitepapers (Israelov et al.).
- CBOE — methodology docs for BXM, PUT, CLL benchmark indices (covered-call / put-write / collar).
- NSE quarterly **derivatives market reports** — concentration, OI, expiry-day volume.

> When in doubt: pricing → Hull/Gatheral; strategy mechanics → Natenberg/Sinclair; backtest hygiene → Bailey & López de Prado (2014).
