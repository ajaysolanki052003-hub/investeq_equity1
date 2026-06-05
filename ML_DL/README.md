# NIFTY Options — Machine Learning & Deep Learning

Spec for the ML/DL workstream on the 6-year NIFTY index options dataset (`..\DATA`, 2020-01-01 → 2026-04-30, 1-minute bars). This README is the canonical brief — every problem formulation, feature, model family, validation protocol, and output convention lives here.

- **Project root:** `C:\Users\User\Desktop\investeq_ajs\ML_DL`
- **Data source:** `C:\Users\User\Desktop\investeq_ajs\DATA` (full schema in `..\DATA\README.md`)
- **Underlying:** NSE NIFTY 50 index options (weekly + monthly), 1-min OHLCV + OI
- **Sibling project:** `..\OPTION_STRATEGY` — rule-based strategies. ML signals here can plug into that engine as entry/exit/sizing modules.
- **Goal:** an end-to-end pipeline (raw parquet → features → labels → train/validate → deploy as inference service) that supports every problem formulation below with a uniform results layout.

---

## 1. Data layout (cheat-sheet)

| Source | Purpose | Join key |
|---|---|---|
| `..\DATA\spot\YYYYMMDD.parquet` | 1-min NIFTY spot OHLCV | `timestamp` |
| `..\DATA\options\YYYYMMDD.parquet` | 1-min OHLCV per contract | `timestamp, expiry, strike, option_type` |
| `..\DATA\oi\YYYYMMDD.parquet` | 1-min OI per contract | same as options |
| `..\DATA\index\expiries.parquet` | full expiry list | — |
| `..\DATA\spot_all.parquet` | rolled-up spot (2024+) | `timestamp, date` |

**Gotchas (carried from `..\DATA\README.md`):**
- Timestamps are **strings, no tz**, IST — cast and localise.
- Contract minutes are **sparse** — reindex against the session grid before sequence modelling.
- **Lot size** changed over the window (75 → 50 → 25 → 75) — load from a date-keyed table, never hardcode.
- **No survivorship adjustment needed** — every traded contract is present, including those that expired worthless.
- Derive the trading calendar from `spot/` filenames, not `trading_days.parquet`.

---

## 2. Project layout (proposed)

```
ML_DL/
├── README.md                         # this file
├── config/
│   ├── data.yaml                     # paths, date range, session window
│   ├── features.yaml                 # feature toggles + windows
│   ├── labels.yaml                   # label definitions per task
│   ├── splits.yaml                   # walk-forward / purged-CV config
│   └── models/                       # one yaml per model+task combo
│       ├── lgbm_spot_direction.yaml
│       ├── tft_iv_forecast.yaml
│       └── ...
├── src/
│   ├── data/
│   │   ├── loader.py                 # lazy parquet readers + cache
│   │   ├── session.py                # market hours, holidays, expiry calendars
│   │   ├── chain.py                  # build option chain at timestamp t
│   │   └── greeks.py                 # IV solver (Black-76) + Δ/Γ/V/Θ
│   ├── features/
│   │   ├── spot.py                   # returns, vol, RSI, MACD, etc.
│   │   ├── options.py                # ATM-IV, skew, term-structure, OI deltas
│   │   ├── microstructure.py         # spread proxy, range, signed volume
│   │   ├── regime.py                 # vol/trend/IV-rank regime tags
│   │   └── calendar.py               # dte, day-of-week, event flags
│   ├── labels/
│   │   ├── direction.py              # up/down/flat over horizon
│   │   ├── triple_barrier.py         # Lopez de Prado
│   │   ├── vol.py                    # realised-vol / IV-change targets
│   │   └── pnl.py                    # strategy-conditional P&L labels
│   ├── splits/
│   │   ├── walk_forward.py
│   │   └── purged_kfold.py           # purge + embargo for leakage control
│   ├── models/
│   │   ├── classical/                # logistic, GBM (lgbm/xgb/cat), RF, SVM
│   │   ├── sequence/                 # LSTM, GRU, TCN, Transformer, TFT, N-BEATS, N-HiTS, PatchTST
│   │   ├── tabular_dl/               # TabNet, FT-Transformer, SAINT, NODE
│   │   ├── generative/               # VAE / Diffusion for IV surface, GAN for synthetic paths
│   │   ├── graph/                    # GNN over option chain (strike-expiry grid)
│   │   ├── rl/                       # DQN, PPO, SAC for sizing/hedging
│   │   └── ensembles.py              # stacking, blending, calibration
│   ├── training/
│   │   ├── trainer.py                # Lightning-style loop (PyTorch) + sklearn wrapper
│   │   ├── early_stop.py
│   │   ├── hpo.py                    # Optuna / Ray Tune
│   │   └── calibration.py            # Platt / Isotonic for probabilistic outputs
│   ├── eval/
│   │   ├── metrics.py                # classification, regression, ranking, financial
│   │   ├── financial.py              # Sharpe / hit-rate / payoff applied to predictions
│   │   ├── attribution.py            # SHAP, permutation importance, integrated gradients
│   │   └── reports.py                # markdown + matplotlib + plotly
│   ├── inference/
│   │   ├── batch.py                  # offline scoring → parquet
│   │   ├── service.py                # FastAPI / gRPC stub
│   │   └── plugin_option_strategy.py # adapter so signals feed ..\OPTION_STRATEGY engine
│   └── cli.py                        # python -m ml_dl <command>
├── notebooks/                        # EDA, ablation, error analysis
├── artifacts/
│   ├── features/                     # cached engineered features (parquet)
│   ├── labels/                       # cached labels
│   └── iv/                           # cached IV surface YYYYMMDD.parquet
├── results/
│   └── <run_id>/                     # one dir per training run
│       ├── config.yaml               # frozen
│       ├── splits.json
│       ├── metrics.json
│       ├── predictions.parquet
│       ├── shap.parquet              # if applicable
│       ├── model.pkl / model.pt
│       └── report.md
└── tests/
```

`run_id` = `{task}_{model}_{features_hash}_{YYYYMMDDHHMM}` — never overwrite a run.

---

## 3. Problem formulations

Every task below should have its own yaml config (`config/models/<task>_<model>.yaml`) and reuse the shared feature/label modules.

### 3.1 Spot-direction classification
- **Targets:** sign of forward return over `H` minutes/days (binary up/down, ternary up/flat/down with dead-zone, multi-class quantile bins).
- **Horizons:** 5m, 15m, 30m, 60m, 1d, 1w, to-expiry.
- **Models:** logistic, LightGBM/XGBoost/CatBoost, RandomForest, LSTM/GRU/TCN, Transformer (PatchTST), TFT, N-HiTS, hybrid CNN-LSTM.
- **Use:** entry trigger for directional spreads or single-leg longs in `..\OPTION_STRATEGY`.

### 3.2 Spot price / return regression
- **Targets:** continuous H-step return, log-return, or path quantiles (P10/P50/P90).
- **Loss:** MSE, MAE, Huber, **quantile loss** (pinball) for distributional output, **CRPS** for probabilistic.
- **Models:** Ridge/Lasso, GBM with quantile loss, DeepAR, TFT, N-BEATS/N-HiTS, PatchTST, Informer.

### 3.3 Realised volatility forecasting
- **Targets:** RV over horizon H (Parkinson, Garman-Klass, Yang-Zhang estimators).
- **Use:** sizing volatility-targeted strategies, gating short-premium entries.
- **Models:** HAR-RV baseline → LightGBM → LSTM → TFT.

### 3.4 Implied-volatility forecasting
- **Targets:**
  - ATM-IV at horizon H
  - Full IV surface change (per (moneyness, dte) cell)
  - IV-rank / IV-percentile move
- **Models:** SVI/SABR parametric baseline → GBM per cell → seq2seq Transformer → graph-NN over chain → VAE/diffusion for full surface.

### 3.5 Option-price prediction (mid/close next-bar)
- **Targets:** next-minute close per contract.
- **Models:** per-contract GBM, panel models with contract embeddings, GNN over chain.
- **Caveat:** mostly a tracking task — value is in residual signal (mispricing).

### 3.6 Mispricing / residual alpha
- **Targets:** residual of `observed_price − model_price` (BS/SABR) — predict mean-reversion of the residual.
- **Models:** quantile GBM, LSTM on residual series.

### 3.7 Greeks prediction / surrogate
- **Targets:** Δ/Γ/V/Θ at horizon (or under perturbed spot).
- **Use:** fast surrogates to avoid re-solving IV during backtests / live hedging.
- **Models:** small MLP, GBM per greek, distillation from analytical solver.

### 3.8 Triple-barrier trade labelling (Lopez de Prado)
- **Targets:** {+1, 0, -1} based on which of (profit-take, stop-loss, time-out) hit first, with **meta-labelling** for size.
- **Use:** turns any directional model into a trade-by-trade classifier with realistic P&L semantics.

### 3.9 Strategy-conditional P&L regression
- **Setup:** for each strategy in `..\OPTION_STRATEGY`, train a model that predicts the next-period P&L of running that strategy under current market state.
- **Use:** **strategy selection** — pick the highest-EV strategy at each entry slot.
- **Models:** GBM with strategy-id as feature, multi-task NN with per-strategy heads.

### 3.10 Regime classification (unsupervised → supervised)
- **Targets:** vol regime, trend regime, liquidity regime — unsupervised cluster → name regimes → use as feature / gating signal.
- **Models:** GMM, HMM, k-means on rolling stats; supervised classifier once regimes are named.

### 3.11 Anomaly / event detection
- **Targets:** unusual OI build-up, IV spike, spread blow-out, microstructure dislocation.
- **Models:** Isolation Forest, Autoencoder, VAE, One-Class SVM, Mahalanobis.
- **Use:** circuit-breakers for live trading; event-study flags for research.

### 3.12 Reinforcement learning
- **Setup:** state = features + position; action = {open, close, roll, size, hedge}; reward = MTM ΔP&L − transaction cost − risk penalty.
- **Variants:**
  - **Delta-hedging RL** — agent decides when/how much to hedge a short straddle (continuous action, SAC/PPO/DDPG).
  - **Sizing RL** — agent decides lots per trade given an external signal.
  - **Multi-leg strategy selection** — discrete action over a strategy menu (DQN/PPO).
  - **Execution RL** — minute-level fill scheduling (PPO with implementation-shortfall reward).
- **Caveats:** simulator fidelity = everything. Use the `..\OPTION_STRATEGY` engine as the env.

### 3.13 Generative
- **Synthetic path generation:** GAN / Diffusion / VAE conditioned on regime → stress-test strategies on plausible counterfactuals.
- **IV surface generation:** conditional VAE / diffusion over the (moneyness, dte) grid.
- **Use:** data augmentation, robustness eval, scenario analysis.

### 3.14 LLM / sequence-of-trade
- Token-style modelling of the order/trade tape with a Transformer to predict next-minute event distribution.
- Niche — only worth attempting once everything above is in place.

---

## 4. Feature catalogue

Toggleable in `config/features.yaml`. Cache each family to `artifacts/features/<family>.parquet`.

### 4.1 Spot / underlying
- Log returns (1, 5, 15, 30, 60, 1d, 5d, 20d)
- Realised vol (rolling std, Parkinson, Garman-Klass, Yang-Zhang) on multiple windows
- RSI, MACD, ADX, ATR, Bollinger %B/width, Donchian, Keltner
- Z-score of return vs rolling mean
- VWAP distance, anchored-VWAP (day / week / expiry)
- Gap (open vs prev close), overnight return
- Session-time features (time-of-day sin/cos, time-since-open, time-to-close)

### 4.2 Option-implied
- ATM-IV (per dte bucket: 0–2d, 3–7d, 8–30d, 30d+)
- IV-rank (1y), IV-percentile, IV-Z
- Term structure (front-month / back-month IV ratio, slope)
- Skew (25Δ put-call IV diff, risk-reversal, butterfly)
- SVI / SABR fitted params per expiry
- IV surface PCA components (level / slope / curvature)
- VIX-style aggregate (own NIFTY VIX proxy from chain) and its return

### 4.3 OI / flow
- Total CE OI, Total PE OI, PCR (OI and Volume)
- Δ OI per strike → strike with largest CE/PE OI build / unwind
- Max-pain strike + distance from spot
- OI concentration (Gini / Herfindahl over strikes)
- Volume-weighted ATM strike

### 4.4 Microstructure (per-contract)
- Per-minute volume, OI delta, signed volume proxy (close vs vwap)
- Range / |close-open| / wick ratios
- Trade intensity (active minutes per hour)
- Spread proxy (high-low / close)

### 4.5 Calendar / event
- DTE (days to expiry) — current week, current month
- Day-of-week, week-of-month, expiry-day flag, day-before-expiry
- Holiday-buffer flags (T-1 / T+1 of NSE holiday)
- Event flags: RBI policy, Fed FOMC, Union Budget, Election results, monthly expiry, GDP/CPI prints

### 4.6 Cross-asset (optional, requires extra data)
- USDINR, Brent, US 10y, SGX-NIFTY pre-open, S&P futures overnight return
- Sector indices (Bank NIFTY especially), India-VIX

### 4.7 Regime tags
- Trend regime (ADX > 25 → trending, else range)
- Vol regime (RV percentile bucket)
- IV regime (IV-rank bucket)
- Liquidity regime (chain-wide volume percentile)

---

## 5. Labelling protocols

Pick one per task — defined in `config/labels.yaml`.

| Protocol | Definition | When to use |
|---|---|---|
| **Fixed-horizon binary** | sign of `r_{t+H}` | quick baseline |
| **Fixed-horizon ternary** | `+1` if `r > θ`, `-1` if `r < -θ`, else `0` | reduces noise around zero |
| **Triple-barrier** | first of (PT, SL, time-out) wins | realistic, P&L-aware |
| **Meta-labelling** | given a primary signal, predict whether to act | sizing on top of any classifier |
| **Quantile bins** | bin returns into 5/10 buckets | ranking / ordinal models |
| **Continuous return** | raw `r_{t+H}` | regression |
| **Distributional** | predict {P10, P50, P90} or full CDF | uncertainty-aware |
| **Vol target** | `RV_{t,t+H}` | vol forecasting |
| **Strategy P&L** | actual P&L from running a rule on the chain | strategy selection |

---

## 6. Validation — leakage-safe

Time-series + overlapping labels make naive CV catastrophic. Defaults:

1. **Walk-forward (anchored or rolling).** Default split:
   - Train 2020-01-01 → 2022-12-31
   - Validate 2023-01-01 → 2023-12-31
   - Test (out-of-sample) 2024-01-01 → 2026-04-30
2. **Rolling re-train** every quarter / month during the test window for production-style backtest.
3. **Purged k-fold** with **embargo** (Lopez de Prado, AFML ch.7) when CV is necessary — purge `H` bars around each fold edge, embargo a small buffer to prevent serial-correlation leakage.
4. **Combinatorial purged CV (CPCV)** for HPO when sample is precious.
5. **Never** shuffle rows.
6. **Never** scale/normalise using future stats — fit scalers per fold on train only.
7. **Feature lag check** — every feature at time `t` must be derivable from data with `timestamp ≤ t`. Unit-test this for the whole feature module.

---

## 7. Metrics

### 7.1 Classification
- Accuracy, balanced accuracy, F1, AUC-ROC, AUC-PR
- Brier score, log-loss (probabilistic calibration)
- Confusion matrix per regime / per dte bucket

### 7.2 Regression
- MAE, RMSE, MAPE (with guard against zero)
- R², adjusted R², MASE
- Pinball loss (quantile), CRPS (distributional)

### 7.3 Ranking
- Spearman / Pearson rank-IC (information coefficient)
- IC by decile, IC-IR (mean/std across periods)
- Top-decile minus bottom-decile spread P&L

### 7.4 Financial (the only ones that ultimately matter)
- Hit-rate, payoff ratio, expectancy per trade
- Equity curve, Sharpe / Sortino / Calmar / MAR
- Max drawdown, time-under-water
- Turnover, transaction-cost drag, capacity
- **Net Sharpe after fees and slippage** — headline metric for every run

Every run report includes both statistical and financial metrics — a model is only "good" if both agree.

---

## 8. Training stack

- **Compute:** local CPU baseline → CUDA GPU for sequence/DL models. Keep models small enough that retrain fits the rolling-window cadence.
- **Frameworks:** scikit-learn, LightGBM/XGBoost/CatBoost, PyTorch + Lightning, Optuna, SHAP, Ray (optional for distributed HPO), Polars/DuckDB for feature pipelines.
- **Tracking:** MLflow or Weights & Biases — log config, metrics, artifacts per `run_id`.
- **Reproducibility:** every run freezes `config.yaml`, env (`requirements.txt` lock), random seeds, and git SHA into `results/<run_id>/`.
- **Determinism:** PyTorch deterministic flags on; CUDA non-determinism documented per model.

---

## 9. Inference / deployment

- **Batch:** `python -m ml_dl score --model <run_id> --date YYYYMMDD` → writes `predictions.parquet` for that day, joined into the strategy engine.
- **Live (future):** FastAPI service that consumes a streaming chain snapshot, returns predictions + confidence + recommended action. Stub in `src/inference/service.py`.
- **Strategy plug-in:** `src/inference/plugin_option_strategy.py` exposes a `Signal` object compatible with `..\OPTION_STRATEGY` so any model can drive entry/exit/sizing without rewriting the strategy.

---

## 10. Quick start — minimal training skeleton

```python
import pandas as pd, numpy as np, glob
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

DATA = Path(r"C:\Users\User\Desktop\investeq_ajs\DATA")

def load_spot(date_str):
    df = pd.read_parquet(DATA / "spot" / f"{date_str}.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def make_features(spot, horizon=15):
    s = spot.set_index("timestamp")["close"].astype(float)
    r1  = s.pct_change()
    feat = pd.DataFrame({
        "ret_1":   r1,
        "ret_5":   s.pct_change(5),
        "ret_15":  s.pct_change(15),
        "vol_30":  r1.rolling(30).std(),
        "rsi_14":  100 - 100/(1 + r1.clip(lower=0).rolling(14).mean()
                                / (-r1.clip(upper=0)).rolling(14).mean()),
    })
    feat["y"] = (s.shift(-horizon) > s).astype(int)
    return feat.dropna()

days = sorted(p.stem for p in (DATA / "spot").glob("2024*.parquet"))
frames = [make_features(load_spot(d)) for d in days]
df = pd.concat(frames).dropna()

split = int(len(df) * 0.7)
X_tr, y_tr = df.iloc[:split].drop(columns="y"), df.iloc[:split]["y"]
X_te, y_te = df.iloc[split:].drop(columns="y"), df.iloc[split:]["y"]

model = LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=-1)
model.fit(X_tr, y_tr)
p = model.predict_proba(X_te)[:, 1]
print("acc:", accuracy_score(y_te, p > 0.5))
print("auc:", roc_auc_score(y_te, p))
```

This is intentionally trivial — real runs should: (a) load via the cached feature pipeline, (b) split with walk-forward + purge, (c) HPO with Optuna, (d) write a `results/<run_id>/` directory.

---

## 11. Roadmap

- [ ] Lock in loader stack (DuckDB for filters + Polars for in-memory; PyArrow dataset fallback).
- [ ] Pre-compute IV surface to `artifacts/iv/YYYYMMDD.parquet` (Black-76 once, reuse forever).
- [ ] Implement `splits/walk_forward.py` and `splits/purged_kfold.py` with embargo and unit tests for leakage.
- [ ] Encode NSE holiday + event calendars (RBI/Fed/Budget/Election) in `config/calendars/`.
- [ ] Build feature pipeline (`src/features/*`) with cache invalidation by `config_hash + file_mtime`.
- [ ] Baselines: HAR-RV (vol), logistic (direction), per-contract GBM (mid-price).
- [ ] First DL milestone: PatchTST or TFT on the spot-direction task with strict walk-forward.
- [ ] Wire `plugin_option_strategy.py` and prove a model → strategy → backtest loop end-to-end.
- [ ] Add MLflow / W&B tracking.
- [ ] HPO with Optuna; persist all trials.
- [ ] SHAP / permutation importance reports auto-generated per run.
- [ ] Stretch: RL delta-hedger on top of a short-straddle env from `..\OPTION_STRATEGY`.

---

## 12. References / prior work

Canonical reading list — pricing baselines, ML-for-finance methodology, sequence models, RL for hedging, and generative work on IV surfaces. Cite when you reuse a method; skim before defending a new design.

### 12.1 ML for finance — methodology & leakage
- López de Prado, M. (2018). **Advances in Financial Machine Learning (AFML).** Wiley — triple-barrier labels, meta-labelling, purged/embargo CV, CPCV, fractional differentiation. Backbone of the validation protocol in §6.
- López de Prado, M. (2020). **Machine Learning for Asset Managers.** CUP.
- Bailey, D. & López de Prado, M. (2014). **The Deflated Sharpe Ratio.** *JPM* — multiple-testing correction for HPO sweeps.
- Israel, R., Kelly, B., Moskowitz, T. (2020). **Can Machines "Learn" Finance?** *Journal of Investment Management*.
- Gu, S., Kelly, B., Xiu, D. (2020). **Empirical Asset Pricing via Machine Learning.** *RFS*, 33(5).
- Heaton, J. B., Polson, N., Witte, J. (2017). **Deep Learning in Finance.**

### 12.2 Time-series forecasting (sequence models)
- Hochreiter, S. & Schmidhuber, J. (1997). **Long Short-Term Memory.** *Neural Computation*.
- Bai, S., Kolter, J. Z., Koltun, V. (2018). **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (TCN).**
- Vaswani et al. (2017). **Attention Is All You Need.**
- Lim, B., Arik, S. O., Loeff, N., Pfister, T. (2021). **Temporal Fusion Transformers (TFT).** *IJF*.
- Oreshkin, B. et al. (2020). **N-BEATS: Neural Basis Expansion Analysis for Interpretable Time-Series Forecasting.** ICLR.
- Challu, C. et al. (2023). **N-HiTS: Neural Hierarchical Interpolation for Time-Series Forecasting.** AAAI.
- Nie, Y. et al. (2023). **A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers (PatchTST).** ICLR.
- Zhou, H. et al. (2021). **Informer.** AAAI.
- Wu, H. et al. (2021). **Autoformer.** NeurIPS.
- Salinas, D. et al. (2020). **DeepAR.** *IJF*.

### 12.3 Tabular deep learning
- Arik, S. O. & Pfister, T. (2021). **TabNet.** AAAI.
- Gorishniy, Y. et al. (2021). **Revisiting Deep Learning Models for Tabular Data (FT-Transformer).** NeurIPS.
- Somepalli, G. et al. (2021). **SAINT: Improved Neural Networks for Tabular Data.**
- Popov, S., Morozov, S., Babenko, A. (2020). **Neural Oblivious Decision Ensembles (NODE).** ICLR.

### 12.4 Gradient-boosted baselines (still the workhorse)
- Chen, T. & Guestrin, C. (2016). **XGBoost.** KDD.
- Ke, G. et al. (2017). **LightGBM.** NeurIPS.
- Prokhorenkova, L. et al. (2018). **CatBoost.** NeurIPS.

### 12.5 Volatility — econometric baselines
- Bollerslev, T. (1986). **GARCH.** *Journal of Econometrics*.
- Andersen, T., Bollerslev, T., Diebold, F., Labys, P. (2003). **Modeling and Forecasting Realized Volatility.** *Econometrica*.
- Corsi, F. (2009). **HAR-RV.** *JFE*. Strong baseline before any DL claim on RV.
- Carr, P. & Wu, L. (2009). **Variance Risk Premiums.** *RFS*.

### 12.6 Implied-volatility surface modelling
- Heston, S. (1993). **Stochastic Volatility Model.** *RFS*.
- Dupire, B. (1994). **Local Volatility.** *Risk Magazine*.
- Hagan, P., Kumar, D., Lesniewski, A., Woodward, D. (2002). **SABR — Managing Smile Risk.** *Wilmott*.
- Gatheral, J. (2004). **SVI Parameterization.** Global Derivatives.
- Gatheral, J. (2006). **The Volatility Surface.** Wiley.
- Bergeron, M., Fung, N. et al. (2021). **Variational Autoencoders: A Hands-Off Approach to Volatility.**
- Cont, R. & Vuletić, M. (2023). **Simulation of Arbitrage-Free Implied Volatility Surfaces.**
- Ackerer, D., Tagasovska, N., Vatter, T. (2020). **Deep Smoothing of the Implied Volatility Surface.** NeurIPS.
- Horvath, B., Muguruza, A., Tomas, M. (2021). **Deep Learning Volatility (rough-vol calibration).** *Quantitative Finance*.

### 12.7 Deep hedging & RL for derivatives
- Buehler, H., Gonon, L., Teichmann, J., Wood, B. (2019). **Deep Hedging.** *Quantitative Finance* — replication under frictions via deep nets.
- Buehler, H. et al. (2019). **Deep Hedging: Hedging Derivatives Under Generic Market Frictions.** SSRN.
- Halperin, I. (2017). **QLBS: Q-Learner in the Black-Scholes World.**
- Cao, J., Chen, J., Hull, J., Poulos, Z. (2020). **Deep Hedging of Derivatives Using Reinforcement Learning.** *JFDS*.
- Kolm, P. & Ritter, G. (2019). **Dynamic Replication and Hedging: A Reinforcement Learning Approach.** *JFDS*.
- Mnih, V. et al. (2015). **Human-Level Control through Deep Reinforcement Learning (DQN).** *Nature*.
- Schulman, J. et al. (2017). **Proximal Policy Optimization (PPO).**
- Haarnoja, T. et al. (2018). **Soft Actor-Critic (SAC).** ICML.

### 12.8 Limit-order book & microstructure DL
- Sirignano, J. & Cont, R. (2019). **Universal Features of Price Formation in Financial Markets.** *Quantitative Finance*.
- Tsantekidis, A. et al. (2017). **Forecasting Stock Prices from the LOB Using CNNs.** CBI.
- Zhang, Z., Zohren, S., Roberts, S. (2019). **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.** *IEEE TSP*.
- Cont, R. (2011). **Statistical Modeling of High-Frequency Financial Data.** *IEEE SPM*.

### 12.9 Interpretability & feature attribution
- Lundberg, S. & Lee, S.-I. (2017). **A Unified Approach to Interpreting Model Predictions (SHAP).** NeurIPS.
- Sundararajan, M., Taly, A., Yan, Q. (2017). **Axiomatic Attribution for Deep Networks (Integrated Gradients).** ICML.
- Breiman, L. (2001). **Random Forests** — permutation importance baseline.

### 12.10 Generative models (paths, surfaces, synthetic data)
- Kingma, D. & Welling, M. (2014). **Auto-Encoding Variational Bayes (VAE).** ICLR.
- Goodfellow, I. et al. (2014). **Generative Adversarial Nets.** NeurIPS.
- Wiese, M. et al. (2020). **Quant GANs: Deep Generation of Financial Time Series.** *Quantitative Finance*.
- Ho, J., Jain, A., Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models (DDPM).** NeurIPS.
- Cont, R. (2001). **Empirical Properties of Asset Returns: Stylized Facts.** — checklist any generator must reproduce.

### 12.11 NIFTY / Indian-market empirics
- Kakati, M. (2006). **An Empirical Analysis of Pricing Indian Index Options.**
- Mishra, B. (2010). **Performance of Option Pricing Models on NIFTY Index Options.** *Decision* (IIM-C).
- Misra, D., Kannan, R., Misra, S. D. (2006). **Implied Volatility Surfaces: A Study of NIFTY Options.** *ICFAI Journal of Derivatives Markets*.
- Tripathi, V. & Gupta, S. (2010). **Effectiveness of the Black-Scholes Model for Pricing NIFTY Index Options.**
- Various IIM/IIT working papers on NIFTY VIX, expiry-day effects, and the Indian VRP — search SSRN with `NIFTY options` for the latest.

### 12.12 Books worth owning
- López de Prado, M. **Advances in Financial Machine Learning.** Wiley.
- de Prado, M. **Machine Learning for Asset Managers.** CUP.
- Dixon, M., Halperin, I., Bilokon, P. **Machine Learning in Finance: From Theory to Practice.** Springer.
- Jansen, S. **Machine Learning for Algorithmic Trading** (2nd ed). Packt — practical pipeline recipes.
- Goodfellow, I., Bengio, Y., Courville, A. **Deep Learning.** MIT Press.
- Hastie, T., Tibshirani, R., Friedman, J. **The Elements of Statistical Learning.**

> When in doubt: validation/labelling → AFML; sequence models → TFT/PatchTST papers; vol-surface DL → Horvath/Bergeron/Ackerer; hedging RL → Buehler "Deep Hedging".
