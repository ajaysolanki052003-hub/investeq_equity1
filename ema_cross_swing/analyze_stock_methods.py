"""Read stock_method_preset_metrics.csv and produce the framework outputs:

  1. Per-stock winning method per preset (raw assignment)
  2. Stability score across 6 presets (consistency check)
  3. Method assignment with confidence (≥30 trades, ≥20% expectancy gap, +EV)
  4. Drop list (no edge / unstable)
  5. Side asymmetry
  6. Structural rule: ATR%-bucket → preferred method
  7. Best universal preset
  8. In-sample vs Out-of-sample reality check

Outputs:
  - stock_method_map.csv    — final per-(Symbol, Side) assignment
  - structural_rule.txt     — rule of thumb
  - analysis_summary.txt    — top-level numbers
"""
import numpy as np
import pandas as pd

MIN_TRADES   = 30
MIN_GAP_PCT  = 0.20   # winning expectancy must be 20% > runner-up
RANK_METRIC  = 'expectancy'   # 'expectancy' or 'total_pnl' or 'mean_pnl'

print('loading stock_method_preset_metrics.csv ...')
df = pd.read_csv('stock_method_preset_metrics.csv')
print(f'rows: {len(df)}   symbols: {df.Symbol.nunique()}   sides: {sorted(df.Side.unique())}')


# ─── 1. Per-stock × per-preset winning method (using ALL window for assignment) ───
all_df = df[df.Sample == 'ALL'].copy()
def best_method_row(sub):
    sub = sub.copy()
    sub = sub.sort_values(RANK_METRIC, ascending=False, na_position='last')
    sub_valid = sub[sub.n_trades >= MIN_TRADES]
    if len(sub_valid) < 2:
        return None
    top = sub_valid.iloc[0]
    runner = sub_valid.iloc[1]
    if not np.isfinite(top[RANK_METRIC]) or top[RANK_METRIC] <= 0:
        return None
    # Gap: winning expectancy must beat runner-up by ≥ MIN_GAP_PCT (relative)
    if runner[RANK_METRIC] > 0 and (top[RANK_METRIC] - runner[RANK_METRIC]) / abs(runner[RANK_METRIC]) < MIN_GAP_PCT:
        # weak win → still record but flag as low_confidence
        flag = 'low_confidence'
    else:
        flag = 'ok'
    return dict(
        winning_method=top.Method, winning_param=(top.target_pct if top.Method=='target_pct'
                                                   else top.R_mult if top.Method=='R'
                                                   else top.atr_mult),
        expectancy=float(top.expectancy), total_pnl=float(top.total_pnl),
        n_trades=int(top.n_trades), win_rate=float(top.win_rate),
        runner_method=runner.Method, runner_expectancy=float(runner.expectancy),
        flag=flag,
    )

per_preset_rows = []
for (sym, side, preset), g in all_df.groupby(['Symbol', 'Side', 'Preset']):
    r = best_method_row(g)
    if r is None:
        continue
    r.update({'Symbol': sym, 'Side': side, 'Preset': preset,
              'atr_pct_med': float(g.atr_pct_med.iloc[0]),
              'avg_price':   float(g.avg_price.iloc[0]),
              'listing_age_y': float(g.listing_age_y.iloc[0])})
    per_preset_rows.append(r)
per_preset = pd.DataFrame(per_preset_rows)
print(f'\n[1] per-preset winners computed: {len(per_preset)} (stock, side, preset) rows '
      f'with ≥{MIN_TRADES} trades and +EV')


# ─── 2. Stability score: out of N presets a stock has a winner, how often is the
#       winning METHOD the same? ───
stability_rows = []
for (sym, side), g in per_preset.groupby(['Symbol', 'Side']):
    n_pres = len(g)
    mode_method = g.winning_method.mode()
    if len(mode_method) == 0:
        continue
    top_method = mode_method.iloc[0]
    agree = int((g.winning_method == top_method).sum())
    # average expectancy of the dominant method across presets where it won
    avg_exp = g[g.winning_method == top_method].expectancy.mean()
    stability_rows.append({
        'Symbol': sym, 'Side': side,
        'dominant_method': top_method,
        'agree_presets': agree, 'total_presets': n_pres,
        'stability_pct': 100.0 * agree / n_pres,
        'avg_expectancy_when_dominant': float(avg_exp),
        'atr_pct_med':   float(g.atr_pct_med.iloc[0]),
        'avg_price':     float(g.avg_price.iloc[0]),
        'listing_age_y': float(g.listing_age_y.iloc[0]),
    })
stability = pd.DataFrame(stability_rows)
print(f'[2] stability computed: {len(stability)} (stock, side) rows')


# ─── 3. Method assignment with confidence ───
STABLE_THRESH = 4    # dominant method must win in ≥4 of 6 presets
def assign_row(r):
    if r.total_presets < 3:           # too few presets even produced a winner
        return 'no_edge'
    if r.agree_presets >= STABLE_THRESH and r.avg_expectancy_when_dominant > 0:
        return 'tradeable'
    if r.avg_expectancy_when_dominant <= 0:
        return 'no_edge'
    return 'fragile'

stability['assignment'] = stability.apply(assign_row, axis=1)


# ─── 4. Drop list & tradeable map ───
tradeable = stability[stability.assignment == 'tradeable'].copy()
fragile   = stability[stability.assignment == 'fragile']
no_edge   = stability[stability.assignment == 'no_edge']

# For tradeable stocks, pull the BEST preset+param for the dominant method (by expectancy)
best_per_stock = []
for _, r in tradeable.iterrows():
    sub = per_preset[(per_preset.Symbol == r.Symbol) &
                     (per_preset.Side == r.Side) &
                     (per_preset.winning_method == r.dominant_method)]
    if sub.empty: continue
    best = sub.sort_values('expectancy', ascending=False).iloc[0]
    best_per_stock.append({
        'Symbol': r.Symbol, 'Side': r.Side,
        'Method': r.dominant_method, 'Param': float(best.winning_param),
        'Preset': best.Preset,
        'Expectancy_IS_OOS_mixed': float(best.expectancy),
        'Total_PnL_IS_OOS_mixed': float(best.total_pnl),
        'Trades': int(best.n_trades),
        'Stability': f'{int(r.agree_presets)}/{int(r.total_presets)}',
        'atr_pct_med': r.atr_pct_med, 'avg_price': r.avg_price,
    })
stock_method_map = pd.DataFrame(best_per_stock)


# ─── 5. Side asymmetry ───
asymmetry = stability.pivot_table(
    index='Symbol', columns='Side', values='assignment', aggfunc='first'
).fillna('—')
asym_counts = asymmetry.apply(lambda r: tuple(sorted([r.get('BUY', '—'), r.get('SELL', '—')])), axis=1).value_counts()


# ─── 6. Structural rule — ATR%-bucket → preferred method ───
def atr_bucket(x):
    if x < 1.5: return 'low (<1.5%)'
    if x < 2.5: return 'med-low (1.5-2.5%)'
    if x < 3.5: return 'med-high (2.5-3.5%)'
    if x < 5.0: return 'high (3.5-5%)'
    return 'very-high (>=5%)'

tradeable['atr_bucket'] = tradeable.atr_pct_med.apply(atr_bucket)
structural_rule = (tradeable.groupby(['atr_bucket', 'Side', 'dominant_method'])
                   .size().unstack(fill_value=0))
structural_rule_pct = structural_rule.div(structural_rule.sum(axis=1), axis=0) * 100

# Price bucket rule
def price_bucket(p):
    if p < 100:   return 'cheap (<₹100)'
    if p < 500:   return 'low (₹100-500)'
    if p < 2000:  return 'mid (₹500-2000)'
    if p < 5000:  return 'high (₹2000-5000)'
    return 'premium (>₹5000)'
tradeable['price_bucket'] = tradeable.avg_price.apply(price_bucket)


# ─── 7. Best universal preset ───
def preset_universal(df_all, sample='ALL'):
    sub = df_all[(df_all.Sample == sample) & (df_all.n_trades >= MIN_TRADES)]
    # For each (stock, side, preset) take the BEST method (max expectancy)
    best_per = sub.loc[sub.groupby(['Symbol','Side','Preset'])['expectancy'].idxmax()]
    return (best_per.groupby('Preset')
            .agg(median_exp=('expectancy','median'),
                 mean_exp=('expectancy','mean'),
                 sum_total_pnl=('total_pnl','sum'),
                 n=('Symbol','count'))
            .sort_values('median_exp', ascending=False))
universal_all = preset_universal(df, 'ALL')


# ─── 8. IS vs OOS reality check ───
def is_oos_pivot(df_in):
    is_d  = df_in[df_in.Sample == 'IS'].set_index(['Symbol','Side','Method','Preset'])
    oos_d = df_in[df_in.Sample == 'OOS'].set_index(['Symbol','Side','Method','Preset'])
    join = is_d[['expectancy','total_pnl','n_trades']].rename(
        columns={'expectancy':'IS_exp','total_pnl':'IS_pnl','n_trades':'IS_n'}).join(
        oos_d[['expectancy','total_pnl','n_trades']].rename(
            columns={'expectancy':'OOS_exp','total_pnl':'OOS_pnl','n_trades':'OOS_n'}),
        how='inner')
    return join.reset_index()
is_oos = is_oos_pivot(df)

# For each tradeable (Symbol, Side, Method) — pull IS vs OOS at best preset
oos_check_rows = []
for _, r in stock_method_map.iterrows():
    sub = is_oos[(is_oos.Symbol == r.Symbol) &
                 (is_oos.Side == r.Side) &
                 (is_oos.Method == r.Method) &
                 (is_oos.Preset == r.Preset)]
    if sub.empty: continue
    rr = sub.iloc[0]
    oos_check_rows.append({
        'Symbol': r.Symbol, 'Side': r.Side, 'Method': r.Method,
        'Preset': r.Preset, 'Param': r.Param,
        'IS_exp':  float(rr.IS_exp)  if pd.notna(rr.IS_exp)  else np.nan,
        'OOS_exp': float(rr.OOS_exp) if pd.notna(rr.OOS_exp) else np.nan,
        'IS_pnl':  float(rr.IS_pnl), 'OOS_pnl': float(rr.OOS_pnl),
        'IS_n': int(rr.IS_n), 'OOS_n': int(rr.OOS_n),
    })
oos_check = pd.DataFrame(oos_check_rows)
if not oos_check.empty:
    oos_check['exp_decay_pct'] = np.where(
        oos_check.IS_exp.abs() > 0,
        (oos_check.OOS_exp - oos_check.IS_exp) / oos_check.IS_exp.abs() * 100, np.nan)
    oos_check['oos_survives'] = (oos_check.OOS_exp > 0) & (oos_check.OOS_n >= 5)


# ─── SAVE OUTPUTS ───
stock_method_map.to_csv('stock_method_map.csv', index=False)
stability.to_csv('stock_stability.csv', index=False)
oos_check.to_csv('stock_oos_check.csv', index=False)

# Validated subset — only OOS-survivors
if not oos_check.empty:
    val = oos_check[oos_check.oos_survives == True][['Symbol', 'Side', 'Method', 'Preset', 'Param']]
    validated = stock_method_map.merge(val, on=['Symbol', 'Side', 'Method', 'Preset', 'Param'])
    validated.to_csv('stock_method_map_validated.csv', index=False)
    print(f'[OUTPUT] stock_method_map_validated.csv  {len(validated)} rows  '
          f'(OOS-validated subset — highest-confidence list)')

with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
    w = f.write
    w('=' * 78 + '\n')
    w('PER-STOCK METHOD ASSIGNMENT — ANALYSIS SUMMARY\n')
    w('=' * 78 + '\n\n')
    w(f'Total stocks processed     : {df.Symbol.nunique()}\n')
    w(f'Total (stock, side) pairs  : {len(stability)}\n')
    w(f'  tradeable                : {(stability.assignment=="tradeable").sum()}\n')
    w(f'  fragile                  : {(stability.assignment=="fragile").sum()}\n')
    w(f'  no_edge                  : {(stability.assignment=="no_edge").sum()}\n\n')

    w('TRADEABLE — method split:\n')
    w((tradeable.dominant_method.value_counts()
       .to_frame('count').reset_index().to_string(index=False)) + '\n\n')

    w('Side asymmetry — (BUY, SELL) assignment combinations:\n')
    w(asym_counts.to_string() + '\n\n')

    w('Structural rule — % of tradeable stocks per (ATR-bucket, side) preferring each method:\n')
    w(structural_rule_pct.round(1).to_string() + '\n\n')

    w('Structural rule — same but by PRICE-bucket:\n')
    pr = (tradeable.groupby(['price_bucket', 'Side', 'dominant_method']).size()
          .unstack(fill_value=0))
    pr_pct = pr.div(pr.sum(axis=1), axis=0) * 100
    w(pr_pct.round(1).to_string() + '\n\n')

    w('BEST UNIVERSAL PRESET (median expectancy across all stocks, best method per stock-preset):\n')
    w(universal_all.round(3).to_string() + '\n\n')

    if not oos_check.empty:
        w('IN-SAMPLE vs OUT-OF-SAMPLE check for tradeable stocks:\n')
        w(f'  median exp decay (OOS-IS)/|IS|  : {oos_check.exp_decay_pct.median():+.1f}%\n')
        w(f'  share where OOS_exp > 0         : '
          f'{100*oos_check.oos_survives.mean():.1f}%\n')
        w(f'  share where IS_exp > 0          : '
          f'{100*(oos_check.IS_exp > 0).mean():.1f}%\n\n')

with open('structural_rule.txt', 'w', encoding='utf-8') as f:
    f.write('STRUCTURAL RULE (ATR%-bucket × Side → preferred exit method)\n')
    f.write('=' * 68 + '\n\n')
    f.write('Reading: each row sums to 100%; the dominant method per bucket-side is the rule.\n\n')
    f.write(structural_rule_pct.round(1).to_string())
    f.write('\n')

print(f"\n[OUTPUT] stock_method_map.csv         {len(stock_method_map)} rows  (use these in the app)")
print(f"[OUTPUT] stock_stability.csv          {len(stability)} rows  (all stock-sides)")
print(f"[OUTPUT] stock_oos_check.csv          {len(oos_check)} rows  (IS vs OOS for tradeable)")
print(f"[OUTPUT] analysis_summary.txt         (top-level numbers)")
print(f"[OUTPUT] structural_rule.txt          (atr-bucket -> method rule)")

print("\n────────────  HEADLINE  ────────────")
print(f'  tradeable: {(stability.assignment=="tradeable").sum():>4}  '
      f'fragile: {(stability.assignment=="fragile").sum():>4}  '
      f'no_edge: {(stability.assignment=="no_edge").sum():>4}')
print(f'  tradeable method split:')
for m, n in tradeable.dominant_method.value_counts().items():
    print(f'    {m:<12} {n:>4}')
print(f'  best universal preset (by median expectancy): {universal_all.index[0]}'
      f'  (median exp={universal_all.iloc[0].median_exp:.3f}%)')
if not oos_check.empty:
    print(f'  OOS survival rate: {100*oos_check.oos_survives.mean():.1f}%')
print()
