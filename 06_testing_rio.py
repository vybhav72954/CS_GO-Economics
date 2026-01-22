"""
Phase 6: Rio Validation
Dataset: Rio 2022 (Holdout - DO NOT TOUCH UNTIL NOW)

Final out-of-sample test of the momentum decomposition finding.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
BASE_DIR = Path(r"Z:\Projects\2025\CS-GO Time Series Analysis")
OUTPUT_DIR = BASE_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# LOAD DATA
# =============================================================================

print("="*70)
print("PHASE 6: RIO VALIDATION (Final Holdout Test)")
print("="*70)

stockholm = pd.read_csv(BASE_DIR / "json_output" / "Stockholm" / "stockholm_rounds.csv")
antwerp = pd.read_csv(BASE_DIR / "json_output" / "Antwerp" / "antwerp_rounds.csv")
rio = pd.read_csv(BASE_DIR / "json_output" / "Rio" / "rio_rounds.csv")

print(f"\nDataset sizes:")
print(f"  Stockholm (training):     {len(stockholm)} rounds, {stockholm['match_id'].nunique()} matches")
print(f"  Antwerp (confirmation):   {len(antwerp)} rounds, {antwerp['match_id'].nunique()} matches")
print(f"  Rio (validation):         {len(rio)} rounds, {rio['match_id'].nunique()} matches")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_features(df):
    df = df.copy()

    # Regime dummies
    df['regime_building'] = (df['ct_economic_regime'] == 'building').astype(int)
    df['regime_full_buy'] = (df['ct_economic_regime'] == 'full_buy').astype(int)
    df['regime_flush'] = (df['ct_economic_regime'] == 'flush').astype(int)

    df['t_regime_building'] = (df['t_economic_regime'] == 'building').astype(int)
    df['t_regime_full_buy'] = (df['t_economic_regime'] == 'full_buy').astype(int)
    df['t_regime_flush'] = (df['t_economic_regime'] == 'flush').astype(int)

    # Continuous
    df['equip_adv_10k'] = df['equip_advantage'] / 10000

    # Map dummies
    map_dummies = pd.get_dummies(df['map_name'], prefix='map', drop_first=True)
    for col in map_dummies.columns:
        df[col] = map_dummies[col].astype(int)

    # Phase dummies
    df['phase_pistol'] = (df['round_phase'] == 'pistol').astype(int)
    df['phase_conversion'] = (df['round_phase'] == 'conversion').astype(int)
    df['phase_overtime'] = (df['round_phase'] == 'overtime').astype(int)

    # Both full buy
    df['both_full_buy'] = ((df['ct_economic_regime'].isin(['full_buy', 'flush'])) &
                           (df['t_economic_regime'].isin(['full_buy', 'flush']))).astype(int)

    return df

rio_analysis = prepare_features(rio.dropna(subset=['ct_won_lag_1']))
print(f"\nRio analysis sample: {len(rio_analysis)} rounds")


# =============================================================================
# MODEL FUNCTION
# =============================================================================

def run_logit(y, X, model_name):
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        results = {
            'model': model_name,
            'n_obs': int(model.nobs),
            'pseudo_r2': model.prsquared,
            'momentum_coef': model.params.get('ct_won_lag_1', np.nan),
            'momentum_se': model.bse.get('ct_won_lag_1', np.nan),
            'momentum_pvalue': model.pvalues.get('ct_won_lag_1', np.nan),
            'momentum_or': np.exp(model.params.get('ct_won_lag_1', 0)),
        }
        return model, results
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


# =============================================================================
# RIO MODELS
# =============================================================================

print("\n" + "="*70)
print("RIO VALIDATION MODELS")
print("="*70)

y = rio_analysis['ct_wins_round']
regime_vars = ['regime_building', 'regime_full_buy', 'regime_flush',
               't_regime_building', 't_regime_full_buy', 't_regime_flush']
map_vars = [c for c in rio_analysis.columns if c.startswith('map_de_')]
phase_vars = ['phase_pistol', 'phase_conversion', 'phase_overtime']

rio_results = []

# 1. Baseline
X = sm.add_constant(rio_analysis[['ct_won_lag_1']].astype(float))
_, res = run_logit(y, X, 'Baseline')
rio_results.append(res)

# 2. + Equipment (PRIMARY)
X = sm.add_constant(rio_analysis[['ct_won_lag_1', 'equip_adv_10k']].astype(float))
_, res = run_logit(y, X, '+ Equipment')
rio_results.append(res)

# 3. + Equipment + Maps + Phase
X = sm.add_constant(rio_analysis[['ct_won_lag_1', 'equip_adv_10k'] + map_vars + phase_vars].astype(float))
_, res = run_logit(y, X, '+ Equip + Maps + Phase')
rio_results.append(res)

# 4. + Regime Dummies
X = sm.add_constant(rio_analysis[['ct_won_lag_1'] + regime_vars].astype(float))
_, res = run_logit(y, X, '+ Regime Dummies')
rio_results.append(res)

# 5. Regime + Equipment
X = sm.add_constant(rio_analysis[['ct_won_lag_1', 'equip_adv_10k'] + regime_vars].astype(float))
_, res = run_logit(y, X, 'Regime + Equipment')
rio_results.append(res)

print(f"\n{'Model':<25} {'Coef':>8} {'SE':>8} {'OR':>8} {'p':>10}")
print("-"*65)

for res in rio_results:
    sig = '***' if res['momentum_pvalue'] < 0.001 else '**' if res['momentum_pvalue'] < 0.01 else '*' if res['momentum_pvalue'] < 0.05 else ''
    print(f"{res['model']:<25} {res['momentum_coef']:>8.4f} {res['momentum_se']:>8.4f} "
          f"{res['momentum_or']:>8.3f} {res['momentum_pvalue']:>9.4f}{sig}")


# =============================================================================
# ROBUSTNESS: BOTH FULL BUY
# =============================================================================

print("\n" + "="*70)
print("ROBUSTNESS: BOTH TEAMS FULL BUY (Rio)")
print("="*70)

fb_df = rio_analysis[rio_analysis['both_full_buy'] == 1]
print(f"Sample: {len(fb_df)} rounds")

y_fb = fb_df['ct_wins_round']

X = sm.add_constant(fb_df[['ct_won_lag_1']].astype(float))
_, res_fb_base = run_logit(y_fb, X, 'Full Buy Baseline')

X = sm.add_constant(fb_df[['ct_won_lag_1', 'equip_adv_10k']].astype(float))
_, res_fb_equip = run_logit(y_fb, X, 'Full Buy + Equip')

print(f"\nBaseline:    coef = {res_fb_base['momentum_coef']:.4f}, OR = {res_fb_base['momentum_or']:.3f}, p = {res_fb_base['momentum_pvalue']:.4f}")
print(f"+ Equipment: coef = {res_fb_equip['momentum_coef']:.4f}, OR = {res_fb_equip['momentum_or']:.3f}, p = {res_fb_equip['momentum_pvalue']:.4f}")


# =============================================================================
# CROSS-TOURNAMENT COMPARISON
# =============================================================================

print("\n" + "="*70)
print("CROSS-TOURNAMENT COMPARISON")
print("="*70)

# Run same models on all three tournaments
def run_tournament_models(df, name):
    df = prepare_features(df.dropna(subset=['ct_won_lag_1']))
    y = df['ct_wins_round']
    regime_vars = ['regime_building', 'regime_full_buy', 'regime_flush',
                   't_regime_building', 't_regime_full_buy', 't_regime_flush']

    results = {}

    # Baseline
    X = sm.add_constant(df[['ct_won_lag_1']].astype(float))
    _, res = run_logit(y, X, 'Baseline')
    results['baseline'] = res

    # + Equipment
    X = sm.add_constant(df[['ct_won_lag_1', 'equip_adv_10k']].astype(float))
    _, res = run_logit(y, X, '+ Equipment')
    results['equipment'] = res

    # + Regime
    X = sm.add_constant(df[['ct_won_lag_1'] + regime_vars].astype(float))
    _, res = run_logit(y, X, '+ Regime')
    results['regime'] = res

    return results

stockholm_res = run_tournament_models(stockholm, 'Stockholm')
antwerp_res = run_tournament_models(antwerp, 'Antwerp')
rio_res = run_tournament_models(rio, 'Rio')

print(f"\n{'Specification':<20} {'Stockholm':>18} {'Antwerp':>18} {'Rio':>18}")
print("-"*75)

for spec in ['baseline', 'equipment', 'regime']:
    s = stockholm_res[spec]
    a = antwerp_res[spec]
    r = rio_res[spec]

    s_str = f"{s['momentum_coef']:.2f} (p={s['momentum_pvalue']:.3f})"
    a_str = f"{a['momentum_coef']:.2f} (p={a['momentum_pvalue']:.3f})"
    r_str = f"{r['momentum_coef']:.2f} (p={r['momentum_pvalue']:.3f})"

    print(f"{spec:<20} {s_str:>18} {a_str:>18} {r_str:>18}")


# =============================================================================
# FULL SAMPLE (ALL THREE TOURNAMENTS)
# =============================================================================

print("\n" + "="*70)
print("FULL SAMPLE (All Three Tournaments)")
print("="*70)

stockholm['tournament'] = 'Stockholm'
antwerp['tournament'] = 'Antwerp'
rio['tournament'] = 'Rio'

full_sample = pd.concat([stockholm, antwerp, rio], ignore_index=True)
full_analysis = prepare_features(full_sample.dropna(subset=['ct_won_lag_1']))

print(f"Full sample: {len(full_analysis)} rounds from {full_analysis['match_id'].nunique()} matches")

y_full = full_analysis['ct_wins_round']
regime_vars = ['regime_building', 'regime_full_buy', 'regime_flush',
               't_regime_building', 't_regime_full_buy', 't_regime_flush']

# Baseline
X = sm.add_constant(full_analysis[['ct_won_lag_1']].astype(float))
_, res_full_base = run_logit(y_full, X, 'Full: Baseline')

# + Equipment
X = sm.add_constant(full_analysis[['ct_won_lag_1', 'equip_adv_10k']].astype(float))
_, res_full_equip = run_logit(y_full, X, 'Full: + Equipment')

# + Regime
X = sm.add_constant(full_analysis[['ct_won_lag_1'] + regime_vars].astype(float))
_, res_full_regime = run_logit(y_full, X, 'Full: + Regime')

# Regime + Equipment
X = sm.add_constant(full_analysis[['ct_won_lag_1', 'equip_adv_10k'] + regime_vars].astype(float))
_, res_full_both = run_logit(y_full, X, 'Full: Regime + Equip')

print(f"\n{'Model':<25} {'Coef':>8} {'SE':>8} {'OR':>8} {'p':>10}")
print("-"*65)

for res in [res_full_base, res_full_equip, res_full_regime, res_full_both]:
    sig = '***' if res['momentum_pvalue'] < 0.001 else '**' if res['momentum_pvalue'] < 0.01 else '*' if res['momentum_pvalue'] < 0.05 else ''
    print(f"{res['model']:<25} {res['momentum_coef']:>8.4f} {res['momentum_se']:>8.4f} "
          f"{res['momentum_or']:>8.3f} {res['momentum_pvalue']:>9.4f}{sig}")


# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tournaments = [
    ('Stockholm', stockholm_res),
    ('Antwerp', antwerp_res),
    ('Rio', rio_res)
]

for ax, (name, res_dict) in zip(axes, tournaments):
    specs = ['baseline', 'equipment', 'regime']
    labels = ['Baseline', '+Equipment', '+Regime']
    coefs = [res_dict[s]['momentum_coef'] for s in specs]
    ses = [res_dict[s]['momentum_se'] for s in specs]
    pvals = [res_dict[s]['momentum_pvalue'] for s in specs]
    colors = ['steelblue' if p < 0.05 else 'lightgray' for p in pvals]

    ax.bar(range(len(specs)), coefs, yerr=[1.96*se for se in ses],
           capsize=4, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Momentum Coefficient')
    ax.set_title(f'{name}')
    ax.set_ylim(-0.8, 1.5)

    for i, (coef, p) in enumerate(zip(coefs, pvals)):
        sig = '***' if p < 0.001 else '*' if p < 0.05 else 'ns'
        ax.annotate(f'{coef:.2f}\n({sig})', (i, coef + 0.2 if coef >= 0 else coef - 0.3),
                    ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_10_rio_validation.png', dpi=150)
plt.close()
print("Saved: fig_10_rio_validation.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("PHASE 6: FINAL VALIDATION SUMMARY")
print("="*70)

print(f"""
RIO VALIDATION RESULTS:

  Baseline:      coef = {rio_results[0]['momentum_coef']:.4f}, OR = {rio_results[0]['momentum_or']:.2f}x, p = {rio_results[0]['momentum_pvalue']:.4f}
  + Equipment:   coef = {rio_results[1]['momentum_coef']:.4f}, OR = {rio_results[1]['momentum_or']:.2f}x, p = {rio_results[1]['momentum_pvalue']:.4f}
  + Regime:      coef = {rio_results[3]['momentum_coef']:.4f}, OR = {rio_results[3]['momentum_or']:.2f}x, p = {rio_results[3]['momentum_pvalue']:.4f}

FULL SAMPLE (N = {len(full_analysis)}):

  Baseline:      coef = {res_full_base['momentum_coef']:.4f}, OR = {res_full_base['momentum_or']:.2f}x, p = {res_full_base['momentum_pvalue']:.4f}
  + Equipment:   coef = {res_full_equip['momentum_coef']:.4f}, OR = {res_full_equip['momentum_or']:.2f}x, p = {res_full_equip['momentum_pvalue']:.4f}
  + Regime:      coef = {res_full_regime['momentum_coef']:.4f}, OR = {res_full_regime['momentum_or']:.2f}x, p = {res_full_regime['momentum_pvalue']:.4f}
  Both:          coef = {res_full_both['momentum_coef']:.4f}, OR = {res_full_both['momentum_or']:.2f}x, p = {res_full_both['momentum_pvalue']:.4f}

REPLICATION STATUS:
""")

# Check replication
rio_baseline_sig = rio_results[0]['momentum_pvalue'] < 0.05
rio_equip_nonsig = rio_results[1]['momentum_pvalue'] >= 0.05
rio_regime_nonsig = rio_results[3]['momentum_pvalue'] >= 0.05

if rio_baseline_sig:
    print("  [PASS] Rio baseline momentum is significant")
else:
    print("  [FAIL] Rio baseline momentum is NOT significant")

if rio_equip_nonsig:
    print("  [PASS] Rio momentum eliminated by equipment control")
else:
    print("  [FAIL] Rio momentum NOT eliminated by equipment control")

if rio_regime_nonsig:
    print("  [PASS] Rio momentum eliminated by regime control")
else:
    print("  [FAIL] Rio momentum NOT eliminated by regime control")


# =============================================================================
# SAVE
# =============================================================================

rio_df = pd.DataFrame(rio_results)
rio_df['tournament'] = 'Rio'
rio_df.to_csv(OUTPUT_DIR / 'phase6_rio_validation.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'phase6_rio_validation.csv'}")

full_results = pd.DataFrame([res_full_base, res_full_equip, res_full_regime, res_full_both])
full_results.to_csv(OUTPUT_DIR / 'phase6_full_sample_results.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'phase6_full_sample_results.csv'}")

print("\nDone.")
