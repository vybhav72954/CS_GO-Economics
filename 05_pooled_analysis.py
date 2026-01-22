"""
Phase 5: Pooled Analysis
Datasets: Stockholm 2021 + Antwerp 2022

Primary specification: Continuous (equipment $)
Secondary: Categorical (regime dummies) for comparison
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
# LOAD AND COMBINE DATA
# =============================================================================

print("=" * 70)
print("PHASE 5: POOLED ANALYSIS")
print("=" * 70)

stockholm = pd.read_csv(BASE_DIR / "json_output" / "Stockholm" / "stockholm_rounds.csv")
antwerp = pd.read_csv(BASE_DIR / "json_output" / "Antwerp" / "antwerp_rounds.csv")

stockholm['tournament'] = 'Stockholm'
antwerp['tournament'] = 'Antwerp'

pooled = pd.concat([stockholm, antwerp], ignore_index=True)

print(f"\nDataset sizes:")
print(f"  Stockholm: {len(stockholm)} rounds, {stockholm['match_id'].nunique()} matches")
print(f"  Antwerp:   {len(antwerp)} rounds, {antwerp['match_id'].nunique()} matches")
print(f"  Pooled:    {len(pooled)} rounds, {pooled['match_id'].nunique()} matches")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_features(df):
    df = df.copy()

    # Regime dummies
    df['regime_building'] = (df['ct_economic_regime']=='building').astype(int)
    df['regime_full_buy'] = (df['ct_economic_regime']=='full_buy').astype(int)
    df['regime_flush'] = (df['ct_economic_regime']=='flush').astype(int)

    df['t_regime_building'] = (df['t_economic_regime']=='building').astype(int)
    df['t_regime_full_buy'] = (df['t_economic_regime']=='full_buy').astype(int)
    df['t_regime_flush'] = (df['t_economic_regime']=='flush').astype(int)

    # Continuous
    df['equip_adv_10k'] = df['equip_advantage'] / 10000

    # Map dummies
    map_dummies = pd.get_dummies(df['map_name'], prefix='map', drop_first=True)
    for col in map_dummies.columns:
        df[col] = map_dummies[col].astype(int)

    # Phase dummies
    df['phase_pistol'] = (df['round_phase']=='pistol').astype(int)
    df['phase_conversion'] = (df['round_phase']=='conversion').astype(int)
    df['phase_overtime'] = (df['round_phase']=='overtime').astype(int)

    # Tournament dummy
    df['is_antwerp'] = (df['tournament']=='Antwerp').astype(int)

    # Both full buy
    df['both_full_buy'] = ((df['ct_economic_regime'].isin(['full_buy', 'flush'])) &
                           (df['t_economic_regime'].isin(['full_buy', 'flush']))).astype(int)

    return df


pooled_analysis = prepare_features(pooled.dropna(subset=['ct_won_lag_1']))
print(f"\nAnalysis sample: {len(pooled_analysis)} rounds")


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def run_logit(y, X, model_name):
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        results = {
            'model':model_name,
            'n_obs':int(model.nobs),
            'pseudo_r2':model.prsquared,
            'aic':model.aic,
            'momentum_coef':model.params.get('ct_won_lag_1', np.nan),
            'momentum_se':model.bse.get('ct_won_lag_1', np.nan),
            'momentum_pvalue':model.pvalues.get('ct_won_lag_1', np.nan),
            'momentum_or':np.exp(model.params.get('ct_won_lag_1', 0)),
        }
        return model, results
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


def print_model(model, results, show_coefs=False):
    sig = '***' if results['momentum_pvalue'] < 0.001 else '**' if results['momentum_pvalue'] < 0.01 else '*' if \
    results['momentum_pvalue'] < 0.05 else ''
    print(f"\n{results['model']}")
    print(f"  N={results['n_obs']}, Pseudo R²={results['pseudo_r2']:.4f}")
    print(
        f"  Momentum: coef={results['momentum_coef']:.4f}, OR={results['momentum_or']:.3f}, p={results['momentum_pvalue']:.4f} {sig}")

    if show_coefs and model is not None:
        print(f"  Coefficients:")
        for var in model.params.index:
            coef = model.params[var]
            p = model.pvalues[var]
            sig_var = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {var:<25} {coef:>8.4f} {sig_var}")


# =============================================================================
# POOLED MODELS
# =============================================================================

print("\n" + "=" * 70)
print("POOLED MODELS (Stockholm + Antwerp)")
print("=" * 70)

y = pooled_analysis['ct_wins_round']
regime_vars = ['regime_building', 'regime_full_buy', 'regime_flush',
               't_regime_building', 't_regime_full_buy', 't_regime_flush']
map_vars = [c for c in pooled_analysis.columns if c.startswith('map_de_')]
phase_vars = ['phase_pistol', 'phase_conversion', 'phase_overtime']

all_results = []

# 1. Baseline
X = sm.add_constant(pooled_analysis[['ct_won_lag_1']].astype(float))
model, res = run_logit(y, X, '1. Baseline')
print_model(model, res)
all_results.append(res)

# 2. + Tournament FE
X = sm.add_constant(pooled_analysis[['ct_won_lag_1', 'is_antwerp']].astype(float))
model, res = run_logit(y, X, '2. + Tournament FE')
print_model(model, res)
all_results.append(res)

# 3. + Equipment (PRIMARY SPECIFICATION)
X = sm.add_constant(pooled_analysis[['ct_won_lag_1', 'is_antwerp', 'equip_adv_10k']].astype(float))
model_primary, res_primary = run_logit(y, X, '3. + Equipment (PRIMARY)')
print_model(model_primary, res_primary, show_coefs=True)
all_results.append(res_primary)

# 4. + Equipment + Maps + Phase
X = sm.add_constant(
    pooled_analysis[['ct_won_lag_1', 'is_antwerp', 'equip_adv_10k'] + map_vars + phase_vars].astype(float))
model, res = run_logit(y, X, '4. + Equipment + Maps + Phase')
print_model(model, res)
all_results.append(res)

# 5. Regime dummies (for comparison)
X = sm.add_constant(pooled_analysis[['ct_won_lag_1', 'is_antwerp'] + regime_vars].astype(float))
model_regime, res_regime = run_logit(y, X, '5. + Regime Dummies')
print_model(model_regime, res_regime, show_coefs=True)
all_results.append(res_regime)

# 6. Regime + Equipment (does equipment explain residual?)
X = sm.add_constant(pooled_analysis[['ct_won_lag_1', 'is_antwerp', 'equip_adv_10k'] + regime_vars].astype(float))
model_both, res_both = run_logit(y, X, '6. Regime + Equipment')
print_model(model_both, res_both, show_coefs=True)
all_results.append(res_both)

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

print(f"\n{'Model':<35} {'Coef':>8} {'SE':>8} {'OR':>8} {'p':>10}")
print("-" * 70)

for res in all_results:
    sig = '***' if res['momentum_pvalue'] < 0.001 else '**' if res['momentum_pvalue'] < 0.01 else '*' if res[
                                                                                                             'momentum_pvalue'] < 0.05 else ''
    print(f"{res['model']:<35} {res['momentum_coef']:>8.4f} {res['momentum_se']:>8.4f} "
          f"{res['momentum_or']:>8.3f} {res['momentum_pvalue']:>9.4f}{sig}")

# =============================================================================
# ROBUSTNESS: BOTH FULL BUY
# =============================================================================

print("\n" + "=" * 70)
print("ROBUSTNESS: BOTH TEAMS FULL BUY (Pooled)")
print("=" * 70)

fb_df = pooled_analysis[pooled_analysis['both_full_buy']==1]
print(f"Sample: {len(fb_df)} rounds")

y_fb = fb_df['ct_wins_round']

# Baseline
X = sm.add_constant(fb_df[['ct_won_lag_1', 'is_antwerp']].astype(float))
_, res_fb_base = run_logit(y_fb, X, 'Full Buy: Baseline')
print(
    f"\nBaseline:    coef = {res_fb_base['momentum_coef']:.4f}, OR = {res_fb_base['momentum_or']:.3f}, p = {res_fb_base['momentum_pvalue']:.4f}")

# + Equipment
X = sm.add_constant(fb_df[['ct_won_lag_1', 'is_antwerp', 'equip_adv_10k']].astype(float))
_, res_fb_equip = run_logit(y_fb, X, 'Full Buy: + Equipment')
print(
    f"+ Equipment: coef = {res_fb_equip['momentum_coef']:.4f}, OR = {res_fb_equip['momentum_or']:.3f}, p = {res_fb_equip['momentum_pvalue']:.4f}")

# =============================================================================
# TOURNAMENT INTERACTION
# =============================================================================

print("\n" + "=" * 70)
print("HETEROGENEITY: MOMENTUM × TOURNAMENT INTERACTION")
print("=" * 70)

pooled_analysis['momentum_x_antwerp'] = pooled_analysis['ct_won_lag_1'] * pooled_analysis['is_antwerp']

X = sm.add_constant(
    pooled_analysis[['ct_won_lag_1', 'is_antwerp', 'momentum_x_antwerp', 'equip_adv_10k']].astype(float))
model_interact, res_interact = run_logit(y, X, 'Interaction Model')

print(
    f"\nMomentum (Stockholm):        coef = {model_interact.params['ct_won_lag_1']:.4f}, p = {model_interact.pvalues['ct_won_lag_1']:.4f}")
print(
    f"Momentum × Antwerp:          coef = {model_interact.params['momentum_x_antwerp']:.4f}, p = {model_interact.pvalues['momentum_x_antwerp']:.4f}")

if model_interact.pvalues['momentum_x_antwerp'] < 0.05:
    print("\n  >> Significant interaction: momentum effect differs between tournaments")
else:
    print("\n  >> No significant interaction: momentum effect is consistent across tournaments")

# =============================================================================
# EFFECT SIZE DECOMPOSITION
# =============================================================================

print("\n" + "=" * 70)
print("EFFECT SIZE DECOMPOSITION")
print("=" * 70)

baseline_coef = all_results[0]['momentum_coef']
primary_coef = res_primary['momentum_coef']

print(f"\nBaseline momentum coefficient: {baseline_coef:.4f}")
print(f"After equipment control:       {primary_coef:.4f}")

if baseline_coef > 0:
    reduction = (baseline_coef - primary_coef) / baseline_coef * 100
    print(f"\nReduction: {reduction:.1f}%")

    if primary_coef <= 0:
        print("Coefficient reduced to zero or negative (fully explained)")
    else:
        residual_pct = (primary_coef / baseline_coef) * 100
        print(f"Residual: {residual_pct:.1f}% of original effect remains")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "=" * 70)
print("CREATING FIGURES")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

models = [r['model'] for r in all_results]
coefs = [r['momentum_coef'] for r in all_results]
ses = [r['momentum_se'] for r in all_results]
colors = ['steelblue' if r['momentum_pvalue'] < 0.05 else 'lightgray' for r in all_results]

y_pos = range(len(all_results))
ax.barh(y_pos, coefs, xerr=[1.96 * se for se in ses], capsize=3,
        color=colors, edgecolor='black', alpha=0.8)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('Momentum Coefficient')
ax.set_title('Pooled Analysis: Momentum Across Specifications (n=3,538)')

for i, (coef, p) in enumerate(zip(coefs, [r['momentum_pvalue'] for r in all_results])):
    sig = '*' if p < 0.05 else ''
    ax.annotate(f'{coef:.3f}{sig}', (coef + 0.05 if coef >= 0 else coef - 0.15, i),
                va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_09_pooled_analysis.png', dpi=150)
plt.close()
print("Saved: fig_09_pooled_analysis.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 5 FINAL SUMMARY")
print("=" * 70)

print(f"""
POOLED ANALYSIS (Stockholm 2021 + Antwerp 2022)
N = {len(pooled_analysis)} rounds from {pooled_analysis['match_id'].nunique()} matches

BASELINE:
  Coefficient: {all_results[0]['momentum_coef']:.4f}
  Odds Ratio:  {all_results[0]['momentum_or']:.2f}x
  P-value:     {all_results[0]['momentum_pvalue']:.4f}

PRIMARY SPECIFICATION (+ Equipment $):
  Coefficient: {res_primary['momentum_coef']:.4f}
  Odds Ratio:  {res_primary['momentum_or']:.2f}x
  P-value:     {res_primary['momentum_pvalue']:.4f}

REGIME DUMMIES:
  Coefficient: {res_regime['momentum_coef']:.4f}
  Odds Ratio:  {res_regime['momentum_or']:.2f}x
  P-value:     {res_regime['momentum_pvalue']:.4f}

REGIME + EQUIPMENT:
  Coefficient: {res_both['momentum_coef']:.4f}
  Odds Ratio:  {res_both['momentum_or']:.2f}x
  P-value:     {res_both['momentum_pvalue']:.4f}

BOTH FULL BUY SUBSAMPLE:
  Baseline:    coef = {res_fb_base['momentum_coef']:.4f}, p = {res_fb_base['momentum_pvalue']:.4f}
  + Equipment: coef = {res_fb_equip['momentum_coef']:.4f}, p = {res_fb_equip['momentum_pvalue']:.4f}
""")

# =============================================================================
# SAVE
# =============================================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / 'phase5_pooled_results.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'phase5_pooled_results.csv'}")

print("\nDone.")
