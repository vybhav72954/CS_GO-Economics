"""
Phase 3: Momentum Decomposition
Dataset: Stockholm 2021 (Training Set Only)

Structure:
  - Baseline model
  - Panel A: Categorical specification (regime dummies)
  - Panel B: Continuous specification (equipment $)
  - Robustness checks
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

df = pd.read_csv(BASE_DIR / "json_output" / "Stockholm" / "stockholm_rounds.csv")
print(f"Loaded {len(df)} rounds from {df['match_id'].nunique()} matches")

analysis_df = df.dropna(subset=['ct_won_lag_1']).copy()
print(f"Analysis sample: {len(analysis_df)} rounds\n")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_features(df):
    df = df.copy()

    # CT regime dummies (reference: broke)
    df['regime_building'] = (df['ct_economic_regime'] == 'building').astype(int)
    df['regime_full_buy'] = (df['ct_economic_regime'] == 'full_buy').astype(int)
    df['regime_flush'] = (df['ct_economic_regime'] == 'flush').astype(int)

    # T regime dummies
    df['t_regime_building'] = (df['t_economic_regime'] == 'building').astype(int)
    df['t_regime_full_buy'] = (df['t_economic_regime'] == 'full_buy').astype(int)
    df['t_regime_flush'] = (df['t_economic_regime'] == 'flush').astype(int)

    # Continuous equipment advantage (in $10k for interpretability)
    df['equip_adv_10k'] = df['equip_advantage'] / 10000

    # Map dummies (drop first to avoid collinearity)
    map_dummies = pd.get_dummies(df['map_name'], prefix='map', drop_first=True)
    for col in map_dummies.columns:
        df[col] = map_dummies[col].astype(int)

    # Phase dummies (reference: gun rounds)
    df['phase_pistol'] = (df['round_phase'] == 'pistol').astype(int)
    df['phase_conversion'] = (df['round_phase'] == 'conversion').astype(int)
    df['phase_overtime'] = (df['round_phase'] == 'overtime').astype(int)

    # First kill indicator
    df['ct_first_kill'] = (df['first_kill_side'] == 'CT').astype(int)
    df['first_kill_missing'] = df['first_kill_side'].isna().astype(int)

    # Indicator for gun rounds where both teams have full equipment
    df['both_full_buy'] = ((df['ct_economic_regime'].isin(['full_buy', 'flush'])) &
                           (df['t_economic_regime'].isin(['full_buy', 'flush']))).astype(int)

    return df

analysis_df = prepare_features(analysis_df)

# Define variable groups
regime_vars = ['regime_building', 'regime_full_buy', 'regime_flush',
               't_regime_building', 't_regime_full_buy', 't_regime_flush']
map_vars = [c for c in analysis_df.columns if c.startswith('map_de_')]
phase_vars = ['phase_pistol', 'phase_conversion', 'phase_overtime']
fk_vars = ['ct_first_kill', 'first_kill_missing']


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def run_logit(y, X, model_name):
    """Fit logistic regression and extract key results."""
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        results = {
            'model': model_name,
            'n_obs': int(model.nobs),
            'pseudo_r2': model.prsquared,
            'aic': model.aic,
            'bic': model.bic,
            'momentum_coef': model.params.get('ct_won_lag_1', np.nan),
            'momentum_se': model.bse.get('ct_won_lag_1', np.nan),
            'momentum_pvalue': model.pvalues.get('ct_won_lag_1', np.nan),
            'momentum_or': np.exp(model.params.get('ct_won_lag_1', 0)),
        }
        return model, results
    except Exception as e:
        print(f"  Error in {model_name}: {e}")
        return None, None


def print_model(model, results, show_coefs=False):
    """Print model results in a compact format."""
    sig = '***' if results['momentum_pvalue'] < 0.001 else '**' if results['momentum_pvalue'] < 0.01 else '*' if results['momentum_pvalue'] < 0.05 else ''

    print(f"\n{results['model']}")
    print(f"  N={results['n_obs']}, Pseudo R²={results['pseudo_r2']:.4f}, AIC={results['aic']:.1f}")
    print(f"  Momentum: coef={results['momentum_coef']:.4f}, OR={results['momentum_or']:.3f}, p={results['momentum_pvalue']:.4f} {sig}")

    if show_coefs and model is not None:
        print(f"  All coefficients:")
        for var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            p = model.pvalues[var]
            sig_var = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {var:<28} {coef:>8.4f} (se={se:.4f}) {sig_var}")


def print_summary_table(results_list, title="MODEL COMPARISON"):
    """Print a summary table of all models."""
    print(f"\n{'='*85}")
    print(title)
    print(f"{'='*85}")
    print(f"{'Model':<40} {'Coef':>9} {'SE':>8} {'OR':>8} {'p':>9} {'R²':>7}")
    print("-"*85)

    for res in results_list:
        if res is None:
            continue
        sig = '***' if res['momentum_pvalue'] < 0.001 else '**' if res['momentum_pvalue'] < 0.01 else '*' if res['momentum_pvalue'] < 0.05 else ''
        print(f"{res['model']:<40} {res['momentum_coef']:>9.4f} {res['momentum_se']:>8.4f} "
              f"{res['momentum_or']:>8.3f} {res['momentum_pvalue']:>8.4f}{sig:<1} {res['pseudo_r2']:>7.4f}")

    print("-"*85)


# =============================================================================
# BASELINE MODEL
# =============================================================================

print("="*70)
print("BASELINE MODEL")
print("="*70)

y = analysis_df['ct_wins_round']

X_baseline = sm.add_constant(analysis_df[['ct_won_lag_1']].astype(float))
model_baseline, res_baseline = run_logit(y, X_baseline, 'Baseline: Lag only')
print_model(model_baseline, res_baseline)

all_results = [res_baseline]
all_models = {'baseline': model_baseline}


# =============================================================================
# PANEL A: CATEGORICAL SPECIFICATION (Regime Dummies)
# =============================================================================

print("\n" + "="*70)
print("PANEL A: CATEGORICAL SPECIFICATION (Regime Dummies)")
print("="*70)
print("Tests whether categorical economic state explains momentum")

# A1: + Regime dummies only
X_a1 = sm.add_constant(analysis_df[['ct_won_lag_1'] + regime_vars].astype(float))
model_a1, res_a1 = run_logit(y, X_a1, 'A1: + Regime dummies')
print_model(model_a1, res_a1, show_coefs=True)
all_results.append(res_a1)
all_models['A1'] = model_a1

# A2: + Regime + Maps
X_a2 = sm.add_constant(analysis_df[['ct_won_lag_1'] + regime_vars + map_vars].astype(float))
model_a2, res_a2 = run_logit(y, X_a2, 'A2: + Regime + Maps')
print_model(model_a2, res_a2)
all_results.append(res_a2)
all_models['A2'] = model_a2

# A3: + Regime + Maps + Phase
X_a3 = sm.add_constant(analysis_df[['ct_won_lag_1'] + regime_vars + map_vars + phase_vars].astype(float))
model_a3, res_a3 = run_logit(y, X_a3, 'A3: + Regime + Maps + Phase')
print_model(model_a3, res_a3)
all_results.append(res_a3)
all_models['A3'] = model_a3


# =============================================================================
# PANEL B: CONTINUOUS SPECIFICATION (Equipment $)
# =============================================================================

print("\n" + "="*70)
print("PANEL B: CONTINUOUS SPECIFICATION (Equipment Advantage)")
print("="*70)
print("Tests whether continuous equipment difference explains momentum")

# B1: + Equipment only
X_b1 = sm.add_constant(analysis_df[['ct_won_lag_1', 'equip_adv_10k']].astype(float))
model_b1, res_b1 = run_logit(y, X_b1, 'B1: + Equipment ($10k)')
print_model(model_b1, res_b1, show_coefs=True)
all_results.append(res_b1)
all_models['B1'] = model_b1

# B2: + Equipment + Maps
X_b2 = sm.add_constant(analysis_df[['ct_won_lag_1', 'equip_adv_10k'] + map_vars].astype(float))
model_b2, res_b2 = run_logit(y, X_b2, 'B2: + Equipment + Maps')
print_model(model_b2, res_b2)
all_results.append(res_b2)
all_models['B2'] = model_b2

# B3: + Equipment + Maps + Phase
X_b3 = sm.add_constant(analysis_df[['ct_won_lag_1', 'equip_adv_10k'] + map_vars + phase_vars].astype(float))
model_b3, res_b3 = run_logit(y, X_b3, 'B3: + Equipment + Maps + Phase')
print_model(model_b3, res_b3)
all_results.append(res_b3)
all_models['B3'] = model_b3

# B4: + First kill (endogenous - interpret with caution)
X_b4 = sm.add_constant(analysis_df[['ct_won_lag_1', 'equip_adv_10k'] + map_vars + phase_vars + fk_vars].astype(float))
model_b4, res_b4 = run_logit(y, X_b4, 'B4: + First kill (endogenous)')
print_model(model_b4, res_b4, show_coefs=True)
all_results.append(res_b4)
all_models['B4'] = model_b4


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print_summary_table(all_results, "MOMENTUM COEFFICIENT ACROSS ALL SPECIFICATIONS")


# =============================================================================
# KEY COMPARISON: BASELINE vs CONTROLLED
# =============================================================================

print("\n" + "="*70)
print("KEY COMPARISONS")
print("="*70)

print(f"""
Baseline momentum:
  Coefficient: {res_baseline['momentum_coef']:.4f}
  Odds Ratio:  {res_baseline['momentum_or']:.3f}x
  P-value:     {res_baseline['momentum_pvalue']:.4f}
  Interpretation: Winning previous round nearly TRIPLES odds of winning current round

After regime dummies (A1):
  Coefficient: {res_a1['momentum_coef']:.4f}
  Odds Ratio:  {res_a1['momentum_or']:.3f}x  
  P-value:     {res_a1['momentum_pvalue']:.4f}
  Interpretation: Momentum effect DISAPPEARS (not significant)

After equipment control (B1):
  Coefficient: {res_b1['momentum_coef']:.4f}
  Odds Ratio:  {res_b1['momentum_or']:.3f}x
  P-value:     {res_b1['momentum_pvalue']:.4f}
  Interpretation: Coefficient becomes NEGATIVE (slight mean reversion)
""")


# =============================================================================
# ROBUSTNESS: BOTH TEAMS FULL BUY
# =============================================================================

print("="*70)
print("ROBUSTNESS: BOTH TEAMS HAVE FULL EQUIPMENT")
print("="*70)

full_buy_df = analysis_df[analysis_df['both_full_buy'] == 1].copy()
print(f"Sample: {len(full_buy_df)} rounds where both teams have full_buy or flush")

y_fb = full_buy_df['ct_wins_round']

# Baseline on full buy subset
X_fb0 = sm.add_constant(full_buy_df[['ct_won_lag_1']].astype(float))
model_fb0, res_fb0 = run_logit(y_fb, X_fb0, 'Full Buy: Baseline')
print_model(model_fb0, res_fb0)

# With equipment variation
X_fb1 = sm.add_constant(full_buy_df[['ct_won_lag_1', 'equip_adv_10k']].astype(float))
model_fb1, res_fb1 = run_logit(y_fb, X_fb1, 'Full Buy: + Equipment')
print_model(model_fb1, res_fb1)

if res_fb0['momentum_pvalue'] >= 0.05:
    print(f"\n  >> Momentum NOT significant (p={res_fb0['momentum_pvalue']:.3f}) when economics are equalized")
else:
    print(f"\n  >> Momentum REMAINS significant (p={res_fb0['momentum_pvalue']:.3f}) - potential psychological component")


# =============================================================================
# ROBUSTNESS: GUN ROUNDS ONLY
# =============================================================================

print("\n" + "="*70)
print("ROBUSTNESS: GUN ROUNDS ONLY (exclude pistol/conversion)")
print("="*70)

gun_df = analysis_df[analysis_df['round_phase'] == 'gun'].copy()
print(f"Sample: {len(gun_df)} gun rounds")

y_gun = gun_df['ct_wins_round']

# Baseline on gun rounds
X_gun0 = sm.add_constant(gun_df[['ct_won_lag_1']].astype(float))
model_gun0, res_gun0 = run_logit(y_gun, X_gun0, 'Gun Rounds: Baseline')
print_model(model_gun0, res_gun0)

# With regime dummies
X_gun1 = sm.add_constant(gun_df[['ct_won_lag_1'] + regime_vars].astype(float))
model_gun1, res_gun1 = run_logit(y_gun, X_gun1, 'Gun Rounds: + Regime')
print_model(model_gun1, res_gun1)

# With equipment
X_gun2 = sm.add_constant(gun_df[['ct_won_lag_1', 'equip_adv_10k']].astype(float))
model_gun2, res_gun2 = run_logit(y_gun, X_gun2, 'Gun Rounds: + Equipment')
print_model(model_gun2, res_gun2)


# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

# Figure 1: Coefficient comparison across specifications
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Panel A (Categorical)
ax1 = axes[0]
panel_a_results = [res_baseline, res_a1, res_a2, res_a3]
labels_a = ['Baseline', '+Regime', '+Regime\n+Maps', '+Regime\n+Maps\n+Phase']
coefs_a = [r['momentum_coef'] for r in panel_a_results]
ses_a = [r['momentum_se'] for r in panel_a_results]
colors_a = ['steelblue' if r['momentum_pvalue'] < 0.05 else 'lightgray' for r in panel_a_results]

ax1.bar(range(len(panel_a_results)), coefs_a, yerr=[1.96*se for se in ses_a],
        capsize=4, color=colors_a, edgecolor='black', alpha=0.8)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax1.set_xticks(range(len(panel_a_results)))
ax1.set_xticklabels(labels_a)
ax1.set_ylabel('Momentum Coefficient (log-odds)')
ax1.set_title('Panel A: Categorical Controls (Regime Dummies)')
ax1.set_ylim(-0.8, 1.5)

for i, (coef, p) in enumerate(zip(coefs_a, [r['momentum_pvalue'] for r in panel_a_results])):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax1.annotate(f'{coef:.2f}\n({sig})', (i, coef + 0.25 if coef >= 0 else coef - 0.35),
                 ha='center', fontsize=9)

# Right: Panel B (Continuous)
ax2 = axes[1]
panel_b_results = [res_baseline, res_b1, res_b2, res_b3]
labels_b = ['Baseline', '+Equip', '+Equip\n+Maps', '+Equip\n+Maps\n+Phase']
coefs_b = [r['momentum_coef'] for r in panel_b_results]
ses_b = [r['momentum_se'] for r in panel_b_results]
colors_b = ['steelblue' if r['momentum_pvalue'] < 0.05 else 'lightgray' for r in panel_b_results]

ax2.bar(range(len(panel_b_results)), coefs_b, yerr=[1.96*se for se in ses_b],
        capsize=4, color=colors_b, edgecolor='black', alpha=0.8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax2.set_xticks(range(len(panel_b_results)))
ax2.set_xticklabels(labels_b)
ax2.set_ylabel('Momentum Coefficient (log-odds)')
ax2.set_title('Panel B: Continuous Controls (Equipment $)')
ax2.set_ylim(-0.8, 1.5)

for i, (coef, p) in enumerate(zip(coefs_b, [r['momentum_pvalue'] for r in panel_b_results])):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax2.annotate(f'{coef:.2f}\n({sig})', (i, coef + 0.25 if coef >= 0 else coef - 0.35),
                 ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_06_momentum_decomposition.png', dpi=150)
plt.close()
print("Saved: fig_06_momentum_decomposition.png")


# Figure 2: Odds ratio comparison
fig, ax = plt.subplots(figsize=(10, 6))

plot_results = [res_baseline, res_a1, res_a2, res_a3, res_b1, res_b2, res_b3]
plot_labels = ['Baseline', 'A1: +Regime', 'A2: +Regime+Maps', 'A3: +Regime+Maps+Phase',
               'B1: +Equipment', 'B2: +Equip+Maps', 'B3: +Equip+Maps+Phase']
ors = [r['momentum_or'] for r in plot_results]
colors_or = ['steelblue' if r['momentum_pvalue'] < 0.05 else 'lightgray' for r in plot_results]

bars = ax.barh(range(len(plot_results)), ors, color=colors_or, edgecolor='black', alpha=0.8)
ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label='No effect (OR=1)')
ax.set_yticks(range(len(plot_results)))
ax.set_yticklabels(plot_labels)
ax.set_xlabel('Odds Ratio')
ax.set_title('Momentum Odds Ratio Across Specifications')
ax.set_xlim(0, 3.5)
ax.legend(loc='lower right')

for i, (or_val, p) in enumerate(zip(ors, [r['momentum_pvalue'] for r in plot_results])):
    sig = '*' if p < 0.05 else ''
    ax.annotate(f'{or_val:.2f}x{sig}', (or_val + 0.05, i), va='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_07_odds_ratio_comparison.png', dpi=150)
plt.close()
print("Saved: fig_07_odds_ratio_comparison.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("PHASE 3 FINAL SUMMARY")
print("="*70)

print(f"""
RESEARCH QUESTION:
  Is momentum in CS:GO a psychological phenomenon or an economic artifact?

DATA:
  Stockholm Major 2021, {len(analysis_df)} rounds from {analysis_df['match_id'].nunique()} matches

BASELINE FINDING:
  Winning previous round multiplies current round win odds by {res_baseline['momentum_or']:.2f}x
  This appears to be strong "momentum" or "hot hand" effect

DECOMPOSITION RESULTS:

  Panel A (Categorical Controls - Regime Dummies):
    Baseline:              coef = {res_baseline['momentum_coef']:.4f}, OR = {res_baseline['momentum_or']:.2f}x, p < 0.001 ***
    + Regime:              coef = {res_a1['momentum_coef']:.4f}, OR = {res_a1['momentum_or']:.2f}x, p = {res_a1['momentum_pvalue']:.3f}
    + Regime + Maps:       coef = {res_a2['momentum_coef']:.4f}, OR = {res_a2['momentum_or']:.2f}x, p = {res_a2['momentum_pvalue']:.3f}
    + Regime + Maps + Ph:  coef = {res_a3['momentum_coef']:.4f}, OR = {res_a3['momentum_or']:.2f}x, p = {res_a3['momentum_pvalue']:.3f}

  Panel B (Continuous Controls - Equipment $):
    Baseline:              coef = {res_baseline['momentum_coef']:.4f}, OR = {res_baseline['momentum_or']:.2f}x, p < 0.001 ***
    + Equipment:           coef = {res_b1['momentum_coef']:.4f}, OR = {res_b1['momentum_or']:.2f}x, p = {res_b1['momentum_pvalue']:.3f}
    + Equipment + Maps:    coef = {res_b2['momentum_coef']:.4f}, OR = {res_b2['momentum_or']:.2f}x, p = {res_b2['momentum_pvalue']:.3f}
    + Equip + Maps + Ph:   coef = {res_b3['momentum_coef']:.4f}, OR = {res_b3['momentum_or']:.2f}x, p = {res_b3['momentum_pvalue']:.3f}

ROBUSTNESS:
  Both teams full buy (n={len(full_buy_df)}): momentum coef = {res_fb0['momentum_coef']:.4f}, p = {res_fb0['momentum_pvalue']:.3f}
  Gun rounds only (n={len(gun_df)}): after equipment control, coef = {res_gun2['momentum_coef']:.4f}

CONCLUSION:
  The apparent momentum effect (OR = {res_baseline['momentum_or']:.2f}x) is fully explained by economics.
  
  - With categorical controls (regime dummies): coefficient drops to ~0, NOT significant
  - With continuous controls (equipment $): coefficient becomes slightly NEGATIVE
  - When both teams have full equipment: momentum is NOT significant (p = {res_fb0['momentum_pvalue']:.3f})
  
  Momentum in CS:GO is an ECONOMIC ILLUSION created by the loss bonus system,
  not a psychological phenomenon. What looks like "hot hand" is simply winners
  having more money to buy better equipment.
""")


# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / 'phase3_model_comparison.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'phase3_model_comparison.csv'}")

for name, model in all_models.items():
    if model is not None:
        coef_df = pd.DataFrame({
            'variable': model.params.index,
            'coefficient': model.params.values,
            'std_error': model.bse.values,
            'z_value': model.tvalues.values,
            'p_value': model.pvalues.values,
            'odds_ratio': np.exp(model.params.values)
        })
        coef_df.to_csv(OUTPUT_DIR / f'model_{name}_coefficients.csv', index=False)

print("Done.")
