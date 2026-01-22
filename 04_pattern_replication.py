"""
Phase 4: Antwerp Replication + Diagnostics
Dataset: Antwerp 2022 (Confirmation Set)

Goals:
  1. Replicate Stockholm findings on independent tournament
  2. Check for multicollinearity concerns
  3. Verify mediation vs confounding
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

print("=" * 70)
print("PHASE 4: ANTWERP REPLICATION")
print("=" * 70)

# Load both datasets
stockholm_df = pd.read_csv(BASE_DIR / "json_output" / "Stockholm" / "stockholm_rounds.csv")
antwerp_df = pd.read_csv(BASE_DIR / "json_output" / "Antwerp" / "antwerp_rounds.csv")

print(f"\nStockholm (training):    {len(stockholm_df)} rounds, {stockholm_df['match_id'].nunique()} matches")
print(f"Antwerp (confirmation):  {len(antwerp_df)} rounds, {antwerp_df['match_id'].nunique()} matches")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_features(df):
    df = df.copy()

    # Regime dummies (reference: broke)
    df['regime_building'] = (df['ct_economic_regime']=='building').astype(int)
    df['regime_full_buy'] = (df['ct_economic_regime']=='full_buy').astype(int)
    df['regime_flush'] = (df['ct_economic_regime']=='flush').astype(int)

    df['t_regime_building'] = (df['t_economic_regime']=='building').astype(int)
    df['t_regime_full_buy'] = (df['t_economic_regime']=='full_buy').astype(int)
    df['t_regime_flush'] = (df['t_economic_regime']=='flush').astype(int)

    # Continuous equipment
    df['equip_adv_10k'] = df['equip_advantage'] / 10000

    # Map dummies
    map_dummies = pd.get_dummies(df['map_name'], prefix='map', drop_first=True)
    for col in map_dummies.columns:
        df[col] = map_dummies[col].astype(int)

    # Phase dummies
    df['phase_pistol'] = (df['round_phase']=='pistol').astype(int)
    df['phase_conversion'] = (df['round_phase']=='conversion').astype(int)
    df['phase_overtime'] = (df['round_phase']=='overtime').astype(int)

    # Both full buy
    df['both_full_buy'] = ((df['ct_economic_regime'].isin(['full_buy', 'flush'])) &
                           (df['t_economic_regime'].isin(['full_buy', 'flush']))).astype(int)

    return df


stockholm_analysis = prepare_features(stockholm_df.dropna(subset=['ct_won_lag_1']))
antwerp_analysis = prepare_features(antwerp_df.dropna(subset=['ct_won_lag_1']))

print(f"\nAnalysis samples:")
print(f"  Stockholm: {len(stockholm_analysis)} rounds")
print(f"  Antwerp:   {len(antwerp_analysis)} rounds")

# =============================================================================
# DIAGNOSTIC 1: CORRELATION BETWEEN MOMENTUM AND ECONOMICS
# =============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 1: CORRELATION MATRIX")
print("=" * 70)
print("Checking if momentum (ct_won_lag_1) is too correlated with regime")


def correlation_diagnostic(df, name):
    print(f"\n{name}:")

    vars_to_check = ['ct_won_lag_1', 'regime_building', 'regime_full_buy',
                     'regime_flush', 'equip_adv_10k']

    corr_matrix = df[vars_to_check].corr()

    print(f"\n  Correlations with ct_won_lag_1:")
    for var in vars_to_check[1:]:
        r = corr_matrix.loc['ct_won_lag_1', var]
        print(f"    {var:<20}: r = {r:>6.3f}")

    return corr_matrix


corr_stockholm = correlation_diagnostic(stockholm_analysis, "Stockholm")
corr_antwerp = correlation_diagnostic(antwerp_analysis, "Antwerp")

# =============================================================================
# DIAGNOSTIC 2: VARIANCE INFLATION FACTOR
# =============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 2: VARIANCE INFLATION FACTOR (VIF)")
print("=" * 70)
print("VIF > 5 suggests problematic multicollinearity")
print("VIF > 10 is severe")


def vif_diagnostic(df, name):
    print(f"\n{name}:")

    regime_vars = ['regime_building', 'regime_full_buy', 'regime_flush',
                   't_regime_building', 't_regime_full_buy', 't_regime_flush']

    X = df[['ct_won_lag_1'] + regime_vars].astype(float)
    X = sm.add_constant(X)

    vif_data = []
    for i, col in enumerate(X.columns):
        if col!='const':
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({'variable':col, 'VIF':vif})
            print(f"    {col:<25}: VIF = {vif:.2f}")

    return pd.DataFrame(vif_data)


vif_stockholm = vif_diagnostic(stockholm_analysis, "Stockholm")
vif_antwerp = vif_diagnostic(antwerp_analysis, "Antwerp")

# =============================================================================
# DIAGNOSTIC 3: CROSS-TABULATION
# =============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 3: MOMENTUM × REGIME CROSS-TABULATION")
print("=" * 70)
print("Checking cell sizes and conditional distributions")


def crosstab_diagnostic(df, name):
    print(f"\n{name}:")

    ct = pd.crosstab(df['ct_won_lag_1'], df['ct_economic_regime'], margins=True)
    ct = ct[['broke', 'building', 'full_buy', 'flush', 'All']]
    print(f"\n  Counts:")
    print(ct.to_string().replace('\n', '\n  '))

    # Conditional probabilities
    ct_norm = pd.crosstab(df['ct_won_lag_1'], df['ct_economic_regime'], normalize='index')
    ct_norm = ct_norm[['broke', 'building', 'full_buy', 'flush']]
    print(f"\n  P(Regime | Momentum):")
    print((ct_norm * 100).round(1).to_string().replace('\n', '\n  '))

    return ct


ct_stockholm = crosstab_diagnostic(stockholm_analysis, "Stockholm")
ct_antwerp = crosstab_diagnostic(antwerp_analysis, "Antwerp")


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


def run_all_models(df, dataset_name):
    """Run the full model sequence on a dataset."""

    print(f"\n{'=' * 70}")
    print(f"{dataset_name.upper()} MODELS")
    print(f"{'=' * 70}")

    y = df['ct_wins_round']
    regime_vars = ['regime_building', 'regime_full_buy', 'regime_flush',
                   't_regime_building', 't_regime_full_buy', 't_regime_flush']
    map_vars = [c for c in df.columns if c.startswith('map_de_')]
    phase_vars = ['phase_pistol', 'phase_conversion', 'phase_overtime']

    results = []

    # Baseline
    X = sm.add_constant(df[['ct_won_lag_1']].astype(float))
    _, res = run_logit(y, X, 'Baseline')
    results.append(res)

    # A1: + Regime
    X = sm.add_constant(df[['ct_won_lag_1'] + regime_vars].astype(float))
    _, res = run_logit(y, X, 'A1: +Regime')
    results.append(res)

    # A2: + Regime + Maps + Phase
    X = sm.add_constant(df[['ct_won_lag_1'] + regime_vars + map_vars + phase_vars].astype(float))
    _, res = run_logit(y, X, 'A2: +Regime+Maps+Phase')
    results.append(res)

    # B1: + Equipment
    X = sm.add_constant(df[['ct_won_lag_1', 'equip_adv_10k']].astype(float))
    _, res = run_logit(y, X, 'B1: +Equipment')
    results.append(res)

    # B2: + Equipment + Maps + Phase
    X = sm.add_constant(df[['ct_won_lag_1', 'equip_adv_10k'] + map_vars + phase_vars].astype(float))
    _, res = run_logit(y, X, 'B2: +Equip+Maps+Phase')
    results.append(res)

    # Print results
    print(f"\n{'Model':<25} {'Coef':>8} {'SE':>8} {'OR':>8} {'p':>10}")
    print("-" * 65)
    for r in results:
        if r:
            sig = '***' if r['momentum_pvalue'] < 0.001 else '**' if r['momentum_pvalue'] < 0.01 else '*' if r[
                                                                                                                 'momentum_pvalue'] < 0.05 else ''
            print(f"{r['model']:<25} {r['momentum_coef']:>8.4f} {r['momentum_se']:>8.4f} "
                  f"{r['momentum_or']:>8.3f} {r['momentum_pvalue']:>9.4f}{sig}")

    return results


# =============================================================================
# RUN MODELS ON BOTH DATASETS
# =============================================================================

stockholm_results = run_all_models(stockholm_analysis, "Stockholm (Training)")
antwerp_results = run_all_models(antwerp_analysis, "Antwerp (Confirmation)")

# =============================================================================
# ROBUSTNESS: BOTH FULL BUY
# =============================================================================

print("\n" + "=" * 70)
print("ROBUSTNESS: BOTH TEAMS FULL BUY")
print("=" * 70)

for name, df in [("Stockholm", stockholm_analysis), ("Antwerp", antwerp_analysis)]:
    fb_df = df[df['both_full_buy']==1]
    print(f"\n{name}: {len(fb_df)} rounds with both teams full buy")

    y = fb_df['ct_wins_round']
    X = sm.add_constant(fb_df[['ct_won_lag_1']].astype(float))
    _, res = run_logit(y, X, f'{name} Full Buy')

    if res:
        sig = '***' if res['momentum_pvalue'] < 0.001 else '**' if res['momentum_pvalue'] < 0.01 else '*' if res[
                                                                                                                 'momentum_pvalue'] < 0.05 else 'ns'
        print(
            f"  Momentum: coef = {res['momentum_coef']:.4f}, OR = {res['momentum_or']:.3f}, p = {res['momentum_pvalue']:.4f} ({sig})")

# =============================================================================
# SIDE-BY-SIDE COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("CROSS-TOURNAMENT COMPARISON")
print("=" * 70)

print(f"\n{'Model':<25} {'Stockholm':>20} {'Antwerp':>20}")
print(f"{'':25} {'Coef (p)':>20} {'Coef (p)':>20}")
print("-" * 65)

for s_res, a_res in zip(stockholm_results, antwerp_results):
    if s_res and a_res:
        s_str = f"{s_res['momentum_coef']:.3f} (p={s_res['momentum_pvalue']:.3f})"
        a_str = f"{a_res['momentum_coef']:.3f} (p={a_res['momentum_pvalue']:.3f})"
        print(f"{s_res['model']:<25} {s_str:>20} {a_str:>20}")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "=" * 70)
print("CREATING FIGURES")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stockholm
ax1 = axes[0]
labels = ['Baseline', '+Regime', '+Regime\n+Maps+Phase', '+Equip', '+Equip\n+Maps+Phase']
coefs_s = [r['momentum_coef'] for r in stockholm_results]
ses_s = [r['momentum_se'] for r in stockholm_results]
colors_s = ['steelblue' if r['momentum_pvalue'] < 0.05 else 'lightgray' for r in stockholm_results]

ax1.bar(range(len(stockholm_results)), coefs_s, yerr=[1.96 * se for se in ses_s],
        capsize=4, color=colors_s, edgecolor='black', alpha=0.8)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax1.set_xticks(range(len(stockholm_results)))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel('Momentum Coefficient')
ax1.set_title('Stockholm 2021 (Training)')
ax1.set_ylim(-0.8, 1.5)

for i, (coef, p) in enumerate(zip(coefs_s, [r['momentum_pvalue'] for r in stockholm_results])):
    sig = '***' if p < 0.001 else '*' if p < 0.05 else 'ns'
    ax1.annotate(f'{coef:.2f}\n({sig})', (i, coef + 0.25 if coef >= 0 else coef - 0.3),
                 ha='center', fontsize=9)

# Antwerp
ax2 = axes[1]
coefs_a = [r['momentum_coef'] for r in antwerp_results]
ses_a = [r['momentum_se'] for r in antwerp_results]
colors_a = ['steelblue' if r['momentum_pvalue'] < 0.05 else 'lightgray' for r in antwerp_results]

ax2.bar(range(len(antwerp_results)), coefs_a, yerr=[1.96 * se for se in ses_a],
        capsize=4, color=colors_a, edgecolor='black', alpha=0.8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax2.set_xticks(range(len(antwerp_results)))
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('Momentum Coefficient')
ax2.set_title('Antwerp 2022 (Confirmation)')
ax2.set_ylim(-0.8, 1.5)

for i, (coef, p) in enumerate(zip(coefs_a, [r['momentum_pvalue'] for r in antwerp_results])):
    sig = '***' if p < 0.001 else '*' if p < 0.05 else 'ns'
    ax2.annotate(f'{coef:.2f}\n({sig})', (i, coef + 0.25 if coef >= 0 else coef - 0.3),
                 ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_08_antwerp_replication.png', dpi=150)
plt.close()
print("Saved: fig_08_antwerp_replication.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 4 SUMMARY")
print("=" * 70)

# Extract key values
s_baseline = stockholm_results[0]
s_regime = stockholm_results[1]
a_baseline = antwerp_results[0]
a_regime = antwerp_results[1]

print(f"""
MULTICOLLINEARITY DIAGNOSTICS:
  Correlation (ct_won_lag_1 vs regime_flush):
    Stockholm: {corr_stockholm.loc['ct_won_lag_1', 'regime_flush']:.3f}
    Antwerp:   {corr_antwerp.loc['ct_won_lag_1', 'regime_flush']:.3f}

  VIF scores are printed above. Values < 5 indicate acceptable collinearity.

REPLICATION RESULTS:

  STOCKHOLM (Training):
    Baseline:    coef = {s_baseline['momentum_coef']:.4f}, OR = {s_baseline['momentum_or']:.2f}x, p = {s_baseline['momentum_pvalue']:.4f}
    +Regime:     coef = {s_regime['momentum_coef']:.4f}, OR = {s_regime['momentum_or']:.2f}x, p = {s_regime['momentum_pvalue']:.4f}

  ANTWERP (Confirmation):
    Baseline:    coef = {a_baseline['momentum_coef']:.4f}, OR = {a_baseline['momentum_or']:.2f}x, p = {a_baseline['momentum_pvalue']:.4f}
    +Regime:     coef = {a_regime['momentum_coef']:.4f}, OR = {a_regime['momentum_or']:.2f}x, p = {a_regime['momentum_pvalue']:.4f}

INTERPRETATION:
  Compare the patterns above. If Antwerp shows similar baseline effect (~2.5-3x)
  that disappears with economic controls, the finding replicates.

  If patterns differ substantially, investigate why (meta differences, 
  team compositions, patch changes between tournaments).
""")

# =============================================================================
# SAVE
# =============================================================================

# Combine results
stockholm_df_results = pd.DataFrame(stockholm_results)
stockholm_df_results['tournament'] = 'Stockholm'
antwerp_df_results = pd.DataFrame(antwerp_results)
antwerp_df_results['tournament'] = 'Antwerp'

combined = pd.concat([stockholm_df_results, antwerp_df_results], ignore_index=True)
combined.to_csv(OUTPUT_DIR / 'phase4_replication_comparison.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'phase4_replication_comparison.csv'}")

print("\nDone.")
