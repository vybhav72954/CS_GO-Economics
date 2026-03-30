"""
Phase 8: Clustered Standard Errors

Problem: Rounds within a match are not independent.
- Same teams, same map, same day
- Momentum/tilt effects carry across rounds
- Standard errors may be underestimated

Solution: Cluster standard errors by match_id

This script re-runs all key regressions with clustered SEs and compares.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(r"Z:\Projects\2025\CS-GO Time Series Analysis")
OUTPUT_DIR = BASE_DIR / "analysis_output"

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 70)
print("PHASE 8: CLUSTERED STANDARD ERRORS")
print("=" * 70)

stockholm = pd.read_csv(BASE_DIR / "json_output" / "Stockholm" / "stockholm_rounds.csv")
antwerp = pd.read_csv(BASE_DIR / "json_output" / "Antwerp" / "antwerp_rounds.csv")
rio = pd.read_csv(BASE_DIR / "json_output" / "Rio" / "rio_rounds.csv")

stockholm['tournament'] = 'Stockholm'
antwerp['tournament'] = 'Antwerp'
rio['tournament'] = 'Rio'

df = pd.concat([stockholm, antwerp, rio], ignore_index=True)
print(f"\nFull sample: {len(df)} rounds from {df['match_id'].nunique()} matches")
print(f"Average rounds per match: {len(df) / df['match_id'].nunique():.1f}")

# =============================================================================
# PREPARE FEATURES
# =============================================================================

df['equip_adv_10k'] = df['equip_advantage'] / 10000

# Regime dummies
df['regime_building'] = (df['ct_economic_regime']=='building').astype(int)
df['regime_full_buy'] = (df['ct_economic_regime']=='full_buy').astype(int)
df['regime_flush'] = (df['ct_economic_regime']=='flush').astype(int)

df['t_regime_building'] = (df['t_economic_regime']=='building').astype(int)
df['t_regime_full_buy'] = (df['t_economic_regime']=='full_buy').astype(int)
df['t_regime_flush'] = (df['t_economic_regime']=='flush').astype(int)

# Force-buy indicator
df['ct_could_force'] = df['ct_economic_regime'].isin(['broke', 'building'])
df['ct_is_forcing'] = (df['ct_could_force'] &
                       ~df['ct_buy_type'].str.lower().str.contains('eco', na=False)).astype(int)

# Analysis sample
analysis_df = df.dropna(subset=['ct_won_lag_1']).copy()
print(f"Analysis sample (with lag): {len(analysis_df)} rounds")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_logit_comparison(y, X, groups, model_name):
    """
    Run logit with both regular and clustered SEs.
    Returns comparison of results.
    """
    # Regular logit
    try:
        model_regular = sm.Logit(y, X).fit(disp=0)
        coef = model_regular.params.get('ct_won_lag_1', np.nan)
        se_regular = model_regular.bse.get('ct_won_lag_1', np.nan)
        p_regular = model_regular.pvalues.get('ct_won_lag_1', np.nan)
    except:
        return None

    # Clustered SEs using robust covariance
    try:
        model_clustered = sm.Logit(y, X).fit(disp=0, cov_type='cluster',
                                             cov_kwds={'groups':groups})
        se_clustered = model_clustered.bse.get('ct_won_lag_1', np.nan)
        p_clustered = model_clustered.pvalues.get('ct_won_lag_1', np.nan)
    except Exception as e:
        print(f"  Clustering error: {e}")
        se_clustered = np.nan
        p_clustered = np.nan

    return {
        'model':model_name,
        'coef':coef,
        'se_regular':se_regular,
        'se_clustered':se_clustered,
        'p_regular':p_regular,
        'p_clustered':p_clustered,
        'se_ratio':se_clustered / se_regular if se_regular > 0 else np.nan,
        'n_obs':len(y),
        'n_clusters':groups.nunique(),
    }


def print_comparison(result):
    """Print comparison of regular vs clustered SEs."""
    if result is None:
        print("  Model failed to converge")
        return

    sig_reg = '***' if result['p_regular'] < 0.001 else '**' if result['p_regular'] < 0.01 else '*' if result[
                                                                                                           'p_regular'] < 0.05 else ''
    sig_clu = '***' if result['p_clustered'] < 0.001 else '**' if result['p_clustered'] < 0.01 else '*' if result[
                                                                                                               'p_clustered'] < 0.05 else ''

    print(f"\n{result['model']}")
    print(f"  N = {result['n_obs']}, Clusters = {result['n_clusters']}")
    print(f"  Coefficient: {result['coef']:.4f}")
    print(f"  Regular SE:   {result['se_regular']:.4f}, p = {result['p_regular']:.4f} {sig_reg}")
    print(f"  Clustered SE: {result['se_clustered']:.4f}, p = {result['p_clustered']:.4f} {sig_clu}")
    print(f"  SE Ratio (clustered/regular): {result['se_ratio']:.2f}x")


# =============================================================================
# PART 1: MOMENTUM DECOMPOSITION WITH CLUSTERED SEs
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: MOMENTUM DECOMPOSITION (Clustered by Match)")
print("=" * 70)

y = analysis_df['ct_wins_round']
groups = analysis_df['match_id']

regime_vars = ['regime_building', 'regime_full_buy', 'regime_flush',
               't_regime_building', 't_regime_full_buy', 't_regime_flush']

results_momentum = []

# Model 1: Baseline
X = sm.add_constant(analysis_df[['ct_won_lag_1']].astype(float))
result = run_logit_comparison(y, X, groups, 'Baseline')
print_comparison(result)
results_momentum.append(result)

# Model 2: + Equipment
X = sm.add_constant(analysis_df[['ct_won_lag_1', 'equip_adv_10k']].astype(float))
result = run_logit_comparison(y, X, groups, '+ Equipment')
print_comparison(result)
results_momentum.append(result)

# Model 3: + Regime dummies
X = sm.add_constant(analysis_df[['ct_won_lag_1'] + regime_vars].astype(float))
result = run_logit_comparison(y, X, groups, '+ Regime Dummies')
print_comparison(result)
results_momentum.append(result)

# Model 4: Regime + Equipment
X = sm.add_constant(analysis_df[['ct_won_lag_1', 'equip_adv_10k'] + regime_vars].astype(float))
result = run_logit_comparison(y, X, groups, 'Regime + Equipment')
print_comparison(result)
results_momentum.append(result)

# =============================================================================
# PART 2: BEHAVIORAL ANALYSIS WITH CLUSTERED SEs
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: BEHAVIORAL ANALYSIS - FORCE-BUY (Clustered by Match)")
print("=" * 70)

# Filter to forceable situations (broke/building), exclude pistols
forceable = analysis_df[(analysis_df['ct_could_force']) &
                        (analysis_df['round_phase']!='pistol')].copy()

print(f"\nForceable sample (no pistols): {len(forceable)} rounds")
print(f"Clusters (matches): {forceable['match_id'].nunique()}")

y_force = forceable['ct_is_forcing']
groups_force = forceable['match_id']

# Model: Force = f(consecutive losses)
X = sm.add_constant(forceable[['ct_consecutive_losses']].astype(float))

try:
    # Regular
    model_reg = sm.Logit(y_force, X).fit(disp=0)
    coef = model_reg.params['ct_consecutive_losses']
    se_reg = model_reg.bse['ct_consecutive_losses']
    p_reg = model_reg.pvalues['ct_consecutive_losses']

    # Clustered
    model_clu = sm.Logit(y_force, X).fit(disp=0, cov_type='cluster',
                                         cov_kwds={'groups':groups_force})
    se_clu = model_clu.bse['ct_consecutive_losses']
    p_clu = model_clu.pvalues['ct_consecutive_losses']

    sig_reg = '***' if p_reg < 0.001 else '**' if p_reg < 0.01 else '*' if p_reg < 0.05 else ''
    sig_clu = '***' if p_clu < 0.001 else '**' if p_clu < 0.01 else '*' if p_clu < 0.05 else ''

    print(f"\nForce = f(Consecutive Losses)")
    print(f"  Coefficient: {coef:.4f}")
    print(f"  OR: {np.exp(coef):.3f}")
    print(f"  Regular SE:   {se_reg:.4f}, p = {p_reg:.4f} {sig_reg}")
    print(f"  Clustered SE: {se_clu:.4f}, p = {p_clu:.4f} {sig_clu}")
    print(f"  SE Ratio: {se_clu / se_reg:.2f}x")

    force_result = {
        'model':'Force ~ Consecutive Losses',
        'coef':coef,
        'or':np.exp(coef),
        'se_regular':se_reg,
        'se_clustered':se_clu,
        'p_regular':p_reg,
        'p_clustered':p_clu,
    }
except Exception as e:
    print(f"Error: {e}")
    force_result = None

# =============================================================================
# PART 3: SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: REGULAR VS CLUSTERED STANDARD ERRORS")
print("=" * 70)

print(f"\n{'Model':<25} {'Coef':>8} {'SE(reg)':>8} {'SE(clu)':>8} {'p(reg)':>8} {'p(clu)':>8} {'Ratio':>6}")
print("-" * 75)

for r in results_momentum:
    if r:
        print(f"{r['model']:<25} {r['coef']:>8.3f} {r['se_regular']:>8.3f} {r['se_clustered']:>8.3f} "
              f"{r['p_regular']:>8.3f} {r['p_clustered']:>8.3f} {r['se_ratio']:>6.2f}")

# =============================================================================
# PART 4: DO CONCLUSIONS CHANGE?
# =============================================================================

print("\n" + "=" * 70)
print("DO CONCLUSIONS CHANGE WITH CLUSTERING?")
print("=" * 70)

print("\n--- MOMENTUM DECOMPOSITION ---")
for r in results_momentum:
    if r:
        sig_reg = r['p_regular'] < 0.05
        sig_clu = r['p_clustered'] < 0.05

        if sig_reg==sig_clu:
            status = "UNCHANGED"
        elif sig_reg and not sig_clu:
            status = "NOW NON-SIGNIFICANT"
        else:
            status = "NOW SIGNIFICANT"

        reg_str = "sig" if sig_reg else "ns"
        clu_str = "sig" if sig_clu else "ns"

        print(f"  {r['model']:<25}: {reg_str} → {clu_str} ({status})")

print("\n--- BEHAVIORAL (FORCE-BUY) ---")
if force_result:
    sig_reg = force_result['p_regular'] < 0.05
    sig_clu = force_result['p_clustered'] < 0.05

    if sig_reg==sig_clu:
        status = "UNCHANGED"
    elif sig_reg and not sig_clu:
        status = "NOW NON-SIGNIFICANT"
    else:
        status = "NOW SIGNIFICANT"

    print(f"  Force ~ Losses: {'sig' if sig_reg else 'ns'} → {'sig' if sig_clu else 'ns'} ({status})")

# =============================================================================
# PART 5: INTERPRETATION
# =============================================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

# Check average SE inflation
se_ratios = [r['se_ratio'] for r in results_momentum if r and not np.isnan(r['se_ratio'])]
avg_ratio = np.mean(se_ratios)

print(f"""
STANDARD ERROR INFLATION:
  Average SE ratio (clustered/regular): {avg_ratio:.2f}x

  Interpretation:
  - Ratio > 1 means regular SEs were underestimated
  - Ratio of ~1.5-2x is common with clustered data
  - Ratio < 1.2 suggests minimal clustering effect
""")

# Final check on main results
baseline_clu_p = results_momentum[0]['p_clustered'] if results_momentum[0] else np.nan
equip_clu_p = results_momentum[1]['p_clustered'] if results_momentum[1] else np.nan

print(f"""
MAIN FINDINGS WITH CLUSTERED SEs:

1. BASELINE MOMENTUM:
   p = {baseline_clu_p:.4f} {'(significant)' if baseline_clu_p < 0.05 else '(not significant)'}

2. AFTER EQUIPMENT CONTROL:
   p = {equip_clu_p:.4f} {'(significant)' if equip_clu_p < 0.05 else '(not significant)'}
""")

if baseline_clu_p < 0.05 and equip_clu_p >= 0.05:
    print("CONCLUSION: Main finding ROBUST to clustering.")
    print("            Momentum exists but is explained by economics.")
elif baseline_clu_p >= 0.05:
    print("CONCLUSION: Baseline momentum no longer significant with clustering!")
    print("            This would change our interpretation.")
else:
    print("CONCLUSION: Check results carefully - something unexpected.")

# Behavioral
if force_result:
    print(f"""
3. BEHAVIORAL (FORCE AFTER LOSSES):
   p = {force_result['p_clustered']:.4f} {'(significant)' if force_result['p_clustered'] < 0.05 else '(not significant)'}
""")
    if force_result['p_clustered'] < 0.05:
        print("   Teams still force significantly more after losses (with clustering).")
    else:
        print("   Force-after-loss effect no longer significant with clustering!")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame(results_momentum)
results_df.to_csv(OUTPUT_DIR / 'phase8_clustered_se_results.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'phase8_clustered_se_results.csv'}")

print("\nDone.")
