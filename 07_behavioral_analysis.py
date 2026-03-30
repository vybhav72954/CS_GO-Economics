"""
07_behavioral_analysis_consolidated.py

Behavioral Analysis: Force-Buy Rationality in CS:GO

Research Question: Do teams exhibit irrational "tilt" behavior by force-buying
more frequently after losses, or is this strategically optimal?

Key methodological note: Pistol rounds must be EXCLUDED from this analysis
because everyone has $800 and "Full Eco" is the only option, artificially
inflating eco win rates.

Findings:
- Teams force more after losses (+18.9pp)
- Force-buying has HIGHER expected value than eco-ing
- This is RATIONAL behavior, not tilt
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

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*70)
print("BEHAVIORAL ANALYSIS: FORCE-BUY RATIONALITY")
print("="*70)

stockholm = pd.read_csv(BASE_DIR / "json_output" / "Stockholm" / "stockholm_rounds.csv")
antwerp = pd.read_csv(BASE_DIR / "json_output" / "Antwerp" / "antwerp_rounds.csv")
rio = pd.read_csv(BASE_DIR / "json_output" / "Rio" / "rio_rounds.csv")

stockholm['tournament'] = 'Stockholm'
antwerp['tournament'] = 'Antwerp'
rio['tournament'] = 'Rio'

df = pd.concat([stockholm, antwerp, rio], ignore_index=True)

print(f"\nFull sample: {len(df)} rounds")
print(f"Pistol rounds: {(df['round_phase'] == 'pistol').sum()}")

# =============================================================================
# EXCLUDE PISTOL ROUNDS
# =============================================================================
# Pistol rounds contaminate eco sample: everyone has $800, "Full Eco" is
# the only option, and win rates are ~50% regardless of strategy.

df_clean = df[df['round_phase'] != 'pistol'].copy()
print(f"Gun rounds (excluding pistols): {len(df_clean)}")

# =============================================================================
# DEFINE FORCE VS ECO
# =============================================================================
# Force = any non-eco buy when in broke/building regime
# Eco = any eco buy when in broke/building regime

df_clean['ct_could_force'] = df_clean['ct_economic_regime'].isin(['broke', 'building'])
df_clean['ct_is_forcing'] = (df_clean['ct_could_force'] & 
                              ~df_clean['ct_buy_type'].str.lower().str.contains('eco', na=False))
df_clean['ct_is_ecoing'] = (df_clean['ct_could_force'] & 
                             df_clean['ct_buy_type'].str.lower().str.contains('eco', na=False))

forceable = df_clean[df_clean['ct_could_force']].copy()

print(f"\nForceable situations (gun rounds): {len(forceable)}")
print(f"  Force rounds: {forceable['ct_is_forcing'].sum()}")
print(f"  Eco rounds: {forceable['ct_is_ecoing'].sum()}")

# =============================================================================
# PART 1: BEHAVIORAL PATTERNS
# =============================================================================

print("\n" + "="*70)
print("PART 1: BEHAVIORAL PATTERNS")
print("="*70)

# --- 1A: Force rate by previous round result ---
forceable_lag = forceable.dropna(subset=['ct_won_lag_1'])
after_win = forceable_lag[forceable_lag['ct_won_lag_1'] == 1]
after_loss = forceable_lag[forceable_lag['ct_won_lag_1'] == 0]

force_rate_after_win = after_win['ct_is_forcing'].mean()
force_rate_after_loss = after_loss['ct_is_forcing'].mean()
force_diff = force_rate_after_loss - force_rate_after_win

print(f"\nForce rate after WIN:  {force_rate_after_win*100:.1f}% (n={len(after_win)})")
print(f"Force rate after LOSS: {force_rate_after_loss*100:.1f}% (n={len(after_loss)})")
print(f"Difference: {force_diff*100:+.1f} percentage points")

# --- 1B: Regression ---
y = forceable_lag['ct_is_forcing'].astype(int)
X = sm.add_constant(forceable_lag[['ct_consecutive_losses']].astype(float))
model = sm.Logit(y, X).fit(disp=0)

loss_coef = model.params['ct_consecutive_losses']
loss_or = np.exp(loss_coef)
loss_p = model.pvalues['ct_consecutive_losses']

print(f"\nRegression: Force ~ Consecutive Losses")
print(f"  Coefficient: {loss_coef:.4f}")
print(f"  Odds Ratio:  {loss_or:.3f}")
print(f"  p-value:     {loss_p:.4f}")

# =============================================================================
# PART 2: SINGLE-ROUND WIN RATES
# =============================================================================

print("\n" + "="*70)
print("PART 2: SINGLE-ROUND WIN RATES")
print("="*70)

force_rounds = forceable[forceable['ct_is_forcing']]
eco_rounds = forceable[forceable['ct_is_ecoing']]

force_wr = force_rounds['ct_wins_round'].mean()
eco_wr = eco_rounds['ct_wins_round'].mean()

print(f"\nForce-buy win rate: {force_wr*100:.1f}% (n={len(force_rounds)})")
print(f"Eco win rate:       {eco_wr*100:.1f}% (n={len(eco_rounds)})")
print(f"Difference:         {(force_wr - eco_wr)*100:+.1f}pp")

# --- Win rate by opponent regime ---
print(f"\nWin Rate by Opponent Regime:")
print(f"{'Opponent':<12} {'Force WR':>10} {'Eco WR':>10} {'Diff':>10}")
print("-"*45)

for regime in ['full_buy', 'flush']:
    f_sub = force_rounds[force_rounds['t_economic_regime'] == regime]
    e_sub = eco_rounds[eco_rounds['t_economic_regime'] == regime]
    
    if len(f_sub) >= 20 and len(e_sub) >= 20:
        f_wr = f_sub['ct_wins_round'].mean() * 100
        e_wr = e_sub['ct_wins_round'].mean() * 100
        print(f"{regime:<12} {f_wr:>9.1f}% {e_wr:>9.1f}% {f_wr-e_wr:>+9.1f}pp")

# =============================================================================
# PART 3: MULTI-ROUND EXPECTED VALUE
# =============================================================================

print("\n" + "="*70)
print("PART 3: MULTI-ROUND EXPECTED VALUE")
print("="*70)

# Prepare next-round data
df_sorted = df_clean.sort_values(['match_id', 'round_num']).copy()
df_sorted['ct_regime_next'] = df_sorted.groupby('match_id')['ct_economic_regime'].shift(-1)
df_sorted['ct_wins_next'] = df_sorted.groupby('match_id')['ct_wins_round'].shift(-1)

analysis_df = df_sorted[df_sorted['ct_could_force'] & df_sorted['ct_regime_next'].notna()].copy()
analysis_df['ct_is_forcing'] = (~analysis_df['ct_buy_type'].str.lower().str.contains('eco', na=False))
analysis_df['ct_is_ecoing'] = ~analysis_df['ct_is_forcing']

print(f"\nAnalysis sample (with next-round data): {len(analysis_df)}")

def calculate_ev(subset):
    """
    Calculate 2-round expected value.
    
    EV = P(win) * (1 + P(win_next|win)) + P(lose) * P(win_next|lose)
    """
    if len(subset) < 20:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    p_win = subset['ct_wins_round'].mean()
    won = subset[subset['ct_wins_round'] == 1]
    lost = subset[subset['ct_wins_round'] == 0]
    
    next_wr_if_win = won['ct_wins_next'].mean() if len(won) > 0 else 0
    next_wr_if_lose = lost['ct_wins_next'].mean() if len(lost) > 0 else 0
    
    ev = p_win * (1 + next_wr_if_win) + (1 - p_win) * next_wr_if_lose
    return ev, p_win, next_wr_if_win, next_wr_if_lose, len(subset)

force_analysis = analysis_df[analysis_df['ct_is_forcing']]
eco_analysis = analysis_df[analysis_df['ct_is_ecoing']]

f_ev, f_wr, f_next_win, f_next_lose, f_n = calculate_ev(force_analysis)
e_ev, e_wr, e_next_win, e_next_lose, e_n = calculate_ev(eco_analysis)

print(f"\n{'Metric':<30} {'Force':>12} {'Eco':>12}")
print("-"*56)
print(f"{'N':<30} {f_n:>12} {e_n:>12}")
print(f"{'This round win rate':<30} {f_wr*100:>11.1f}% {e_wr*100:>11.1f}%")
print(f"{'Next round WR if WIN':<30} {f_next_win*100:>11.1f}% {e_next_win*100:>11.1f}%")
print(f"{'Next round WR if LOSE':<30} {f_next_lose*100:>11.1f}% {e_next_lose*100:>11.1f}%")
print(f"{'2-Round Expected Value':<30} {f_ev:>12.3f} {e_ev:>12.3f}")

ev_diff = f_ev - e_ev
print(f"\nDifference: {ev_diff:+.3f} rounds in favor of {'FORCE' if ev_diff > 0 else 'ECO'}")

# =============================================================================
# PART 4: BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================

print("\n" + "="*70)
print("PART 4: BOOTSTRAP CONFIDENCE INTERVAL")
print("="*70)

def bootstrap_ev_diff(force_df, eco_df, n_bootstrap=2000):
    """Bootstrap the EV difference between force and eco."""
    np.random.seed(42)
    diffs = []
    
    for _ in range(n_bootstrap):
        f_sample = force_df.sample(n=len(force_df), replace=True)
        e_sample = eco_df.sample(n=len(eco_df), replace=True)
        
        f_ev_b, _, _, _, _ = calculate_ev(f_sample)
        e_ev_b, _, _, _, _ = calculate_ev(e_sample)
        
        if not np.isnan(f_ev_b) and not np.isnan(e_ev_b):
            diffs.append(f_ev_b - e_ev_b)
    
    return np.array(diffs)

print("\nRunning 2000 bootstrap iterations...")
bootstrap_diffs = bootstrap_ev_diff(force_analysis, eco_analysis, n_bootstrap=2000)

ci_95_lower = np.percentile(bootstrap_diffs, 2.5)
ci_95_upper = np.percentile(bootstrap_diffs, 97.5)
ci_90_lower = np.percentile(bootstrap_diffs, 5)
ci_90_upper = np.percentile(bootstrap_diffs, 95)
pct_force_better = (bootstrap_diffs > 0).mean()

print(f"\nEV Difference (Force - Eco):")
print(f"  Point estimate: {ev_diff:+.3f}")
print(f"  Bootstrap mean: {np.mean(bootstrap_diffs):+.3f}")
print(f"  Bootstrap std:  {np.std(bootstrap_diffs):.3f}")
print(f"  90% CI:         [{ci_90_lower:+.3f}, {ci_90_upper:+.3f}]")
print(f"  95% CI:         [{ci_95_lower:+.3f}, {ci_95_upper:+.3f}]")
print(f"  P(Force > Eco): {pct_force_better*100:.1f}%")

# =============================================================================
# PART 5: ROBUSTNESS CHECKS
# =============================================================================

print("\n" + "="*70)
print("PART 5: ROBUSTNESS CHECKS")
print("="*70)

# --- By Tournament ---
print(f"\nBy Tournament:")
print(f"{'Tournament':<12} {'Force EV':>10} {'Eco EV':>10} {'Diff':>10}")
print("-"*45)

for tournament in ['Stockholm', 'Antwerp', 'Rio']:
    t_df = analysis_df[analysis_df['tournament'] == tournament]
    f_ev_t, _, _, _, _ = calculate_ev(t_df[t_df['ct_is_forcing']])
    e_ev_t, _, _, _, _ = calculate_ev(t_df[t_df['ct_is_ecoing']])
    
    if not np.isnan(f_ev_t) and not np.isnan(e_ev_t):
        print(f"{tournament:<12} {f_ev_t:>10.3f} {e_ev_t:>10.3f} {f_ev_t-e_ev_t:>+10.3f}")

# --- By Consecutive Losses ---
print(f"\nBy Consecutive Losses:")
print(f"{'Losses':<10} {'Force EV':>10} {'Eco EV':>10} {'Diff':>10} {'Better':>10}")
print("-"*55)

for n_losses in [1, 2, 3, 4]:
    subset = analysis_df[analysis_df['ct_consecutive_losses'] == n_losses]
    f_ev_l, _, _, _, _ = calculate_ev(subset[subset['ct_is_forcing']])
    e_ev_l, _, _, _, _ = calculate_ev(subset[subset['ct_is_ecoing']])
    
    if not np.isnan(f_ev_l) and not np.isnan(e_ev_l):
        diff_l = f_ev_l - e_ev_l
        better = "FORCE" if diff_l > 0 else "ECO"
        print(f"{n_losses:<10} {f_ev_l:>10.3f} {e_ev_l:>10.3f} {diff_l:>+10.3f} {better:>10}")

# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================

print("\n" + "="*70)
print("CREATING FIGURE")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Force rate after win vs loss
ax1 = axes[0, 0]
categories = ['After Win', 'After Loss']
rates = [force_rate_after_win * 100, force_rate_after_loss * 100]
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(categories, rates, color=colors, edgecolor='black', alpha=0.8)
ax1.set_ylabel('Force-Buy Rate (%)')
ax1.set_title('Force-Buy Rate by Previous Round Result')
ax1.set_ylim(0, max(rates) * 1.2)
for i, v in enumerate(rates):
    ax1.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')

# Plot 2: Win rate comparison
ax2 = axes[0, 1]
categories = ['Force-Buy', 'Eco']
win_rates = [force_wr * 100, eco_wr * 100]
colors = ['#3498db', '#95a5a6']
bars = ax2.bar(categories, win_rates, color=colors, edgecolor='black', alpha=0.8)
ax2.set_ylabel('Win Rate (%)')
ax2.set_title('Single-Round Win Rate (Gun Rounds)')
ax2.set_ylim(0, max(win_rates) * 1.3)
for i, v in enumerate(win_rates):
    ax2.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')

# Plot 3: Expected value comparison
ax3 = axes[1, 0]
categories = ['Force-Buy', 'Eco']
evs = [f_ev, e_ev]
colors = ['#3498db', '#95a5a6']
bars = ax3.bar(categories, evs, color=colors, edgecolor='black', alpha=0.8)
ax3.set_ylabel('Expected Rounds Won (2-Round Horizon)')
ax3.set_title('Multi-Round Expected Value')
ax3.set_ylim(0, max(evs) * 1.2)
for i, v in enumerate(evs):
    ax3.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')

# Plot 4: Bootstrap distribution
ax4 = axes[1, 1]
ax4.hist(bootstrap_diffs, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
ax4.axvline(x=ev_diff, color='green', linestyle='-', linewidth=2, label=f'Point estimate: {ev_diff:+.3f}')
ax4.axvline(x=ci_95_lower, color='orange', linestyle=':', linewidth=2)
ax4.axvline(x=ci_95_upper, color='orange', linestyle=':', linewidth=2, label=f'95% CI')
ax4.set_xlabel('EV Difference (Force - Eco)')
ax4.set_ylabel('Frequency')
ax4.set_title('Bootstrap Distribution of EV Difference')
ax4.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_12_behavioral_ev_analysis.png', dpi=150)
plt.close()
print("Saved: fig_12_behavioral_ev_analysis.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"""
BEHAVIORAL PATTERN:
  Force rate after WIN:   {force_rate_after_win*100:.1f}%
  Force rate after LOSS:  {force_rate_after_loss*100:.1f}%
  Difference:             {force_diff*100:+.1f}pp
  
  Regression (Force ~ Losses):
    Coefficient: {loss_coef:.4f}, p = {loss_p:.4f}

STRATEGIC ANALYSIS (excluding pistol rounds):
  Sample: {f_n} force rounds, {e_n} eco rounds
  
  Single-round win rate:
    Force: {force_wr*100:.1f}%
    Eco:   {eco_wr*100:.1f}%
    
  2-Round Expected Value:
    Force EV: {f_ev:.3f} rounds
    Eco EV:   {e_ev:.3f} rounds
    Diff:     {ev_diff:+.3f} rounds
    
  Bootstrap 95% CI: [{ci_95_lower:+.3f}, {ci_95_upper:+.3f}]
  P(Force > Eco):   {pct_force_better*100:.1f}%
""")

if ci_95_lower > 0:
    verdict = "FORCE > ECO is STATISTICALLY SIGNIFICANT (p < 0.05)"
    interpretation = "Teams that force more after losses are making the RATIONAL choice."
    tilt_status = "NOT TILT - strategically optimal behavior"
elif ci_90_lower > 0:
    verdict = "FORCE > ECO is marginally significant (p < 0.10)"
    interpretation = "Evidence suggests forcing is better, but CI includes zero at 95%."
    tilt_status = "LIKELY NOT TILT - probably rational"
else:
    verdict = "Cannot determine if Force > Eco"
    interpretation = "Strategies may be approximately equivalent."
    tilt_status = "INCONCLUSIVE"

print(f"VERDICT: {verdict}")
print(f"INTERPRETATION: {interpretation}")
print(f"TILT STATUS: {tilt_status}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results = {
    # Behavioral pattern
    'force_rate_after_win': force_rate_after_win,
    'force_rate_after_loss': force_rate_after_loss,
    'force_rate_diff': force_diff,
    'loss_streak_coef': loss_coef,
    'loss_streak_or': loss_or,
    'loss_streak_p': loss_p,
    # Strategic analysis
    'force_n': f_n,
    'eco_n': e_n,
    'force_wr': f_wr,
    'eco_wr': e_wr,
    'force_ev': f_ev,
    'eco_ev': e_ev,
    'ev_diff': ev_diff,
    # Bootstrap
    'ci_95_lower': ci_95_lower,
    'ci_95_upper': ci_95_upper,
    'ci_90_lower': ci_90_lower,
    'ci_90_upper': ci_90_upper,
    'pct_force_better': pct_force_better,
    # Verdict
    'verdict': verdict,
    'tilt_status': tilt_status,
}

pd.DataFrame([results]).to_csv(OUTPUT_DIR / 'phase7_behavioral_consolidated.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'phase7_behavioral_consolidated.csv'}")

print("\nDone.")
