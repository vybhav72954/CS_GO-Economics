"""
Phase 7: Behavioral Analysis
Question: Do teams make irrational economic decisions after losses (tilt)?
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
print("PHASE 7: BEHAVIORAL ANALYSIS")
print("="*70)
print("\nQuestion: Do teams exhibit irrational 'tilt' behavior after losses?")

stockholm = pd.read_csv(BASE_DIR / "json_output" / "Stockholm" / "stockholm_rounds.csv")
antwerp = pd.read_csv(BASE_DIR / "json_output" / "Antwerp" / "antwerp_rounds.csv")
rio = pd.read_csv(BASE_DIR / "json_output" / "Rio" / "rio_rounds.csv")

stockholm['tournament'] = 'Stockholm'
antwerp['tournament'] = 'Antwerp'
rio['tournament'] = 'Rio'

df = pd.concat([stockholm, antwerp, rio], ignore_index=True)
print(f"\nFull sample: {len(df)} rounds from {df['match_id'].nunique()} matches")

# =============================================================================
# DEFINE FORCE-BUY
# =============================================================================

df['ct_could_force'] = df['ct_economic_regime'].isin(['broke', 'building'])
df['ct_is_forcing'] = (df['ct_could_force'] &
                       ~df['ct_buy_type'].str.lower().str.contains('eco', na=False))

# =============================================================================
# ANALYSIS 1: FORCE-BUY RATE BY LOSS STREAK
# =============================================================================

print("\n" + "="*70)
print("ANALYSIS 1: FORCE-BUY RATE BY CONSECUTIVE LOSSES")
print("="*70)

ct_forceable = df[df['ct_could_force']].copy()
print(f"\nCT rounds where force-buy is possible: {len(ct_forceable)}")

force_by_losses = ct_forceable.groupby('ct_consecutive_losses').agg(
    force_rate=('ct_is_forcing', 'mean'),
    n_rounds=('ct_is_forcing', 'count'),
    win_rate=('ct_wins_round', 'mean')
).reset_index()

print(f"\n{'Consecutive Losses':<20} {'Force Rate':>12} {'Win Rate':>12} {'N':>8}")
print("-"*55)

for _, row in force_by_losses.iterrows():
    if row['n_rounds'] >= 30:
        print(f"{int(row['ct_consecutive_losses']):<20} {row['force_rate']*100:>11.1f}% {row['win_rate']*100:>11.1f}% {int(row['n_rounds']):>8}")

# =============================================================================
# ANALYSIS 2: FORCE-BUY RATE BY PREVIOUS ROUND RESULT
# =============================================================================

print("\n" + "="*70)
print("ANALYSIS 2: FORCE-BUY RATE BY PREVIOUS ROUND RESULT")
print("="*70)

ct_forceable_lag = ct_forceable.dropna(subset=['ct_won_lag_1'])
after_win = ct_forceable_lag[ct_forceable_lag['ct_won_lag_1'] == 1]
after_loss = ct_forceable_lag[ct_forceable_lag['ct_won_lag_1'] == 0]

print(f"\nAfter WIN (when in broke/building):")
print(f"  Force rate: {after_win['ct_is_forcing'].mean()*100:.1f}%")
print(f"  N = {len(after_win)}")

print(f"\nAfter LOSS (when in broke/building):")
print(f"  Force rate: {after_loss['ct_is_forcing'].mean()*100:.1f}%")
print(f"  N = {len(after_loss)}")

force_diff = after_loss['ct_is_forcing'].mean() - after_win['ct_is_forcing'].mean()
print(f"\nDifference: {force_diff*100:+.1f} percentage points")

if force_diff > 0.05:
    print("  >> Teams force MORE after losses - potential tilt behavior")
elif force_diff < -0.05:
    print("  >> Teams force LESS after losses - disciplined/rational")
else:
    print("  >> No significant difference in force rate")

# =============================================================================
# ANALYSIS 3: FORCE-BUY SUCCESS RATE
# =============================================================================

print("\n" + "="*70)
print("ANALYSIS 3: FORCE-BUY WIN RATE BY CONTEXT")
print("="*70)

force_rounds = ct_forceable[ct_forceable['ct_is_forcing']].copy()
print(f"\nTotal force-buy rounds: {len(force_rounds)}")

force_success = force_rounds.groupby('ct_consecutive_losses').agg(
    win_rate=('ct_wins_round', 'mean'),
    n=('ct_wins_round', 'count')
).reset_index()

print(f"\n{'Consecutive Losses':<20} {'Force Win Rate':>15} {'N':>8}")
print("-"*45)

for _, row in force_success.iterrows():
    if row['n'] >= 20:
        print(f"{int(row['ct_consecutive_losses']):<20} {row['win_rate']*100:>14.1f}% {int(row['n']):>8}")

# =============================================================================
# ANALYSIS 4: REGRESSION - DOES LOSS STREAK PREDICT FORCING?
# =============================================================================

print("\n" + "="*70)
print("ANALYSIS 4: LOGISTIC REGRESSION - LOSS STREAK → FORCING")
print("="*70)

reg_df = ct_forceable.dropna(subset=['ct_won_lag_1']).copy()
reg_df['ct_is_forcing_int'] = reg_df['ct_is_forcing'].astype(int)

y = reg_df['ct_is_forcing_int']
X1 = sm.add_constant(reg_df[['ct_consecutive_losses']])
model1 = sm.Logit(y, X1).fit(disp=0)

print("\nModel: Force = f(consecutive losses)")
print(f"  Loss streak coef: {model1.params['ct_consecutive_losses']:.4f}")
print(f"  OR: {np.exp(model1.params['ct_consecutive_losses']):.3f}")
print(f"  p-value: {model1.pvalues['ct_consecutive_losses']:.4f}")

if model1.pvalues['ct_consecutive_losses'] < 0.05 and model1.params['ct_consecutive_losses'] > 0:
    print("  >> Significant: Teams force MORE as loss streak grows")
elif model1.pvalues['ct_consecutive_losses'] < 0.05 and model1.params['ct_consecutive_losses'] < 0:
    print("  >> Significant: Teams force LESS as loss streak grows")
else:
    print("  >> Not significant")

# =============================================================================
# ANALYSIS 5: FORCE-BUY TIMING BY GAME SITUATION
# =============================================================================

print("\n" + "="*70)
print("ANALYSIS 5: FORCE-BUY TIMING BY GAME SITUATION")
print("="*70)

ct_forceable['score_close'] = abs(ct_forceable['ct_score'] - ct_forceable['t_score']) <= 3
ct_forceable['losing_badly'] = (ct_forceable['t_score'] - ct_forceable['ct_score']) >= 5
ct_forceable['match_point'] = (ct_forceable['t_score'] >= 15) | (ct_forceable['ct_score'] >= 15)

print("\nForce rate by game situation:")
print(f"\n{'Situation':<25} {'Force Rate':>12} {'N':>10}")
print("-"*50)

for name, col in [('Score close (±3)', 'score_close'),
                  ('Losing badly (5+ down)', 'losing_badly'),
                  ('Match point', 'match_point')]:
    subset = ct_forceable[ct_forceable[col]]
    if len(subset) > 50:
        rate = subset['ct_is_forcing'].mean()
        print(f"{name:<25} {rate*100:>11.1f}% {len(subset):>10}")

# =============================================================================
# ANALYSIS 6: TILT DETECTION
# =============================================================================

print("\n" + "="*70)
print("ANALYSIS 6: TILT DETECTION")
print("="*70)
print("\nDefine 'tilt' as: forcing when you probably shouldn't")

# Bad force situation: early in loss streak AND not desperate
ct_forceable['bad_force_situation'] = (
    (ct_forceable['ct_consecutive_losses'] <= 1) &
    (~ct_forceable['score_close']) &
    (~ct_forceable['match_point'])
)

bad_situation_rounds = ct_forceable[ct_forceable['bad_force_situation']]
print(f"\nRounds where forcing is suboptimal: {len(bad_situation_rounds)}")

if len(bad_situation_rounds) > 100:
    bad_force_rate = bad_situation_rounds['ct_is_forcing'].mean()
    print(f"Force rate in these situations: {bad_force_rate*100:.1f}%")

    force_in_bad = bad_situation_rounds[bad_situation_rounds['ct_is_forcing']]
    if len(force_in_bad) > 10:
        bad_force_winrate = force_in_bad['ct_wins_round'].mean()
        print(f"Win rate on these 'bad' forces: {bad_force_winrate*100:.1f}%")

    eco_rounds = bad_situation_rounds[~bad_situation_rounds['ct_is_forcing']]
    if len(eco_rounds) > 50:
        eco_winrate = eco_rounds['ct_wins_round'].mean()
        print(f"Win rate when eco-ing instead: {eco_winrate*100:.1f}%")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Force rate by consecutive losses
ax1 = axes[0, 0]
valid_data = force_by_losses[force_by_losses['n_rounds'] >= 30]
ax1.bar(valid_data['ct_consecutive_losses'], valid_data['force_rate'] * 100,
        color='steelblue', edgecolor='black', alpha=0.8)
ax1.set_xlabel('Consecutive Losses')
ax1.set_ylabel('Force-Buy Rate (%)')
ax1.set_title('Force-Buy Rate by Loss Streak')

# Plot 2: Force success rate
ax2 = axes[0, 1]
valid_success = force_success[force_success['n'] >= 20]
ax2.bar(valid_success['ct_consecutive_losses'], valid_success['win_rate'] * 100,
        color='coral', edgecolor='black', alpha=0.8)
ax2.axhline(y=35, color='red', linestyle='--', label='Expected (~35%)')
ax2.set_xlabel('Consecutive Losses')
ax2.set_ylabel('Force-Buy Win Rate (%)')
ax2.set_title('Force-Buy Success Rate by Loss Streak')
ax2.legend()

# Plot 3: Force rate after win vs loss
ax3 = axes[1, 0]
categories = ['After Win', 'After Loss']
rates = [after_win['ct_is_forcing'].mean() * 100, after_loss['ct_is_forcing'].mean() * 100]
colors = ['green', 'red']
ax3.bar(categories, rates, color=colors, edgecolor='black', alpha=0.7)
ax3.set_ylabel('Force-Buy Rate (%)')
ax3.set_title('Force-Buy Rate by Previous Round Result')
for i, v in enumerate(rates):
    ax3.text(i, v + 1, f'{v:.1f}%', ha='center')

# Plot 4: Win rate comparison
ax4 = axes[1, 1]
force_wr = ct_forceable[ct_forceable['ct_is_forcing']]['ct_wins_round'].mean() * 100
eco_wr = ct_forceable[~ct_forceable['ct_is_forcing']]['ct_wins_round'].mean() * 100
ax4.bar(['Force Buy', 'Eco'], [force_wr, eco_wr],
        color=['steelblue', 'lightgray'], edgecolor='black', alpha=0.8)
ax4.set_ylabel('Win Rate (%)')
ax4.set_title('Win Rate: Force-Buy vs Eco (in broke/building)')
for i, v in enumerate([force_wr, eco_wr]):
    ax4.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_11_behavioral_analysis.png', dpi=150)
plt.close()
print("Saved: fig_11_behavioral_analysis.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("PHASE 7: BEHAVIORAL ANALYSIS SUMMARY")
print("="*70)

print(f"""
FINDINGS:

1. FORCE-BUY RATE AFTER WIN VS LOSS:
   After win:  {after_win['ct_is_forcing'].mean()*100:.1f}%
   After loss: {after_loss['ct_is_forcing'].mean()*100:.1f}%
   Difference: {force_diff*100:+.1f} pp

2. REGRESSION (Loss Streak → Forcing):
   Coefficient: {model1.params['ct_consecutive_losses']:.4f}
   OR: {np.exp(model1.params['ct_consecutive_losses']):.3f}
   p-value: {model1.pvalues['ct_consecutive_losses']:.4f}

3. WIN RATES:
   Force-buy win rate: {force_wr:.1f}%
   Eco win rate: {eco_wr:.1f}%

4. INTERPRETATION:
""")

if model1.pvalues['ct_consecutive_losses'] < 0.05 and model1.params['ct_consecutive_losses'] > 0:
    print("   Teams DO show tilt behavior - they force more as losses accumulate.")
    print("   This is irrational if force-buys have lower expected value than saving.")
else:
    print("   No significant tilt behavior detected.")

# =============================================================================
# SAVE
# =============================================================================

behavioral_results = {
    'force_rate_after_win': after_win['ct_is_forcing'].mean(),
    'force_rate_after_loss': after_loss['ct_is_forcing'].mean(),
    'force_rate_diff': force_diff,
    'loss_streak_coef': model1.params['ct_consecutive_losses'],
    'loss_streak_or': np.exp(model1.params['ct_consecutive_losses']),
    'loss_streak_pvalue': model1.pvalues['ct_consecutive_losses'],
    'force_win_rate': force_wr,
    'eco_win_rate': eco_wr,
}

pd.DataFrame([behavioral_results]).to_csv(OUTPUT_DIR / 'phase7_behavioral_results.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'phase7_behavioral_results.csv'}")

print("\nDone.")
