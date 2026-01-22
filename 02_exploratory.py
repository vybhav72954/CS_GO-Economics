"""
Phase 2: Exploratory Analysis
Dataset: Stockholm 2021 (Training Set Only)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

BASE_DIR = Path(r"Z:\Projects\2025\CS-GO Time Series Analysis")
OUTPUT_DIR = BASE_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load Stockholm data only
df = pd.read_csv(BASE_DIR / "json_output" / "Stockholm" / "stockholm_rounds.csv")
print(f"Loaded {len(df)} rounds from {df['match_id'].nunique()} matches")


# =============================================================================
# SECTION 1: DESCRIPTIVE STATISTICS
# =============================================================================

def descriptive_statistics(df):
    """Generate descriptive statistics for the dataset."""

    print("\n" + "=" * 70)
    print("SECTION 1: DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # Basic counts
    print("\n1.1 Dataset Overview")
    print("-" * 40)
    print(f"Total rounds: {len(df)}")
    print(f"Total matches: {df['match_id'].nunique()}")
    print(f"Maps: {df['map_name'].nunique()}")
    print(f"Unique teams (CT): {df['ct_team'].nunique()}")

    # Outcome distribution
    print("\n1.2 Round Outcomes")
    print("-" * 40)
    ct_wins = df['ct_wins_round'].sum()
    t_wins = len(df) - ct_wins
    print(f"CT wins: {ct_wins} ({ct_wins / len(df) * 100:.1f}%)")
    print(f"T wins:  {t_wins} ({t_wins / len(df) * 100:.1f}%)")

    # By map
    print("\n1.3 CT Win Rate by Map")
    print("-" * 40)
    map_stats = df.groupby('map_name')['ct_wins_round'].agg(['mean', 'count'])
    map_stats.columns = ['ct_win_rate', 'rounds']
    map_stats = map_stats.sort_values('ct_win_rate', ascending=False)
    for map_name, row in map_stats.iterrows():
        print(f"  {map_name:<15}: {row['ct_win_rate']:.1%} (n={row['rounds']:.0f})")

    # By round phase
    print("\n1.4 CT Win Rate by Round Phase")
    print("-" * 40)
    phase_stats = df.groupby('round_phase')['ct_wins_round'].agg(['mean', 'count'])
    for phase in ['pistol', 'conversion', 'gun', 'overtime']:
        if phase in phase_stats.index:
            row = phase_stats.loc[phase]
            print(f"  {phase:<12}: {row['mean']:.1%} (n={row['count']:.0f})")

    # Economic regime distribution
    print("\n1.5 Economic Regime Distribution")
    print("-" * 40)
    print("CT Side:")
    ct_regime = df['ct_economic_regime'].value_counts()
    for regime in ['broke', 'building', 'full_buy', 'flush']:
        count = ct_regime.get(regime, 0)
        print(f"  {regime:<12}: {count:>4} ({count / len(df) * 100:>5.1f}%)")

    print("T Side:")
    t_regime = df['t_economic_regime'].value_counts()
    for regime in ['broke', 'building', 'full_buy', 'flush']:
        count = t_regime.get(regime, 0)
        print(f"  {regime:<12}: {count:>4} ({count / len(df) * 100:>5.1f}%)")

    # Win rate by economic regime
    print("\n1.6 CT Win Rate by Economic Regime")
    print("-" * 40)
    regime_wins = df.groupby('ct_economic_regime')['ct_wins_round'].agg(['mean', 'count'])
    for regime in ['broke', 'building', 'full_buy', 'flush']:
        if regime in regime_wins.index:
            row = regime_wins.loc[regime]
            print(f"  CT {regime:<10}: {row['mean']:.1%} win rate (n={row['count']:.0f})")

    print()
    regime_wins_t = df.groupby('t_economic_regime')['ct_wins_round'].agg(['mean', 'count'])
    for regime in ['broke', 'building', 'full_buy', 'flush']:
        if regime in regime_wins_t.index:
            row = regime_wins_t.loc[regime]
            t_win_rate = 1 - row['mean']
            print(f"  T {regime:<10}: {t_win_rate:.1%} win rate (n={row['count']:.0f})")

    # Equipment value statistics
    print("\n1.7 Equipment Value Statistics")
    print("-" * 40)
    print(f"CT: mean=${df['ct_equip_value'].mean():,.0f}, median=${df['ct_equip_value'].median():,.0f}")
    print(f"T:  mean=${df['t_equip_value'].mean():,.0f}, median=${df['t_equip_value'].median():,.0f}")
    print(f"Advantage range: ${df['equip_advantage'].min():,.0f} to ${df['equip_advantage'].max():,.0f}")

    # Loss streak statistics
    print("\n1.8 Loss Streak Statistics")
    print("-" * 40)
    print("CT consecutive losses:")
    ct_loss_dist = df['ct_consecutive_losses'].value_counts().sort_index()
    for losses, count in ct_loss_dist.items():
        if losses <= 5:
            print(f"  {losses} losses: {count} rounds ({count / len(df) * 100:.1f}%)")

    return map_stats, phase_stats


# =============================================================================
# SECTION 2: VISUALIZATIONS
# =============================================================================

def create_visualizations(df, output_dir):
    """Generate key visualizations for exploratory analysis."""

    print("\n" + "=" * 70)
    print("SECTION 2: VISUALIZATIONS")
    print("=" * 70)

    # 2.1 Win rate by equipment advantage
    print("\n2.1 Creating: Win Rate by Equipment Advantage")
    fig, ax = plt.subplots(figsize=(10, 6))

    df['equip_adv_bin'] = pd.cut(df['equip_advantage'],
                                 bins=[-30000, -20000, -10000, -5000, 0, 5000, 10000, 20000, 30000],
                                 labels=['<-20k', '-20k to -10k', '-10k to -5k', '-5k to 0',
                                         '0 to 5k', '5k to 10k', '10k to 20k', '>20k'])

    equip_win_rate = df.groupby('equip_adv_bin', observed=True)['ct_wins_round'].agg(['mean', 'count'])

    bars = ax.bar(range(len(equip_win_rate)), equip_win_rate['mean'], color='steelblue', edgecolor='black')
    ax.axhline(y=0.5, color='red', linestyle='--', label='50% baseline')
    ax.set_xticks(range(len(equip_win_rate)))
    ax.set_xticklabels(equip_win_rate.index, rotation=45, ha='right')
    ax.set_xlabel('CT Equipment Advantage ($)')
    ax.set_ylabel('CT Win Rate')
    ax.set_title('CT Win Rate by Equipment Advantage (Stockholm 2021)')
    ax.set_ylim(0, 1)
    ax.legend()

    for i, (idx, row) in enumerate(equip_win_rate.iterrows()):
        ax.annotate(f'n={row["count"]:.0f}', (i, row['mean'] + 0.03), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_01_win_rate_by_equipment.png', dpi=150)
    plt.close()

    # 2.2 Economic regime transition matrix
    print("2.2 Creating: Economic Regime Transitions")
    fig, ax = plt.subplots(figsize=(8, 6))

    df_sorted = df.sort_values(['match_id', 'round_num'])
    df_sorted['ct_regime_next'] = df_sorted.groupby('match_id')['ct_economic_regime'].shift(-1)
    transitions = df_sorted.dropna(subset=['ct_regime_next'])

    regime_order = ['broke', 'building', 'full_buy', 'flush']
    transition_matrix = pd.crosstab(
        transitions['ct_economic_regime'],
        transitions['ct_regime_next'],
        normalize='index'
    )
    transition_matrix = transition_matrix.reindex(index=regime_order, columns=regime_order, fill_value=0)

    sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=regime_order, yticklabels=regime_order)
    ax.set_xlabel('Next Round Regime')
    ax.set_ylabel('Current Round Regime')
    ax.set_title('CT Economic Regime Transition Probabilities')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_02_regime_transitions.png', dpi=150)
    plt.close()

    # 2.3 Win rate by round phase and regime
    print("2.3 Creating: Win Rate by Round Phase")
    fig, ax = plt.subplots(figsize=(10, 6))

    phase_regime = df.groupby(['round_phase', 'ct_economic_regime'])['ct_wins_round'].mean().unstack()
    phase_order = ['pistol', 'conversion', 'gun']
    phase_regime = phase_regime.reindex(phase_order)

    phase_regime.plot(kind='bar', ax=ax, width=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Round Phase')
    ax.set_ylabel('CT Win Rate')
    ax.set_title('CT Win Rate by Round Phase and Economic Regime')
    ax.set_xticklabels(phase_order, rotation=0)
    ax.legend(title='CT Regime', bbox_to_anchor=(1.02, 1))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_03_phase_regime_winrate.png', dpi=150)
    plt.close()

    # 2.4 Momentum: Win rate after win vs after loss
    print("2.4 Creating: Momentum Effect Visualization")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: simple lag effect
    lag_effect = df.groupby('ct_won_lag_1')['ct_wins_round'].mean()
    colors = ['#e74c3c', '#27ae60']
    axes[0].bar(['After Loss', 'After Win'], [lag_effect.get(0, 0), lag_effect.get(1, 0)],
                color=colors, edgecolor='black')
    axes[0].axhline(y=0.5, color='gray', linestyle='--')
    axes[0].set_ylabel('CT Win Rate')
    axes[0].set_title('CT Win Rate: After Win vs After Loss')
    axes[0].set_ylim(0, 0.8)

    for i, val in enumerate([lag_effect.get(0, 0), lag_effect.get(1, 0)]):
        axes[0].annotate(f'{val:.1%}', (i, val + 0.02), ha='center', fontsize=12, fontweight='bold')

    # Right: by loss streak
    loss_streak_win = df.groupby('ct_consecutive_losses')['ct_wins_round'].agg(['mean', 'count'])
    loss_streak_win = loss_streak_win[loss_streak_win['count'] >= 20]

    axes[1].bar(loss_streak_win.index, loss_streak_win['mean'], color='steelblue', edgecolor='black')
    axes[1].axhline(y=0.5, color='gray', linestyle='--')
    axes[1].set_xlabel('CT Consecutive Losses')
    axes[1].set_ylabel('CT Win Rate')
    axes[1].set_title('CT Win Rate by Loss Streak Length')
    axes[1].set_ylim(0, 0.8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_04_momentum_effect.png', dpi=150)
    plt.close()

    # 2.5 Equipment distribution by regime
    print("2.5 Creating: Equipment Distribution by Regime")
    fig, ax = plt.subplots(figsize=(10, 6))

    regime_order = ['broke', 'building', 'full_buy', 'flush']
    df['ct_economic_regime'] = pd.Categorical(df['ct_economic_regime'], categories=regime_order, ordered=True)

    sns.boxplot(data=df, x='ct_economic_regime', y='ct_equip_value', order=regime_order, ax=ax)
    ax.set_xlabel('CT Economic Regime')
    ax.set_ylabel('CT Equipment Value ($)')
    ax.set_title('Equipment Value Distribution by Regime Classification')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_05_equipment_by_regime.png', dpi=150)
    plt.close()

    print(f"\nAll figures saved to: {output_dir}")


# =============================================================================
# SECTION 3: MOMENTUM ANALYSIS (MODEL A - BASELINE)
# =============================================================================

def momentum_baseline_analysis(df):
    """
    Model A: Baseline momentum test.
    Tests whether winning round N predicts winning round N+1.
    """

    print("\n" + "=" * 70)
    print("SECTION 3: MOMENTUM ANALYSIS (MODEL A - BASELINE)")
    print("=" * 70)

    # Prepare data (drop rows with missing lag values)
    analysis_df = df.dropna(subset=['ct_won_lag_1']).copy()
    print(f"\nAnalysis sample: {len(analysis_df)} rounds (excluding first round of each match)")

    # Model A: Simple logistic regression
    print("\n3.1 Model A: Baseline Momentum")
    print("-" * 40)

    X = sm.add_constant(analysis_df['ct_won_lag_1'])
    y = analysis_df['ct_wins_round']

    model_a = sm.Logit(y, X).fit(disp=0)

    print(f"\nLogistic Regression Results:")
    print(f"  Observations: {model_a.nobs:.0f}")
    print(f"  Pseudo R-squared: {model_a.prsquared:.4f}")
    print(f"\n  {'Variable':<20} {'Coef':>10} {'Std Err':>10} {'z':>10} {'P>|z|':>10}")
    print(f"  {'-' * 60}")
    for var in model_a.params.index:
        coef = model_a.params[var]
        se = model_a.bse[var]
        z = model_a.tvalues[var]
        p = model_a.pvalues[var]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {var:<20} {coef:>10.4f} {se:>10.4f} {z:>10.2f} {p:>10.4f} {sig}")

    # Interpretation
    momentum_coef = model_a.params['ct_won_lag_1']
    momentum_or = np.exp(momentum_coef)

    print(f"\n3.2 Interpretation")
    print("-" * 40)
    print(f"  Momentum coefficient: {momentum_coef:.4f}")
    print(f"  Odds ratio: {momentum_or:.3f}")
    print(f"  Interpretation: Winning the previous round multiplies the odds of")
    print(f"                  winning the current round by {momentum_or:.2f}x")

    # Marginal effects
    print(f"\n3.3 Conditional Win Probabilities")
    print("-" * 40)

    p_after_loss = 1 / (1 + np.exp(-(model_a.params['const'])))
    p_after_win = 1 / (1 + np.exp(-(model_a.params['const'] + model_a.params['ct_won_lag_1'])))

    print(f"  P(CT wins | lost previous round) = {p_after_loss:.1%}")
    print(f"  P(CT wins | won previous round)  = {p_after_win:.1%}")
    print(f"  Difference: {(p_after_win - p_after_loss) * 100:.1f} percentage points")

    # Store results for later comparison
    results = {
        'model':'A',
        'description':'Baseline (no controls)',
        'n_obs':model_a.nobs,
        'momentum_coef':momentum_coef,
        'momentum_se':model_a.bse['ct_won_lag_1'],
        'momentum_pvalue':model_a.pvalues['ct_won_lag_1'],
        'odds_ratio':momentum_or,
        'pseudo_r2':model_a.prsquared,
        'aic':model_a.aic,
        'bic':model_a.bic
    }

    return model_a, results


# =============================================================================
# SECTION 4: PRELIMINARY CONTROL VARIABLE ANALYSIS
# =============================================================================

def control_variable_preview(df):
    """
    Preview the relationship between control variables and outcomes.
    This informs which controls to include in Models B-F.
    """

    print("\n" + "=" * 70)
    print("SECTION 4: CONTROL VARIABLE PREVIEW")
    print("=" * 70)

    analysis_df = df.dropna(subset=['ct_won_lag_1']).copy()

    # 4.1 Equipment advantage correlation
    print("\n4.1 Equipment Advantage")
    print("-" * 40)
    corr = analysis_df['equip_advantage'].corr(analysis_df['ct_wins_round'])
    print(f"  Correlation with CT win: {corr:.3f}")

    # Binned analysis
    bins = pd.qcut(analysis_df['equip_advantage'], q=5, duplicates='drop')
    binned = analysis_df.groupby(bins, observed=True)['ct_wins_round'].agg(['mean', 'count'])
    print(f"\n  Win rate by equipment advantage quintile:")
    for idx, row in binned.iterrows():
        print(f"    {str(idx):<25}: {row['mean']:.1%} (n={row['count']:.0f})")

    # 4.2 Economic regime effect
    print("\n4.2 Economic Regime Effect")
    print("-" * 40)

    regime_stats = analysis_df.groupby('ct_economic_regime')['ct_wins_round'].agg(['mean', 'count'])
    for regime in ['broke', 'building', 'full_buy', 'flush']:
        if regime in regime_stats.index:
            row = regime_stats.loc[regime]
            print(f"  CT {regime:<10}: {row['mean']:.1%} (n={row['count']:.0f})")

    # 4.3 Round phase effect
    print("\n4.3 Round Phase Effect")
    print("-" * 40)

    phase_stats = analysis_df.groupby('round_phase')['ct_wins_round'].agg(['mean', 'count'])
    for phase in ['pistol', 'conversion', 'gun', 'overtime']:
        if phase in phase_stats.index:
            row = phase_stats.loc[phase]
            print(f"  {phase:<12}: {row['mean']:.1%} (n={row['count']:.0f})")

    # 4.4 Momentum by economic regime (interaction preview)
    print("\n4.4 Momentum Effect by Economic Regime")
    print("-" * 40)
    print("  (CT win rate after winning previous round)")

    for regime in ['broke', 'building', 'full_buy', 'flush']:
        regime_df = analysis_df[analysis_df['ct_economic_regime']==regime]
        if len(regime_df) > 50:
            after_win = regime_df[regime_df['ct_won_lag_1']==1]['ct_wins_round'].mean()
            after_loss = regime_df[regime_df['ct_won_lag_1']==0]['ct_wins_round'].mean()
            diff = after_win - after_loss
            print(f"  {regime:<10}: after win={after_win:.1%}, after loss={after_loss:.1%}, diff={diff:+.1%}")

    # 4.5 First kill effect
    print("\n4.5 First Kill Effect")
    print("-" * 40)

    fk_ct = analysis_df[analysis_df['first_kill_side']=='CT']['ct_wins_round'].mean()
    fk_t = analysis_df[analysis_df['first_kill_side']=='T']['ct_wins_round'].mean()
    print(f"  CT gets first kill: {fk_ct:.1%} CT win rate")
    print(f"  T gets first kill:  {fk_t:.1%} CT win rate")
    print(f"  Difference: {(fk_ct - fk_t) * 100:.1f} percentage points")


# =============================================================================
# SECTION 5: SUMMARY AND NEXT STEPS
# =============================================================================

def print_summary(model_a_results):
    """Print summary and next steps."""

    print("\n" + "=" * 70)
    print("SECTION 5: SUMMARY")
    print("=" * 70)

    print("\n5.1 Key Findings from Exploratory Analysis")
    print("-" * 40)
    print(f"  1. Baseline momentum coefficient: {model_a_results['momentum_coef']:.4f}")
    print(f"  2. Odds ratio: {model_a_results['odds_ratio']:.3f}")
    print(f"  3. This is the coefficient to track through Models B-F")

    print("\n5.2 Variables for Momentum Decomposition (Phase 3)")
    print("-" * 40)
    print("  Model B: + ct_economic_regime (categorical)")
    print("  Model C: + equip_advantage (continuous)")
    print("  Model D: + rank_diff (requires HLTV data)")
    print("  Model E: + baseline_ct_win_prob (map/side)")
    print("  Model F: + round_phase interactions")

    print("\n5.3 Data Gaps to Address")
    print("-" * 40)
    print("  - HLTV rankings not yet added (needed for Model D)")
    print("  - Force-buy identification (needed for Phase 5)")

    print("\n" + "=" * 70)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__=="__main__":
    # Section 1: Descriptive Statistics
    map_stats, phase_stats = descriptive_statistics(df)

    # Section 2: Visualizations
    create_visualizations(df, OUTPUT_DIR)

    # Section 3: Momentum Baseline
    model_a, model_a_results = momentum_baseline_analysis(df)

    # Section 4: Control Variable Preview
    control_variable_preview(df)

    # Section 5: Summary
    print_summary(model_a_results)

    # Save Model A results for Phase 3 comparison
    results_df = pd.DataFrame([model_a_results])
    results_df.to_csv(OUTPUT_DIR / 'model_results.csv', index=False)
    print(f"\nModel results saved to: {OUTPUT_DIR / 'model_results.csv'}")
