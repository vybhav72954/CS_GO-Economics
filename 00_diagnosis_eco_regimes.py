"""
Diagnostic: Why does categorical control fail in Antwerp?

Hypothesis: Regime thresholds calibrated on Stockholm don't fit Antwerp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(r"Z:\Projects\2025\CS-GO Time Series Analysis")

stockholm = pd.read_csv(BASE_DIR / "json_output" / "Stockholm" / "stockholm_rounds.csv")
antwerp = pd.read_csv(BASE_DIR / "json_output" / "Antwerp" / "antwerp_rounds.csv")

print("=" * 70)
print("DIAGNOSTIC: ECONOMIC DISTRIBUTIONS")
print("=" * 70)

# =============================================================================
# 1. EQUIPMENT VALUE DISTRIBUTIONS
# =============================================================================

print("\n1. CT EQUIPMENT VALUE STATISTICS")
print("-" * 50)
print(f"{'Stat':<15} {'Stockholm':>15} {'Antwerp':>15}")
print("-" * 50)

for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
    s_val = stockholm['ct_equip_value'].describe()[stat]
    a_val = antwerp['ct_equip_value'].describe()[stat]
    print(f"{stat:<15} {s_val:>15,.0f} {a_val:>15,.0f}")

# =============================================================================
# 2. REGIME DISTRIBUTIONS
# =============================================================================

print("\n\n2. CT REGIME DISTRIBUTION")
print("-" * 50)

s_regime = stockholm['ct_economic_regime'].value_counts(normalize=True).sort_index()
a_regime = antwerp['ct_economic_regime'].value_counts(normalize=True).sort_index()

print(f"{'Regime':<15} {'Stockholm':>15} {'Antwerp':>15} {'Diff':>10}")
print("-" * 50)
for regime in ['broke', 'building', 'full_buy', 'flush']:
    s_pct = s_regime.get(regime, 0) * 100
    a_pct = a_regime.get(regime, 0) * 100
    diff = a_pct - s_pct
    print(f"{regime:<15} {s_pct:>14.1f}% {a_pct:>14.1f}% {diff:>+9.1f}%")

# =============================================================================
# 3. EQUIPMENT BY REGIME
# =============================================================================

print("\n\n3. MEAN CT EQUIPMENT VALUE BY REGIME")
print("-" * 50)

s_by_regime = stockholm.groupby('ct_economic_regime')['ct_equip_value'].mean()
a_by_regime = antwerp.groupby('ct_economic_regime')['ct_equip_value'].mean()

print(f"{'Regime':<15} {'Stockholm':>15} {'Antwerp':>15} {'Diff':>10}")
print("-" * 50)
for regime in ['broke', 'building', 'full_buy', 'flush']:
    s_val = s_by_regime.get(regime, 0)
    a_val = a_by_regime.get(regime, 0)
    diff = a_val - s_val
    print(f"{regime:<15} ${s_val:>13,.0f} ${a_val:>13,.0f} ${diff:>+8,.0f}")

# =============================================================================
# 4. WIN RATE BY REGIME (THE KEY CHECK)
# =============================================================================

print("\n\n4. CT WIN RATE BY REGIME")
print("-" * 50)

s_winrate = stockholm.groupby('ct_economic_regime')['ct_wins_round'].mean()
a_winrate = antwerp.groupby('ct_economic_regime')['ct_wins_round'].mean()

print(f"{'Regime':<15} {'Stockholm':>15} {'Antwerp':>15} {'Diff':>10}")
print("-" * 50)
for regime in ['broke', 'building', 'full_buy', 'flush']:
    s_val = s_winrate.get(regime, 0) * 100
    a_val = a_winrate.get(regime, 0) * 100
    diff = a_val - s_val
    print(f"{regime:<15} {s_val:>14.1f}% {a_val:>14.1f}% {diff:>+9.1f}%")

# =============================================================================
# 5. MOMENTUM EFFECT WITHIN EACH REGIME
# =============================================================================

print("\n\n5. MOMENTUM EFFECT BY REGIME (Win rate after win - after loss)")
print("-" * 70)


def momentum_by_regime(df, name):
    df = df.dropna(subset=['ct_won_lag_1'])
    results = []

    for regime in ['broke', 'building', 'full_buy', 'flush']:
        regime_df = df[df['ct_economic_regime']==regime]
        if len(regime_df) > 30:
            after_win = regime_df[regime_df['ct_won_lag_1']==1]['ct_wins_round'].mean()
            after_loss = regime_df[regime_df['ct_won_lag_1']==0]['ct_wins_round'].mean()
            n_win = len(regime_df[regime_df['ct_won_lag_1']==1])
            n_loss = len(regime_df[regime_df['ct_won_lag_1']==0])
            diff = after_win - after_loss
            results.append({
                'regime':regime,
                'after_win':after_win,
                'after_loss':after_loss,
                'diff':diff,
                'n_win':n_win,
                'n_loss':n_loss
            })

    return pd.DataFrame(results)


s_momentum = momentum_by_regime(stockholm, "Stockholm")
a_momentum = momentum_by_regime(antwerp, "Antwerp")

print(f"{'Regime':<12} {'Stockholm':>25} {'Antwerp':>25}")
print(f"{'':12} {'After W':>10} {'After L':>8} {'Diff':>7} {'After W':>10} {'After L':>8} {'Diff':>7}")
print("-" * 70)

for regime in ['broke', 'building', 'full_buy', 'flush']:
    s_row = s_momentum[s_momentum['regime']==regime].iloc[0] if len(
        s_momentum[s_momentum['regime']==regime]) > 0 else None
    a_row = a_momentum[a_momentum['regime']==regime].iloc[0] if len(
        a_momentum[a_momentum['regime']==regime]) > 0 else None

    if s_row is not None and a_row is not None:
        print(
            f"{regime:<12} {s_row['after_win'] * 100:>9.1f}% {s_row['after_loss'] * 100:>7.1f}% {s_row['diff'] * 100:>+6.1f}% "
            f"{a_row['after_win'] * 100:>9.1f}% {a_row['after_loss'] * 100:>7.1f}% {a_row['diff'] * 100:>+6.1f}%")

# =============================================================================
# 6. CELL SIZES IN CROSS-TAB
# =============================================================================

print("\n\n6. CELL SIZES: Momentum × Regime")
print("-" * 50)

print("\nStockholm (after loss → regime):")
s_after_loss = stockholm[stockholm['ct_won_lag_1']==0]['ct_economic_regime'].value_counts()
for regime in ['broke', 'building', 'full_buy', 'flush']:
    print(f"  {regime}: n = {s_after_loss.get(regime, 0)}")

print("\nAntwerp (after loss → regime):")
a_after_loss = antwerp[antwerp['ct_won_lag_1']==0]['ct_economic_regime'].value_counts()
for regime in ['broke', 'building', 'full_buy', 'flush']:
    print(f"  {regime}: n = {a_after_loss.get(regime, 0)}")

# =============================================================================
# 7. THE KEY QUESTION: FLUSH REGIME
# =============================================================================

print("\n\n" + "=" * 70)
print("KEY FINDING: FLUSH REGIME ANALYSIS")
print("=" * 70)

print("\nFlush regime is where most 'after win' observations fall.")
print("If momentum persists in flush, that's the residual effect.\n")

for name, df in [("Stockholm", stockholm), ("Antwerp", antwerp)]:
    df = df.dropna(subset=['ct_won_lag_1'])
    flush_df = df[df['ct_economic_regime']=='flush']

    after_win = flush_df[flush_df['ct_won_lag_1']==1]['ct_wins_round']
    after_loss = flush_df[flush_df['ct_won_lag_1']==0]['ct_wins_round']

    print(f"{name} - FLUSH regime:")
    print(f"  After win:  {after_win.mean() * 100:.1f}% win rate (n={len(after_win)})")
    print(f"  After loss: {after_loss.mean() * 100:.1f}% win rate (n={len(after_loss)})")
    print(f"  Difference: {(after_win.mean() - after_loss.mean()) * 100:+.1f} pp")
    print()

# =============================================================================
# 8. SUMMARY
# =============================================================================

print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

print("""
Questions to answer:

1. Are equipment distributions similar? 
   → Check section 1 & 3

2. Are regime distributions similar?
   → Check section 2

3. Does momentum exist WITHIN regimes?
   → Check section 5
   → If yes, regime dummies can't fully capture it

4. Is flush regime the problem?
   → Check section 7
   → Large momentum diff in flush = residual effect
""")
