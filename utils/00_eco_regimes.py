"""
Fix Economic Regime Classification
Run this after the main pipeline to recalculate regimes with correct team-level thresholds.
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# CORRECTED THRESHOLDS (Team Equipment Value)
# =============================================================================

REGIME_THRESHOLDS = {
    'ct':{
        'broke':(0, 8000),
        'building':(8000, 18000),
        'full_buy':(18000, 28000),
        'flush':(28000, float('inf'))
    },
    't':{
        'broke':(0, 6000),
        'building':(6000, 14000),
        'full_buy':(14000, 22000),
        'flush':(22000, float('inf'))
    }
}


def classify_economic_regime(equip_value: float, side: str) -> str:
    """Classify economic regime based on team equipment value."""
    thresholds = REGIME_THRESHOLDS.get(side, REGIME_THRESHOLDS['ct'])

    for regime, (low, high) in thresholds.items():
        if low <= equip_value < high:
            return regime

    return 'flush'


def fix_regimes_in_csv(csv_path: Path) -> pd.DataFrame:
    """Recalculate economic regimes in a CSV file."""
    df = pd.read_csv(csv_path)

    # Recalculate regimes
    df['ct_economic_regime'] = df['ct_equip_value'].apply(
        lambda x:classify_economic_regime(x, 'ct')
    )
    df['t_economic_regime'] = df['t_equip_value'].apply(
        lambda x:classify_economic_regime(x, 't')
    )

    # Save back
    df.to_csv(csv_path, index=False)

    return df


def main():
    base_dir = Path(r"Z:\Projects\2025\CS-GO Time Series Analysis\json_output")

    print("=" * 60)
    print("FIXING ECONOMIC REGIME CLASSIFICATIONS")
    print("=" * 60)

    # Fix all CSVs
    csv_files = [
        base_dir / "Stockholm" / "stockholm_rounds.csv",
        base_dir / "Antwerp" / "antwerp_rounds.csv",
        base_dir / "Rio" / "rio_rounds.csv",
        base_dir / "csv_exports" / "stockholm_2021_major_rounds.csv",
        base_dir / "csv_exports" / "antwerp_2022_major_rounds.csv",
        base_dir / "csv_exports" / "rio_2022_major_rounds.csv",
    ]

    for csv_path in csv_files:
        if csv_path.exists():
            print(f"\nProcessing: {csv_path.name}")
            df = fix_regimes_in_csv(csv_path)

            # Show new distribution
            ct_dist = df['ct_economic_regime'].value_counts()
            print(f"  CT Regimes: {ct_dist.to_dict()}")

            t_dist = df['t_economic_regime'].value_counts()
            print(f"  T Regimes:  {t_dist.to_dict()}")
        else:
            print(f" Not found: {csv_path}")

    print("\n" + "=" * 60)
    print("REGIME FIX COMPLETE")
    print("=" * 60)

    # Show combined distribution for Stockholm (training set)
    stockholm_df = pd.read_csv(base_dir / "Stockholm" / "stockholm_rounds.csv")

    print("\nStockholm Distribution (Training Set):")
    print("\nCT Economic Regimes:")
    ct_counts = stockholm_df['ct_economic_regime'].value_counts()
    for regime in ['broke', 'building', 'full_buy', 'flush']:
        count = ct_counts.get(regime, 0)
        pct = count / len(stockholm_df) * 100
        print(f"  {regime:<10}: {count:>4} ({pct:>5.1f}%)")

    print("\nT Economic Regimes:")
    t_counts = stockholm_df['t_economic_regime'].value_counts()
    for regime in ['broke', 'building', 'full_buy', 'flush']:
        count = t_counts.get(regime, 0)
        pct = count / len(stockholm_df) * 100
        print(f"  {regime:<10}: {count:>4} ({pct:>5.1f}%)")

    # Equipment value distribution for validation
    print("\nEquipment Value Statistics:")
    print(f"  CT mean: ${stockholm_df['ct_equip_value'].mean():,.0f}")
    print(f"  CT median: ${stockholm_df['ct_equip_value'].median():,.0f}")
    print(f"  CT min/max: ${stockholm_df['ct_equip_value'].min():,.0f} / ${stockholm_df['ct_equip_value'].max():,.0f}")
    print(f"  T mean: ${stockholm_df['t_equip_value'].mean():,.0f}")
    print(f"  T median: ${stockholm_df['t_equip_value'].median():,.0f}")


if __name__=="__main__":
    main()

