# =============================================================================
# CS:GO MAJOR DEMO PARSING PIPELINE
# =============================================================================
# Project: The Economics of Momentum: Evidence from Professional Esports
# Version: 2.1
#
# Tournament Roles:
#   - Stockholm 2021: TRAINING (exploratory, model development)
#   - Antwerp 2022:   CONFIRMATION (pattern matching, replication)
#   - Rio 2022:       VALIDATION (final holdout, out-of-sample test)
#
# Compatible with: awpy 1.2.3 (for CS:GO / Source 1 demos)
#
# Usage:
#   python 01_pipeline.py          # Load from cache if available, else parse
#   python 01_pipeline.py --parse  # Force re-parse all demos
#   python 01_pipeline.py --load   # Load from cache only (error if not found)
# =============================================================================

import os
import sys
import json
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# awpy 1.2.x imports (for CS:GO Source 1 demos)
from awpy.parser import DemoParser

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths - UPDATE THESE FOR YOUR SYSTEM
BASE_DIR = Path(r"Z:\Projects\2025\CS-GO Time Series Analysis")
DEMO_DIR = BASE_DIR / "demos"
OUTPUT_DIR = BASE_DIR / "json_output"

# Tournament configuration with roles
TOURNAMENTS = {
    "Stockholm":{
        "folder":"Stockholm",  # Make sure to rename from "Stolkhom"
        "event_id":5552,
        "role":"training",
        "dates":"Oct 26 - Nov 7, 2021",
        "winner":"Natus Vincere"
    },
    "Antwerp":{
        "folder":"Antwerp",
        "event_id":6372,
        "role":"confirmation",
        "dates":"May 9-22, 2022",
        "winner":"FaZe Clan"
    },
    "Rio":{
        "folder":"Rio",
        "event_id":6586,
        "role":"validation",
        "dates":"Oct 31 - Nov 13, 2022",
        "winner":"Outsiders"
    }
}

# Economic regime thresholds (TEAM equipment value, not per-player)
# These are based on typical CS:GO buy patterns:
# - CT full buy: ~$5,500/player × 5 = ~$27,500
# - T full buy: ~$4,200/player × 5 = ~$21,000
REGIME_THRESHOLDS = {
    'ct':{
        'broke':(0, 8000),  # Eco - pistols, minimal utility
        'building':(8000, 18000),  # Force/half-buy - SMGs, partial rifles
        'full_buy':(18000, 28000),  # Full buy - rifles + AWP + utility
        'flush':(28000, float('inf'))  # Excess wealth, double AWP setups
    },
    't':{
        'broke':(0, 6000),  # Eco
        'building':(6000, 14000),  # Force
        'full_buy':(14000, 22000),  # Full buy (T side is cheaper)
        'flush':(22000, float('inf'))
    }
}

# Map-side baseline CT win rates (approximate)
MAP_CT_WIN_RATES = {
    "de_ancient":0.52,
    "de_dust2":0.48,
    "de_inferno":0.50,
    "de_mirage":0.49,
    "de_nuke":0.57,
    "de_overpass":0.52,
    "de_vertigo":0.53,
    "default":0.50
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def classify_economic_regime(total_money: float, side: str) -> str:
    """
    Classify economic regime based on game mechanics thresholds.

    Args:
        total_money: Equipment value + cash (or just equipment if cash unavailable)
        side: 'ct' or 't'

    Returns:
        Regime string: 'broke', 'building', 'full_buy', or 'flush'
    """
    thresholds = REGIME_THRESHOLDS.get(side, REGIME_THRESHOLDS['ct'])

    for regime, (low, high) in thresholds.items():
        if low <= total_money < high:
            return regime

    return 'flush'  # Default to flush if above all thresholds


def classify_round_phase(round_num: int) -> str:
    """
    Classify round into game phase.

    Pistol: Economy reset, pure skill/chaos
    Conversion: Winner of pistol has huge advantage (anti-eco)
    Gun: Actual competitive rounds where momentum matters
    Overtime: Special OT rounds
    """
    if round_num in [1, 16]:
        return 'pistol'
    elif round_num in [2, 3, 17, 18]:
        return 'conversion'
    elif round_num > 30:
        return 'overtime'
    else:
        return 'gun'


def get_map_baseline_ct_win_rate(map_name: str) -> float:
    """Get baseline CT win rate for map."""
    return MAP_CT_WIN_RATES.get(map_name, MAP_CT_WIN_RATES['default'])


def calculate_half_number(round_num: int) -> int:
    """Calculate which half the round belongs to."""
    if round_num <= 15:
        return 1
    elif round_num <= 30:
        return 2
    else:
        return 3  # Overtime


def calculate_round_in_half(round_num: int) -> int:
    """Calculate round number within the half."""
    if round_num <= 15:
        return round_num
    elif round_num <= 30:
        return round_num - 15
    else:
        return ((round_num - 31) % 6) + 1  # OT rounds


# =============================================================================
# MAIN PARSING FUNCTION (awpy 1.2.x compatible - for CS:GO demos)
# =============================================================================

def parse_single_demo(demofile_path: str, tournament_name: str) -> dict:
    """
    Parse a single CS:GO demo file and extract comprehensive round-level features.
    Compatible with awpy 1.2.3 (CS:GO / Source 1 demos)

    Args:
        demofile_path: Full path to the .dem file
        tournament_name: Name of the tournament for metadata

    Returns:
        Dictionary containing match metadata and round data
    """
    try:
        # outpath set to temp dir to prevent awpy from writing JSON to working directory
        temp_dir = tempfile.gettempdir()
        demo_parser = DemoParser(demofile=demofile_path, outpath=temp_dir, parse_rate=128, trade_time=5)
        data = demo_parser.parse()
    except Exception as e:
        print(f"  Error parsing {demofile_path}: {e}")
        return None

    if not data.get("gameRounds"):
        print(f"  No game rounds found in {demofile_path}")
        return None

    # Extract match metadata
    match_id = os.path.basename(demofile_path).replace(".dem", "")
    map_name = data.get("mapName", "unknown").lower()

    # Get team names from first round
    first_round = data["gameRounds"][0]
    ct_team_name = first_round.get("ctTeam", "CT")
    t_team_name = first_round.get("tTeam", "T")

    # Track loss streaks for each side
    ct_consecutive_losses = 0
    t_consecutive_losses = 0

    rounds_data = []

    for i, r in enumerate(data["gameRounds"]):
        round_num = r.get("roundNum", i + 1)

        # Skip warmup rounds
        if r.get("isWarmup", False):
            continue

        # Basic round info
        ct_score = r.get("ctScore", 0)
        t_score = r.get("tScore", 0)
        winning_side = r.get("winningSide", "").upper()

        # Equipment values (from round-level data)
        ct_equip_value = r.get("ctFreezeTimeEndEqVal", 0) or 0
        t_equip_value = r.get("tFreezeTimeEndEqVal", 0) or 0

        # Buy types (already classified by awpy parser)
        ct_buy_type = r.get("ctBuyType", "unknown")
        t_buy_type = r.get("tBuyType", "unknown")

        # Round spend money (useful for economy tracking)
        ct_spend = r.get("ctRoundSpendMoney", 0) or 0
        t_spend = r.get("tRoundSpendMoney", 0) or 0

        # Bomb events
        bomb_planted = r.get("bombPlantTick") is not None

        # First kill analysis
        first_kill_side = None
        kills = r.get("kills", [])
        if kills and len(kills) > 0:
            first_kill = kills[0]
            attacker_side = first_kill.get("attackerSide", "")
            if attacker_side:
                first_kill_side = attacker_side.upper()

        # Calculate derived features
        half_number = calculate_half_number(round_num)
        round_in_half = calculate_round_in_half(round_num)
        is_overtime = round_num > 30
        is_pistol = round_num in [1, 16]
        is_conversion = round_num in [2, 3, 17, 18]
        round_phase = classify_round_phase(round_num)

        # Economic regimes
        ct_economic_regime = classify_economic_regime(ct_equip_value, 'ct')
        t_economic_regime = classify_economic_regime(t_equip_value, 't')

        # Equipment advantage
        equip_advantage = ct_equip_value - t_equip_value

        # Score difference
        score_diff = ct_score - t_score

        # Map baseline CT win rate
        baseline_ct_win_prob = get_map_baseline_ct_win_rate(map_name)

        # Determine outcome
        ct_wins_round = 1 if winning_side=="CT" else 0

        # Build round record
        round_record = {
            # Identifiers
            "match_id":match_id,
            "tournament":tournament_name,
            "map_name":map_name,
            "round_num":round_num,

            # Teams
            "ct_team":ct_team_name,
            "t_team":t_team_name,

            # Scores (at start of round)
            "ct_score":ct_score,
            "t_score":t_score,
            "score_diff":score_diff,

            # Economic data
            "ct_equip_value":ct_equip_value,
            "t_equip_value":t_equip_value,
            "equip_advantage":equip_advantage,
            "ct_spend":ct_spend,
            "t_spend":t_spend,
            "ct_buy_type":ct_buy_type,
            "t_buy_type":t_buy_type,

            # Economic regimes
            "ct_economic_regime":ct_economic_regime,
            "t_economic_regime":t_economic_regime,

            # Loss streaks (updated after this round)
            "ct_consecutive_losses":ct_consecutive_losses,
            "t_consecutive_losses":t_consecutive_losses,

            # Round structure
            "half_number":half_number,
            "round_in_half":round_in_half,
            "is_overtime":is_overtime,
            "is_pistol":is_pistol,
            "is_conversion":is_conversion,
            "round_phase":round_phase,

            # Map-side baseline
            "baseline_ct_win_prob":baseline_ct_win_prob,

            # In-round events
            "bomb_planted":bomb_planted,
            "first_kill_side":first_kill_side,

            # Outcome
            "winning_side":winning_side,
            "ct_wins_round":ct_wins_round
        }

        rounds_data.append(round_record)

        # Update loss streaks for next round
        if winning_side=="CT":
            ct_consecutive_losses = 0
            t_consecutive_losses += 1
        elif winning_side=="T":
            t_consecutive_losses = 0
            ct_consecutive_losses += 1

        # Reset loss streaks at half time (round 16 starts fresh)
        if round_num==15:
            ct_consecutive_losses = 0
            t_consecutive_losses = 0

    # Get final scores
    if rounds_data:
        last_round = data["gameRounds"][-1]
        final_ct_score = last_round.get("endCTScore", 0)
        final_t_score = last_round.get("endTScore", 0)
    else:
        final_ct_score = 0
        final_t_score = 0

    # Build match-level metadata
    match_data = {
        "metadata":{
            "match_id":match_id,
            "tournament":tournament_name,
            "tournament_config":TOURNAMENTS[tournament_name],
            "map_name":map_name,
            "ct_team":ct_team_name,
            "t_team":t_team_name,
            "final_score":{
                "ct":final_ct_score,
                "t":final_t_score
            },
            "total_rounds":len(rounds_data),
            "parsed_at":datetime.now().isoformat()
        },
        "rounds":rounds_data
    }

    return match_data


# =============================================================================
# FEATURE ENGINEERING (Post-Processing)
# =============================================================================

def add_lag_features(df: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """
    Add lagged outcome features within each match.

    Args:
        df: DataFrame with rounds
        max_lag: Maximum number of lags to create

    Returns:
        DataFrame with lag features added
    """
    df = df.sort_values(by=['match_id', 'round_num']).copy()

    for lag in range(1, max_lag + 1):
        df[f'ct_won_lag_{lag}'] = df.groupby('match_id')['ct_wins_round'].shift(lag)

    return df


def add_win_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win streak features within each match.
    """
    df = df.sort_values(by=['match_id', 'round_num']).copy()

    # Calculate streak for CT side
    df['ct_streak_reset'] = (df['ct_wins_round']!=df.groupby('match_id')['ct_wins_round'].shift(1)).astype(int)
    df['ct_streak_id'] = df.groupby('match_id')['ct_streak_reset'].cumsum()
    df['ct_win_streak'] = df.groupby(['match_id', 'ct_streak_id']).cumcount()

    # If CT didn't win, streak is negative (T streak)
    df.loc[df['ct_wins_round']==0, 'ct_win_streak'] = -df.loc[df['ct_wins_round']==0, 'ct_win_streak']

    # Clean up temp columns
    df = df.drop(columns=['ct_streak_reset', 'ct_streak_id'])

    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def parse_tournament(tournament_name: str) -> tuple:
    """
    Parse all demos from a single tournament.

    Returns:
        Tuple of (list of match dictionaries, summary stats)
    """
    config = TOURNAMENTS[tournament_name]
    folder_path = DEMO_DIR / config['folder']

    if not folder_path.exists():
        print(f"Tournament folder not found: {folder_path}")
        # Return empty but valid summary
        return [], {
            "tournament":tournament_name,
            "role":config['role'],
            "total_files":0,
            "successful_parses":0,
            "failed_parses":0,
            "failed_files":[],
            "total_rounds":0,
            "error":f"Folder not found: {folder_path}"
        }

    demo_files = list(folder_path.glob("*.dem"))
    print(f"\n{'=' * 60}")
    print(f"Processing {tournament_name} ({config['role'].upper()})")
    print(f"Found {len(demo_files)} demo files in {folder_path}")
    print(f"{'=' * 60}")

    matches = []
    failed = []

    for demo_file in tqdm(demo_files, desc=f"Parsing {tournament_name}"):
        result = parse_single_demo(str(demo_file), tournament_name)
        if result:
            matches.append(result)
        else:
            failed.append(demo_file.name)

    summary = {
        "tournament":tournament_name,
        "role":config['role'],
        "total_files":len(demo_files),
        "successful_parses":len(matches),
        "failed_parses":len(failed),
        "failed_files":failed,
        "total_rounds":sum(len(m['rounds']) for m in matches)
    }

    print(f"\n{tournament_name} complete: {len(matches)} matches, {summary['total_rounds']} rounds")
    if failed:
        print(f"   Failed to parse: {failed[:5]}{'...' if len(failed) > 5 else ''}")

    return matches, summary


def save_tournament_data(matches: list, tournament_name: str, output_dir: Path):
    """
    Save tournament data to JSON files.
    """
    if not matches:
        print(f"  No matches to save for {tournament_name}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual match files
    matches_dir = output_dir / tournament_name / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    for match in matches:
        match_id = match['metadata']['match_id']
        match_file = matches_dir / f"{match_id}.json"
        with open(match_file, 'w') as f:
            json.dump(match, f, indent=2)

    # Save combined rounds DataFrame as JSON
    all_rounds = []
    for match in matches:
        all_rounds.extend(match['rounds'])

    # Convert to DataFrame, add engineered features
    df = pd.DataFrame(all_rounds)
    df = add_lag_features(df)
    df = add_win_streak_features(df)

    # Save as JSON (for consistency)
    combined_file = output_dir / tournament_name / f"{tournament_name.lower()}_rounds.json"
    df.to_json(combined_file, orient='records', indent=2)

    # Also save as CSV for convenience
    csv_file = output_dir / tournament_name / f"{tournament_name.lower()}_rounds.csv"
    df.to_csv(csv_file, index=False)

    print(f"  Saved {len(matches)} match files to {matches_dir}")
    print(f"  Saved combined data to {combined_file}")
    print(f"  Saved CSV to {csv_file}")

    return df


def run_full_pipeline():
    """
    Run the complete parsing pipeline for all tournaments.
    """
    print("\n" + "=" * 70)
    print("CS:GO MAJOR DEMO PARSING PIPELINE")
    print("=" * 70)
    print(f"Demo directory: {DEMO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Tournaments: {list(TOURNAMENTS.keys())}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_dfs = {}

    for tournament_name in TOURNAMENTS:
        matches, summary = parse_tournament(tournament_name)
        all_summaries.append(summary)

        if matches:
            df = save_tournament_data(matches, tournament_name, OUTPUT_DIR)
            if df is not None:
                all_dfs[tournament_name] = df

    # Save pipeline summary
    summary_file = OUTPUT_DIR / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "run_timestamp":datetime.now().isoformat(),
            "tournaments":all_summaries,
            "total_matches":sum(s['successful_parses'] for s in all_summaries),
            "total_rounds":sum(s['total_rounds'] for s in all_summaries)
        }, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n{'Tournament':<15} {'Role':<15} {'Matches':<10} {'Rounds':<10}")
    print("-" * 50)
    for s in all_summaries:
        print(f"{s['tournament']:<15} {s['role']:<15} {s['successful_parses']:<10} {s['total_rounds']:<10}")
    print("-" * 50)
    print(
        f"{'TOTAL':<30} {sum(s['successful_parses'] for s in all_summaries):<10} {sum(s['total_rounds'] for s in all_summaries):<10}")

    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"Summary saved to: {summary_file}")

    return all_dfs


# =============================================================================
# SHAREABLE CSV EXPORT
# =============================================================================

def export_shareable_csvs(dataframes: dict, output_dir: Path):
    """
    Export three clean, shareable CSV files for independent analysis.
    These are standalone files with all features included.
    """
    export_dir = output_dir / "csv_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPORTING SHAREABLE CSVs")
    print("=" * 70)

    # Define export configurations
    exports = {
        "Stockholm":{
            "filename":"stockholm_2021_major_rounds.csv",
            "description":"PGL Major Stockholm 2021 - TRAINING dataset"
        },
        "Antwerp":{
            "filename":"antwerp_2022_major_rounds.csv",
            "description":"PGL Major Antwerp 2022 - CONFIRMATION dataset"
        },
        "Rio":{
            "filename":"rio_2022_major_rounds.csv",
            "description":"IEM Rio Major 2022 - VALIDATION dataset (holdout)"
        }
    }

    exported_files = []

    for tournament_name, df in dataframes.items():
        if tournament_name not in exports:
            continue

        config = exports[tournament_name]
        filepath = export_dir / config['filename']

        # Reorder columns for clean export
        export_columns = [
            # Identifiers
            'match_id', 'tournament', 'map_name', 'round_num',

            # Teams
            'ct_team', 't_team',

            # Scores
            'ct_score', 't_score', 'score_diff',

            # Economic data
            'ct_equip_value', 't_equip_value', 'equip_advantage',
            'ct_spend', 't_spend',
            'ct_buy_type', 't_buy_type',
            'ct_economic_regime', 't_economic_regime',

            # Loss tracking
            'ct_consecutive_losses', 't_consecutive_losses',

            # Round structure
            'half_number', 'round_in_half', 'round_phase',
            'is_pistol', 'is_conversion', 'is_overtime',

            # Map baseline
            'baseline_ct_win_prob',

            # In-round events
            'bomb_planted', 'first_kill_side',

            # Momentum features
            'ct_win_streak',
            'ct_won_lag_1', 'ct_won_lag_2', 'ct_won_lag_3',

            # Outcome (target variable)
            'winning_side', 'ct_wins_round'
        ]

        # Select only columns that exist
        available_columns = [col for col in export_columns if col in df.columns]
        export_df = df[available_columns].copy()

        # Sort for consistency
        export_df = export_df.sort_values(['match_id', 'round_num']).reset_index(drop=True)

        # Save CSV
        export_df.to_csv(filepath, index=False)

        exported_files.append({
            'tournament':tournament_name,
            'filename':config['filename'],
            'filepath':str(filepath),
            'description':config['description'],
            'rows':len(export_df),
            'matches':export_df['match_id'].nunique(),
            'columns':len(available_columns)
        })

        print(f"\n{config['filename']}")
        print(f"   {config['description']}")
        print(
            f"   Rows: {len(export_df):,} | Matches: {export_df['match_id'].nunique()} | Columns: {len(available_columns)}")

    # Create a README for the exports
    readme_content = f"""# CS:GO Major Tournament Round-Level Data

## Dataset Description

This dataset contains round-by-round data from three CS:GO Major tournaments,
extracted from official demo files using the awpy parser.

**Project:** The Economics of Momentum: Evidence from Professional Esports

## Files

| File | Tournament | Role | Matches | Rounds |
|------|------------|------|---------|--------|
| `stockholm_2021_major_rounds.csv` | PGL Major Stockholm 2021 | Training | {exported_files[0]['matches'] if len(exported_files) > 0 else 'N/A'} | {exported_files[0]['rows'] if len(exported_files) > 0 else 'N/A'} |
| `antwerp_2022_major_rounds.csv` | PGL Major Antwerp 2022 | Confirmation | {exported_files[1]['matches'] if len(exported_files) > 1 else 'N/A'} | {exported_files[1]['rows'] if len(exported_files) > 1 else 'N/A'} |
| `rio_2022_major_rounds.csv` | IEM Rio Major 2022 | Validation | {exported_files[2]['matches'] if len(exported_files) > 2 else 'N/A'} | {exported_files[2]['rows'] if len(exported_files) > 2 else 'N/A'} |

## Scope

All datasets include **Legends Stage + Champions Stage** only (top 16 teams).
Challengers Stage is excluded for consistency.

## Column Descriptions

### Identifiers
- `match_id`: Unique match identifier (team1-vs-team2-map format)
- `tournament`: Tournament name
- `map_name`: Map played (de_inferno, de_mirage, etc.)
- `round_num`: Round number (1-30, or 31+ for overtime)

### Teams & Scores
- `ct_team`, `t_team`: Team names
- `ct_score`, `t_score`: Score at START of round
- `score_diff`: CT score minus T score

### Economic Data
- `ct_equip_value`, `t_equip_value`: Equipment value at freeze time end ($)
- `equip_advantage`: CT equipment minus T equipment
- `ct_spend`, `t_spend`: Money spent this round
- `ct_buy_type`, `t_buy_type`: Buy classification (Eco/Force/Full/etc.)
- `ct_economic_regime`, `t_economic_regime`: broke/building/full_buy/flush

### Loss Tracking
- `ct_consecutive_losses`, `t_consecutive_losses`: Current loss streak (resets at halftime)

### Round Structure
- `half_number`: 1 (rounds 1-15), 2 (rounds 16-30), 3 (overtime)
- `round_in_half`: Round number within the half
- `round_phase`: pistol/conversion/gun/overtime
- `is_pistol`, `is_conversion`, `is_overtime`: Boolean flags

### Map Baseline
- `baseline_ct_win_prob`: Historical CT-side win rate for this map

### In-Round Events
- `bomb_planted`: Was the bomb planted this round?
- `first_kill_side`: Which side got the first kill (CT/T)

### Momentum Features
- `ct_win_streak`: Current win streak (negative = T streak)
- `ct_won_lag_1/2/3`: Did CT win 1/2/3 rounds ago? (NaN for early rounds)

### Outcome
- `winning_side`: CT or T
- `ct_wins_round`: 1 if CT won, 0 if T won (target variable)

## Recommended Usage

1. **Training (Stockholm)**: Use for exploratory analysis and model development
2. **Confirmation (Antwerp)**: Test if patterns replicate
3. **Validation (Rio)**: Final holdout test - do not touch until paper is complete

## Citation

If you use this data, please cite:
- Original demo files from HLTV.org
- awpy parser: https://github.com/pnxenopoulos/awpy

## Generated

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    readme_path = export_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"\nREADME saved to: {readme_path}")
    print(f"\nAll shareable CSVs exported to: {export_dir}")

    return exported_files


# =============================================================================
# LOAD FROM CACHE (Skip parsing if CSVs exist)
# =============================================================================

def check_cache_exists() -> bool:
    """Check if all tournament CSVs exist."""
    for tournament_name in TOURNAMENTS:
        csv_path = OUTPUT_DIR / tournament_name / f"{tournament_name.lower()}_rounds.csv"
        if not csv_path.exists():
            return False
    return True


def load_from_cache() -> dict:
    """Load tournament data from existing CSV files."""

    print("\n" + "=" * 70)
    print("LOADING FROM CACHE")
    print("=" * 70)

    dataframes = {}

    for tournament_name, config in TOURNAMENTS.items():
        csv_path = OUTPUT_DIR / tournament_name / f"{tournament_name.lower()}_rounds.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dataframes[tournament_name] = df
            print(f"  {tournament_name}: {len(df)} rounds, {df['match_id'].nunique()} matches")
        else:
            print(f"  {tournament_name}: NOT FOUND at {csv_path}")

    print("\n" + "=" * 70)
    print("CACHE LOAD COMPLETE")
    print("=" * 70)

    total_rounds = sum(len(df) for df in dataframes.values())
    total_matches = sum(df['match_id'].nunique() for df in dataframes.values())
    print(f"\nTotal: {total_matches} matches, {total_rounds} rounds")

    return dataframes


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__=="__main__":

    # Parse command line arguments
    force_parse = "--parse" in sys.argv
    force_load = "--load" in sys.argv

    if force_parse and force_load:
        print("Error: Cannot use both --parse and --load")
        sys.exit(1)

    # Determine mode
    cache_exists = check_cache_exists()

    if force_load:
        if not cache_exists:
            print("Error: --load specified but cache not found. Run without --load first.")
            sys.exit(1)
        dataframes = load_from_cache()

    elif force_parse:
        print("Force parsing all demos (--parse flag)")
        dataframes = run_full_pipeline()
        if dataframes:
            export_shareable_csvs(dataframes, OUTPUT_DIR)

    else:
        # Auto mode: load from cache if exists, otherwise parse
        if cache_exists:
            print("Cache found. Loading from CSV files.")
            print("(Use --parse to force re-parsing demos)")
            dataframes = load_from_cache()
        else:
            print("No cache found. Parsing demo files.")
            dataframes = run_full_pipeline()
            if dataframes:
                export_shareable_csvs(dataframes, OUTPUT_DIR)

    # Quick validation
    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    if not dataframes:
        print("\nNo data was parsed successfully!")
        print("Make sure:")
        print("  1. You renamed 'Stolkhom' folder to 'Stockholm'")
        print("  2. You have awpy 1.2.3 installed (not awpy 2.0+)")
        print("     Run: pip uninstall awpy && pip install awpy==1.2.3")
    else:
        for name, df in dataframes.items():
            print(f"\n{name}:")
            print(f"  Rounds: {len(df)}")
            print(f"  Matches: {df['match_id'].nunique()}")
            print(f"  Maps: {df['map_name'].unique().tolist()}")
            print(f"  Equipment values present: {(df['ct_equip_value'] > 0).sum()} / {len(df)}")
            print(f"  Economic regimes: {df['ct_economic_regime'].value_counts().to_dict()}")
