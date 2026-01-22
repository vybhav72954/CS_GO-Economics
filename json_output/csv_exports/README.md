# CS:GO Major Tournament Round-Level Data

## Dataset Description

This dataset contains round-by-round data from three CS:GO Major tournaments,
extracted from official demo files using the awpy parser.

**Project:** The Economics of Momentum: Evidence from Professional Esports

## Files

| File | Tournament | Role | Matches | Rounds |
|------|------------|------|---------|--------|
| `stockholm_2021_major_rounds.csv` | PGL Major Stockholm 2021 | Training | 65 | 1733 |
| `antwerp_2022_major_rounds.csv` | PGL Major Antwerp 2022 | Confirmation | 71 | 1941 |
| `rio_2022_major_rounds.csv` | IEM Rio Major 2022 | Validation | 68 | 1804 |

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
