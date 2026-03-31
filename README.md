# The Economics of Momentum: Evidence from Professional Esports

> Is momentum in competitive gaming a psychological phenomenon — or an economic illusion?

This repository contains the full analysis pipeline for our study decomposing "momentum" in professional CS:GO into its economic and psychological components. We analyze 5,478 rounds across three Major tournaments using a train/confirm/validate design and find that the entire momentum effect (OR ≈ 3.2×) is explained by the in-game economic system.

---

## Key Finding

| Without Economic Controls | With Economic Controls |
|--------------------------|----------------------|
| Winning previous round **triples** odds of winning current round (OR = 2.95, p < 0.001) | Momentum coefficient drops to **zero** (β = −0.001, p = 0.993) |

The loss bonus system creates a mechanical pathway: **winning → income → equipment → more winning**. What looks like psychological momentum is an economic artifact.

---

## Repository Structure

```
├── run_all.py                    # Master pipeline runner
├── 01_pipeline.py                # Phase 1: Demo parsing (.dem → CSV)
├── 02_exploratory.py             # Phase 2: Descriptive stats & visualizations
├── 03_decomposition.py           # Phase 3: Momentum decomposition (training set)
├── 04_pattern_replication.py     # Phase 4: Antwerp replication (confirmation set)
├── 05_pooled_analysis.py         # Phase 5: Pooled analysis (Stockholm + Antwerp)
├── 06_testing_rio.py             # Phase 6: Rio validation (holdout test)
├── 07_behavioral_analysis.py     # Phase 7: Force-buy rationality & expected value
├── 08_clustered_errors.py        # Phase 8: Clustered standard errors robustness
├── 09_hltv_rankings.py           # Phase 9: HLTV ranking controls robustness
│
├── utils/
│   ├── 00_eco_regimes.py         # Economic regime classification logic
│   ├── 00_diagnosis_eco_regimes.py  # Regime threshold diagnostics
│   └── 00_organize.py            # Data organization utilities
│
├── json_output/
│   └── csv_exports/              # Pre-parsed round-level CSVs (see below)
│
├── analysis_output/              # All figures (PNG) and model coefficients (CSV)
│
├── requirements.txt              # Python dependencies
└── LICENSE                       # All rights reserved
```

## Data

### Pre-Parsed Round-Level Data

The `json_output/csv_exports/` directory contains ready-to-use round-level CSVs extracted from official demo files:

| File | Tournament | Role | Matches | Rounds |
|------|------------|------|---------|--------|
| `stockholm_2021_major_rounds.csv` | PGL Major Stockholm 2021 | Training | 65 | 1,733 |
| `antwerp_2022_major_rounds.csv` | PGL Major Antwerp 2022 | Confirmation | 71 | 1,941 |
| `rio_2022_major_rounds.csv` | IEM Rio Major 2022 | Validation | 68 | 1,804 |

Each CSV contains 30+ columns per round including equipment values, buy types, economic regimes, loss streaks, round phase, first kill data, and momentum lag features. See [`json_output/csv_exports/README.md`](json_output/csv_exports/README.md) for full column descriptions.

### Source Data

Original `.dem` demo files are not included due to size (~50 GB). They can be downloaded from [HLTV.org](https://www.hltv.org/) match pages. Phase 1 (`01_pipeline.py`) handles parsing if you have the raw demos.

---

## Quick Start

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/CS_GO-Economics.git
cd CS_GO-Economics
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Run all analysis phases (skips demo parsing if CSVs exist)
python run_all.py

# Start from a specific phase
python run_all.py --from 3

# Run only one phase
python run_all.py --only 7

# Force re-parse demos (requires .dem files + awpy==1.2.3)
python run_all.py --parse
```

The pipeline automatically detects cached CSVs in `json_output/` and skips Phase 1 unless `--parse` is specified.

---

## Pipeline Phases

### Phase 1 — Demo Parsing (`01_pipeline.py`)
Parses raw `.dem` files using [awpy](https://github.com/pnxenopoulos/awpy) v1.2.3. Extracts round-level features including equipment values at freeze time, buy types, economic regime classifications, kill events, and momentum lag variables. Outputs CSV files to `json_output/`.

### Phase 2 — Exploratory Analysis (`02_exploratory.py`)
Stockholm-only. Computes descriptive statistics, CT win rates by map/regime/phase, equipment distributions, and the baseline momentum model (Model A). Generates figures 1–5.

### Phase 3 — Momentum Decomposition (`03_decomposition.py`)
The core analysis. Tests whether momentum survives economic controls using two approaches:
- **Panel A (Categorical):** Regime dummies for both CT and T sides → momentum drops from β = 1.08 to β = −0.001
- **Panel B (Continuous):** Equipment advantage in $10k → momentum reverses to β = −0.28

Also runs robustness checks on the full-buy subsample and gun-rounds-only subset. Generates figures 6–7.

### Phase 4 — Antwerp Replication (`04_pattern_replication.py`)
Applies Stockholm model specifications to Antwerp without modification. Tests independent replication of the decomposition result. Generates figure 8.

### Phase 5 — Pooled Analysis (`05_pooled_analysis.py`)
Combines Stockholm + Antwerp for higher-powered estimation. Tests tournament interaction effects and pooled model stability. Generates figure 9.

### Phase 6 — Rio Validation (`06_testing_rio.py`)
Final holdout test on the Rio dataset. This data was not examined until all model specifications were finalized. Generates figure 10.

### Phase 7 — Behavioral Analysis (`07_behavioral_analysis.py`)
Tests whether force-buying after losses is irrational "tilt" or strategically optimal. Computes two-round expected values with bootstrap confidence intervals (2,000 iterations). Finding: Force EV = 0.829 vs. Eco EV = 0.612, P(Force > Eco) = 100%. Generates figure 12.

### Phase 8 — Clustered Standard Errors (`08_clustered_errors.py`)
Re-estimates all full-sample models with standard errors clustered by match (204 clusters). Reports both regular and clustered SEs.

### Phase 9 — HLTV Rankings (`09_hltv_rankings.py`)
Adds HLTV world ranking difference as a team skill control. Tests whether momentum results are confounded by team quality. Finding: rank coefficient is never significant (all p > 0.28), and momentum results are unchanged.

---

## Output

### Figures

| Figure | File | Description |
|--------|------|-------------|
| 1 | `fig_01_win_rate_by_equipment.png` | CT win rate by equipment advantage bin |
| 2 | `fig_02_regime_transitions.png` | Economic regime transition heatmap |
| 3 | `fig_03_phase_regime_winrate.png` | Win rate by round phase and regime |
| 4 | `fig_04_momentum_effect.png` | Raw momentum effect visualization |
| 5 | `fig_05_equipment_by_regime.png` | Equipment value distributions by regime |
| 6 | `fig_06_momentum_decomposition.png` | Panel A vs Panel B decomposition |
| 7 | `fig_07_odds_ratio_comparison.png` | Odds ratio across all specifications |
| 8 | `fig_08_antwerp_replication.png` | Cross-tournament replication |
| 9 | `fig_09_pooled_analysis.png` | Pooled analysis results |
| 10 | `fig_10_rio_validation.png` | Rio holdout validation |
| 12 | `fig_12_behavioral_ev_analysis.png` | Force-buy expected value analysis |

### Model Coefficients (CSV)

All model coefficients are saved to `analysis_output/` with columns: `variable`, `coefficient`, `std_error`, `z_value`, `p_value`, `odds_ratio`.

| File | Description |
|------|-------------|
| `model_baseline_coefficients.csv` | Baseline momentum-only model |
| `model_A1_coefficients.csv` | + Regime dummies |
| `model_A2_coefficients.csv` | + Regime + Maps |
| `model_A3_coefficients.csv` | + Regime + Maps + Phase |
| `model_B1_coefficients.csv` | + Equipment ($10k) |
| `model_B2_coefficients.csv` | + Equipment + Maps |
| `model_B3_coefficients.csv` | + Equipment + Maps + Phase |
| `model_B4_coefficients.csv` | + Equipment + Maps + Phase + First Kill |
| `model_full_with_rank_coefficients.csv` | Full model + HLTV rank control |

### Summary Tables (CSV)

| File | Description |
|------|-------------|
| `phase3_model_comparison.csv` | All Stockholm models side-by-side |
| `phase4_replication_comparison.csv` | Stockholm vs. Antwerp comparison |
| `phase5_pooled_results.csv` | Pooled (Stockholm + Antwerp) results |
| `phase6_full_sample_results.csv` | Full sample (all 3 tournaments) |
| `phase6_rio_validation.csv` | Rio holdout test results |
| `phase7_behavioral_consolidated.csv` | Force-buy rationality results |
| `phase8_clustered_se_results.csv` | Regular vs. clustered SEs |
| `phase9_rank_controlled_results.csv` | HLTV rank-controlled models |

---

## Methodology

### Train / Confirm / Validate Design

```
Stockholm 2021 ──── Training set: all model development
        │
Antwerp 2022  ──── Confirmation set: independent replication
        │
Rio 2022      ──── Validation set: final holdout (not examined until specifications finalized)
```

### Model Sequence

```
Baseline (lag-1 only)
    │
    ├── Panel A: + Regime dummies (categorical economic controls)
    │       └── + Maps → + Phase
    │
    └── Panel B: + Equipment advantage (continuous economic control)
            └── + Maps → + Phase → + First Kill (endogenous)
```

---

## License

All rights reserved