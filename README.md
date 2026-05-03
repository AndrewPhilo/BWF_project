# BWF World Tour Match Outcome Analysis

A data science project investigating whether recent match load affects a player's probability of winning their next match on the BWF (Badminton World Federation) World Tour.

---

## Research Question

> Does the volume and intensity of matches played in the weeks before a tournament affect a player's probability of winning their next match on the BWF World Tour?

---

## Project Structure

```
BWF_project/
├── Data Science Professional Practice Summative Assessment.ipynb  # Main analysis notebook
├── requirements.txt                                               # Python dependencies
├── README.md                                                      # This file
├── data/
│   ├── ms.csv              # Raw men's singles match data (2018–2021)
│   ├── data_dictionary.csv # Column definitions for the raw dataset
│   ├── names.csv           # Player name reference table
│   ├── unique_rounds.csv   # Unique tournament round structures
│   └── cs_df_1.csv         # Intermediate cross-sectional player-level table
└── bwf_env/                # Python virtual environment
```

---

## Dataset

- **Source:** BWF World Tour men's singles matches
- **Coverage:** 2018–2021
- **Raw size:** 3,761 matches
- **Processed size:** 7,522 player-level rows (one row per player per match)

The raw dataset (`data/ms.csv`) contains match-level records including player names, nationalities, tournament names, tournament rounds, game-by-game scores, and match outcomes.

---

## Methodology

The analysis follows a structured pipeline across 12 sections in the notebook:

### 1. Setup
Imports all required libraries for data processing, feature engineering, visualisation, and modelling.

### 2. Load Raw Match Data
Reads the source CSV file containing match-level information, player identifiers, scores, and round data.

### 3. Clean and Transform Match Data
- **Round ranking:** Tournament round names (e.g., `Round of 16`, `Semi final`, `Final`) are mapped to a consistent ordinal integer rank across six distinct tournament formats to allow cross-tournament comparison.
- **Retired match resolution:** Matches where the outcome field is `0` (retirement/walkover) are resolved by tracing which player appeared in the next round of the same tournament and updating the winner and retired fields accordingly.

### 4. Feature Engineering — Iteration 1: Match Intensity Features
33 new features are derived from the point-by-point score progressions stored in each game's score column. For each game (1, 2, 3) and at the match aggregate level:

| Feature group | Variables created |
|---|---|
| Closeness | `closeness`, `closeness_ratio`, `avg_score_diff`, `std_score_diff`, `final_score_diff` |
| Momentum | `momentum_swings`, `lead_changes` |
| Lead extremes | `max_lead_team1`, `max_lead_team2` |
| Match-level aggregates | `match_avg_closeness`, `match_total_momentum_swings`, `match_total_lead_changes`, `match_avg_score_diff`, `match_close_games_count`, `match_max_momentum_swings` |

### 5. Reshape to a Player-Level Modelling Table
The match-level table (one row per match) is converted to a cross-sectional table with one row per player per match. This assigns each player a binary `winner` target (1 = won, 0 = lost) and attaches all match-level features from their perspective.

### 6. Remove Columns Not Needed for Modelling
Identifier columns, raw score strings, game score fields, and intermediate derived columns are dropped to produce a clean modelling table.

### 7. Feature Engineering — Iteration 2: Pre-Match Workload Features
Rolling lookback features are computed for each player row using only matches that occurred **before** the current match date (no data leakage). A 21-day lookback window is used.

| Feature | Description |
|---|---|
| `matches_last_3_weeks` | Number of matches played in the 21 days before this match |
| `consecutive_matches_played` | Length of the current run of back-to-back matches (max gap 3 days) |
| `avg_total_points_recent` | Average total points played per match in the lookback window |
| `avg_closeness_recent` | Average match closeness score in the lookback window |
| `avg_nb_sets_recent` | Average number of sets played per match in the lookback window |

### 8. Screen Candidate Predictors
A full correlation heatmap is produced to show the association of every numeric column with the binary `winner` target, guiding feature selection.

### 9. Modelling — Iteration 1: Baseline Logistic Regression
A scikit-learn `Pipeline` with median imputation, standard scaling, and logistic regression is fitted on the six pre-match features. The model is evaluated on a held-out 20% test split with stratification.

### 10. Modelling — Iteration 2: Balanced Logistic Regression
The same pipeline is repeated with `class_weight='balanced'` to correct for the slight class imbalance and test whether recall for winners improves.

### 11. Validation and Player Spot Checks
A substring search function allows individual player records to be inspected in the processed table, used to sanity-check whether feature values are plausible for known players.

### 12. Summary of Findings
A structured summary of correlation findings, model results, residual analysis, and conclusions.

---

## Results

### Correlation with match outcome

| Feature | Pearson r with `winner` |
|---|---|
| `matches_last_3_weeks` | +0.07 |
| `consecutive_matches_played` | +0.04 |
| `avg_closeness_recent` | +0.03 |
| `avg_nb_sets_recent` | +0.03 |
| `avg_total_points_recent` | +0.03 |

All pre-match workload features show very weak correlations with outcome.

### Model performance

| Model | Accuracy | ROC AUC | Recall (winners) |
|---|---|---|---|
| Baseline logistic regression | 55.8% | 0.521 | 26% |
| Balanced logistic regression | 52.6% | 0.521 | 40% |

The balanced model improves recall for winners by 14 percentage points at the cost of 3.2 percentage points of overall accuracy. The ROC AUC is identical (0.521, near chance), confirming the improvement is a decision-threshold shift rather than a gain in signal.

### Residuals

Both models produce the expected bimodal residual distribution for a near-balanced binary target. No systematic directional bias is detected in either model — the issue is feature weakness rather than model misspecification.

---

## Conclusion

Recent match load, on its own, has very limited predictive power over match outcomes on the BWF World Tour. A ROC AUC of 0.521 for both models indicates the selected pre-match features carry almost no discriminatory information beyond chance.

This is a substantive finding: elite players appear able to maintain consistent performance regardless of recent match volume, or recovery and scheduling practices at the World Tour level effectively mitigate fatigue effects.

### Suggested next steps

1. Add player ranking or seeding as a predictor — likely to explain far more variance than load alone.
2. Include days since last match as a direct measure of rest.
3. Test non-linear models (Random Forest, XGBoost) to capture interaction effects between load variables.
4. Investigate sub-groups such as younger versus more experienced players who may respond differently to match load.

---

## Requirements

Python 3.14 is used with a local virtual environment (`bwf_env/`). All dependencies are pinned in `requirements.txt`.

Key packages:

| Package | Version | Purpose |
|---|---|---|
| pandas | 3.0.1 | Data manipulation |
| numpy | 2.4.3 | Numerical operations |
| scikit-learn | 1.8.0 | Modelling and preprocessing |
| matplotlib | 3.10.9 | Visualisation |
| seaborn | 0.13.2 | Statistical plots |
| scipy | 1.17.1 | Statistical utilities |
| ipykernel | 7.2.0 | Jupyter kernel |

### Installation

```bash
# Create and activate the virtual environment
python -m venv bwf_env
bwf_env\Scripts\activate      # Windows
# source bwf_env/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the notebook

Open `Data Science Professional Practice Summative Assessment.ipynb` in VS Code or Jupyter and select the `bwf_env` kernel. Run all cells in order from top to bottom.

---

## Academic Context

This project was produced as part of the **Data Science Professional Practice** summative assessment for the BPP Data Science programme.
