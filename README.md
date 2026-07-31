# NFL Prediction Model

An NFL prediction model focused on ATS (against the spread) prediction. Uses play-by-play data from nflverse (2006–2025) via `nfl_data_py`, with scikit-learn for modeling. Train set: 2006–2021, test set: 2022–2025.

## Model Leaderboard

All models evaluated on home-team perspective only (one row per game), week 4+, test set 2022–2025.

| Rank | Model | Features | ATS | n | Train Window |
|------|-------|----------|-----|---|--------------|
| 1 | `epa_formation_first_down_model.py` | Formation matchup EPA (nickel/base/11/12) + off 1st down EPA | 52.5% | 871 | 2016–2021 |
| 2 | Formation + CPOE | Formation matchup EPA (nickel/base/11/12) + CPOE | 52.6%* | 871 | 2016–2021 |
| 3 | `epa_qbr_first_down_pace_model.py` | EPA + QBR + off 1st down EPA + pace | 51.8% | 757 | 2006–2021 |
| 4 | EPA + 1st down + pace + QBR | EPA + off/def 1st down EPA + pace + QBR | 51.9% | 757 | 2006–2021 |
| 5 | CPOE only | CPOE rolling | 51.3% | 872 | 2006–2021 |
| 6 | EPA + QBR | EPA + QBR rolling | 50.9% | 757 | 2006–2021 |
| 7 | EPA + TOP | EPA + time of possession | 50.8% | 872 | 2006–2021 |
| 8 | EPA + rest | EPA + rest days | 50.6% | 872 | 2006–2021 |
| 9 | EPA + 1st down + pace | EPA + off/def 1st down EPA + pace | 50.6% | 872 | 2006–2021 |
| 10 | `epa_model.py` | EPA only (baseline) | 50.3% | 872 | 2006–2021 |

*Formation + CPOE not yet saved as a formal model file.

**Key insight**: Formation matchup features (how a team's offense performs against specific defensive packages, and vice versa) provide the strongest signal beyond raw EPA. Personnel data is only available from 2016 onward, which limits the training window for these models.

## Project Structure

```
betting-models/
├── requirements.txt
├── config.py                  # Seasons, feature lists, model params
├── data/
│   └── (auto-downloaded cached data)
├── src/
│   ├── data_loader.py         # Fetch & cache nflverse data
│   ├── feature_engineering.py # Build team-level features per game
│   ├── model.py               # Train/evaluate/predict
│   └── predict.py             # Generate predictions for upcoming games
├── notebooks/
│   └── exploration.ipynb      # EDA and model analysis
└── tests/
    └── test_features.py
```

## How It Works

### Data Pipeline
1. **Data Loader** — pulls play-by-play and schedule data from nflverse, caches locally
2. **Feature Engineering** — builds rolling team-level stats computed *before* each game (no data leakage)
3. **Model** — Ridge regression trained via walk-forward validation
4. **Predictions** — generates predictions for upcoming games and compares against betting lines

### Features

The core features are rolling averages of EPA (Expected Points Added per play) from each team's **previous 3 games**. EPA values come pre-calculated from nflverse and are averaged at the game level, then rolled — so each game's features reflect raw per-game EPA averages, not averages of averages.

- **off_epa_rolling** — team's offensive EPA/play, avg of prior 3 games
- **def_epa_rolling** — team's defensive EPA/play allowed, avg of prior 3 games
- **opp_off_epa_rolling** — opponent's offensive EPA/play, avg of their prior 3 games
- **opp_def_epa_rolling** — opponent's defensive EPA/play allowed, avg of their prior 3 games

For early-season games the window is smaller: week 2 uses 1 prior game, week 3 uses 2, week 4+ uses 3. The window resets at the start of each season.

**Contextual:**
- Home/away indicator

### Model Design
- **Predicts each team's score independently** — spread and total derived from the same prediction
- **Walk-forward validation** — trains on past seasons, validates on future (never peeks at future data)
- **Evaluation**: MAE on scores, ATS (against the spread) accuracy, O/U accuracy

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Download and cache data
python -m src.data_loader

# Train model and evaluate
python -m src.model

# Predict upcoming games
python -m src.predict
```

## Key Design Decisions
1. **Predict team scores, not margins** — gives us both spread and total from one model
2. **Rolling windows, not season averages** — captures recent form, avoids early-season noise
3. **Ridge regression first** — interpretable baseline; can upgrade to XGBoost later
4. **Walk-forward validation** — simulates real betting by never training on future data
5. **EPA-based features** — more predictive than raw box score stats per NFL analytics research
