# NFL Prediction Model

An NFL prediction model focused on ATS (against the spread) prediction. Uses play-by-play data from nflverse (2006–2025) via `nfl_data_py`, with scikit-learn for modeling. Train set: 2006–2021, test set: 2022–2025.

## Model Leaderboard

All models evaluated on home-team perspective only (one row per game), week 4+, test set 2022–2025. Break-even at standard −110 juice is 52.38%.

All models evaluated on home-team perspective only (one row per game), weeks 4–16/17, test set 2022–2025. Final week of each season is excluded (week 18 from 2021+, week 17 pre-2021) — resting starters and meaningless games break model signal.

| Rank | Model | Features | ATS | n | E>3 ATS | E>3 n | Train |
|------|-------|----------|-----|---|---------|-------|-------|
| 1 | weighted+raw + pd + qbr | off_11/12_epa_rolling + off_11/12_weighted_epa_rolling, point_diff_rolling, qbr_rolling | **55.0%** | 687 | 52.0% | 346 | 2016–2021 |
| 2 | off_11/12 + def_vs_11/12 + pd + qbr | off_11/12_epa_rolling, def_vs_11/12_epa_rolling, point_diff_rolling, qbr_rolling | 54.7% | 687 | 54.0% | 337 | 2016–2021 |
| 3 | `off_formation_point_diff_qbr_model.py` | off_11/12_epa_rolling, point_diff_rolling, qbr_rolling | 54.6% | 687 | 53.2% | 348 | 2016–2021 |
| 4 | off_11/12 + def_vs_11/12 + pd + qbr + cpoe | off_11/12_epa_rolling, def_vs_11/12_epa_rolling, point_diff_rolling, qbr_rolling, cpoe_rolling | 54.4% | 687 | **55.1%** | 334 | 2016–2021 |
| 5 | off_11/12 + def_vs_11/12 + pd | off_11/12_epa_rolling, def_vs_11/12_epa_rolling, point_diff_rolling | 53.8% | 796 | 53.3% | 398 | 2016–2021 |
| 6 | off_11/12_w5 + pd + qbr | off_11/12_epa_rolling_w5, point_diff_rolling, qbr_rolling | 53.4% | 697 | 54.2% | 334 | 2016–2021 |
| 7 | off_11/12 + def_vs_11/12 + cpoe | off_11/12_epa_rolling, def_vs_11/12_epa_rolling, cpoe_rolling | 53.3% | 796 | 53.0% | 419 | 2016–2021 |
| 8 | `point_diff_cpoe_qbr_model.py` | point_diff_rolling, cpoe_rolling, qbr_rolling | 52.7% | 698 | 52.9% | 340 | 2006–2021 |
| 9 | first_down_rate + pd + qbr | first_down_rate_rolling, point_diff_rolling, qbr_rolling | 52.4% | 698 | **55.1%** | 334 | 2016–2021 |
| 10 | off_11/12 + def_vs_11/12 | off_11/12_epa_rolling, def_vs_11/12_epa_rolling | 52.4% | 796 | 52.7% | 438 | 2016–2021 |

**E>3** = games where `|predicted_margin − spread_line| ≥ 3` (high-conviction bets only).

### ATS by Week (best model, test 2022–2025)

| Weeks | ATS | n | E>3 ATS | E>3 n |
|-------|-----|---|---------|-------|
| Wk 4–6 | 53.3% | 137 | 51.4% | 74 |
| Wk 7–9 | 50.4% | 139 | 51.6% | 64 |
| Wk 10–12 | 55.5% | 146 | 52.3% | 65 |
| Wk 13–15 | 55.8% | 154 | 53.6% | 84 |
| Wk 16–17 | **61.3%** | 111 | 50.8% | 59 |

**Key insights:**
- Model improves meaningfully mid-to-late season — weeks 10–17 are substantially stronger than weeks 4–9
- Weeks 16–17 hit 61.3% — rolling averages are most informative once teams have 10+ games of data
- Final week excluded: week 18 (2021+) and week 17 (pre-2021) destroyed signal due to resting starters
- Offensive formation EPA (11/12 personnel) is the strongest signal — captures scheme efficiency beyond raw EPA
- 3-game rolling window consistently beats expanding or 5-game windows for overall ATS
- Personnel data available from 2016 onward, limiting the training window for formation models

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
