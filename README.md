# NFL Prediction Model

An NFL prediction model focused on ATS (against the spread) prediction. Uses play-by-play data from nflverse (2006–2025) via `nfl_data_py`, with scikit-learn for modeling. Train set: 2006–2021, test set: 2022–2025.

## Model Leaderboard

All models evaluated on home-team perspective only (one row per game), week 4+, test set 2022–2025. Break-even at standard −110 juice is 52.38%.

All models evaluated on home-team perspective only (one row per game), weeks 4–16/17, test set 2022–2025. Final week of each season is excluded (week 18 from 2021+, week 17 pre-2021) — resting starters and meaningless games break model signal.

| Rank | Model | ATS | N | Units | E>3 ATS | E>3 N | E>3 Units | Train |
|------|-------|-----|---|-------|---------|-------|-----------|-------|
| 1 | weighted+raw + pd + qbr | **55.0%** | 687 | **+34.6** | 52.0% | 346 | −2.4 | 2016–2021 |
| 2 | off_11/12 + def + pd + qbr | 54.7% | 687 | +30.8 | 54.0% | 337 | +10.5 | 2016–2021 |
| 3 | `off_formation_point_diff_qbr_model.py` | 54.6% | 687 | +28.9 | 53.2% | 348 | +5.2 | 2016–2021 |
| 4 | off_11/12 + def + pd + qbr + cpoe | 54.4% | 687 | +27.0 | **55.1%** | 334 | **+17.3** | 2016–2021 |
| 5 | off_11/12 + def + pd | 53.8% | 796 | +21.1 | 53.3% | 398 | +6.7 | 2016–2021 |
| 6 | off_11/12_w5 + pd + qbr | 53.4% | 697 | +13.2 | 54.2% | 334 | +11.5 | 2016–2021 |
| 7 | off_11/12 + def + cpoe | 53.3% | 796 | +13.5 | 53.0% | 419 | +4.8 | 2016–2021 |
| 8 | `point_diff_cpoe_qbr_model.py` | 52.7% | 698 | +4.5 | 52.9% | 340 | +3.6 | 2006–2021 |
| 9 | first_down + pd + qbr | 52.4% | 698 | +0.7 | **55.1%** | 334 | **+17.3** | 2016–2021 |
| 10 | off_11/12 + def | 52.4% | 796 | +0.1 | 52.7% | 438 | +3.0 | 2016–2021 |

Units at −110 juice: `wins × (1/1.1) − losses`. **E>3** = games where `|predicted_margin − spread_line| ≥ 3`.

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

## Diagnostic Findings

### Where the model wins and loses (best model, test 2022–2025)

**Model vs line agreement (ATS pick direction):**

| Situation | N | ATS | Units |
|-----------|---|-----|-------|
| Model bets same ATS side as line | 159 | **62.3%** | **+30.0** |
| Model fades line (bets opposite ATS side) | 528 | 52.8% | +4.6 |

Within fades, there are two very different sub-groups:

| Fade type | N | ATS | Units |
|-----------|---|-----|-------|
| Same outright winner, model under the spread (e.g. line −7, model −5 → bet underdog) | 338 | 55.0% | +17.1 |
| Model picks a **different outright winner** than the line | 190 | 48.9% | −12.5 |

The model loses money specifically when it disagrees with the line on *who wins outright* — those contrarian picks are wrong more than right. Games where the model agrees on the winner but thinks the spread is too large are still profitable. **Untested idea: filter out games where `sign(predicted_margin) ≠ sign(spread_line)`.**

**By spread size:**

| Spread | N | ATS | Units |
|--------|---|-----|-------|
| Big dog >7 | 42 | 61.9% | +7.6 |
| Dog 3–7 | 104 | 58.7% | +12.5 |
| Dog <3 | 131 | 51.1% | −3.1 |
| Fav <3 | 91 | 59.3% | +12.1 |
| Fav 3–7 | 197 | 51.8% | −2.3 |
| Big fav >7 | 122 | 55.7% | +7.8 |

Near pick-em games (spread within 3 in either direction) are net losers — the model has less signal when the line is tight. **Untested idea: filter out spreads between −3 and +3.**

**By season:**

| Season | N | ATS | Units |
|--------|---|-----|-------|
| 2022 | 189 | 57.7% | +19.1 |
| 2023 | 177 | 53.1% | +2.5 |
| 2024 | 174 | 53.4% | +3.5 |
| 2025 | 147 | 55.8% | +9.5 |

**Worst teams (by units lost, test 2022–2025):** NE −9.5u (28.6% ATS), CIN −7.5u (31.6%), LV −6.5u (31.2%). These teams appear to have drifted significantly from their historical profiles in the training data.

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
