# NFL Prediction Model

An NFL prediction model focused on ATS (against the spread) prediction. Uses play-by-play data from nflverse (2006–2025) via `nfl_data_py`, with scikit-learn for modeling. Train set: 2006–2021, test set: 2022–2025.

## Game Filters

All models are evaluated on the home-team perspective only (one row per game), test set 2022–2025. The following games are excluded from both train and test sets:

- **Weeks 1–3**: Dropped — rolling windows require at least 3 prior games; early-season features are too noisy.
- **Final week of each season**: Dropped — week 18 (2021+) and week 17 (pre-2021). Resting starters and meaningless games destroy signal.
- **Backup QB starts (injury or benching)**: Dropped — rolling EPA features are built from the starter's history. When a backup plays due to injury or benching, those features no longer represent the team on the field. See `exports/backup_qb_games.csv` for the full list (categorized as injury, benched, controversy, returning, or rest). Only injury and benched are excluded; controversy and returning QB situations remain in the dataset and are partially addressed via QB-specific rolling features.

Break-even at standard −110 juice is **52.38%**. Units at −110 juice: `wins × (1/1.1) − losses`. **E>3** = games where `|predicted_margin − spread_line| ≥ 3`.

## Model Leaderboard

Three model types have been tested: **Team** (team-level rolling EPA only), **QB** (QB-specific rolling EPA that follows the QB across teams and seasons), and **Combined** (both team and QB features together).

### Top 10 by Overall ATS

Includes playoff games (WC, DIV, CON, SB). Train 2016–2021, test 2022–2025.

| Rank | Model | N | ATS | Units | E>3 ATS | E>3 N | E>3 Units |
|------|-------|---|-----|-------|---------|-------|-----------|
| 1 | off_11/12 + qb_off + def + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 558 | **56.5%** | **+43.4** | 55.6% | 286 | +18.2 |
| 2 | off_11/12 + qb_off + def + pd + qbr + cpoe + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `cpoe_rolling`, `def_vs_12_epa_rolling`</sub> | 558 | 56.5% | +43.4 | **58.1%** | 277 | **+31.0** |
| 3 | off_11/12 + qb_off + def + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 558 | 56.3% | +41.5 | 57.2% | 275 | +26.2 |
| 4 | off_11/12 + qb_off + pd + qbr + TO<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `turnovers_committed_rolling`</sub> | 558 | 56.1% | +39.5 | 59.0% | 289 | +37.2 |
| 5 | off_11/12 + qb_off + def + pd + qbr + cpoe<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `cpoe_rolling`</sub> | 558 | 55.7% | +35.7 | 57.0% | 279 | +25.3 |
| 6 | off_11/12 + qb_off + def + pd + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `def_vs_12_epa_rolling`</sub> | 558 | 54.9% | +30.8 | 56.3% | 275 | +23.7 |
| 7 | off_11/12 + qb_off + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 558 | 54.8% | +26.2 | 58.3% | 282 | +32.0 |
| 8 | off_11/12 + qb_off + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 558 | 54.7% | +24.3 | 55.5% | 284 | +17.3 |
| 9 | off_11/12 + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 573 | 54.5% | +22.6 | 55.2% | 285 | +15.5 |
| 10 | off_11/12 + def + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 573 | 53.4% | +11.2 | 55.9% | 284 | +19.4 |

### Top 10 by E>3 ATS (selective high-confidence bets)

| Rank | Model | N | ATS | Units | E>3 ATS | E>3 N | E>3 Units |
|------|-------|---|-----|-------|---------|-------|-----------|
| 1 | off_11/12 + qb_off + pd + qbr + TO<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `turnovers_committed_rolling`</sub> | 558 | 56.1% | +39.5 | **59.0%** | 289 | **+37.2** |
| 2 | off_11/12 + qb_off + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 558 | 54.8% | +26.2 | 58.3% | 282 | +32.0 |
| 3 | off_11/12 + qb_off + def + pd + qbr + cpoe + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `cpoe_rolling`, `def_vs_12_epa_rolling`</sub> | 558 | 56.5% | +43.4 | 58.1% | 277 | +31.0 |
| 4 | off_11/12 + qb_off + def + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 558 | 56.3% | +41.5 | 57.2% | 275 | +26.2 |
| 5 | off_11/12 + qb_off + def + pd + qbr + cpoe<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `cpoe_rolling`</sub> | 558 | 55.7% | +35.7 | 57.0% | 279 | +25.3 |
| 6 | off_11/12 + qb_off + def + pd + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `def_vs_12_epa_rolling`</sub> | 558 | 54.9% | +30.8 | 56.3% | 275 | +23.7 |
| 7 | off_11/12 + qb_off + def + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 558 | 56.5% | +43.4 | 55.6% | 286 | +18.2 |
| 8 | off_11/12 + qb_off + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 558 | 54.7% | +24.3 | 55.5% | 284 | +17.3 |
| 9 | off_11/12 + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 573 | 54.5% | +22.6 | 55.2% | 285 | +15.5 |
| 10 | off_11/12 + def + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 573 | 53.4% | +11.2 | 55.9% | 284 | +19.4 |

**Notable:** `off_11/12 + qb_off + pd + qbr + TO` is the new E>3 leader at 59.0% / +37.2u with playoffs included — offensive turnover rate adds stable signal beyond what point differential alone captures. `def_vs_12_epa_rolling` holds three of the top four E>3 spots. Adding playoff games consistently improved all metrics across the board.

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
