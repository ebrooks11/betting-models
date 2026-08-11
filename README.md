# NFL Prediction Model

An NFL ATS (against the spread) prediction model using play-by-play data from nflverse (2016–2025) via `nfl_data_py`, with scikit-learn for modeling. Train set: 2016–2022, test set: 2023–2025.

## Game Filters

All models are evaluated on the home-team perspective (one row per game). The following games are excluded from both train and test sets:

- **Weeks 1–3**: Dropped — rolling windows require at least 3 prior games; early-season features are too noisy.
- **Final week of each season**: Dropped — week 18 (2021+) and week 17 (pre-2021). Resting starters and meaningless games destroy signal.
- **Backup QB starts (injury or benching)**: Dropped — rolling EPA features are built from the starter's history. When a backup plays, those features no longer represent the team on the field. See `exports/backup_qb_games.csv` for the full list. Only injury and benched starts are excluded; controversy and returning QB situations remain in the dataset.

Break-even at standard −110 juice is **52.38%**. Units at −110 juice: `wins / 1.1 − losses`. **E>3** = games where `|predicted_margin − spread_line| ≥ 3`.

## Model Leaderboard

Three model types have been tested: **Team** (team-level rolling EPA only), **QB** (QB-specific rolling EPA that follows the QB across teams and seasons), and **Combined** (both team and QB features together).

### Top 10 by Overall ATS

Includes playoff games (WC, DIV, CON, SB). Train 2016–2022, test 2023–2025.

| Rank | Model | N | ATS | Units | E>3 ATS | E>3 N | E>3 Units |
|------|-------|---|-----|-------|---------|-------|-----------|
| 1 | off_11/12 + qb_off + def + cpoe<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `cpoe_rolling`</sub> | 492 | **53.7%** | **+12.0** | 52.0% | 275 | -2.0 |
| 2 | off_11/12 + qb_off + def + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 488 | 53.3% | +8.4 | 53.8% | 253 | +6.6 |
| 3 | off_11/12 + qb_off + def + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 488 | 53.1% | +6.5 | 52.9% | 255 | +2.7 |
| 4 | off_11/12 + qb_off + def + pd<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`</sub> | 489 | 53.0% | +5.5 | 54.2% | 251 | +8.6 |
| 5 | off_11/12 + def_vs_11/12 + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `def_vs_11_epa_rolling`, `def_vs_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 504 | 52.8% | +3.8 | 51.0% | 259 | -7.0 |
| 6 | off_11/12 + qb_off + pd + qbr + TO<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `turnovers_committed_rolling`</sub> | 488 | 52.7% | +2.6 | 54.8% | 261 | +12.0 |
| 7 | off_11/12 + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 504 | 52.6% | +1.9 | 50.0% | 258 | -11.7 |
| 8 | off_11/12 + qb_off + def + pd + qbr + cpoe<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `cpoe_rolling`</sub> | 488 | 52.5% | +0.7 | 53.3% | 242 | +4.3 |
| 9 | off_11/12 + qb_off + def + pd + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `def_vs_12_epa_rolling`</sub> | 489 | 52.4% | -0.3 | 54.2% | 240 | +8.2 |
| 10 | off_11/12 + def + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 504 | 52.2% | -1.9 | 50.2% | 265 | -11.1 |

### Top 10 by E>3 ATS (high-confidence bets)

| Rank | Model | N | ATS | Units | E>3 ATS | E>3 N | E>3 Units |
|------|-------|---|-----|-------|---------|-------|-----------|
| 1 | qb_off + def + pd<br><sub>`qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`</sub> | 489 | 51.5% | -7.9 | **55.8%** | 260 | **+16.8** |
| 2 | off_11/12 + qb_off + pd + qbr + TO<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `turnovers_committed_rolling`</sub> | 488 | 52.7% | +2.6 | 54.8% | 261 | +12.0 |
| 3 | off_11/12 + qb_off + def + pd + qbr + cpoe + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `cpoe_rolling`, `def_vs_12_epa_rolling`</sub> | 488 | 52.0% | -3.1 | 54.8% | 241 | +11.0 |
| 4 | off_11/12 + qb_off + def + pd<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`</sub> | 489 | 53.0% | +5.5 | 54.2% | 251 | +8.6 |
| 5 | off_11/12 + qb_off + def + pd + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `def_vs_12_epa_rolling`</sub> | 489 | 52.4% | -0.3 | 54.2% | 240 | +8.2 |
| 6 | off_11/12 + qb_off + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 488 | 51.4% | -8.8 | 54.2% | 253 | +8.5 |
| 7 | qb_weighted_raw + pd + qbr<br><sub>`qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `qb_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 488 | 50.6% | -16.5 | 54.2% | 253 | +8.5 |
| 8 | qb_first_down + pd + qbr<br><sub>`qb_epa_rolling`, `def_epa_first_down_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 507 | 50.7% | -16.4 | 54.0% | 261 | +8.2 |
| 9 | off_11/12 + qb_off + def + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 488 | 53.3% | +8.4 | 53.8% | 253 | +6.6 |
| 10 | off_11/12 (w5) + qb_off + pd + qbr<br><sub>`off_11_epa_rolling_w5`, `off_12_epa_rolling_w5`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 488 | 51.2% | -10.7 | 53.4% | 232 | +4.7 |

### ATS by Week (best model, test 2023–2025)

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
- Offensive formation EPA (11/12 personnel) is the strongest signal — captures scheme efficiency beyond raw EPA
- 3-game rolling window consistently beats expanding or 5-game windows for overall ATS

## Diagnostic Findings

### Where the model wins and loses (best model, test 2023–2025)

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

The model loses money specifically when it disagrees with the line on *who wins outright*. Games where the model agrees on the winner but thinks the spread is too large are still profitable. **Untested idea: filter out games where `sign(predicted_margin) ≠ sign(spread_line)`.**

**By spread size:**

| Spread | N | ATS | Units |
|--------|---|-----|-------|
| Big dog >7 | 42 | 61.9% | +7.6 |
| Dog 3–7 | 104 | 58.7% | +12.5 |
| Dog <3 | 131 | 51.1% | −3.1 |
| Fav <3 | 91 | 59.3% | +12.1 |
| Fav 3–7 | 197 | 51.8% | −2.3 |
| Big fav >7 | 122 | 55.7% | +7.8 |

Near pick-em games (spread within 3 in either direction) are net losers. **Untested idea: filter out spreads between −3 and +3.**

**By season:**

| Season | N | ATS | Units |
|--------|---|-----|-------|
| 2023 | 177 | 53.1% | +2.5 |
| 2024 | 174 | 53.4% | +3.5 |
| 2025 | 147 | 55.8% | +9.5 |

**Worst teams (by units lost, test 2023–2025):** NE −9.5u (28.6% ATS), CIN −7.5u (31.6%), LV −6.5u (31.2%). These teams appear to have drifted significantly from their historical profiles in the training data.

## Project Structure

```
betting-models/
├── config.py                        # Shared constants (seasons, paths)
├── requirements.txt
│
├── pipeline/                        # Data building — run first
│   ├── export_rolling_epa.py        # Builds exports/features.parquet
│   └── generate_game_table.py       # Builds docs/data/games.json for browser explorer
│
├── models/
│   └── margin/                      # Margin prediction model (predict home_score - away_score)
│       ├── run_all_models.py        # Sweep all feature combinations, update README leaderboard
│       ├── cross_validate.py        # Walk-forward CV across seasons
│       └── generate_predictions.py  # Regenerate weekly pick JSONs for the UI
│
├── src/                             # Shared library
│   ├── data_loader.py               # Fetch & cache nflverse data
│   ├── model_utils.py               # build_dataset(), ats_accuracy(), load_features()
│   └── predictions_generator.py    # Core prediction logic used by generate_predictions.py
│
├── analysis/                        # One-off exploratory scripts
├── deprecated/                      # Superseded experiment files and old src code
│
├── exports/                         # Generated artifacts (not committed)
│   ├── features.parquet             # Master feature store
│   └── backup_qb_games.csv         # Identified backup QB starts
│
├── data/                            # Raw cached nflverse downloads (not committed)
├── docs/                            # Browser UI (GitHub Pages)
│   ├── index.html                   # Weekly picks dashboard
│   ├── season.html                  # Season-level ATS results
│   ├── explore.html                 # Game explorer with sortable feature table
│   └── data/                        # Generated JSON consumed by the UI
└── tests/
    └── test_features.py
```

## How It Works

### Data Pipeline

1. **Data Loader** (`src/data_loader.py`) — pulls play-by-play, schedule, and QBR data from nflverse and caches locally in `data/`
2. **Feature Builder** (`pipeline/export_rolling_epa.py`) — aggregates play-by-play into per-game team stats, computes 3-game rolling averages (shifted to prevent leakage), and writes `exports/features.parquet`
3. **Dataset Builder** (`src/model_utils.py`) — joins features to the schedule, pairs home and away team stats into one row per game, creating `opp_*` mirror columns for the opponent
4. **Model** — OLS linear regression predicting `home_score − away_score` (the margin). The model never sees the spread as a feature.
5. **ATS Evaluation** — `predicted_margin` is compared to `spread_line`; if `predicted_margin > spread_line` the model picks home, otherwise away

### Features

All features are rolling averages of the prior 3 games, computed within-season (window resets each year). Each game uses only data from games already played — no leakage.

- **off_11_epa_rolling** — offensive EPA/play in 11 personnel (1 RB, 1 TE, 3 WR)
- **off_12_epa_rolling** — offensive EPA/play in 12 personnel (1 RB, 2 TE, 2 WR)
- **qb_off_11_epa_rolling** — starting QB's EPA/play in 11 personnel (follows QB across teams)
- **qb_off_12_epa_rolling** — starting QB's EPA/play in 12 personnel
- **def_epa_rolling** — defensive EPA/play allowed
- **cpoe_rolling** — completion percentage over expected (QB accuracy vs. difficulty)
- **point_diff_rolling** — recent scoring margin (proxy for overall team quality)

Personnel data is available from 2016 onward, which sets the effective training window start.

### Key Design Decisions

1. **Predict margin directly** — one model predicts `home_score − away_score`; simpler and more stable than two separate score models
2. **Rolling 3-game window** — captures recent form; consistently beats expanding windows or season averages for ATS accuracy
3. **OLS linear regression** — interpretable and sufficient given the feature count; Ridge regularization was tested and showed no improvement
4. **Personnel-split EPA** — `off_11_epa` and `off_12_epa` separately outperform raw `off_epa`; formation captures scheme efficiency beyond volume
5. **Walk-forward validation** — trains on past seasons only, tests on future seasons; never peeks at future data
6. **Spread not used as a feature** — keeps the model market-independent; we want to predict what will happen, not echo what the market already knows

## Weekly Workflow

Each week, refresh the data and regenerate predictions:

```bash
# 1. Refresh raw data (picks up new scores and upcoming lines)
python -m src.data_loader --refresh

# 2. Rebuild the feature store
python pipeline/export_rolling_epa.py

# 3. Regenerate weekly pick JSONs for the UI
python models/margin/generate_predictions.py

# 4. Commit and push — GitHub Pages picks it up automatically
git add docs/data/
git commit -m "Week N predictions"
git push
```

## Setup

```bash
pip install -r requirements.txt
```

## Picks UI

A static dashboard lives in `docs/` and is served via **GitHub Pages** at `https://ebrooks11.github.io/betting-models/`.

- **index.html** — upcoming game cards with spread, predicted margin, and model picks
- **season.html** — full-season ATS results by week and model
- **explore.html** — interactive game table with sortable/filterable feature columns

### Enable GitHub Pages

1. Go to **Settings → Pages** in the GitHub repo
2. Set source to **Deploy from a branch**, branch: `main`, folder: `/docs`
