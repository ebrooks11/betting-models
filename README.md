# NFL Prediction Model

An NFL ATS (against the spread) prediction model using play-by-play data from nflverse (2016–2025) via `nfl_data_py`, with scikit-learn for modeling. Train set: 2016–2022, test set: 2023–2025.

## To-Do

- [ ] **Power rating system** — single points-based team rating derived from preseason win totals, QB quality, coaching, and iterative performance against opponents (power records). Rating difference = predicted margin.
- [ ] **Injury adjustments for power ratings** — preseason win totals assume full health; adjust ratings down when key players (especially QB) are out or limited.
- [ ] **Prop models** — starting with rush yards, then receiving yards.
- [ ] **Model gap analysis** — review what the current margin and score models are capturing and identify blind spots.
- [ ] **Preseason win total in opponent strength** — incorporate preseason win totals into the iterative opponent-adjustment calculation alongside EPA, so opponent quality reflects market-assessed talent not just in-season performance.

## Game Filters

All models are evaluated on the home-team perspective (one row per game). The following games are excluded from both train and test sets:

- **Weeks 1–3**: Dropped — rolling windows require at least 3 prior games; early-season features are too noisy.
- **Final week of each season**: Dropped — week 18 (2021+) and week 17 (pre-2021). Resting starters and meaningless games destroy signal.

See `models/margin/RESULTS.md` for the full margin model leaderboard.

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
│   └── features.parquet             # Master feature store
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
