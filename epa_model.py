"""
Minimal EPA model: 3-game rolling EPA features, linear regression, train 2015-2021, test 2022-2024.
Evaluates MAE and ATS accuracy to verify the Google Sheets result in Python.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.data_loader import get_pbp_data, get_schedule_data
from config import SEASONS

WINDOW = 3
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
TRAIN_SEASONS = range(2015, 2022)
TEST_SEASONS = range(2022, 2025)
FEATURES = ["off_epa_rolling", "def_epa_rolling", "opp_off_epa_rolling", "opp_def_epa_rolling"]


def build_game_epa(pbp):
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    offense = (
        plays.groupby(["season", "week", "posteam"])
        .agg(off_epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    defense = (
        plays.groupby(["season", "week", "defteam"])
        .agg(def_epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    game_epa = offense.merge(defense, on=["season", "week", "team"], how="outer")
    game_epa["team"] = game_epa["team"].replace(TEAM_MAP)
    game_epa = game_epa.sort_values(["team", "season", "week"]).reset_index(drop=True)

    game_epa["off_epa_rolling"] = (
        game_epa.groupby(["team", "season"])["off_epa_per_play"]
        .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
    )
    game_epa["def_epa_rolling"] = (
        game_epa.groupby(["team", "season"])["def_epa_per_play"]
        .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
    )
    return game_epa


def build_dataset(schedules, game_epa):
    games = schedules[schedules["game_type"] == "REG"].copy()

    home = games[["season", "week", "home_team", "away_team",
                   "home_score", "away_score", "spread_line"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent",
                                 "home_score": "score", "away_score": "opp_score"})
    home["is_home"] = 1

    team_games = home.copy()
    team_games["team"] = team_games["team"].replace(TEAM_MAP)
    team_games["opponent"] = team_games["opponent"].replace(TEAM_MAP)
    team_games["margin"] = team_games["score"] - team_games["opp_score"]

    # Merge home team rolling EPA
    team_games = team_games.merge(
        game_epa[["season", "week", "team", "off_epa_rolling", "def_epa_rolling"]],
        on=["season", "week", "team"], how="left"
    )

    # Merge opponent rolling EPA
    opp_epa = game_epa[["season", "week", "team", "off_epa_rolling", "def_epa_rolling"]].copy()
    opp_epa = opp_epa.rename(columns={
        "team": "opponent",
        "off_epa_rolling": "opp_off_epa_rolling",
        "def_epa_rolling": "opp_def_epa_rolling",
    })
    team_games = team_games.merge(opp_epa, on=["season", "week", "opponent"], how="left")

    team_games = team_games.dropna(subset=FEATURES + ["margin", "spread_line"])
    return team_games


def ats_accuracy(df, pred_col="predicted_margin"):
    """Fraction of games where model picks the correct ATS side (pushes excluded)."""
    margin = df["margin"]
    spread = df["spread_line"]
    pred = df[pred_col]

    covered = margin + spread
    model_pick = pred + spread

    push = covered == 0
    hit = (model_pick > 0) == (covered > 0)

    non_push = ~push
    return hit[non_push].mean(), non_push.sum(), push.sum()


print("Loading data...")
pbp = get_pbp_data(SEASONS)
schedules = get_schedule_data(SEASONS)

print("Building features...")
game_epa = build_game_epa(pbp)
df = build_dataset(schedules, game_epa)

train = df[df["season"].isin(TRAIN_SEASONS)]
test = df[df["season"].isin(TEST_SEASONS)]

print(f"\nTrain: {len(train)} games ({train['season'].min()}-{train['season'].max()})")
print(f"Test:  {len(test)} games ({test['season'].min()}-{test['season'].max()})")

model = LinearRegression()
model.fit(train[FEATURES], train["margin"])

print(f"\nCoefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  intercept: {model.intercept_:.4f}")

train = train.copy()
test = test.copy()
train["predicted_margin"] = model.predict(train[FEATURES])
test["predicted_margin"] = model.predict(test[FEATURES])

train_mae = mean_absolute_error(train["margin"], train["predicted_margin"])
test_mae = mean_absolute_error(test["margin"], test["predicted_margin"])
print(f"\nMAE — Train: {train_mae:.2f} pts, Test: {test_mae:.2f} pts")

train_ats, train_n, train_push = ats_accuracy(train)
test_ats, test_n, test_push = ats_accuracy(test)
print(f"\nATS accuracy:")
print(f"  Train: {train_ats:.1%}  ({train_n} non-push, {train_push} push)")
print(f"  Test:  {test_ats:.1%}  ({test_n} non-push, {test_push} push)")

# Sanity check 1: naive baseline — always predict home wins by 3
test = test.copy()
test["naive_margin"] = 3.0
naive_ats, _, _ = ats_accuracy(test, pred_col="naive_margin")
print(f"\nSanity checks (test set):")
print(f"  Naive (always home +3): {naive_ats:.1%}")

# Sanity check 2: shuffled predictions
rng = np.random.default_rng(42)
test["shuffled_margin"] = rng.permutation(test["predicted_margin"].values)
shuffled_ats, _, _ = ats_accuracy(test, pred_col="shuffled_margin")
print(f"  Shuffled predictions:   {shuffled_ats:.1%}")

# Sanity check 3: spread distribution
print(f"\nSpread distribution (test set):")
print(test["spread_line"].describe().to_string())
print(f"  Games with |spread| <= 3: {(test['spread_line'].abs() <= 3).sum()} ({(test['spread_line'].abs() <= 3).mean():.1%})")
