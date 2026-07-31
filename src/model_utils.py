"""Shared utilities for NFL margin prediction models."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

FEATURES_PATH = Path("exports/features.parquet")
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
TRAIN_SEASONS = range(2006, 2022)
TEST_SEASONS = range(2022, 2026)


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run export_rolling_epa.py first."
        )
    return pd.read_parquet(FEATURES_PATH)


def build_dataset(schedules: pd.DataFrame, features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build one row per home game with team and opponent features merged in.

    feature_cols: columns from features.parquet to include (must exist in the file).
    Opponent versions are added automatically with an opp_ prefix.
    """
    games = schedules[schedules["game_type"] == "REG"].copy()

    home = games[["season", "week", "home_team", "away_team",
                   "home_score", "away_score", "spread_line"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent",
                                 "home_score": "score", "away_score": "opp_score"})
    home["team"] = home["team"].replace(TEAM_MAP)
    home["opponent"] = home["opponent"].replace(TEAM_MAP)
    home["margin"] = home["score"] - home["opp_score"]

    team_features = features[["season", "week", "team"] + feature_cols]

    # Home team features
    home = home.merge(team_features, on=["season", "week", "team"], how="left")

    # Opponent features (opp_ prefixed)
    opp_features = team_features.rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in feature_cols}}
    )
    home = home.merge(opp_features, on=["season", "week", "opponent"], how="left")

    opp_cols = [f"opp_{c}" for c in feature_cols]
    home = home.dropna(subset=feature_cols + opp_cols + ["margin", "spread_line"])
    home = home[home["week"] >= 4]

    # Matchup formation score: how well the home offense's personnel packages match up
    # against the opponent's defensive tendencies, weighted by usage rate.
    # (off_11_epa - opp_def_vs_11_epa) * off_11_rate + (off_12_epa - opp_def_vs_12_epa) * off_12_rate
    formation_cols = ["off_11_epa_rolling", "off_12_epa_rolling",
                      "off_11_rate_rolling", "off_12_rate_rolling",
                      "opp_def_vs_11_epa_rolling", "opp_def_vs_12_epa_rolling"]
    if all(c in home.columns for c in formation_cols):
        home["matchup_formation_score"] = (
            (home["off_11_epa_rolling"] - home["opp_def_vs_11_epa_rolling"]) * home["off_11_rate_rolling"] +
            (home["off_12_epa_rolling"] - home["opp_def_vs_12_epa_rolling"]) * home["off_12_rate_rolling"]
        )
        # Opponent's matchup score (their offense vs our defense)
        home["opp_matchup_formation_score"] = (
            (home["opp_off_11_epa_rolling"] - home["def_vs_11_epa_rolling"]) * home["opp_off_11_rate_rolling"] +
            (home["opp_off_12_epa_rolling"] - home["def_vs_12_epa_rolling"]) * home["opp_off_12_rate_rolling"]
        )

    return home


def ats_accuracy(df: pd.DataFrame, pred_col: str = "predicted_margin", min_edge: float = 0.0):
    """Fraction of games where model picks the correct ATS side (pushes excluded).

    nflverse spread_line is positive when home is favored.
    Home covers when margin > spread_line.
    min_edge: only evaluate games where abs(predicted_margin - spread_line) >= min_edge.
    """
    covered = df["margin"] - df["spread_line"]
    model_pick = df[pred_col] - df["spread_line"]
    if min_edge > 0:
        mask = model_pick.abs() >= min_edge
        covered = covered[mask]
        model_pick = model_pick[mask]
    push = covered == 0
    hit = (model_pick > 0) == (covered > 0)
    non_push = ~push
    return hit[non_push].mean(), non_push.sum(), push.sum()


def evaluate(train: pd.DataFrame, test: pd.DataFrame, model, feature_cols: list[str]):
    """Print coefficients, MAE, ATS accuracy, and edge-filtered ATS accuracy."""
    all_cols = feature_cols + [f"opp_{c}" for c in feature_cols]

    print("\nCoefficients:")
    for feat, coef in zip(all_cols, model.coef_):
        print(f"  {feat}: {coef:.4f}")
    print(f"  intercept: {model.intercept_:.4f}")

    train_mae = mean_absolute_error(train["margin"], train["predicted_margin"])
    test_mae = mean_absolute_error(test["margin"], test["predicted_margin"])
    print(f"\nMAE — Train: {train_mae:.2f} pts, Test: {test_mae:.2f} pts")

    train_ats, train_n, train_push = ats_accuracy(train)
    test_ats, test_n, test_push = ats_accuracy(test)
    print(f"\nATS accuracy (all games):")
    print(f"  Train: {train_ats:.1%}  ({train_n} non-push, {train_push} push)")
    print(f"  Test:  {test_ats:.1%}  ({test_n} non-push, {test_push} push)")

    train_e3, train_n3, train_p3 = ats_accuracy(train, min_edge=3.0)
    test_e3, test_n3, test_p3 = ats_accuracy(test, min_edge=3.0)
    print(f"\nATS accuracy (edge > 3 pts):")
    print(f"  Train: {train_e3:.1%}  ({train_n3} non-push, {train_p3} push)")
    print(f"  Test:  {test_e3:.1%}  ({test_n3} non-push, {test_p3} push)")

