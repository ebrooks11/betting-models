"""Build rolling team-level features from play-by-play data."""

import pandas as pd
import numpy as np

from config import ROLLING_WINDOW, ALL_FEATURES


def compute_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-by-play data into per-team, per-game stats."""
    # Filter to real plays (exclude timeouts, penalties, etc.)
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    offense = (
        plays.groupby(["season", "week", "posteam"])
        .agg(
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
            off_yards_per_play=("yards_gained", "mean"),
            off_plays=("play_id", "count"),
            off_turnovers=("interception", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Add fumbles lost to turnovers
    fumbles = (
        plays[plays["fumble_lost"] == 1]
        .groupby(["season", "week", "posteam"])
        .size()
        .reset_index(name="fumbles_lost")
        .rename(columns={"posteam": "team"})
    )
    offense = offense.merge(fumbles, on=["season", "week", "team"], how="left")
    offense["fumbles_lost"] = offense["fumbles_lost"].fillna(0)
    offense["off_turnovers"] = offense["off_turnovers"] + offense["fumbles_lost"]
    offense = offense.drop(columns=["fumbles_lost"])

    # Third down conversion rate
    third_downs = plays[plays["down"] == 3]
    third_down_rate = (
        third_downs.groupby(["season", "week", "posteam"])
        .agg(
            third_down_attempts=("play_id", "count"),
            third_down_conversions=("third_down_converted", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    third_down_rate["off_third_down_rate"] = (
        third_down_rate["third_down_conversions"] / third_down_rate["third_down_attempts"]
    )
    offense = offense.merge(
        third_down_rate[["season", "week", "team", "off_third_down_rate"]],
        on=["season", "week", "team"],
        how="left",
    )

    defense = (
        plays.groupby(["season", "week", "defteam"])
        .agg(
            def_epa_per_play=("epa", "mean"),
            def_success_rate=("success", "mean"),
            def_yards_per_play=("yards_gained", "mean"),
            def_takeaways=("interception", "sum"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # Add fumble recoveries to takeaways
    def_fumbles = (
        plays[plays["fumble_lost"] == 1]
        .groupby(["season", "week", "defteam"])
        .size()
        .reset_index(name="fumbles_recovered")
        .rename(columns={"defteam": "team"})
    )
    defense = defense.merge(def_fumbles, on=["season", "week", "team"], how="left")
    defense["fumbles_recovered"] = defense["fumbles_recovered"].fillna(0)
    defense["def_takeaways"] = defense["def_takeaways"] + defense["fumbles_recovered"]
    defense = defense.drop(columns=["fumbles_recovered"])

    game_stats = offense.merge(defense, on=["season", "week", "team"], how="outer")
    return game_stats


def add_scores_and_context(
    game_stats: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Merge in final scores, home/away, rest days, and betting lines."""
    games = schedules[schedules["game_type"] == "REG"].copy()

    # Build home team rows
    home = games[["season", "week", "home_team", "away_team", "home_score",
                   "away_score", "spread_line", "total_line", "gameday"]].copy()
    home = home.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_score": "score",
        "away_score": "opponent_score",
    })
    home["is_home"] = 1

    # Build away team rows
    away = games[["season", "week", "home_team", "away_team", "home_score",
                   "away_score", "spread_line", "total_line", "gameday"]].copy()
    away = away.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_score": "score",
        "home_score": "opponent_score",
    })
    away["is_home"] = 0
    # Flip spread for away team perspective
    away["spread_line"] = -away["spread_line"]

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])

    # Rest days
    team_games = team_games.sort_values(["team", "season", "week"])
    team_games["rest_days"] = (
        team_games.groupby("team")["gameday"].diff().dt.days.fillna(7)
    )

    # Win streak
    team_games["won"] = (team_games["score"] > team_games["opponent_score"]).astype(int)
    streaks = []
    for _, group in team_games.groupby("team"):
        streak = 0
        team_streaks = []
        for won in group["won"]:
            team_streaks.append(streak)
            streak = streak + 1 if won else streak - 1 if not won else 0
        streaks.extend(team_streaks)
    team_games["win_streak"] = streaks

    merged = team_games.merge(game_stats, on=["season", "week", "team"], how="left")
    return merged


def build_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Compute rolling averages for each team, shifted so each row uses only prior games."""
    df = df.sort_values(["team", "season", "week"]).copy()

    stat_cols = [
        "off_epa_per_play", "off_success_rate", "off_yards_per_play",
        "off_turnovers", "off_third_down_rate",
        "def_epa_per_play", "def_success_rate", "def_yards_per_play",
        "def_takeaways",
    ]

    rename_map = {
        "off_turnovers": "off_turnovers_per_game",
        "def_takeaways": "def_takeaways_per_game",
    }

    for col in stat_cols:
        new_col = rename_map.get(col, col)
        df[new_col] = df.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        if col in rename_map:
            df = df.drop(columns=[col])

    df["off_points_per_game"] = df.groupby("team")["score"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df["def_points_per_game"] = df.groupby("team")["opponent_score"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    # Drop raw stat columns that weren't renamed
    raw_to_drop = [c for c in stat_cols if c not in rename_map]
    df = df.drop(columns=raw_to_drop, errors="ignore")
    return df


def build_feature_matrix(
    pbp: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Full pipeline: pbp + schedules → feature matrix ready for modeling."""
    game_stats = compute_game_stats(pbp)
    df = add_scores_and_context(game_stats, schedules)
    df = build_rolling_features(df)

    # Drop rows with no rolling data (first games of a team's history)
    df = df.dropna(subset=ALL_FEATURES)

    return df


if __name__ == "__main__":
    from config import SEASONS
    from src.data_loader import get_pbp_data, get_schedule_data

    pbp = get_pbp_data(SEASONS)
    schedules = get_schedule_data(SEASONS)

    features = build_feature_matrix(pbp, schedules)
    print(f"Feature matrix: {features.shape[0]:,} rows, {features.shape[1]} columns")
    print(f"\nFeature columns used for modeling:")
    for f in ALL_FEATURES:
        print(f"  {f}: {features[f].describe().to_dict()}")
    print(f"\nTarget (score) distribution:")
    print(features["score"].describe())
