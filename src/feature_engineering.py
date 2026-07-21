"""Build rolling team-level features from play-by-play data."""

import pandas as pd
import numpy as np

from config import ROLLING_WINDOW, ALL_FEATURES, GAME_FEATURES


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

    # Per-down success rates (1st, 2nd, 3rd)
    for down_num, col_name in [(1, "off_first_down_rate"), (2, "off_second_down_rate"), (3, "off_third_down_rate")]:
        down_plays = plays[plays["down"] == down_num]
        down_rate = (
            down_plays.groupby(["season", "week", "posteam"])["success"]
            .mean()
            .reset_index()
            .rename(columns={"posteam": "team", "success": col_name})
        )
        offense = offense.merge(down_rate, on=["season", "week", "team"], how="left")

    # CPOE (completion percentage over expected) — QB efficiency metric
    pass_plays = plays[(plays["play_type"] == "pass") & plays["cpoe"].notna()]
    cpoe = (
        pass_plays.groupby(["season", "week", "posteam"])["cpoe"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "cpoe": "off_cpoe"})
    )
    offense = offense.merge(cpoe, on=["season", "week", "team"], how="left")

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
        "off_turnovers", "off_first_down_rate", "off_second_down_rate", "off_third_down_rate",
        "off_cpoe",
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

    return df


def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge in the opponent's rolling stats and rest days for each game."""
    team_cols = (
        [c for c in df.columns if c.startswith("off_") or c.startswith("def_")]
    )

    opp_stats = df[["season", "week", "team", "rest_days"] + team_cols].copy()
    opp_stats = opp_stats.rename(
        columns={c: f"opp_{c}" for c in team_cols}
    )
    opp_stats = opp_stats.rename(columns={"team": "opponent", "rest_days": "opp_rest_days"})

    df = df.merge(opp_stats, on=["season", "week", "opponent"], how="left")
    df["rest_advantage"] = df["rest_days"] - df["opp_rest_days"]
    return df


def build_feature_matrix(
    pbp: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Full pipeline: pbp + schedules → team-level feature matrix."""
    game_stats = compute_game_stats(pbp)
    df = add_scores_and_context(game_stats, schedules)
    df = build_rolling_features(df)
    df = add_opponent_features(df)
    df = df.dropna(subset=ALL_FEATURES)
    return df


def build_personnel_matrix(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Build a game-level matrix using only QB and coach identity as features.

    For each game: identify which QB started (most pass attempts), pull the
    head coaches from the schedule, compute rest advantage, and set margin as target.
    """
    games = schedules[schedules["game_type"] == "REG"].copy()

    # Check coach columns exist
    if "home_coach" not in games.columns or "away_coach" not in games.columns:
        raise ValueError("Schedule data missing home_coach/away_coach columns")

    # Starting QB = QB with most pass attempts in each game
    pass_plays = pbp[pbp["play_type"] == "pass"].copy()
    qb_attempts = (
        pass_plays.groupby(["season", "week", "posteam", "passer_player_name"])
        .size()
        .reset_index(name="attempts")
    )
    starting_qbs = (
        qb_attempts.sort_values("attempts", ascending=False)
        .groupby(["season", "week", "posteam"])
        .first()
        .reset_index()[["season", "week", "posteam", "passer_player_name"]]
    )

    # Merge starting QB for home and away
    games = games.merge(
        starting_qbs.rename(columns={"posteam": "home_team", "passer_player_name": "home_qb"}),
        on=["season", "week", "home_team"], how="left",
    )
    games = games.merge(
        starting_qbs.rename(columns={"posteam": "away_team", "passer_player_name": "away_qb"}),
        on=["season", "week", "away_team"], how="left",
    )

    # Rest advantage (home rest days - away rest days)
    games["gameday"] = pd.to_datetime(games["gameday"])
    games = games.sort_values(["season", "week"])
    home_rest = (
        games[["season", "week", "home_team", "gameday"]]
        .rename(columns={"home_team": "team"})
        .sort_values(["team", "season", "week"])
    )
    home_rest["rest_days"] = home_rest.groupby("team")["gameday"].diff().dt.days.fillna(7)

    away_rest = (
        games[["season", "week", "away_team", "gameday"]]
        .rename(columns={"away_team": "team"})
        .sort_values(["team", "season", "week"])
    )
    away_rest["rest_days"] = away_rest.groupby("team")["gameday"].diff().dt.days.fillna(7)

    games = games.merge(
        home_rest[["season", "week", "team", "rest_days"]].rename(
            columns={"team": "home_team", "rest_days": "home_rest"}
        ),
        on=["season", "week", "home_team"], how="left",
    )
    games = games.merge(
        away_rest[["season", "week", "team", "rest_days"]].rename(
            columns={"team": "away_team", "rest_days": "away_rest"}
        ),
        on=["season", "week", "away_team"], how="left",
    )
    games["rest_advantage"] = games["home_rest"] - games["away_rest"]
    games["margin"] = games["home_score"] - games["away_score"]

    cols = [
        "season", "week", "home_team", "away_team",
        "home_qb", "away_qb", "home_coach", "away_coach",
        "rest_advantage", "margin", "spread_line",
    ]
    result = games[cols].dropna(subset=["home_qb", "away_qb", "margin", "spread_line"])
    return result.reset_index(drop=True)


def build_game_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse team-level rows to one row per game (home perspective).

    Target is margin = home_score - away_score. Each row contains both
    teams' rolling stats via the opp_* columns already present.
    """
    games = df[df["is_home"] == 1].copy()
    games["margin"] = games["score"] - games["opponent_score"]
    games = games.dropna(subset=GAME_FEATURES + ["margin", "spread_line"])
    return games


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
