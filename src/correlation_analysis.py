"""Compute correlation between available features and team points scored."""

import pandas as pd
import numpy as np

from config import SEASONS
from src.data_loader import get_pbp_data, get_schedule_data, get_qbr_data, get_pfr_advstats


def add_q3_stats(base: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Add Q1-Q3 EPA and success rate per team per game."""
    plays = pbp[pbp["play_type"].isin(["pass", "run"]) & (pbp["qtr"] <= 3)].copy()
    q3_stats = (
        plays.groupby(["season", "week", "posteam"])
        .agg(
            epa_per_play_q3=("epa", "mean"),
            success_rate_q3=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    return base.merge(q3_stats, on=["season", "week", "team"], how="left")


def build_team_game_scores(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with final score."""
    games = schedules[schedules["game_type"] == "REG"].copy()
    home = games[["season", "week", "home_team", "away_team", "home_score",
                  "wind", "temp", "roof", "div_game"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_score": "points"}
    )
    away = games[["season", "week", "away_team", "home_team", "away_score",
                  "wind", "temp", "roof", "div_game"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_score": "points"}
    )
    return pd.concat([home, away], ignore_index=True)


def add_pbp_stats(base: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Add per-game team stats from play-by-play."""
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    team_stats = (
        plays.groupby(["season", "week", "posteam"])
        .agg(
            epa_per_play=("epa", "mean"),
            success_rate=("success", "mean"),
            yards_per_play=("yards_gained", "mean"),
            cpoe=("cpoe", "mean"),
            qb_epa=("qb_epa", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    pass_plays = plays[plays["play_type"] == "pass"]
    pass_stats = (
        pass_plays.groupby(["season", "week", "posteam"])
        .agg(
            air_yards=("air_yards", "mean"),
            pass_epa=("epa", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    run_plays = plays[plays["play_type"] == "run"]
    run_stats = (
        run_plays.groupby(["season", "week", "posteam"])
        .agg(run_epa=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    base = base.merge(team_stats, on=["season", "week", "team"], how="left")
    base = base.merge(pass_stats, on=["season", "week", "team"], how="left")
    base = base.merge(run_stats, on=["season", "week", "team"], how="left")
    return base


def add_qbr(base: pd.DataFrame, qbr: pd.DataFrame) -> pd.DataFrame:
    """Add ESPN QBR — take the highest QBR per team per game (starting QB)."""
    reg = qbr[qbr["season_type"] == "Regular"].copy()
    top_qbr = (
        reg.sort_values("qbr_total", ascending=False)
        .groupby(["season", "game_week", "team_abb"])
        .first()
        .reset_index()[["season", "game_week", "team_abb", "qbr_total", "pts_added"]]
        .rename(columns={"game_week": "week", "team_abb": "team"})
    )
    return base.merge(top_qbr, on=["season", "week", "team"], how="left")


def add_pfr(base: pd.DataFrame, pfr: pd.DataFrame) -> pd.DataFrame:
    """Add PFR team-level aggregates per game."""
    reg = pfr[pfr["game_type"] == "REG"].copy()

    # Pass stats — aggregate to team level (sum pressures, average rates)
    pass_pfr = (
        reg[reg["stat_type"] == "pass"]
        .groupby(["season", "week", "team"])
        .agg(
            times_pressured=("times_pressured", "sum"),
            times_pressured_pct=("times_pressured_pct", "mean"),
            passing_bad_throw_pct=("passing_bad_throw_pct", "mean"),
            passing_drop_pct=("passing_drop_pct", "mean"),
        )
        .reset_index()
    )

    # Rush stats — aggregate to team level
    rush_pfr = (
        reg[reg["stat_type"] == "rush"]
        .groupby(["season", "week", "team"])
        .agg(
            rushing_yac_avg=("rushing_yards_after_contact_avg", "mean"),
            rushing_broken_tackles=("rushing_broken_tackles", "sum"),
        )
        .reset_index()
    )

    base = base.merge(pass_pfr, on=["season", "week", "team"], how="left")
    base = base.merge(rush_pfr, on=["season", "week", "team"], how="left")
    return base


def add_lagged_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add next_game_points and rolling feature averages for each team.

    Each row gets:
    - next_game_points: points scored in the following game
    - rolling_{col}: average of the previous `window` games for each stat
    """
    df = df.sort_values(["team", "season", "week"]).copy()
    df["next_game_points"] = df.groupby("team")["points"].shift(-1)

    # Drop rows where next game crosses a season boundary
    df["next_season"] = df.groupby("team")["season"].shift(-1)
    df.loc[df["next_season"] != df["season"], "next_game_points"] = None
    df = df.drop(columns=["next_season"])

    # Compute rolling averages for all numeric feature columns
    feature_cols = [
        "epa_per_play", "epa_per_play_q3",
        "success_rate", "success_rate_q3",
        "yards_per_play", "cpoe", "qb_epa",
        "air_yards", "pass_epa", "run_epa", "qbr_total", "pts_added",
        "times_pressured_pct", "passing_bad_throw_pct", "passing_drop_pct",
        "rushing_yac_avg", "rushing_broken_tackles", "wind", "temp",
    ]
    available = [c for c in feature_cols if c in df.columns]
    for col in available:
        df[f"rolling_{col}"] = df.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )

    return df


def print_correlations(df: pd.DataFrame):
    feature_cols = [
        "epa_per_play", "epa_per_play_q3",
        "success_rate", "success_rate_q3",
        "yards_per_play", "cpoe", "qb_epa",
        "air_yards", "pass_epa", "run_epa",
        "qbr_total", "pts_added",
        "times_pressured_pct", "passing_bad_throw_pct", "passing_drop_pct",
        "rushing_yac_avg", "rushing_broken_tackles",
        "wind", "temp", "div_game",
    ]

    available = [c for c in feature_cols if c in df.columns]

    print(f"\n{'Feature':<30} {'Same game':>10}  {'Lag 1 game':>10}  {'Lag 5 avg':>10}  {'Strength (5-game)'}")
    print("-" * 90)

    rows = []
    for col in available:
        same = df[["points", col]].dropna()
        lagged1 = df[["next_game_points", col]].dropna()
        lagged5 = df[["next_game_points", f"rolling_{col}"]].dropna() if f"rolling_{col}" in df.columns else pd.DataFrame()

        if len(same) < 100:
            continue

        same_corr = same["points"].corr(same[col])
        lag1_corr = lagged1["next_game_points"].corr(lagged1[col]) if len(lagged1) > 100 else float("nan")
        lag5_corr = lagged5["next_game_points"].corr(lagged5[f"rolling_{col}"]) if len(lagged5) > 100 else float("nan")

        rows.append((col, same_corr, lag1_corr, lag5_corr))

    rows.sort(key=lambda x: abs(x[3]) if not np.isnan(x[3]) else 0, reverse=True)

    for feat, same_corr, lag1_corr, lag5_corr in rows:
        bar = "█" * int(abs(lag5_corr) * 20) if not np.isnan(lag5_corr) else ""
        direction = "+" if lag5_corr > 0 else "-"
        strength = "strong" if abs(lag5_corr) > 0.3 else "moderate" if abs(lag5_corr) > 0.15 else "weak"
        lag1_str = f"{lag1_corr:>+10.3f}" if not np.isnan(lag1_corr) else "       N/A"
        lag5_str = f"{direction}{abs(lag5_corr):>9.3f}" if not np.isnan(lag5_corr) else "       N/A"
        print(f"{feat:<30} {same_corr:>+10.3f}  {lag1_str}  {lag5_str}  {bar} {strength}")


if __name__ == "__main__":
    print("Loading data...")
    pbp = get_pbp_data(SEASONS)
    schedules = get_schedule_data(SEASONS)
    qbr = get_qbr_data(SEASONS)
    pfr = get_pfr_advstats(SEASONS)

    print("Building game-level stats...")
    df = build_team_game_scores(schedules)
    df = add_pbp_stats(df, pbp)
    df = add_qbr(df, qbr)
    df = add_pfr(df, pfr)

    df = add_q3_stats(df, pbp)
    print(f"Dataset: {len(df):,} team-game rows, {df['points'].notna().sum():,} with scores")
    df = add_lagged_points(df)
    print_correlations(df)
