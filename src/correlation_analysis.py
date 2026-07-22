"""Compute correlation between available features and team points scored."""

import pandas as pd
import numpy as np

from config import SEASONS
from src.data_loader import get_pbp_data, get_schedule_data, get_qbr_data, get_pfr_advstats


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


def print_correlations(df: pd.DataFrame):
    feature_cols = [
        "epa_per_play", "success_rate", "yards_per_play", "cpoe", "qb_epa",
        "air_yards", "pass_epa", "run_epa",
        "qbr_total", "pts_added",
        "times_pressured_pct", "passing_bad_throw_pct", "passing_drop_pct",
        "rushing_yac_avg", "rushing_broken_tackles",
        "wind", "temp", "div_game",
    ]

    available = [c for c in feature_cols if c in df.columns]
    correlations = {}
    for col in available:
        valid = df[["points", col]].dropna()
        if len(valid) > 100:
            correlations[col] = valid["points"].corr(valid[col])

    ranked = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<30} {'Correlation':>12}  {'Strength'}")
    print("-" * 60)
    for feat, corr in ranked:
        bar = "█" * int(abs(corr) * 20)
        direction = "+" if corr > 0 else "-"
        strength = "strong" if abs(corr) > 0.4 else "moderate" if abs(corr) > 0.2 else "weak"
        print(f"{feat:<30} {direction}{abs(corr):>10.3f}  {bar} {strength}")


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

    print(f"Dataset: {len(df):,} team-game rows, {df['points'].notna().sum():,} with scores")
    print_correlations(df)
