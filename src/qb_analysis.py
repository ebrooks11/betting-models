"""Analyze how many training games featured a non-starting QB."""

import pandas as pd

from config import SEASONS
from src.data_loader import get_pbp_data


def identify_backup_starts(pbp: pd.DataFrame) -> pd.DataFrame:
    """Flag games where a team's primary season QB did not play."""
    passes = pbp[
        (pbp["play_type"] == "pass") & pbp["passer_player_name"].notna()
    ].copy()

    # Attempts per QB per team per game
    game_qb = (
        passes.groupby(["season", "week", "posteam", "passer_player_name"])
        .size()
        .reset_index(name="attempts")
    )

    # Primary starter = QB with most attempts across the full season
    season_qb = (
        passes.groupby(["season", "posteam", "passer_player_name"])
        .size()
        .reset_index(name="season_attempts")
    )
    starters = (
        season_qb.sort_values("season_attempts", ascending=False)
        .groupby(["season", "posteam"])
        .first()
        .reset_index()
        .rename(columns={"passer_player_name": "starter", "posteam": "team"})
    )

    # For each game, find which QB had most attempts
    game_starter = (
        game_qb.sort_values("attempts", ascending=False)
        .groupby(["season", "week", "posteam"])
        .first()
        .reset_index()
        .rename(columns={"passer_player_name": "game_qb", "posteam": "team"})
    )

    merged = game_starter.merge(starters[["season", "team", "starter"]], on=["season", "team"])
    merged["backup_started"] = merged["game_qb"] != merged["starter"]

    return merged


def print_summary(df: pd.DataFrame):
    total_games = len(df)
    backup_games = df["backup_started"].sum()
    print(f"\nTotal team-game entries: {total_games:,}")
    print(f"Games with non-starter QB: {backup_games:,} ({backup_games/total_games*100:.1f}%)")

    print("\nBy season:")
    by_season = df.groupby("season")["backup_started"].agg(["sum", "count"])
    by_season["pct"] = (by_season["sum"] / by_season["count"] * 100).round(1)
    by_season.columns = ["backup_games", "total_games", "pct"]
    print(by_season.to_string())

    print("\nSample backup starts:")
    sample = df[df["backup_started"]].head(20)[
        ["season", "week", "team", "starter", "game_qb"]
    ]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    print("Loading play-by-play data...")
    pbp = get_pbp_data(SEASONS)

    print("Identifying backup starts...")
    results = identify_backup_starts(pbp)

    print_summary(results)
