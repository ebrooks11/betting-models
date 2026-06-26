"""Fetch and cache NFL data from nflverse via nfl_data_py."""

import os
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_pbp_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "pbp.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading play-by-play data for {seasons[0]}-{seasons[-1]}...")
    pbp = nfl.import_pbp_data(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    pbp.to_parquet(cache_path, index=False)
    print(f"Cached {len(pbp):,} plays to {cache_path}")
    return pbp


def get_schedule_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "schedules.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading schedule data for {seasons[0]}-{seasons[-1]}...")
    schedules = nfl.import_schedules(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    schedules.to_parquet(cache_path, index=False)
    print(f"Cached {len(schedules):,} games to {cache_path}")
    return schedules


if __name__ == "__main__":
    from config import SEASONS

    pbp = get_pbp_data(SEASONS)
    print(f"Play-by-play: {pbp.shape[0]:,} plays, {pbp.shape[1]} columns")
    print(f"Seasons: {pbp['season'].unique()}")

    schedules = get_schedule_data(SEASONS)
    print(f"\nSchedules: {schedules.shape[0]:,} games")
    print(schedules[["season", "game_type"]].value_counts().sort_index())
