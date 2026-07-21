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


def get_qbr_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "qbr.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading QBR data for {seasons[0]}-{seasons[-1]}...")
    qbr = nfl.import_qbr(seasons, level="nfl", frequency="weekly")
    os.makedirs(DATA_DIR, exist_ok=True)
    qbr.to_parquet(cache_path, index=False)
    print(f"Cached {len(qbr):,} QB-game rows to {cache_path}")
    return qbr


def get_roster_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "rosters.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading roster data for {seasons[0]}-{seasons[-1]}...")
    rosters = nfl.import_players()
    os.makedirs(DATA_DIR, exist_ok=True)
    rosters.to_parquet(cache_path, index=False)
    print(f"Cached {len(rosters):,} roster rows to {cache_path}")
    return rosters


def get_snap_counts(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "snap_counts.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading snap count data for {seasons[0]}-{seasons[-1]}...")
    snaps = nfl.import_snap_counts(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    snaps.to_parquet(cache_path, index=False)
    print(f"Cached {len(snaps):,} snap count rows to {cache_path}")
    return snaps


def get_pfr_advstats(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "pfr_advstats.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading PFR advanced stats for {seasons[0]}-{seasons[-1]}...")
    dfs = []
    # Try both known function names and stat_type variants
    for stat_type in ["pass", "rush", "def"]:
        for fn_name in ["import_pfr_advstats", "import_advstats"]:
            fn = getattr(nfl, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn(seasons, stat_type=stat_type)
                df["stat_type"] = stat_type
                dfs.append(df)
                print(f"  {stat_type}: {len(df):,} rows")
                break
            except Exception as e:
                print(f"  {stat_type} via {fn_name}: skipped ({e})")

    if not dfs:
        print("  No PFR advanced stats available in this version of nfl_data_py")
        available = [x for x in dir(nfl) if "pfr" in x.lower() or "adv" in x.lower()]
        print(f"  Related functions found: {available}")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    print(f"Cached {len(combined):,} PFR rows to {cache_path}")
    return combined


if __name__ == "__main__":
    from config import SEASONS

    print("=== Play-by-play ===")
    pbp = get_pbp_data(SEASONS)
    print(f"{pbp.shape[0]:,} plays, {pbp.shape[1]} columns\n")

    print("=== Schedules ===")
    schedules = get_schedule_data(SEASONS)
    print(f"{schedules.shape[0]:,} games\n")

    print("=== QBR (weekly) ===")
    qbr = get_qbr_data(SEASONS)
    print(f"{qbr.shape[0]:,} QB-game rows")
    print(f"Columns: {list(qbr.columns)}\n")

    print("=== Rosters ===")
    rosters = get_roster_data(SEASONS)
    print(f"{rosters.shape[0]:,} roster rows")
    print(f"Columns: {list(rosters.columns)}\n")

    print("=== Snap Counts ===")
    snaps = get_snap_counts(SEASONS)
    print(f"{snaps.shape[0]:,} snap count rows")
    print(f"Columns: {list(snaps.columns)}\n")

    print("=== PFR Advanced Stats ===")
    pfr = get_pfr_advstats(SEASONS)
    if not pfr.empty:
        print(f"{pfr.shape[0]:,} rows")
        print(f"Columns: {list(pfr.columns)}")
    else:
        print("Not available in this version of nfl_data_py")
