"""Export per-game rolling EPA features (3-game window, shift 1) to parquet.

For each game, off_epa_per_play and def_epa_per_play are the mean of the
raw EPA values from the prior 1-3 games (not averages of averages).
"""

from pathlib import Path
import pandas as pd
from src.data_loader import get_pbp_data
from config import SEASONS

WINDOW = 3

print("Loading play-by-play data...")
pbp = get_pbp_data(SEASONS)

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

TEAM_MAP = {
    "OAK": "LV",
    "SD":  "LAC",
    "STL": "LA",
}

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

result = game_epa[["season", "week", "team", "off_epa_rolling", "def_epa_rolling"]].copy()
result = result.sort_values(["season", "week", "team"]).reset_index(drop=True)

out_path = Path("exports/rolling_epa.parquet")
out_path.parent.mkdir(exist_ok=True)
result.to_parquet(out_path, index=False)

print(f"Exported {len(result):,} rows to {out_path}")
print("\nSample — DAL weeks 1-8 of 2022:")
sample = result[(result["team"] == "DAL") & (result["season"] == 2022)].head(8)
print(sample.to_string(index=False))
