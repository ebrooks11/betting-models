"""Export per-game team EPA aggregates (no rolling, no shifting) to parquet."""

from pathlib import Path
import pandas as pd
from src.data_loader import get_pbp_data
from config import SEASONS

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

game_epa = offense.merge(defense, on=["season", "week", "team"], how="outer")
game_epa = game_epa.sort_values(["season", "week", "team"]).reset_index(drop=True)

out_path = Path("exports/game_epa.parquet")
out_path.parent.mkdir(exist_ok=True)
game_epa.to_parquet(out_path, index=False)

print(f"Exported {len(game_epa):,} rows to {out_path}")
print(game_epa.head(10).to_string())
