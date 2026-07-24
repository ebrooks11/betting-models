"""Combine rolling EPA features with schedule data into a single CSV."""

from pathlib import Path
import pandas as pd
from src.data_loader import get_schedule_data
from config import SEASONS

OUTPUT_DIR = Path("exports")
OUTPUT_DIR.mkdir(exist_ok=True)

TEAM_MAP = {
    "OAK": "LV",
    "SD":  "LAC",
    "STL": "LA",
}

print("Loading data...")
schedules = get_schedule_data(SEASONS)
rolling_epa = pd.read_parquet(OUTPUT_DIR / "rolling_epa.parquet")

rolling_epa["team"] = rolling_epa["team"].replace(TEAM_MAP)

games = schedules[schedules["game_type"] == "REG"][
    ["season", "week", "home_team", "away_team", "home_score", "away_score", "spread_line"]
].copy()

# Build one row per team per game
home = games.rename(columns={
    "home_team": "team", "away_team": "opponent",
    "home_score": "score", "away_score": "opponent_score",
})
away = games.rename(columns={
    "away_team": "team", "home_team": "opponent",
    "away_score": "score", "home_score": "opponent_score",
})
away["spread_line"] = -away["spread_line"]

team_games = pd.concat([home, away], ignore_index=True)

# Merge team's own rolling EPA
team_games = team_games.merge(rolling_epa, on=["season", "week", "team"], how="left")

# Merge opponent's rolling EPA
opp_epa = rolling_epa.rename(columns={
    "team": "opponent",
    "off_epa_rolling": "opp_off_epa_rolling",
    "def_epa_rolling": "opp_def_epa_rolling",
})
team_games = team_games.merge(opp_epa, on=["season", "week", "opponent"], how="left")

cols = [
    "season", "week", "team", "opponent", "spread_line",
    "score", "opponent_score",
    "off_epa_rolling", "def_epa_rolling",
    "opp_off_epa_rolling", "opp_def_epa_rolling",
]
result = team_games[cols].sort_values(["season", "week", "team"]).reset_index(drop=True)

out_path = OUTPUT_DIR / "nfl_combined.csv"
result.to_csv(out_path, index=False)

print(f"Exported {len(result):,} rows x {len(result.columns)} columns to {out_path}")
print(result.head(5).to_string())
