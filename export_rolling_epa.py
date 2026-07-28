"""Export per-game rolling EPA features (3-game window, shift 1) to parquet.

For each game, off_epa_per_play and def_epa_per_play are the mean of the
raw EPA values from the prior 1-3 games (not averages of averages).
"""

from pathlib import Path
import pandas as pd
from src.data_loader import get_pbp_data, get_schedule_data
from config import SEASONS

WINDOW = 3

print("Loading data...")
pbp = get_pbp_data(SEASONS)
schedules = get_schedule_data(SEASONS)

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

# Verification: spot-check that week 4+ rolling = mean of prior 3 raw EPAs
print("\nVerifying rolling values (checking week 4 of each team/season)...")
errors = 0
for (team, season), grp in game_epa.groupby(["team", "season"]):
    grp = grp.sort_values("week").reset_index(drop=True)
    for i in range(len(grp)):
        prior = grp["off_epa_per_play"].iloc[:i].tail(WINDOW)
        expected = prior.mean() if len(prior) > 0 else float("nan")
        actual = grp["off_epa_rolling"].iloc[i]
        if i == 0:
            if pd.notna(actual):
                print(f"  FAIL {team} {season} week {grp['week'].iloc[i]}: expected NaN, got {actual:.4f}")
                errors += 1
        else:
            if abs(actual - expected) > 1e-6:
                print(f"  FAIL {team} {season} week {grp['week'].iloc[i]}: expected {expected:.4f}, got {actual:.4f}")
                errors += 1
if errors == 0:
    print("  All rolling values verified correct.")
else:
    print(f"  {errors} errors found.")

# Rest days (within season only)
games = schedules[schedules["game_type"] == "REG"].copy()
games["gameday"] = pd.to_datetime(games["gameday"])
games["home_team"] = games["home_team"].replace(TEAM_MAP)
games["away_team"] = games["away_team"].replace(TEAM_MAP)

home = games[["season", "week", "home_team", "away_team", "gameday"]].rename(
    columns={"home_team": "team", "away_team": "opponent"}
)
away = games[["season", "week", "away_team", "home_team", "gameday"]].rename(
    columns={"away_team": "team", "home_team": "opponent"}
)
team_games = pd.concat([home, away], ignore_index=True)
team_games = team_games.sort_values(["team", "season", "week"]).reset_index(drop=True)

team_games["rest_days"] = (
    team_games.groupby(["team", "season"])["gameday"]
    .diff().dt.days.fillna(7)
)

# Merge opponent rest days and compute rest advantage
opp_rest = team_games[["season", "week", "team", "rest_days"]].copy()
team_games = team_games.merge(
    opp_rest.rename(columns={"team": "opponent", "rest_days": "opp_rest_days"}),
    on=["season", "week", "opponent"], how="left"
)
team_games["rest_advantage"] = team_games["rest_days"] - team_games["opp_rest_days"]

game_epa = game_epa.merge(
    team_games[["season", "week", "team", "rest_days", "rest_advantage"]],
    on=["season", "week", "team"], how="left"
)

result = game_epa[["season", "week", "team",
                    "off_epa_per_play", "off_epa_rolling",
                    "def_epa_per_play", "def_epa_rolling",
                    "rest_days", "rest_advantage"]].copy()
result = result.sort_values(["season", "week", "team"]).reset_index(drop=True)

out_path = Path("exports/features.parquet")
out_path.parent.mkdir(exist_ok=True)
result.to_parquet(out_path, index=False)

print(f"\nExported {len(result):,} rows to {out_path}")
print("\nSample — DAL weeks 1-5 of 2022:")
sample = result[(result["team"] == "DAL") & (result["season"] == 2022)].head(5)
print(sample[["season", "week", "team", "off_epa_rolling", "def_epa_rolling", "rest_days", "rest_advantage"]].to_string(index=False))
