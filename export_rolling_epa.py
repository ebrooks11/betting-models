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
pass_plays = plays[(plays["play_type"] == "pass") & plays["cpoe"].notna()].copy()
non_to_plays = plays[plays["fumble_lost"] == 0].copy()

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

cpoe = (
    pass_plays.groupby(["season", "week", "posteam"])
    .agg(cpoe_per_game=("cpoe", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

offense_no_to = (
    non_to_plays.groupby(["season", "week", "posteam"])
    .agg(off_epa_no_to=("epa", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

defense_no_to = (
    non_to_plays.groupby(["season", "week", "defteam"])
    .agg(def_epa_no_to=("epa", "mean"))
    .reset_index()
    .rename(columns={"defteam": "team"})
)

early_down_plays = plays[plays["down"].isin([1, 2])].copy()

offense_early = (
    early_down_plays.groupby(["season", "week", "posteam"])
    .agg(off_epa_early_down=("epa", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

defense_early = (
    early_down_plays.groupby(["season", "week", "defteam"])
    .agg(def_epa_early_down=("epa", "mean"))
    .reset_index()
    .rename(columns={"defteam": "team"})
)

plays_per_game = (
    plays.groupby(["season", "week", "posteam"])
    .size()
    .reset_index(name="plays")
    .rename(columns={"posteam": "team"})
)

# Time of possession: sum drive_time_of_possession (MM:SS) per team per game
drives = pbp[pbp["drive_time_of_possession"].notna() & pbp["posteam"].notna()].copy()
drives = drives[["season", "week", "posteam", "fixed_drive", "drive_time_of_possession"]].drop_duplicates()

def parse_mmss(s):
    parts = str(s).split(":")
    return int(parts[0]) * 60 + int(parts[1])

drives["top_seconds"] = drives["drive_time_of_possession"].apply(parse_mmss)
top = (
    drives.groupby(["season", "week", "posteam"])["top_seconds"]
    .sum()
    .reset_index()
    .rename(columns={"posteam": "team", "top_seconds": "top_seconds_per_game"})
)

TEAM_MAP = {
    "OAK": "LV",
    "SD":  "LAC",
    "STL": "LA",
}

game_epa = offense.merge(defense, on=["season", "week", "team"], how="outer")
game_epa = game_epa.merge(cpoe, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(offense_no_to, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(defense_no_to, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(offense_early, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(defense_early, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(plays_per_game, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(top, on=["season", "week", "team"], how="left")
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
game_epa["cpoe_rolling"] = (
    game_epa.groupby(["team", "season"])["cpoe_per_game"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["off_epa_no_to_rolling"] = (
    game_epa.groupby(["team", "season"])["off_epa_no_to"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["def_epa_no_to_rolling"] = (
    game_epa.groupby(["team", "season"])["def_epa_no_to"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["plays_per_game_rolling"] = (
    game_epa.groupby(["team", "season"])["plays"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["off_epa_early_down_rolling"] = (
    game_epa.groupby(["team", "season"])["off_epa_early_down"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["def_epa_early_down_rolling"] = (
    game_epa.groupby(["team", "season"])["def_epa_early_down"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["top_rolling"] = (
    game_epa.groupby(["team", "season"])["top_seconds_per_game"]
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
                    "cpoe_per_game", "cpoe_rolling",
                    "off_epa_no_to", "off_epa_no_to_rolling",
                    "def_epa_no_to", "def_epa_no_to_rolling",
                    "off_epa_early_down", "off_epa_early_down_rolling",
                    "def_epa_early_down", "def_epa_early_down_rolling",
                    "plays", "plays_per_game_rolling",
                    "top_seconds_per_game", "top_rolling",
                    "rest_days", "rest_advantage"]].copy()
result = result.sort_values(["season", "week", "team"]).reset_index(drop=True)

out_path = Path("exports/features.parquet")
out_path.parent.mkdir(exist_ok=True)
result.to_parquet(out_path, index=False)

print(f"\nExported {len(result):,} rows to {out_path}")
print("\nSample — DAL weeks 1-5 of 2022:")
sample = result[(result["team"] == "DAL") & (result["season"] == 2022)].head(5)
print(sample[["season", "week", "team", "off_epa_rolling", "def_epa_rolling", "rest_days", "rest_advantage"]].to_string(index=False))
