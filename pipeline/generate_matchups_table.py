"""
Generate docs/data/matchups.json for the new docs/matchups.html page, built
from data/injury_reports.parquet.

Not the {n, columns, data} columnar shape used by generate_game_table.py and
its descendants (generate_players_table.py, generate_coordinators_table.py,
generate_team_games_table.py) — that convention fits a sortable flat table,
and a matchups page is a list of per-game cards, not a table. Shape here:

{
  "weeks": [1, 2, 3],
  "current_week": 3,
  "games_by_week": {
    "1": [
      {
        "game_date": "...", "kickoff_et": "...",
        "away_team": "NE", "home_team": "SEA",
        "away_injuries": [ {player, pos, status, injury, practice: [d1,d2,d3],
                             game_day_inactive, expected_return, notes}, ... ],
        "home_injuries": [ ... ]
      }, ...
    ],
    "2": [...], "3": [...]
  }
}

current_week = the max week present, used as the page's default selection —
"upcoming" in practice, since the sheet only ever has data through the
nearest not-yet-fully-played week (see build_injury_reports_table.py's
docstring on how new weeks get added).

Within each team's injury list, rows are ordered by (1) confirmed
game-day-inactive first — regardless of status/type, since "not playing
today" is the single most actionable fact for a fantasy manager reading
this ahead of kickoff — then (2) _STATUS_RANK (the actual weekly
game-status designations first — Out down to No designation — then
healthy scratches, then season-long reserve-list entries), then (3)
player name. This ordering is a display judgment call made here, not
baked into injury_reports.parquet itself, so it's easy to change later
without rebuilding the underlying table.
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data" / "matchups.json"

_STATUS_RANK = {
    "Out": 0,
    "Doubtful": 1,
    "Questionable": 2,
    "No designation": 3,
    "Inactive (healthy/other)": 4,
    "Injured Reserve": 5,
    "IR - Designated to Return": 5,
    "PUP": 6,
    "NFI": 6,
    "Suspended": 7,
}


def _clean(v):
    return None if pd.isna(v) else v


def _player_row(r):
    return {
        "player": r["player"],
        "pos": _clean(r["pos"]),
        "status": r["status"],
        "injury": _clean(r["injury"]),
        "practice": [_clean(r["practice_day_1"]), _clean(r["practice_day_2"]), _clean(r["practice_day_3"])],
        "game_day_inactive": bool(r["game_day_inactive"]),
        "expected_return": _clean(r["expected_return"]),
        "notes": _clean(r["notes"]),
    }


def _team_injuries(df, season, week, game, team):
    rows = df[(df.season == season) & (df.week == week) & (df.game == game) & (df.team == team)].copy()
    rows["_inactive_rank"] = ~rows["game_day_inactive"]
    rows["_rank"] = rows["status"].map(_STATUS_RANK).fillna(9)
    rows = rows.sort_values(["_inactive_rank", "_rank", "player"])
    return [_player_row(r) for _, r in rows.iterrows()]


def build():
    df = pd.read_parquet(DATA_DIR / "injury_reports.parquet")

    season = int(df["season"].max())
    df = df[df.season == season]

    weeks = sorted(df["week"].unique().tolist())
    games_by_week = {}
    for week in weeks:
        wk_df = df[df.week == week]
        games = wk_df[["game", "game_date", "kickoff_et", "away_team", "home_team"]].drop_duplicates().sort_values("game_date")
        week_games = []
        for _, g in games.iterrows():
            week_games.append({
                "game_date": g["game_date"],
                "kickoff_et": g["kickoff_et"],
                "away_team": g["away_team"],
                "home_team": g["home_team"],
                "away_injuries": _team_injuries(df, season, week, g["game"], g["away_team"]),
                "home_injuries": _team_injuries(df, season, week, g["game"], g["home_team"]),
            })
        games_by_week[str(week)] = week_games

    out = {
        "season": season,
        "weeks": weeks,
        "current_week": max(weeks),
        "games_by_week": games_by_week,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    n_games = sum(len(v) for v in games_by_week.values())
    print(f"Wrote {n_games} games across {len(weeks)} weeks -> {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build()
