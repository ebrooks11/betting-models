"""Build an injury-report table: one row per (season, week, team, player),
parsed from data/raw/injury_reports/*.csv. Writes data/injury_reports.parquet
so it's automatically picked up as a table by pipeline/build_duckdb.py.

Run after new raw CSVs land in data/raw/injury_reports/:
    python3 pipeline/build_injury_reports_table.py

Source of the raw CSVs
-----------------------
Unlike every other table in this pipeline, this one has no headless fetch
script. The data lives in a Google Sheet per week (in the user's Drive
folder "NFL Injury Reports 2026", populated by the user's own scheduled
task), and this session's only access to Google Drive is through the
assistant's Drive connector tools available in conversation — not
something a plain `python3 pipeline/fetch_*.py` script run via cron can
reach without separate service-account credentials. So "fetching" is an
assistant-driven step: read each week's sheet via the Drive connector,
save it as data/raw/injury_reports/{season}_week{week:02d}.csv, and only
then does this build script (ordinary, credential-free pandas) take over.
Refreshing later (e.g. once Week 3's report is finalized) means repeating
that manual save step for the updated sheet, not just re-running this
script.

Filename convention
--------------------
data/raw/injury_reports/2026_week03.csv -> season=2026, week=3. This is
the only place season/week are recorded — the sheet itself has no such
columns, only a Game Date/title that implies them.

Column mapping
---------------
The sheet's "Game" column is a "AWAY @ HOME" matchup string (e.g.
"ATL @ GB"), split here into away_team/home_team. team is the row's own
team, i.e. whichever side of the matchup this player belongs to. All three
team columns are normalized the same way as every other table in this
pipeline (_TEAM_NORM) — the sheet uses 'LAR' for the Rams where the rest
of this pipeline uses 'LA'; applying the full historical relocation map
here too is harmless (unmatched codes pass through) and future-proofs
against older seasons being added later.

Status / List values seen across the first three 2026 weeks: No
designation, Questionable, Doubtful, Out, Inactive (healthy/other),
Injured Reserve, IR - Designated to Return, PUP, NFI, Suspended. The
first four are the actual weekly injury-report designations; the rest are
season-long roster states (reserve lists, healthy scratches) included in
the same sheet — kept as-is here, not filtered out, since which of these
matter is a display decision for whatever reads this table, not a
judgment call to bake into the data layer.

game_day_inactive is 'Y'/blank in the source, converted to bool here. It's
only populated once inactives are actually posted (~90 min before
kickoff) — null/False for a game that hasn't reached that point yet is
"not yet known", not "not inactive".
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "injury_reports"
OUT_PATH = DATA_DIR / "injury_reports.parquet"

# Same normalization convention as build_team_games_table.py / build_offensive_coordinators_table.py.
_TEAM_NORM = {
    "OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA",
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA",
}


def _norm_team(code):
    if pd.isna(code):
        return code
    code = code.strip()
    return _TEAM_NORM.get(code, code)


def _load_one(path: Path) -> pd.DataFrame:
    m = re.match(r"(\d{4})_week(\d{2})\.csv$", path.name)
    if not m:
        raise ValueError(f"Unrecognized raw injury-report filename: {path.name}")
    season, week = int(m.group(1)), int(m.group(2))

    df = pd.read_csv(path)
    away, home = zip(*df["Game"].str.split(" @ ", n=1))

    return pd.DataFrame({
        "season": season,
        "week": week,
        "game_date": df["Game Date"],
        "kickoff_et": df["Kickoff (ET)"],
        "game": df["Game"],
        "away_team": [_norm_team(t) for t in away],
        "home_team": [_norm_team(t) for t in home],
        "team": df["Team"].map(_norm_team),
        "player": df["Player"],
        "pos": df["Pos"],
        "status": df["Status / List"],
        "injury": df["Injury"],
        "practice_day_1": df["Practice Day 1"],
        "practice_day_2": df["Practice Day 2"],
        "practice_day_3": df["Practice Day 3"],
        "game_day_inactive": df["Game-Day Inactive"].fillna("") == "Y",
        "expected_return": df["Expected Return"],
        "notes": df["Notes"],
        "source": df["Source"],
        "updated_et": df["Updated (ET)"],
    })


def build_injury_reports_table():
    files = sorted(RAW_DIR.glob("*_week*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw injury-report CSVs found in {RAW_DIR}")

    df = pd.concat([_load_one(f) for f in files], ignore_index=True)
    df = df.sort_values(["season", "week", "game", "team", "player"]).reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)
    n_weeks = df[["season", "week"]].drop_duplicates().shape[0]
    print(f"Wrote {len(df):,} injury-report rows across {n_weeks} weeks to {OUT_PATH}")
    return df


if __name__ == "__main__":
    build_injury_reports_table()
