"""Fetch and merge just the current NFL season's schedules, pbp, NGS, and
coordinator data, then rebuild that season's slice of team_games.parquet
and splice it into the existing table. Built for
.github/workflows/weekly_stats_refresh.yml — see that workflow and
build_team_games_table.py's docstring ("Building from a partial
pbp.parquet") for why this exists instead of just re-running
build_team_games_table.py directly: doing that needs the FULL historical
pbp.parquet (hundreds of MB across ~20 seasons) present locally, which
this pipeline deliberately never persists — too large to commit to git,
too slow to refetch every run. This script instead:
  1. Fetches ONLY the current season's schedules/pbp/NGS/coordinators (a
     few thousand rows total, seconds — not the full historical archive)
  2. Runs build_team_games_table.py's exact SQL against just that slice
  3. Splices the result into data/team_games.parquet and
     data/coordinators.parquet, replacing only that season's rows and
     leaving every other season exactly as already there

data/team_games.parquet and data/coordinators.parquet are small enough
(2MB and 20KB as of writing) to be the two exceptions carved out of this
pipeline's usual "data/ is gitignored, always regenerable, never
committed" rule (see .gitignore) — specifically so a fresh checkout
already has full history in them, and only the current season needs
refreshing here.

Run locally to test — writes the current season's raw schedules/pbp/NGS
to data/raw/_current_season/, NOT data/schedules.parquet or
data/pbp.parquet directly, so it can't clobber a full local historical
copy of those you might already have for other pages:
    python3 pipeline/fetch_current_season_data.py
"""

import sys
from datetime import date
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.build_team_games_table import build_team_games_table
from pipeline.fetch_coordinators import ALL_TEAMS, scrape_team_season

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCRATCH_DIR = DATA_DIR / "raw" / "_current_season"


def current_season() -> int:
    """NFL seasons are labeled by the year they start in (games run
    roughly September through the following February) — treating
    Jan/Feb as still "last year's" season, matching how nflverse itself
    numbers seasons."""
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


def _fetch_schedules(season: int) -> pd.DataFrame:
    return nfl.import_schedules([season])


def _fetch_pbp(season: int) -> pd.DataFrame:
    # include_participation=False: the participation-data merge (which
    # supplies offense_personnel/offense_formation, among others) doesn't
    # exist yet for an in-progress season — confirmed directly, requesting
    # it for 2026 raises a 404 (masked by a real exception-handling bug in
    # nfl_data_py that surfaces as a confusing NameError instead). Those
    # columns just come back null for the current season's team_games
    # rows, same "floor" pattern as several other columns already
    # documented in build_team_games_table.py.
    df = nfl.import_pbp_data([season], downcast=True, include_participation=False)
    # build_team_games_table.py's SQL references offense_personnel by
    # name unconditionally (it handles the column being present-but-null,
    # not being absent outright) — add it as an all-null column so that
    # query still runs; the resulting personnel_* stats correctly come
    # back null for the current season either way.
    if "offense_personnel" not in df.columns:
        # Must be a proper (nullable) string dtype, not bare Python None —
        # an all-None object column round-trips through parquet as a type
        # DuckDB's regexp_extract() can't bind against.
        df["offense_personnel"] = pd.Series([pd.NA] * len(df), dtype="string")
    return df


def _fetch_ngs(season: int) -> pd.DataFrame:
    dfs = []
    for stat_type in ["passing", "rushing", "receiving"]:
        df = nfl.import_ngs_data(stat_type, [season])
        df["stat_type"] = stat_type
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _scrape_coordinators(season: int) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research scraper)"})
    rows = []
    for team in ALL_TEAMS:
        try:
            rows.extend(scrape_team_season(team, season, session))
        except Exception as e:
            print(f"  {season} {team}: ERROR - {e}")
    return pd.DataFrame(rows)


def _splice(existing_path: Path, fresh: pd.DataFrame, season: int, sort_cols: list) -> pd.DataFrame:
    """Replace `season`'s rows in the committed table at existing_path
    with `fresh`, leaving every other season's rows untouched."""
    existing = pd.read_parquet(existing_path)
    existing = existing[existing.season != season]
    combined = pd.concat([existing, fresh], ignore_index=True)
    return combined.sort_values(sort_cols).reset_index(drop=True)


def refresh():
    season = current_season()
    print(f"Refreshing current-season ({season}) data...")
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    schedules = _fetch_schedules(season)
    schedules_path = SCRATCH_DIR / "schedules.parquet"
    schedules.to_parquet(schedules_path, index=False)
    print(f"  schedules: {len(schedules)} rows")

    pbp = _fetch_pbp(season)
    pbp_path = SCRATCH_DIR / "pbp.parquet"
    pbp.to_parquet(pbp_path, index=False)
    print(f"  pbp: {len(pbp)} rows")

    ngs = _fetch_ngs(season)
    ngs_path = SCRATCH_DIR / "ngs.parquet"
    ngs.to_parquet(ngs_path, index=False)
    print(f"  ngs: {len(ngs)} rows")

    coordinators_path = DATA_DIR / "coordinators.parquet"
    coord_new = _scrape_coordinators(season)
    if coord_new.empty:
        print(f"  coordinators: no rows scraped for {season} (Wikipedia's staff section likely not published yet for this season) — leaving existing data as-is")
    else:
        combined_coord = _splice(coordinators_path, coord_new, season, ["season", "team", "role_category"])
        combined_coord.to_parquet(coordinators_path, index=False)
        print(f"  coordinators: {len(coord_new)} rows scraped for {season}")

    fresh_team_games = build_team_games_table(
        pbp_path=pbp_path,
        schedules_path=schedules_path,
        coordinators_path=coordinators_path,
        ngs_path=ngs_path,
        out_path=SCRATCH_DIR / "team_games.parquet",
    )

    team_games_path = DATA_DIR / "team_games.parquet"
    combined_tg = _splice(team_games_path, fresh_team_games, season, ["season", "week", "team"])
    combined_tg.to_parquet(team_games_path, index=False)
    print(f"Spliced {len(fresh_team_games)} {season} team-game rows into {team_games_path} ({len(combined_tg):,} total)")


if __name__ == "__main__":
    refresh()
