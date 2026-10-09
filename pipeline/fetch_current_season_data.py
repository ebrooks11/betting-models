"""Fetch and merge just the current NFL season's schedules, pbp, NGS,
coordinator, and preseason-win-total data, then rebuild that season's
slice of team_games.parquet and splice it into the existing table. Built
for .github/workflows/weekly_stats_refresh.yml — see that workflow and
build_team_games_table.py's docstring ("Building from a partial
pbp.parquet") for why this exists instead of just re-running
build_team_games_table.py directly: doing that needs the FULL historical
pbp.parquet (hundreds of MB across ~20 seasons) present locally, which
this pipeline deliberately never persists — too large to commit to git,
too slow to refetch every run. This script instead:
  1. Fetches ONLY the current season's schedules/pbp/NGS/coordinators/win
     totals (a few thousand rows total, seconds — not the full historical
     archive)
  2. Runs build_team_games_table.py's exact SQL against just the
     schedules/pbp/NGS/coordinators slice
  3. Splices every result into its existing committed table
     (team_games.parquet, coordinators.parquet, win_totals.parquet,
     snap_counts.parquet), replacing only that season's rows and leaving
     every other season exactly as already there
  4. Also refreshes data/coordinators.parquet's current-season rows from
     Wikipedia's per-team staff navbox when the usual season-article
     scrape comes up empty (see fetch_coordinators.scrape_current_staff),
     and data/depth_chart.parquet from ESPN's depth chart pages (see
     fetch_depth_chart.py) — both independent of pbp/schedules, but run
     here so one script keeps everything current-season current.

data/team_games.parquet, data/coordinators.parquet,
data/win_totals.parquet, data/snap_counts.parquet, and
data/depth_chart.parquet are small enough to be the exceptions carved out
of this pipeline's usual "data/ is gitignored, always regenerable, never
committed" rule (see .gitignore) — specifically so a fresh checkout
already has full history in them, and only the current season needs
refreshing here. data/depth_chart.parquet accumulates one snapshot per
week (see _splice_week) rather than being a reconstructable rebuild, the
same reasoning as data/injury_reports.parquet.

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
from pipeline import fetch_depth_chart
from pipeline.build_team_games_table import build_team_games_table
from pipeline.fetch_coordinators import ALL_TEAMS, scrape_current_staff, scrape_team_season
from pipeline.fetch_win_totals import scrape_season

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


def _fetch_snap_counts(season: int) -> pd.DataFrame:
    # PFR publishes snap counts within a day or two of each game, so this
    # is available well before the season ends — used by
    # generate_matchups_table.py to tell which injured players were
    # playing starter-level snaps before they got hurt.
    return nfl.import_snap_counts([season])


def _scrape_coordinators(season: int) -> pd.DataFrame:
    # scrape_team_season() reads the season article's own staff table,
    # which Wikipedia editors only fill in at/after a season ends — for
    # an in-progress season it comes back empty for every team (confirmed
    # directly). Fall back to scrape_current_staff(), which reads each
    # team's continuously-maintained "Template:{Team} staff" navbox
    # instead — see that function's docstring for why it's reliable
    # mid-season despite being a different Wikipedia page than the
    # historical scrape uses. Only used as a fallback (not preferred
    # outright) so a season that HAS finished still gets the season
    # article's more stable, dated snapshot rather than a navbox that
    # already reflects next season's hires.
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research scraper)"})
    rows = []
    for team in ALL_TEAMS:
        try:
            team_rows = scrape_team_season(team, season, session)
            if not team_rows:
                team_rows = scrape_current_staff(team, season, session)
            rows.extend(team_rows)
        except Exception as e:
            print(f"  {season} {team}: ERROR - {e}")
    return pd.DataFrame(rows)


def _scrape_win_totals(season: int) -> pd.DataFrame:
    # Preseason win totals are published before the season starts and
    # never change — only needs fetching once per season, but no harm in
    # re-fetching it weekly along with everything else here.
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research scraper)"})
    rows = scrape_season(season, session)
    return pd.DataFrame(rows)


def _splice(existing_path: Path, fresh: pd.DataFrame, season: int, sort_cols: list) -> pd.DataFrame:
    """Replace `season`'s rows in the committed table at existing_path
    with `fresh`, leaving every other season's rows untouched."""
    existing = pd.read_parquet(existing_path)
    existing = existing[existing.season != season]
    combined = pd.concat([existing, fresh], ignore_index=True)
    return combined.sort_values(sort_cols).reset_index(drop=True)


def _splice_week(existing_path: Path, fresh: pd.DataFrame, season: int, week: int, sort_cols: list) -> pd.DataFrame:
    """Like _splice, but replaces only `season`'s `week` — used for
    data/depth_chart.parquet, which (unlike team_games.parquet etc.)
    accumulates one snapshot per week rather than being fully
    reconstructable from a current-season slice, the same reasoning as
    build_injury_reports_table.py's week-level splice. ESPN's depth chart
    only ever shows "right now" — there's no historical endpoint to
    backfill from — so each week this runs is the only chance to capture
    that week's snapshot durably; overwriting the whole season on every
    run (like _splice does) would silently throw away every earlier
    week's already-captured snapshot instead of just the one being
    refreshed.
    """
    if not existing_path.exists():
        return fresh.sort_values(sort_cols).reset_index(drop=True)
    existing = pd.read_parquet(existing_path)
    existing = existing[~((existing.season == season) & (existing.week == week))]
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

    # Also splice into the real, tracked data/schedules.parquet — not just
    # the scratch copy above, which only exists as an input to
    # build_team_games_table() below. generate_matchups_table.py reads
    # data/schedules.parquet directly for its games list (home/away teams,
    # dates, scores), so that file needs the current season's latest
    # results too, not just team_games.parquet.
    committed_schedules_path = DATA_DIR / "schedules.parquet"
    combined_schedules = _splice(committed_schedules_path, schedules, season, ["season", "week", "gameday", "home_team"])
    combined_schedules.to_parquet(committed_schedules_path, index=False)

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

    win_totals_path = DATA_DIR / "win_totals.parquet"
    win_new = _scrape_win_totals(season)
    if win_new.empty:
        print(f"  win totals: no rows scraped for {season} — leaving existing data as-is")
    else:
        combined_win = _splice(win_totals_path, win_new, season, ["season", "team"])
        combined_win.to_parquet(win_totals_path, index=False)
        print(f"  win totals: {len(win_new)} rows scraped for {season}")

    snap_counts_path = DATA_DIR / "snap_counts.parquet"
    snaps_new = _fetch_snap_counts(season)
    if snaps_new.empty:
        print(f"  snap counts: no rows for {season} yet — leaving existing data as-is")
    else:
        combined_snaps = _splice(snap_counts_path, snaps_new, season, ["season", "week", "team", "player"])
        combined_snaps.to_parquet(snap_counts_path, index=False)
        print(f"  snap counts: {len(snaps_new)} rows for {season}")

    # Same "earliest week with a game not yet played" rule
    # generate_matchups_table.py uses for current_week — tags this run's
    # depth-chart scrape (see below) with which week it's a snapshot of.
    unplayed = schedules[schedules["home_score"].isna()]
    current_week = int(unplayed["week"].min()) if not unplayed.empty else int(schedules["week"].max())

    depth_chart_path = DATA_DIR / "depth_chart.parquet"
    depth_new = pd.DataFrame(fetch_depth_chart.scrape_all(season))
    if depth_new.empty:
        print(f"  depth chart: no rows scraped — leaving existing data as-is")
    else:
        depth_new["week"] = current_week
        combined_depth = _splice_week(
            depth_chart_path, depth_new, season, current_week,
            ["season", "week", "team", "side", "slot_index", "depth_rank"],
        )
        combined_depth.to_parquet(depth_chart_path, index=False)
        print(f"  depth chart: {len(depth_new)} rows for {season} week {current_week}")

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
