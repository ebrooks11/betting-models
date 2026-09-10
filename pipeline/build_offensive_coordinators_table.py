"""Build an OC-season table: one row per (team, season) with that team's
offensive coordinator(s) and the primary QB/RB/WR/TE's stats that season.
Writes data/offensive_coordinators.parquet so it's automatically picked up
as a table by pipeline/build_duckdb.py.

Run after the four position tables (build_quarterbacks_table.py,
build_running_backs_table.py, build_wide_receivers_table.py,
build_tight_ends_table.py) have been built — this script only reads their
output plus coordinators.parquet and win_totals.parquet, no raw pbp work.
    python3 pipeline/build_offensive_coordinators_table.py

Grain: one row per (team, season) that has an OC on record in
coordinators.parquet. Mid-season OC changes (e.g. Carolina 2019) are
combined into one "Name A; Name B" string per team-season, same convention
already used for the oc_name column embedded in the position tables — the
underlying player stats are already only available at team-season
granularity, not split by coordinator tenure, so a finer grain isn't
buildable from what we have.

"Primary" player per position = whoever led that team-season in the
position's main usage stat: attempts for QB, carries for RB, targets for
WR/TE. A team with a committee backfield or a timeshare at WR/TE only shows
the single top player, not a split.

Each position's stat block is a curated subset (name, games, core volume/
efficiency stats, fantasy points and points/game), not every column from
the full position tables — a full 250+ column row (52 QB + 75 RB + 75 WR +
75 TE) would defeat the point of an OC-level summary table. fantasy points
per game is computed inline here since only the quarterbacks table has that
column already built in; RB/WR/TE compute it the same way
(fantasy_points/games) rather than requiring changes to those tables.
"""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "offensive_coordinators.parquet"

# The position tables (built from seasonal_rosters) use non-canonical team
# codes in two ways coordinators.parquet/win_totals.parquet don't:
# (1) the historical code for relocated franchises (e.g. "OAK" for a 2010
#     QB, vs. the canonical current "LV"), and
# (2) nflverse's own pre-2016 codes for five franchises that don't match
#     their post-2016 codes at all (ARZ/BLT/CLV/HST/SL before 2016, vs.
#     ARI/BAL/CLE/HOU/LA from 2016 on) — confirmed by inspection, not a
#     relocation, nflverse just changed its abbreviation convention.
# Both need normalizing on the position-table side of the join, or ~11%
# (relocations only) to ~7.5% (plus the code-convention change) of
# team-seasons silently fail to find their primary players.
_TEAM_NORM = """CASE team
    WHEN 'OAK' THEN 'LV' WHEN 'SD' THEN 'LAC' WHEN 'STL' THEN 'LA'
    WHEN 'ARZ' THEN 'ARI' WHEN 'BLT' THEN 'BAL' WHEN 'CLV' THEN 'CLE'
    WHEN 'HST' THEN 'HOU' WHEN 'SL' THEN 'LA'
    ELSE team END"""

QUERY = f"""
WITH oc AS (
    SELECT team, season, string_agg(DISTINCT name, '; ') AS oc_name
    FROM read_parquet('{DATA_DIR}/coordinators.parquet')
    WHERE role_category = 'OC'
    GROUP BY team, season
),
primary_qb AS (
    SELECT * FROM (
        SELECT
            {_TEAM_NORM} AS team, season, player_name, games, attempts, passing_yards, passing_tds,
            interceptions, passing_epa,
            completion_percentage_above_expectation AS cpoe,
            fantasy_points, fantasy_points / NULLIF(games, 0) AS fantasy_points_per_game,
            row_number() OVER (PARTITION BY {_TEAM_NORM}, season ORDER BY attempts DESC) AS rn
        FROM read_parquet('{DATA_DIR}/quarterbacks.parquet')
    ) WHERE rn = 1
),
primary_rb AS (
    SELECT * FROM (
        SELECT
            {_TEAM_NORM} AS team, season, player_name, games, carries, rushing_yards, rushing_tds,
            rushing_epa, targets, receiving_yards,
            fantasy_points, fantasy_points / NULLIF(games, 0) AS fantasy_points_per_game,
            row_number() OVER (PARTITION BY {_TEAM_NORM}, season ORDER BY carries DESC) AS rn
        FROM read_parquet('{DATA_DIR}/running_backs.parquet')
    ) WHERE rn = 1
),
primary_wr AS (
    SELECT * FROM (
        SELECT
            {_TEAM_NORM} AS team, season, player_name, games, targets, receptions, receiving_yards,
            receiving_tds, receiving_epa,
            fantasy_points, fantasy_points / NULLIF(games, 0) AS fantasy_points_per_game,
            row_number() OVER (PARTITION BY {_TEAM_NORM}, season ORDER BY targets DESC) AS rn
        FROM read_parquet('{DATA_DIR}/wide_receivers.parquet')
    ) WHERE rn = 1
),
primary_te AS (
    SELECT * FROM (
        SELECT
            {_TEAM_NORM} AS team, season, player_name, games, targets, receptions, receiving_yards,
            receiving_tds, receiving_epa,
            fantasy_points, fantasy_points / NULLIF(games, 0) AS fantasy_points_per_game,
            row_number() OVER (PARTITION BY {_TEAM_NORM}, season ORDER BY targets DESC) AS rn
        FROM read_parquet('{DATA_DIR}/tight_ends.parquet')
    ) WHERE rn = 1
)
SELECT
    oc.season, oc.team, oc.oc_name,
    wt.win_total AS preseason_win_total,

    qb.player_name AS qb_name, qb.games AS qb_games, qb.attempts AS qb_attempts,
    qb.passing_yards AS qb_passing_yards, qb.passing_tds AS qb_passing_tds,
    qb.interceptions AS qb_interceptions, qb.passing_epa AS qb_passing_epa,
    qb.cpoe AS qb_cpoe, qb.fantasy_points AS qb_fantasy_points,
    qb.fantasy_points_per_game AS qb_fantasy_points_per_game,

    rb.player_name AS rb_name, rb.games AS rb_games, rb.carries AS rb_carries,
    rb.rushing_yards AS rb_rushing_yards, rb.rushing_tds AS rb_rushing_tds,
    rb.rushing_epa AS rb_rushing_epa, rb.targets AS rb_targets,
    rb.receiving_yards AS rb_receiving_yards, rb.fantasy_points AS rb_fantasy_points,
    rb.fantasy_points_per_game AS rb_fantasy_points_per_game,

    wr.player_name AS wr_name, wr.games AS wr_games, wr.targets AS wr_targets,
    wr.receptions AS wr_receptions, wr.receiving_yards AS wr_receiving_yards,
    wr.receiving_tds AS wr_receiving_tds, wr.receiving_epa AS wr_receiving_epa,
    wr.fantasy_points AS wr_fantasy_points, wr.fantasy_points_per_game AS wr_fantasy_points_per_game,

    te.player_name AS te_name, te.games AS te_games, te.targets AS te_targets,
    te.receptions AS te_receptions, te.receiving_yards AS te_receiving_yards,
    te.receiving_tds AS te_receiving_tds, te.receiving_epa AS te_receiving_epa,
    te.fantasy_points AS te_fantasy_points, te.fantasy_points_per_game AS te_fantasy_points_per_game

FROM oc
LEFT JOIN primary_qb qb ON oc.team = qb.team AND oc.season = qb.season
LEFT JOIN primary_rb rb ON oc.team = rb.team AND oc.season = rb.season
LEFT JOIN primary_wr wr ON oc.team = wr.team AND oc.season = wr.season
LEFT JOIN primary_te te ON oc.team = te.team AND oc.season = te.season
LEFT JOIN read_parquet('{DATA_DIR}/win_totals.parquet') wt
    ON oc.team = wt.team AND oc.season = wt.season
ORDER BY oc.season DESC, oc.team
"""


def build_offensive_coordinators_table():
    con = duckdb.connect(":memory:")
    df = con.execute(QUERY).fetchdf()
    con.close()

    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} OC-season rows ({df['season'].min()}-{df['season'].max()}) to {OUT_PATH}")
    return df


if __name__ == "__main__":
    build_offensive_coordinators_table()
