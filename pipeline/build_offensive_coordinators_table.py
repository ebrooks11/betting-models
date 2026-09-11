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

RB/run-game metrics (run rate, formation splits, screen rate, etc.) are
documented in detail in COORDINATOR_RB_METRICS.md in this directory,
including which ones have a season floor (formations 2016+, screens 2022+,
yards-before-contact 2018+) and which requested metric (route participation
by RBs) isn't derivable from any data we have and was replaced with a
clearly-labeled substitute (RB target share). Read that file before
changing any of the run-game CTEs below — several of the definitions in
there (rush % denominator, explosive-run threshold, formation list) are
judgment calls, not standards everyone agrees on.
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
rb_ranked AS (
    SELECT
        {_TEAM_NORM} AS team, season, player_name, games, carries, rushing_yards, rushing_tds,
        rushing_epa, targets, receiving_yards,
        fantasy_points, fantasy_points / NULLIF(games, 0) AS fantasy_points_per_game,
        row_number() OVER (PARTITION BY {_TEAM_NORM}, season ORDER BY carries DESC) AS rn
    FROM read_parquet('{DATA_DIR}/running_backs.parquet')
),
primary_rb AS (SELECT * FROM rb_ranked WHERE rn = 1),
secondary_rb AS (SELECT * FROM rb_ranked WHERE rn = 2),
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
),
-- pbp.posteam and seasonal_pfr.tm are both already canonical team codes
-- (verified directly — unlike seasonal_rosters, no _TEAM_NORM needed here).
rb_lookup AS (
    SELECT DISTINCT player_id AS gsis_id, season
    FROM read_parquet('{DATA_DIR}/seasonal_rosters.parquet')
    WHERE position = 'RB'
),
team_play_rates AS (
    SELECT
        posteam AS team, season,
        COUNT(*) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0) AS rush_plays,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0) AS pass_plays,
        COUNT(*) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0 AND yardline_100 <= 20) AS rush_rz20,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0 AND yardline_100 <= 20) AS pass_rz20,
        COUNT(*) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0 AND yardline_100 <= 10) AS rush_rz10,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0 AND yardline_100 <= 10) AS pass_rz10,
        COUNT(*) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0 AND yardline_100 <= 5) AS rush_rz5,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0 AND yardline_100 <= 5) AS pass_rz5
    FROM read_parquet('{DATA_DIR}/pbp.parquet')
    WHERE season_type = 'REG' AND (rush_attempt = 1 OR pass_attempt = 1)
    GROUP BY posteam, season
),
-- Formation values observed in the data (2016+ only, see COORDINATOR_RB_METRICS.md):
-- SHOTGUN, SINGLEBACK, UNDER CENTER, I_FORM, EMPTY, PISTOL, JUMBO, WILDCAT.
formation_stats AS (
    SELECT
        posteam AS team, season,
        COUNT(*) FILTER (WHERE offense_formation IS NOT NULL) AS formation_known_plays,
        COUNT(*) FILTER (WHERE offense_formation = 'SHOTGUN') AS f_shotgun_n,
        AVG(epa) FILTER (WHERE offense_formation = 'SHOTGUN') AS f_shotgun_epa,
        AVG(success) FILTER (WHERE offense_formation = 'SHOTGUN') AS f_shotgun_success,
        COUNT(*) FILTER (WHERE offense_formation = 'SINGLEBACK') AS f_singleback_n,
        AVG(epa) FILTER (WHERE offense_formation = 'SINGLEBACK') AS f_singleback_epa,
        AVG(success) FILTER (WHERE offense_formation = 'SINGLEBACK') AS f_singleback_success,
        COUNT(*) FILTER (WHERE offense_formation = 'UNDER CENTER') AS f_under_center_n,
        AVG(epa) FILTER (WHERE offense_formation = 'UNDER CENTER') AS f_under_center_epa,
        AVG(success) FILTER (WHERE offense_formation = 'UNDER CENTER') AS f_under_center_success,
        COUNT(*) FILTER (WHERE offense_formation = 'I_FORM') AS f_i_form_n,
        AVG(epa) FILTER (WHERE offense_formation = 'I_FORM') AS f_i_form_epa,
        AVG(success) FILTER (WHERE offense_formation = 'I_FORM') AS f_i_form_success,
        COUNT(*) FILTER (WHERE offense_formation = 'EMPTY') AS f_empty_n,
        AVG(epa) FILTER (WHERE offense_formation = 'EMPTY') AS f_empty_epa,
        AVG(success) FILTER (WHERE offense_formation = 'EMPTY') AS f_empty_success,
        COUNT(*) FILTER (WHERE offense_formation = 'PISTOL') AS f_pistol_n,
        AVG(epa) FILTER (WHERE offense_formation = 'PISTOL') AS f_pistol_epa,
        AVG(success) FILTER (WHERE offense_formation = 'PISTOL') AS f_pistol_success,
        COUNT(*) FILTER (WHERE offense_formation = 'JUMBO') AS f_jumbo_n,
        AVG(epa) FILTER (WHERE offense_formation = 'JUMBO') AS f_jumbo_epa,
        AVG(success) FILTER (WHERE offense_formation = 'JUMBO') AS f_jumbo_success,
        COUNT(*) FILTER (WHERE offense_formation = 'WILDCAT') AS f_wildcat_n,
        AVG(epa) FILTER (WHERE offense_formation = 'WILDCAT') AS f_wildcat_epa,
        AVG(success) FILTER (WHERE offense_formation = 'WILDCAT') AS f_wildcat_success
    FROM read_parquet('{DATA_DIR}/pbp.parquet')
    WHERE season_type = 'REG' AND (rush_attempt = 1 OR pass_attempt = 1)
    GROUP BY posteam, season
),
screen_stats AS (
    SELECT
        p.posteam AS team, p.season,
        COUNT(*) FILTER (WHERE p.pass_attempt = 1) AS pass_plays,
        COUNT(*) FILTER (WHERE p.pass_attempt = 1 AND f.is_screen_pass) AS screen_plays
    FROM read_parquet('{DATA_DIR}/pbp.parquet') p
    JOIN read_parquet('{DATA_DIR}/ftn.parquet') f
        ON p.game_id = f.nflverse_game_id AND p.play_id = f.nflverse_play_id
    WHERE p.season_type = 'REG'
    GROUP BY p.posteam, p.season
),
rb_rush_stats AS (
    SELECT
        p.posteam AS team, p.season,
        COUNT(*) AS rb_group_carries,
        AVG(p.epa) AS rb_group_rush_epa,
        COUNT(*) FILTER (WHERE p.yards_gained >= 10) * 1.0 / COUNT(*) AS rb_group_explosive_rate
    FROM read_parquet('{DATA_DIR}/pbp.parquet') p
    JOIN rb_lookup r ON p.rusher_player_id = r.gsis_id AND p.season = r.season
    WHERE p.season_type = 'REG' AND p.rush_attempt = 1 AND p.two_point_attempt = 0
    GROUP BY p.posteam, p.season
),
rb_target_stats AS (
    SELECT
        p.posteam AS team, p.season,
        COUNT(*) AS rb_group_targets
    FROM read_parquet('{DATA_DIR}/pbp.parquet') p
    JOIN rb_lookup r ON p.receiver_player_id = r.gsis_id AND p.season = r.season
    WHERE p.season_type = 'REG' AND p.pass_attempt = 1 AND p.two_point_attempt = 0
    GROUP BY p.posteam, p.season
),
pfr_rb_ybc AS (
    SELECT
        tm AS team, season,
        SUM(ybc) AS rb_group_ybc_total,
        SUM(att) AS rb_group_att_total
    FROM read_parquet('{DATA_DIR}/seasonal_pfr.parquet')
    WHERE stat_type = 'rush' AND pos = 'RB'
    GROUP BY tm, season
)
SELECT
    oc.season, oc.team, oc.oc_name,

    rb.carries / NULLIF(tpr.rush_plays, 0) AS primary_rb_rush_share,
    rb2.player_name AS secondary_rb_name, rb2.carries AS secondary_rb_carries,
    rb2.carries / NULLIF(tpr.rush_plays, 0) AS secondary_rb_rush_share,

    tpr.rush_plays / NULLIF(tpr.rush_plays + tpr.pass_plays, 0) AS run_rate,
    tpr.rush_rz20 / NULLIF(tpr.rush_rz20 + tpr.pass_rz20, 0) AS rush_rate_inside_20,
    tpr.rush_rz10 / NULLIF(tpr.rush_rz10 + tpr.pass_rz10, 0) AS rush_rate_inside_10,
    tpr.rush_rz5 / NULLIF(tpr.rush_rz5 + tpr.pass_rz5, 0) AS rush_rate_inside_5,

    scr.screen_plays / NULLIF(scr.pass_plays, 0) AS screen_rate,
    rbt.rb_group_targets / NULLIF(tpr.pass_plays, 0) AS rb_target_share,

    rrs.rb_group_carries, rrs.rb_group_rush_epa, rrs.rb_group_explosive_rate,
    ybc.rb_group_ybc_total / NULLIF(ybc.rb_group_att_total, 0) AS rb_group_ybc_per_rush,

    fs.formation_known_plays,
    fs.f_shotgun_n / NULLIF(fs.formation_known_plays, 0) AS formation_shotgun_rate,
    fs.f_shotgun_epa AS formation_shotgun_epa,
    fs.f_shotgun_success AS formation_shotgun_success_rate,
    fs.f_singleback_n / NULLIF(fs.formation_known_plays, 0) AS formation_singleback_rate,
    fs.f_singleback_epa AS formation_singleback_epa,
    fs.f_singleback_success AS formation_singleback_success_rate,
    fs.f_under_center_n / NULLIF(fs.formation_known_plays, 0) AS formation_under_center_rate,
    fs.f_under_center_epa AS formation_under_center_epa,
    fs.f_under_center_success AS formation_under_center_success_rate,
    fs.f_i_form_n / NULLIF(fs.formation_known_plays, 0) AS formation_i_form_rate,
    fs.f_i_form_epa AS formation_i_form_epa,
    fs.f_i_form_success AS formation_i_form_success_rate,
    fs.f_empty_n / NULLIF(fs.formation_known_plays, 0) AS formation_empty_rate,
    fs.f_empty_epa AS formation_empty_epa,
    fs.f_empty_success AS formation_empty_success_rate,
    fs.f_pistol_n / NULLIF(fs.formation_known_plays, 0) AS formation_pistol_rate,
    fs.f_pistol_epa AS formation_pistol_epa,
    fs.f_pistol_success AS formation_pistol_success_rate,
    fs.f_jumbo_n / NULLIF(fs.formation_known_plays, 0) AS formation_jumbo_rate,
    fs.f_jumbo_epa AS formation_jumbo_epa,
    fs.f_jumbo_success AS formation_jumbo_success_rate,
    fs.f_wildcat_n / NULLIF(fs.formation_known_plays, 0) AS formation_wildcat_rate,
    fs.f_wildcat_epa AS formation_wildcat_epa,
    fs.f_wildcat_success AS formation_wildcat_success_rate,

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
LEFT JOIN secondary_rb rb2 ON oc.team = rb2.team AND oc.season = rb2.season
LEFT JOIN primary_wr wr ON oc.team = wr.team AND oc.season = wr.season
LEFT JOIN primary_te te ON oc.team = te.team AND oc.season = te.season
LEFT JOIN read_parquet('{DATA_DIR}/win_totals.parquet') wt
    ON oc.team = wt.team AND oc.season = wt.season
LEFT JOIN team_play_rates tpr ON oc.team = tpr.team AND oc.season = tpr.season
LEFT JOIN formation_stats fs ON oc.team = fs.team AND oc.season = fs.season
LEFT JOIN screen_stats scr ON oc.team = scr.team AND oc.season = scr.season
LEFT JOIN rb_rush_stats rrs ON oc.team = rrs.team AND oc.season = rrs.season
LEFT JOIN rb_target_stats rbt ON oc.team = rbt.team AND oc.season = rbt.season
LEFT JOIN pfr_rb_ybc ybc ON oc.team = ybc.team AND oc.season = ybc.season
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
