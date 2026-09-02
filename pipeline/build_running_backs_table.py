"""Build an RB-season table joining core stats, tracking/advanced metrics,
team/coaching context, and background info from every relevant source in
data/*.parquet. One row per (player, season). Writes data/running_backs.parquet
so it's automatically picked up as a table by pipeline/build_duckdb.py.

Run after data_loader.py has populated data/ (this script does no fetching,
just joins what's already cached):
    python3 pipeline/build_running_backs_table.py

Grain and population: a row exists only for players who (a) are tagged
position='RB' in seasonal_rosters and (b) had a carry or a target as an RB
that year (per pbp) — an RB on the roster who never touched the ball has
nothing to show here and is excluded. Fullbacks are not split out
separately; if seasonal_rosters tags a player 'RB' they're included.

Core box-score stats (rushing AND receiving — a modern RB's receiving work
is core, not incidental) are computed directly from pbp.parquet rather than
nflverse's own seasonal.parquet, for the same reason as the quarterbacks
table: that release has stalled (no 2025 player_stats file published by
nflverse as of this writing, confirmed against the nflverse-data GitHub
release directly), while pbp/schedules/rosters are current. See
build_quarterbacks_table.py's docstring for the full investigation.

Known conventions/approximations, validated the same way as the QB table
(see build_quarterbacks_table.py — the 2pt-attempt and sack exclusions
below were confirmed there against nflverse's own seasonal.parquet numbers,
then carried over here since they're the same underlying pbp quirks):
  - 2-point conversion attempts are flagged rush_attempt=1/pass_attempt=1 in
    raw pbp same as any other play, but are excluded from
    carries/rushing_yards/targets/receiving_yards/etc. and tracked
    separately via rushing_2pt_conversions/receiving_2pt_conversions.
  - receiving_epa sums EPA on all targets (completions AND incompletions
    thrown to this player), not just receptions — standard convention.
  - fantasy_points is standard scoring; fantasy_points_ppr adds 1 point per
    reception; fantasy_points_half_ppr adds 0.5. Unlike the QB table, these
    genuinely differ here since receiving volume matters for RBs.
  - games = count of distinct regular-season weeks with a carry or a target
    as this player — an approximation of games played.
  - Filtered to season_type='REG', matching nflverse's own seasonal.parquet
    default (s_type='REG').

Sources joined and how each is deduplicated to one row per key (mirrors
build_quarterbacks_table.py's approach — see that file for more detail on
why each of these has real duplicates in the raw data):
  - seasonal_rosters : RB population, name/team/bio. Trades produce one row
                        per team; keeps the team the player finished the
                        season with (max week).
  - pbp               : core rushing + receiving stats, computed directly.
  - ngs               : tracking metrics, joined separately for rushing and
                        receiving stat_types. week=0 is the season
                        aggregate row (weeks 1-17 are the weekly rows in
                        the same file).
  - seasonal_pfr      : PFR's exclusive advanced metrics not derivable from
                        pbp (yards before/after contact, broken tackles,
                        target depth) — NOT attempt/yardage totals, which
                        already come from pbp above. Keyed by pfr_id via
                        the ids bridge table. Mid-season trades produce a
                        combined "2TM"-style row alongside per-team rows;
                        prefers the combined row.
  - ids               : gsis_id <-> pfr_id bridge table. Has multiple
                        snapshot rows per player over time (a db_season
                        column); keeps the most recent snapshot.
  - coordinators      : team's OC that season (not DC — this is an
                        offensive-production table). Mid-season coordinator
                        changes are combined into one "Name A; Name B"
                        string rather than fanning out rows.
  - win_totals        : team's preseason win total that season.
  - draft_picks       : draft round/pick, keyed by gsis_id directly.
  - combine           : pre-draft measurables, keyed by pfr_id via the ids
                        bridge.
  - contracts         : keyed by gsis_id directly, but contracts aren't
                        season-indexed — this uses the player's single most
                        recent known contract, not the one specifically
                        active in that season. Treat as rough context only.
"""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "running_backs.parquet"

QUERY = f"""
WITH roster_rb AS (
    SELECT *
    FROM read_parquet('{DATA_DIR}/seasonal_rosters.parquet')
    WHERE position = 'RB'
    QUALIFY row_number() OVER (PARTITION BY player_id, season ORDER BY week DESC) = 1
),
pbp_rush AS (
    SELECT
        rusher_player_id AS gsis_id, season,
        COUNT(*) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0) AS carries,
        SUM(yards_gained) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0) AS rushing_yards,
        SUM(rush_touchdown) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0) AS rushing_tds,
        SUM(first_down_rush) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0) AS rushing_first_downs,
        SUM(epa) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0) AS rushing_epa,
        SUM(fumble) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0) AS rushing_fumbles,
        SUM(fumble_lost) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0) AS rushing_fumbles_lost,
        COUNT(*) FILTER (
            WHERE rush_attempt = 1 AND two_point_attempt = 1 AND two_point_conv_result = 'success'
        ) AS rushing_2pt_conversions
    FROM read_parquet('{DATA_DIR}/pbp.parquet')
    WHERE rusher_player_id IS NOT NULL AND season_type = 'REG'
    GROUP BY rusher_player_id, season
),
pbp_rec AS (
    SELECT
        receiver_player_id AS gsis_id, season,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0) AS targets,
        SUM(complete_pass) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0) AS receptions,
        SUM(yards_gained) FILTER (WHERE complete_pass = 1 AND two_point_attempt = 0) AS receiving_yards,
        SUM(pass_touchdown) FILTER (WHERE complete_pass = 1 AND two_point_attempt = 0) AS receiving_tds,
        SUM(air_yards) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0) AS receiving_air_yards,
        SUM(yards_after_catch) FILTER (WHERE complete_pass = 1) AS receiving_yards_after_catch,
        SUM(first_down_pass) FILTER (WHERE complete_pass = 1 AND two_point_attempt = 0) AS receiving_first_downs,
        SUM(epa) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0) AS receiving_epa,
        SUM(fumble) FILTER (WHERE complete_pass = 1) AS receiving_fumbles,
        SUM(fumble_lost) FILTER (WHERE complete_pass = 1) AS receiving_fumbles_lost,
        COUNT(*) FILTER (
            WHERE pass_attempt = 1 AND two_point_attempt = 1 AND two_point_conv_result = 'success'
        ) AS receiving_2pt_conversions
    FROM read_parquet('{DATA_DIR}/pbp.parquet')
    WHERE receiver_player_id IS NOT NULL AND season_type = 'REG'
    GROUP BY receiver_player_id, season
),
pbp_games AS (
    SELECT gsis_id, season, COUNT(DISTINCT week) AS games
    FROM (
        SELECT rusher_player_id AS gsis_id, season, week
        FROM read_parquet('{DATA_DIR}/pbp.parquet')
        WHERE rusher_player_id IS NOT NULL AND rush_attempt = 1 AND season_type = 'REG'
        UNION ALL
        SELECT receiver_player_id AS gsis_id, season, week
        FROM read_parquet('{DATA_DIR}/pbp.parquet')
        WHERE receiver_player_id IS NOT NULL AND pass_attempt = 1 AND season_type = 'REG'
    )
    GROUP BY gsis_id, season
),
pbp_rb_season AS (
    SELECT
        COALESCE(u.gsis_id, c.gsis_id) AS gsis_id,
        COALESCE(u.season, c.season) AS season,
        COALESCE(u.carries, 0) AS carries,
        COALESCE(u.rushing_yards, 0) AS rushing_yards,
        COALESCE(u.rushing_tds, 0) AS rushing_tds,
        COALESCE(u.rushing_first_downs, 0) AS rushing_first_downs,
        COALESCE(u.rushing_epa, 0) AS rushing_epa,
        COALESCE(u.rushing_fumbles, 0) AS rushing_fumbles,
        COALESCE(u.rushing_fumbles_lost, 0) AS rushing_fumbles_lost,
        COALESCE(u.rushing_2pt_conversions, 0) AS rushing_2pt_conversions,
        COALESCE(c.targets, 0) AS targets,
        COALESCE(c.receptions, 0) AS receptions,
        COALESCE(c.receiving_yards, 0) AS receiving_yards,
        COALESCE(c.receiving_tds, 0) AS receiving_tds,
        COALESCE(c.receiving_air_yards, 0) AS receiving_air_yards,
        COALESCE(c.receiving_yards_after_catch, 0) AS receiving_yards_after_catch,
        COALESCE(c.receiving_first_downs, 0) AS receiving_first_downs,
        COALESCE(c.receiving_epa, 0) AS receiving_epa,
        COALESCE(c.receiving_fumbles, 0) AS receiving_fumbles,
        COALESCE(c.receiving_fumbles_lost, 0) AS receiving_fumbles_lost,
        COALESCE(c.receiving_2pt_conversions, 0) AS receiving_2pt_conversions,
        (COALESCE(u.rushing_yards, 0) / 10.0
            + COALESCE(u.rushing_tds, 0) * 6
            + COALESCE(c.receiving_yards, 0) / 10.0
            + COALESCE(c.receiving_tds, 0) * 6
            - (COALESCE(u.rushing_fumbles_lost, 0) + COALESCE(c.receiving_fumbles_lost, 0)) * 2
            + (COALESCE(u.rushing_2pt_conversions, 0) + COALESCE(c.receiving_2pt_conversions, 0)) * 2
        ) AS fantasy_points,
        (COALESCE(u.rushing_yards, 0) / 10.0
            + COALESCE(u.rushing_tds, 0) * 6
            + COALESCE(c.receiving_yards, 0) / 10.0
            + COALESCE(c.receiving_tds, 0) * 6
            + COALESCE(c.receptions, 0)
            - (COALESCE(u.rushing_fumbles_lost, 0) + COALESCE(c.receiving_fumbles_lost, 0)) * 2
            + (COALESCE(u.rushing_2pt_conversions, 0) + COALESCE(c.receiving_2pt_conversions, 0)) * 2
        ) AS fantasy_points_ppr,
        (COALESCE(u.rushing_yards, 0) / 10.0
            + COALESCE(u.rushing_tds, 0) * 6
            + COALESCE(c.receiving_yards, 0) / 10.0
            + COALESCE(c.receiving_tds, 0) * 6
            + COALESCE(c.receptions, 0) * 0.5
            - (COALESCE(u.rushing_fumbles_lost, 0) + COALESCE(c.receiving_fumbles_lost, 0)) * 2
            + (COALESCE(u.rushing_2pt_conversions, 0) + COALESCE(c.receiving_2pt_conversions, 0)) * 2
        ) AS fantasy_points_half_ppr
    FROM pbp_rush u
    FULL OUTER JOIN pbp_rec c ON u.gsis_id = c.gsis_id AND u.season = c.season
),
ids_bridge AS (
    SELECT gsis_id, pfr_id
    FROM read_parquet('{DATA_DIR}/ids.parquet')
    WHERE gsis_id IS NOT NULL
    QUALIFY row_number() OVER (PARTITION BY gsis_id ORDER BY db_season DESC) = 1
),
ngs_rush AS (
    SELECT
        player_gsis_id AS gsis_id, season,
        efficiency, percent_attempts_gte_eight_defenders, avg_time_to_los,
        expected_rush_yards, rush_yards_over_expected,
        rush_yards_over_expected_per_att, rush_pct_over_expected
    FROM read_parquet('{DATA_DIR}/ngs.parquet')
    WHERE stat_type = 'rushing' AND week = 0
),
ngs_rec AS (
    SELECT
        player_gsis_id AS gsis_id, season,
        avg_cushion, avg_separation, avg_intended_air_yards AS avg_target_air_yards,
        percent_share_of_intended_air_yards, catch_percentage,
        avg_yac, avg_expected_yac, avg_yac_above_expectation
    FROM read_parquet('{DATA_DIR}/ngs.parquet')
    WHERE stat_type = 'receiving' AND week = 0
),
pfr_rush_rb AS (
    SELECT
        pfr_id, season,
        ybc AS pfr_rush_yards_before_contact, ybc_att AS pfr_rush_yards_before_contact_per_att,
        yac AS pfr_rush_yards_after_contact, yac_att AS pfr_rush_yards_after_contact_per_att,
        brk_tkl AS pfr_rush_broken_tackles, att_br AS pfr_rush_atts_per_broken_tackle
    FROM read_parquet('{DATA_DIR}/seasonal_pfr.parquet')
    WHERE stat_type = 'rush'
    QUALIFY row_number() OVER (
        PARTITION BY pfr_id, season ORDER BY (tm LIKE '%TM') DESC, att DESC
    ) = 1
),
pfr_rec_rb AS (
    SELECT
        pfr_id, season,
        ybc_r AS pfr_rec_yards_before_catch_per_rec, yac_r AS pfr_rec_yards_after_catch_per_rec,
        adot AS pfr_rec_avg_depth_of_target, rec_br AS pfr_rec_broken_tackles,
        drop AS pfr_rec_drops, drop_percent AS pfr_rec_drop_pct
    FROM read_parquet('{DATA_DIR}/seasonal_pfr.parquet')
    WHERE stat_type = 'rec'
    QUALIFY row_number() OVER (
        PARTITION BY pfr_id, season ORDER BY (tm LIKE '%TM') DESC, rec DESC
    ) = 1
),
coord AS (
    SELECT team, season, string_agg(DISTINCT name, '; ') AS oc_name
    FROM read_parquet('{DATA_DIR}/coordinators.parquet')
    WHERE role_category = 'OC'
    GROUP BY team, season
),
draft AS (
    SELECT gsis_id, round AS draft_round, pick AS draft_pick, season AS draft_year
    FROM read_parquet('{DATA_DIR}/draft_picks.parquet')
    WHERE gsis_id IS NOT NULL
    QUALIFY row_number() OVER (PARTITION BY gsis_id ORDER BY season) = 1
),
combine_rb AS (
    SELECT
        pfr_id, forty, bench, vertical, broad_jump, cone, shuttle,
        ht AS combine_height, wt AS combine_weight
    FROM read_parquet('{DATA_DIR}/combine.parquet')
    WHERE pfr_id IS NOT NULL
    QUALIFY row_number() OVER (PARTITION BY pfr_id ORDER BY season) = 1
),
contract_latest AS (
    SELECT gsis_id, apy AS contract_apy, year_signed AS contract_year_signed, years AS contract_years
    FROM read_parquet('{DATA_DIR}/contracts.parquet')
    WHERE gsis_id IS NOT NULL
    QUALIFY row_number() OVER (PARTITION BY gsis_id ORDER BY year_signed DESC) = 1
)
SELECT
    s.season,
    r.team,
    r.player_id AS gsis_id,
    r.player_name,
    r.football_name,
    r.jersey_number,
    r.birth_date,
    r.college,
    r.years_exp,

    s.carries, s.rushing_yards, s.rushing_tds, s.rushing_first_downs,
    s.rushing_epa, s.rushing_fumbles, s.rushing_fumbles_lost, s.rushing_2pt_conversions,

    s.targets, s.receptions, s.receiving_yards, s.receiving_tds,
    s.receiving_air_yards, s.receiving_yards_after_catch, s.receiving_first_downs,
    s.receiving_epa, s.receiving_fumbles, s.receiving_fumbles_lost, s.receiving_2pt_conversions,

    ngsr.efficiency, ngsr.percent_attempts_gte_eight_defenders, ngsr.avg_time_to_los,
    ngsr.expected_rush_yards, ngsr.rush_yards_over_expected,
    ngsr.rush_yards_over_expected_per_att, ngsr.rush_pct_over_expected,

    ngsc.avg_cushion, ngsc.avg_separation, ngsc.avg_target_air_yards,
    ngsc.percent_share_of_intended_air_yards, ngsc.catch_percentage,
    ngsc.avg_yac, ngsc.avg_expected_yac, ngsc.avg_yac_above_expectation,

    pfru.pfr_rush_yards_before_contact, pfru.pfr_rush_yards_before_contact_per_att,
    pfru.pfr_rush_yards_after_contact, pfru.pfr_rush_yards_after_contact_per_att,
    pfru.pfr_rush_broken_tackles, pfru.pfr_rush_atts_per_broken_tackle,

    pfrc.pfr_rec_yards_before_catch_per_rec, pfrc.pfr_rec_yards_after_catch_per_rec,
    pfrc.pfr_rec_avg_depth_of_target, pfrc.pfr_rec_broken_tackles,
    pfrc.pfr_rec_drops, pfrc.pfr_rec_drop_pct,

    wt.win_total AS preseason_win_total,
    coord.oc_name,

    d.draft_round, d.draft_pick, d.draft_year,
    cb.forty, cb.bench, cb.vertical, cb.broad_jump, cb.cone, cb.shuttle,
    cb.combine_height, cb.combine_weight,
    ct.contract_apy, ct.contract_year_signed, ct.contract_years,

    s.fantasy_points, s.fantasy_points_ppr, s.fantasy_points_half_ppr, g.games

FROM roster_rb r
JOIN pbp_rb_season s
    ON r.player_id = s.gsis_id AND r.season = s.season
LEFT JOIN pbp_games g ON r.player_id = g.gsis_id AND r.season = g.season
LEFT JOIN ids_bridge ib ON r.player_id = ib.gsis_id
LEFT JOIN ngs_rush ngsr ON r.player_id = ngsr.gsis_id AND r.season = ngsr.season
LEFT JOIN ngs_rec ngsc ON r.player_id = ngsc.gsis_id AND r.season = ngsc.season
LEFT JOIN pfr_rush_rb pfru ON ib.pfr_id = pfru.pfr_id AND r.season = pfru.season
LEFT JOIN pfr_rec_rb pfrc ON ib.pfr_id = pfrc.pfr_id AND r.season = pfrc.season
LEFT JOIN read_parquet('{DATA_DIR}/win_totals.parquet') wt
    ON r.team = wt.team AND r.season = wt.season
LEFT JOIN coord ON r.team = coord.team AND r.season = coord.season
LEFT JOIN draft d ON r.player_id = d.gsis_id
LEFT JOIN combine_rb cb ON ib.pfr_id = cb.pfr_id
LEFT JOIN contract_latest ct ON r.player_id = ct.gsis_id
ORDER BY s.season, r.team, r.player_name
"""


def build_running_backs_table():
    con = duckdb.connect(":memory:")
    df = con.execute(QUERY).fetchdf()
    con.close()

    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} RB-season rows ({df['season'].min()}-{df['season'].max()}) to {OUT_PATH}")
    return df


if __name__ == "__main__":
    build_running_backs_table()
