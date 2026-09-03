"""Build a QB-season table joining core stats, tracking/advanced metrics,
QBR, team/coaching context, and background info from every relevant source
in data/*.parquet. One row per (player, season). Writes data/quarterbacks.parquet
so it's automatically picked up as a table by pipeline/build_duckdb.py.

Run after data_loader.py has populated data/ (this script does no fetching,
just joins what's already cached):
    python3 pipeline/build_quarterbacks_table.py

Grain and population: a QB-season row exists only for players who (a) are
tagged position='QB' in seasonal_rosters and (b) threw a pass or ran the ball
as a QB that year (per pbp) — a QB on the roster who never played has
nothing to show here and is excluded.

Core box-score stats are computed directly from pbp.parquet, NOT from
nflverse's own seasonal.parquet, because that release has stalled — as of
this writing nflverse hasn't published a 2025 player_stats file at all
(checked directly against the nflverse-data GitHub release; the newest
per-season asset is player_stats_2024.parquet from May 2025), even though
pbp/schedules/rosters are current. Recomputing from pbp keeps this table
current without depending on that release. Known differences from what
nflverse's own seasonal.parquet would have produced:
  - dakota (nflverse's proprietary EPA+CPOE composite, fit via regression)
    is dropped entirely — not reproducible without their model weights.
  - fantasy_points/fantasy_points_ppr/fantasy_points_half_ppr are recomputed
    via standard scoring (pass yards/25, pass TD*4, INT*-2, rush yards/10,
    rush TD*6, lost fumble*-2, 2pt*2) rather than copied from nflverse's own
    calculation. QB receiving stats aren't tracked (trick-play catches by a
    QB are rare enough to ignore), so all three scoring variants are the
    same value here — PPR/half-PPR only differ from standard when
    receptions are involved. fantasy_points_per_game = fantasy_points/games
    (NULL when games is 0).
  - A "sack" in raw pbp is flagged with pass_attempt=1 (confirmed by direct
    inspection), so pass attempts/completions/yards all explicitly exclude
    sack plays (sack=0) to match traditional passing-stat conventions —
    sacks are tracked separately via sacks/sack_yards. Two EPA columns are
    provided: passing_epa excludes sacks (matches the attempts/completions
    convention above); passing_epa_with_sacks includes them, since a sack
    is a real drop in win probability charged to the passing play and some
    analyses want that counted.
  - 2-point conversion attempts are also flagged with pass_attempt=1 (or
    rush_attempt=1) in raw pbp but are excluded from attempts/completions/
    yards/etc. — validated directly against nflverse's seasonal.parquet,
    which does the same (a 469-attempt season showing up as 479 in raw pbp
    was exactly 10 2pt attempts). Tracked separately via
    passing_2pt_conversions/rushing_2pt_conversions instead.
  - games = count of distinct regular-season weeks with a pass or rush
    attempt as this player — an approximation of games played, not
    necessarily identical to nflverse's own definition.
  - Filtered to season_type='REG', matching nflverse's own seasonal.parquet
    default (s_type='REG').

Sources joined and how each is deduplicated to one row per key (several
raw tables have real duplicates — mid-season trades, weekly rows mixed with
season totals, regular season vs. playoffs — see comments below):
  - seasonal_rosters : QB population, name/team/bio. Trades produce one row
                        per team; keeps the team the player finished the
                        season with (max week).
  - pbp               : core box-score stats, computed directly (see above).
  - ngs               : tracking metrics. week=0 is the season aggregate row
                        (weeks 1-17 are the weekly rows in the same file).
  - seasonal_pfr      : PFR pressure/accuracy metrics, keyed by pfr_id (not
                        gsis) via the ids bridge table. Mid-season trades
                        produce a combined "2TM"-style row alongside
                        per-team rows; prefers the combined row.
  - qbr_season        : ESPN QBR, keyed by ESPN's player_id via the ids
                        bridge. Has separate Regular/Playoffs rows; keeps
                        Regular only.
  - ids               : gsis_id <-> pfr_id/espn_id bridge table. Has
                        multiple snapshot rows per player over time (a
                        db_season column); keeps the most recent snapshot.
  - coordinators      : team's OC that season. Mid-season coordinator
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
OUT_PATH = DATA_DIR / "quarterbacks.parquet"

QUERY = f"""
WITH roster_qb AS (
    SELECT *
    FROM read_parquet('{DATA_DIR}/seasonal_rosters.parquet')
    WHERE position = 'QB'
    QUALIFY row_number() OVER (PARTITION BY player_id, season ORDER BY week DESC) = 1
),
pbp_pass AS (
    SELECT
        passer_player_id AS gsis_id, season,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND sack = 0 AND two_point_attempt = 0) AS attempts,
        SUM(complete_pass) FILTER (WHERE pass_attempt = 1 AND sack = 0 AND two_point_attempt = 0) AS completions,
        SUM(yards_gained) FILTER (WHERE pass_attempt = 1 AND sack = 0 AND two_point_attempt = 0) AS passing_yards,
        SUM(pass_touchdown) FILTER (WHERE pass_attempt = 1 AND sack = 0 AND two_point_attempt = 0) AS passing_tds,
        SUM(interception) FILTER (WHERE pass_attempt = 1 AND sack = 0 AND two_point_attempt = 0) AS interceptions,
        SUM(air_yards) FILTER (WHERE pass_attempt = 1 AND sack = 0 AND two_point_attempt = 0) AS passing_air_yards,
        SUM(yards_after_catch) FILTER (WHERE complete_pass = 1) AS passing_yards_after_catch,
        SUM(first_down_pass) FILTER (WHERE pass_attempt = 1 AND sack = 0 AND two_point_attempt = 0) AS passing_first_downs,
        SUM(epa) FILTER (WHERE pass_attempt = 1 AND sack = 0 AND two_point_attempt = 0) AS passing_epa,
        SUM(epa) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0) AS passing_epa_with_sacks,
        COUNT(*) FILTER (WHERE sack = 1) AS sacks,
        -SUM(yards_gained) FILTER (WHERE sack = 1) AS sack_yards,
        SUM(fumble) FILTER (WHERE sack = 1) AS sack_fumbles,
        SUM(fumble_lost) FILTER (WHERE sack = 1) AS sack_fumbles_lost,
        COUNT(*) FILTER (
            WHERE pass_attempt = 1 AND two_point_attempt = 1 AND two_point_conv_result = 'success'
        ) AS passing_2pt_conversions
    FROM read_parquet('{DATA_DIR}/pbp.parquet')
    WHERE passer_player_id IS NOT NULL AND season_type = 'REG'
    GROUP BY passer_player_id, season
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
pbp_games AS (
    SELECT gsis_id, season, COUNT(DISTINCT week) AS games
    FROM (
        SELECT passer_player_id AS gsis_id, season, week
        FROM read_parquet('{DATA_DIR}/pbp.parquet')
        WHERE passer_player_id IS NOT NULL AND pass_attempt = 1 AND season_type = 'REG'
        UNION ALL
        SELECT rusher_player_id AS gsis_id, season, week
        FROM read_parquet('{DATA_DIR}/pbp.parquet')
        WHERE rusher_player_id IS NOT NULL AND rush_attempt = 1 AND season_type = 'REG'
    )
    GROUP BY gsis_id, season
),
pbp_qb_season AS (
    SELECT
        COALESCE(p.gsis_id, r.gsis_id) AS gsis_id,
        COALESCE(p.season, r.season) AS season,
        COALESCE(p.attempts, 0) AS attempts,
        COALESCE(p.completions, 0) AS completions,
        COALESCE(p.passing_yards, 0) AS passing_yards,
        COALESCE(p.passing_tds, 0) AS passing_tds,
        COALESCE(p.interceptions, 0) AS interceptions,
        COALESCE(p.passing_air_yards, 0) AS passing_air_yards,
        COALESCE(p.passing_yards_after_catch, 0) AS passing_yards_after_catch,
        COALESCE(p.passing_first_downs, 0) AS passing_first_downs,
        COALESCE(p.passing_epa, 0) AS passing_epa,
        COALESCE(p.passing_epa_with_sacks, 0) AS passing_epa_with_sacks,
        COALESCE(p.sacks, 0) AS sacks,
        COALESCE(p.sack_yards, 0) AS sack_yards,
        COALESCE(p.sack_fumbles, 0) AS sack_fumbles,
        COALESCE(p.sack_fumbles_lost, 0) AS sack_fumbles_lost,
        COALESCE(p.passing_2pt_conversions, 0) AS passing_2pt_conversions,
        CASE WHEN COALESCE(p.passing_air_yards, 0) = 0 THEN NULL
             ELSE p.passing_yards / p.passing_air_yards END AS pacr,
        COALESCE(r.carries, 0) AS carries,
        COALESCE(r.rushing_yards, 0) AS rushing_yards,
        COALESCE(r.rushing_tds, 0) AS rushing_tds,
        COALESCE(r.rushing_fumbles, 0) AS rushing_fumbles,
        COALESCE(r.rushing_fumbles_lost, 0) AS rushing_fumbles_lost,
        COALESCE(r.rushing_first_downs, 0) AS rushing_first_downs,
        COALESCE(r.rushing_epa, 0) AS rushing_epa,
        COALESCE(r.rushing_2pt_conversions, 0) AS rushing_2pt_conversions,
        COALESCE(p.passing_yards, 0) / 25.0
            + COALESCE(p.passing_tds, 0) * 4
            - COALESCE(p.interceptions, 0) * 2
            + COALESCE(r.rushing_yards, 0) / 10.0
            + COALESCE(r.rushing_tds, 0) * 6
            - (COALESCE(p.sack_fumbles_lost, 0) + COALESCE(r.rushing_fumbles_lost, 0)) * 2
            + (COALESCE(p.passing_2pt_conversions, 0) + COALESCE(r.rushing_2pt_conversions, 0)) * 2
            AS fantasy_points
    FROM pbp_pass p
    FULL OUTER JOIN pbp_rush r ON p.gsis_id = r.gsis_id AND p.season = r.season
),
ids_bridge AS (
    SELECT gsis_id, pfr_id, CAST(espn_id AS BIGINT) AS espn_id
    FROM read_parquet('{DATA_DIR}/ids.parquet')
    WHERE gsis_id IS NOT NULL
    QUALIFY row_number() OVER (PARTITION BY gsis_id ORDER BY db_season DESC) = 1
),
ngs_qb AS (
    SELECT
        player_gsis_id AS gsis_id, season,
        avg_time_to_throw, avg_completed_air_yards, avg_intended_air_yards,
        avg_air_yards_differential, aggressiveness, avg_air_yards_to_sticks,
        completion_percentage_above_expectation, expected_completion_percentage,
        avg_time_to_los, percent_attempts_gte_eight_defenders
    FROM read_parquet('{DATA_DIR}/ngs.parquet')
    WHERE stat_type = 'passing' AND week = 0
),
pfr_qb AS (
    SELECT
        pfr_id, season,
        times_pressured, pressure_pct, times_blitzed, times_hit,
        bad_throw_pct, on_tgt_pct, pocket_time,
        pa_pass_att, pa_pass_yards, rpo_pass_att, rpo_pass_yards, scrambles
    FROM read_parquet('{DATA_DIR}/seasonal_pfr.parquet')
    WHERE stat_type = 'pass'
    QUALIFY row_number() OVER (
        PARTITION BY pfr_id, season ORDER BY (team LIKE '%TM') DESC, pass_attempts DESC
    ) = 1
),
qbr_reg AS (
    SELECT
        player_id, season,
        qbr_total, qbr_raw, pts_added, epa_total AS qbr_epa_total,
        pass AS qbr_pass_epa, run AS qbr_run_epa, qb_plays, qualified AS qbr_qualified
    FROM read_parquet('{DATA_DIR}/qbr_season.parquet')
    WHERE season_type = 'Regular'
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
combine_qb AS (
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

    s.completions, s.attempts, s.passing_yards, s.passing_tds, s.interceptions,
    s.sacks, s.sack_yards, s.sack_fumbles, s.sack_fumbles_lost,
    s.passing_air_yards, s.passing_yards_after_catch, s.passing_first_downs,
    s.passing_epa, s.passing_epa_with_sacks, s.passing_2pt_conversions, s.pacr,
    s.carries, s.rushing_yards, s.rushing_tds, s.rushing_fumbles, s.rushing_fumbles_lost,
    s.rushing_first_downs, s.rushing_epa, s.rushing_2pt_conversions,
    s.fantasy_points, s.fantasy_points AS fantasy_points_ppr, s.fantasy_points AS fantasy_points_half_ppr, g.games,
    s.fantasy_points / NULLIF(g.games, 0) AS fantasy_points_per_game,

    ngs.avg_time_to_throw, ngs.avg_completed_air_yards, ngs.avg_intended_air_yards,
    ngs.avg_air_yards_differential, ngs.aggressiveness, ngs.avg_air_yards_to_sticks,
    ngs.completion_percentage_above_expectation, ngs.expected_completion_percentage,
    ngs.avg_time_to_los, ngs.percent_attempts_gte_eight_defenders,

    pfr.times_pressured, pfr.pressure_pct, pfr.times_blitzed, pfr.times_hit,
    pfr.bad_throw_pct, pfr.on_tgt_pct, pfr.pocket_time,
    pfr.pa_pass_att, pfr.pa_pass_yards, pfr.rpo_pass_att, pfr.rpo_pass_yards, pfr.scrambles,

    qbr.qbr_total, qbr.qbr_raw, qbr.pts_added, qbr.qbr_epa_total,
    qbr.qbr_pass_epa, qbr.qbr_run_epa, qbr.qb_plays, qbr.qbr_qualified,

    wt.win_total AS preseason_win_total,
    coord.oc_name,

    d.draft_round, d.draft_pick, d.draft_year,
    cb.forty, cb.bench, cb.vertical, cb.broad_jump, cb.cone, cb.shuttle,
    cb.combine_height, cb.combine_weight,
    ct.contract_apy, ct.contract_year_signed, ct.contract_years

FROM roster_qb r
JOIN pbp_qb_season s
    ON r.player_id = s.gsis_id AND r.season = s.season
LEFT JOIN pbp_games g ON r.player_id = g.gsis_id AND r.season = g.season
LEFT JOIN ids_bridge ib ON r.player_id = ib.gsis_id
LEFT JOIN ngs_qb ngs ON r.player_id = ngs.gsis_id AND r.season = ngs.season
LEFT JOIN pfr_qb pfr ON ib.pfr_id = pfr.pfr_id AND r.season = pfr.season
LEFT JOIN qbr_reg qbr ON ib.espn_id = qbr.player_id AND r.season = qbr.season
LEFT JOIN read_parquet('{DATA_DIR}/win_totals.parquet') wt
    ON r.team = wt.team AND r.season = wt.season
LEFT JOIN coord ON r.team = coord.team AND r.season = coord.season
LEFT JOIN draft d ON r.player_id = d.gsis_id
LEFT JOIN combine_qb cb ON ib.pfr_id = cb.pfr_id
LEFT JOIN contract_latest ct ON r.player_id = ct.gsis_id
ORDER BY s.season, r.team, r.player_name
"""


def build_quarterbacks_table():
    con = duckdb.connect(":memory:")
    df = con.execute(QUERY).fetchdf()
    con.close()

    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} QB-season rows ({df['season'].min()}-{df['season'].max()}) to {OUT_PATH}")
    return df


if __name__ == "__main__":
    build_quarterbacks_table()
