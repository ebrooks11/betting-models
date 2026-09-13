"""Build a team-game table: one row per team per regular-season game (i.e.
two rows per actual game — one from each team's perspective), reshaped from
schedules.parquet's home/away columns into a team-centric fact table.
Writes data/team_games.parquet so it's automatically picked up as a table
by pipeline/build_duckdb.py.

Run after schedules.parquet and coordinators.parquet exist:
    python3 pipeline/build_team_games_table.py

This is meant as a general-purpose foundation, not an OC-specific table:
oc_name is just one feature column on it, the same shape any future
feature (dc_name, hc_name, or eventually a join to player-level game logs
for player-specific splits) would plug into. Season- and career-level OC
views should aggregate *this* table rather than re-deriving points scored
from schedules.parquet independently, so all three grains (game/season/
career) stay consistent by construction instead of drifting apart as
separate hand-written queries.

Grain: (season, week, team). REG season only, matching every other table
in this pipeline's convention (playoffs and week 1-3/final-week filtering
happen at the model-dataset layer, not here — this is meant as a raw-ish
per-game fact table, not a model-ready one).

oc_name is attributed at team-season granularity, same combined
"Name A; Name B" string used in offensive_coordinators.parquet when a
team had more than one OC that season — coordinators.parquet has no
week-level data, so every game in a split season shows the same combined
string; a game log for such a season is not a clean per-OC split, it's the
team's full schedule with all of that season's OCs listed for every game.

Play-calling mix (rushes/passes/rates)
---------------------------------------
Computed from pbp.parquet per game, joined on (game_id, posteam=team) —
posteam is already a canonical team code (verified elsewhere in this
pipeline), so no team-code normalization is needed for this join. Two
judgment calls, both documented here rather than picked silently:

- Sacks are flagged pass_attempt=1 in raw pbp (same quirk noted in the QB
  table's build script) and are counted as passes here — a sack reflects a
  called passing play that failed, which is what "pass rate" as a
  play-calling metric should measure, unlike the passing *box-score* stats
  elsewhere in this pipeline which deliberately exclude sacks. 2-point
  conversion attempts are excluded from every count (same convention used
  everywhere else here).
- "Neutral game script" = plays where the score was within one possession
  (|score_differential| <= 8) at the time of the play, using pbp's own
  score_differential (posteam_score - defteam_score, verified directly),
  AND excludes the last 2 minutes of the 2nd quarter and the last 4 minutes
  of the 4th quarter regardless of score — two-minute-drill and
  four-minute-offense situations are clock-driven play-calling (hurry-up to
  score before half, or run-out-the-clock to protect a lead) even when the
  score itself is close, so they're not "neutral" in the sense this metric
  is after. Checked directly: in qtr=2, quarter_seconds_remaining equals
  half_seconds_remaining (Q2 is the last quarter of the first half), so
  qtr=2 AND quarter_seconds_remaining<=120 correctly targets the last two
  minutes before halftime specifically, not just any two-minutes-remaining
  point earlier in the half. OT (qtr 5/6) is excluded from neutral script
  entirely by the same logic as "first 3 quarters" below. Chose a
  score-differential threshold over nflverse's built-in win probability
  columns (wp/home_wp) on purpose — simpler to explain to this site's
  fantasy-manager audience, and doesn't require trusting a proprietary WP
  model's internals we can't fully audit (same reasoning applied earlier to
  ESPN's QBR). 8 points, and the 2/4-minute cutoffs, are common conventions,
  not universally agreed-upon standards; different thresholds are a
  reasonable ask.
- "First 3 quarters" = qtr IN (1,2,3), i.e. excludes Q4 and any overtime
  (qtr 5/6) entirely, regardless of score — a purely time-based way to look
  past game-script effects, independent of and complementary to the
  score-based neutral-script definition above (both exist because they
  isolate different things: one ignores the score, the other ignores when
  in the game a play happened).

Overall offensive efficiency (epa_per_play, success_rate)
------------------------------------------------------------
Same play set as play_rates (rush_attempt=1 OR pass_attempt=1,
two_point_attempt=0, REG season only), but additionally excludes qb_kneel
and qb_spike plays — both are automatic, non-competitive negative/neutral
plays (kneeling out a win, stopping the clock) that would drag down a
team's efficiency numbers without reflecting actual offensive quality.
This is a team-level, all-personnel/all-formation efficiency number,
distinct from and a superset of the personnel-specific EPA/success columns
below. rush_epa/rush_success_rate and pass_epa/pass_success_rate split the
same play set by rush_attempt vs pass_attempt — sacks count as pass plays
here (pass_attempt=1 in raw pbp, same quirk noted for pass_rate above), so
pass_epa/pass_success_rate reflect the full passing-down outcome including
sacks, not just completed/attempted throws.

Big plays (explosive_run_rate, explosive_pass_rate, explosive_play_rate,
plays_20plus, plays_40plus)
------------------------------------------------------------------------
Same 10-yard rushing / 20-yard passing thresholds already used for
rb_group_explosive_rate in build_offensive_coordinators_table.py, for
consistency rather than introducing a second explosive-play standard.
explosive_run_rate/explosive_pass_rate are out of rush_plays/pass_plays —
this file's own kneel/spike/two-point-excluded play counts (see
"Overall offensive efficiency" above), NOT pr.rushes/pr.passes from
play_rates above, which intentionally include kneels for play-calling
rate purposes. plays_20plus/plays_40plus are raw counts (not rates) of any
rush or pass gaining that many yards, regardless of down/distance —
meant to surface boom-or-bust offenses that a rate alone can mask.

Personnel groupings (11/12/21/other rate, EPA, success rate)
--------------------------------------------------------------
offense_personnel (pbp.parquet) floors at 2016, same as offense_formation.
Classified from the RB/TE/WR counts in that string via regex, matching the
exact convention already used by parse_off_personnel() in
export_rolling_epa.py (independent (\\d+) RB / (\\d+) TE / (\\d+) WR
extraction, not string equality) — checked directly that this is necessary:
many rows list extra offensive-line detail in the same string (e.g.
"1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR" is still 11 personnel), which would
break an exact-string-match approach. Only three groupings are broken out
individually (11 = 1 RB/1 TE/3 WR, 12 = 1 RB/2 TE/2 WR, 21 = 2 RB/1 TE/2 WR)
because those are overwhelmingly the most common and the ones with
established football meaning; every other combination (jumbo, 10 personnel,
empty backfield, etc.) is bucketed as "other" rather than enumerated, same
simplification the existing parser already makes. personnel_known_plays is
the denominator (plays with a non-null offense_personnel) so rates are
computed only over games/seasons where this data actually exists.

Expected rush yards (expected_rush_yards)
--------------------------------------------
Team total from ngs.parquet (NextGen Stats' tracking-data-based rushing
expectation model — a related but distinct concept from PFR's charted
yards-before-contact). NGS publishes a row per rushing player per team per
week going back to 2016, but expected_rush_yards itself is null in every
2016-2017 row — checked directly, floors at 2018. Even from 2018 on,
coverage isn't 100%: NGS's own feed is missing some team-weeks entirely
(e.g. 468 of a possible 544 team-weeks in 2024 — about 86%), presumably a
minimum-attempts publication threshold on NGS's end, not a join bug here.
This is SUM(expected_rush_yards) across every rusher NGS did publish for
that team-week — filtered to season_type='REG' and week BETWEEN 1 AND 18
to exclude NGS's own week=0 season-aggregate rows and playoff weeks,
joined on (season, week, team) since NGS has no game_id. team_abbr uses
'LAR' for the Rams where every other table in this pipeline uses 'LA' —
normalized here the same way home_team/away_team are normalized above.
"""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "team_games.parquet"

# Same normalization as build_offensive_coordinators_table.py — schedules.parquet
# only has the relocation-era codes (OAK/SD/STL), not the ARZ/BLT/CLV/HST/SL
# ones, but applying all of them is harmless (unmatched codes pass through).
_TEAM_NORM_HOME = """CASE home_team
    WHEN 'OAK' THEN 'LV' WHEN 'SD' THEN 'LAC' WHEN 'STL' THEN 'LA'
    WHEN 'ARZ' THEN 'ARI' WHEN 'BLT' THEN 'BAL' WHEN 'CLV' THEN 'CLE'
    WHEN 'HST' THEN 'HOU' WHEN 'SL' THEN 'LA'
    ELSE home_team END"""
_TEAM_NORM_AWAY = _TEAM_NORM_HOME.replace("home_team", "away_team")

# ngs.parquet's team_abbr uses 'LAR' for the Rams (unlike every other table
# in this pipeline, which uses 'LA' — see build_team_games_table.py's
# module docstring, "Expected rush yards" section).
_TEAM_NORM_NGS = """CASE team_abbr
    WHEN 'OAK' THEN 'LV' WHEN 'SD' THEN 'LAC' WHEN 'STL' THEN 'LA' WHEN 'LAR' THEN 'LA'
    WHEN 'ARZ' THEN 'ARI' WHEN 'BLT' THEN 'BAL' WHEN 'CLV' THEN 'CLE'
    WHEN 'HST' THEN 'HOU' WHEN 'SL' THEN 'LA'
    ELSE team_abbr END"""

# "Neutral game script": within one score, and not in a clock-driven
# situation (2-minute drill before half, 4-minute offense to close the
# game) even if the score itself is close. See this file's module
# docstring for the full reasoning behind each piece of this.
_NEUTRAL_SCRIPT = """
    abs(score_differential) <= 8
    AND qtr NOT IN (5, 6)
    AND NOT (qtr = 2 AND quarter_seconds_remaining <= 120)
    AND NOT (qtr = 4 AND quarter_seconds_remaining <= 240)
"""

QUERY = f"""
WITH oc AS (
    SELECT team, season, string_agg(DISTINCT name, '; ') AS oc_name
    FROM read_parquet('{DATA_DIR}/coordinators.parquet')
    WHERE role_category = 'OC'
    GROUP BY team, season
),
team_games AS (
    SELECT
        game_id, season, week, game_type,
        {_TEAM_NORM_HOME} AS team,
        {_TEAM_NORM_AWAY} AS opponent,
        true AS is_home,
        home_score AS points_scored,
        away_score AS points_allowed
    FROM read_parquet('{DATA_DIR}/schedules.parquet')
    WHERE game_type = 'REG' AND home_score IS NOT NULL

    UNION ALL

    SELECT
        game_id, season, week, game_type,
        {_TEAM_NORM_AWAY} AS team,
        {_TEAM_NORM_HOME} AS opponent,
        false AS is_home,
        away_score AS points_scored,
        home_score AS points_allowed
    FROM read_parquet('{DATA_DIR}/schedules.parquet')
    WHERE game_type = 'REG' AND away_score IS NOT NULL
),
play_rates AS (
    SELECT
        game_id, posteam AS team,
        COUNT(*) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0) AS rushes,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0) AS passes,
        COUNT(*) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0 AND qtr IN (1, 2, 3)) AS rushes_first_3q,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0 AND qtr IN (1, 2, 3)) AS passes_first_3q,
        COUNT(*) FILTER (WHERE rush_attempt = 1 AND two_point_attempt = 0 AND {_NEUTRAL_SCRIPT}) AS rushes_neutral,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND two_point_attempt = 0 AND {_NEUTRAL_SCRIPT}) AS passes_neutral
    FROM read_parquet('{DATA_DIR}/pbp.parquet')
    WHERE season_type = 'REG' AND (rush_attempt = 1 OR pass_attempt = 1)
    GROUP BY game_id, posteam
),
efficiency AS (
    SELECT
        game_id, posteam AS team,
        AVG(epa) AS epa_per_play,
        AVG(success) AS success_rate,
        AVG(epa) FILTER (WHERE rush_attempt = 1) AS rush_epa,
        AVG(success) FILTER (WHERE rush_attempt = 1) AS rush_success_rate,
        AVG(epa) FILTER (WHERE pass_attempt = 1) AS pass_epa,
        AVG(success) FILTER (WHERE pass_attempt = 1) AS pass_success_rate,
        COUNT(*) FILTER (WHERE rush_attempt = 1) AS rush_plays,
        COUNT(*) FILTER (WHERE pass_attempt = 1) AS pass_plays,
        COUNT(*) FILTER (WHERE rush_attempt = 1 AND yards_gained >= 10) AS explosive_runs,
        COUNT(*) FILTER (WHERE pass_attempt = 1 AND yards_gained >= 20) AS explosive_passes,
        COUNT(*) FILTER (WHERE yards_gained >= 20) AS plays_20plus,
        COUNT(*) FILTER (WHERE yards_gained >= 40) AS plays_40plus
    FROM read_parquet('{DATA_DIR}/pbp.parquet')
    WHERE season_type = 'REG' AND (rush_attempt = 1 OR pass_attempt = 1)
      AND two_point_attempt = 0 AND qb_kneel = 0 AND qb_spike = 0
    GROUP BY game_id, posteam
),
ngs_rush AS (
    SELECT
        season, week,
        {_TEAM_NORM_NGS} AS team,
        SUM(expected_rush_yards) AS expected_rush_yards
    FROM read_parquet('{DATA_DIR}/ngs.parquet')
    WHERE stat_type = 'rushing' AND season_type = 'REG' AND week BETWEEN 1 AND 18
    GROUP BY season, week, team_abbr
),
personnel AS (
    SELECT
        game_id, posteam AS team,
        CASE
            WHEN COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\\d+) RB', 1) AS INT), 0) = 1
             AND COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\\d+) TE', 1) AS INT), 0) = 1
             AND COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\\d+) WR', 1) AS INT), 0) = 3
            THEN '11'
            WHEN COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\\d+) RB', 1) AS INT), 0) = 1
             AND COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\\d+) TE', 1) AS INT), 0) = 2
             AND COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\\d+) WR', 1) AS INT), 0) = 2
            THEN '12'
            WHEN COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\\d+) RB', 1) AS INT), 0) = 2
             AND COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\\d+) TE', 1) AS INT), 0) = 1
             AND COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\\d+) WR', 1) AS INT), 0) = 2
            THEN '21'
            ELSE 'other'
        END AS personnel_group,
        epa, success
    FROM read_parquet('{DATA_DIR}/pbp.parquet')
    WHERE season_type = 'REG' AND (rush_attempt = 1 OR pass_attempt = 1)
      AND two_point_attempt = 0 AND offense_personnel IS NOT NULL
),
personnel_stats AS (
    SELECT
        game_id, team,
        COUNT(*) AS personnel_known_plays,
        COUNT(*) FILTER (WHERE personnel_group = '11') AS personnel_11_plays,
        AVG(epa) FILTER (WHERE personnel_group = '11') AS personnel_11_epa,
        AVG(success) FILTER (WHERE personnel_group = '11') AS personnel_11_success_rate,
        COUNT(*) FILTER (WHERE personnel_group = '12') AS personnel_12_plays,
        AVG(epa) FILTER (WHERE personnel_group = '12') AS personnel_12_epa,
        AVG(success) FILTER (WHERE personnel_group = '12') AS personnel_12_success_rate,
        COUNT(*) FILTER (WHERE personnel_group = '21') AS personnel_21_plays,
        AVG(epa) FILTER (WHERE personnel_group = '21') AS personnel_21_epa,
        AVG(success) FILTER (WHERE personnel_group = '21') AS personnel_21_success_rate,
        COUNT(*) FILTER (WHERE personnel_group = 'other') AS personnel_other_plays,
        AVG(epa) FILTER (WHERE personnel_group = 'other') AS personnel_other_epa,
        AVG(success) FILTER (WHERE personnel_group = 'other') AS personnel_other_success_rate
    FROM personnel
    GROUP BY game_id, team
)
SELECT
    tg.season, tg.week, tg.team, tg.opponent, tg.is_home,
    oc.oc_name,
    tg.points_scored, tg.points_allowed,

    pr.rushes, pr.passes,
    pr.rushes / NULLIF(pr.rushes + pr.passes, 0) AS rush_rate,
    pr.passes / NULLIF(pr.rushes + pr.passes, 0) AS pass_rate,
    pr.rushes_first_3q / NULLIF(pr.rushes_first_3q + pr.passes_first_3q, 0) AS rush_rate_first_3_quarters,
    pr.passes_first_3q / NULLIF(pr.rushes_first_3q + pr.passes_first_3q, 0) AS pass_rate_first_3_quarters,
    pr.rushes_neutral / NULLIF(pr.rushes_neutral + pr.passes_neutral, 0) AS rush_rate_neutral_game_script,
    pr.passes_neutral / NULLIF(pr.rushes_neutral + pr.passes_neutral, 0) AS pass_rate_neutral_game_script,

    ef.epa_per_play,
    ef.success_rate,
    ef.rush_epa,
    ef.rush_success_rate,
    ef.pass_epa,
    ef.pass_success_rate,
    ef.explosive_runs / NULLIF(ef.rush_plays, 0) AS explosive_run_rate,
    ef.explosive_passes / NULLIF(ef.pass_plays, 0) AS explosive_pass_rate,
    (ef.explosive_runs + ef.explosive_passes) / NULLIF(ef.rush_plays + ef.pass_plays, 0) AS explosive_play_rate,
    ef.plays_20plus,
    ef.plays_40plus,

    ngs.expected_rush_yards,

    ps.personnel_known_plays,
    ps.personnel_11_plays / NULLIF(ps.personnel_known_plays, 0) AS personnel_11_rate,
    ps.personnel_11_epa,
    ps.personnel_11_success_rate,
    ps.personnel_12_plays / NULLIF(ps.personnel_known_plays, 0) AS personnel_12_rate,
    ps.personnel_12_epa,
    ps.personnel_12_success_rate,
    ps.personnel_21_plays / NULLIF(ps.personnel_known_plays, 0) AS personnel_21_rate,
    ps.personnel_21_epa,
    ps.personnel_21_success_rate,
    ps.personnel_other_plays / NULLIF(ps.personnel_known_plays, 0) AS personnel_other_rate,
    ps.personnel_other_epa,
    ps.personnel_other_success_rate,

    tg.game_id
FROM team_games tg
LEFT JOIN oc ON tg.team = oc.team AND tg.season = oc.season
LEFT JOIN play_rates pr ON tg.game_id = pr.game_id AND tg.team = pr.team
LEFT JOIN efficiency ef ON tg.game_id = ef.game_id AND tg.team = ef.team
LEFT JOIN ngs_rush ngs ON tg.season = ngs.season AND tg.week = ngs.week AND tg.team = ngs.team
LEFT JOIN personnel_stats ps ON tg.game_id = ps.game_id AND tg.team = ps.team
ORDER BY tg.season, tg.week, tg.team
"""


def build_team_games_table():
    con = duckdb.connect(":memory:")
    df = con.execute(QUERY).fetchdf()
    con.close()

    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} team-game rows ({df['season'].min()}-{df['season'].max()}) to {OUT_PATH}")
    return df


if __name__ == "__main__":
    build_team_games_table()
