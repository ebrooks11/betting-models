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
)
SELECT
    tg.season, tg.week, tg.team, tg.opponent, tg.is_home,
    oc.oc_name,
    tg.points_scored, tg.points_allowed,
    tg.game_id
FROM team_games tg
LEFT JOIN oc ON tg.team = oc.team AND tg.season = oc.season
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
