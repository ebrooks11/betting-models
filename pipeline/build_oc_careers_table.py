"""Build an OC-career table: one row per named offensive coordinator,
aggregated across every team-season they held that role. Writes
data/oc_careers.parquet so it's automatically picked up as a table by
pipeline/build_duckdb.py.

Run after data/team_games.parquet exists:
    python3 pipeline/build_oc_careers_table.py

Grain: one row per distinct OC name — aggregated from team_games.parquet
(not re-derived from schedules.parquet independently), so this stays
consistent with the season-level team_points_per_game in
offensive_coordinators.parquet by construction.

team_games.oc_name combines co-OCs from a split season into one
"Name A; Name B" string (team_games has no week-level OC data to split on
more precisely — see build_team_games_table.py). Splitting that string
back apart here means each named person gets that season's full game log
credited to their own career row, rather than being merged under the
combined string as if "Name A; Name B" were a single entity.

Known limitation: a coordinator with a shared season shows that whole
season's games in their career total, not just the games they personally
coached — we don't have data at finer grain than team-season to do better.

This table is meant to grow: today it's just career points per game (see
COORDINATOR_RB_METRICS.md and build_offensive_coordinators_table.py for
the season-level metrics already built) — more career-level OC stats are
expected to land here as new columns later.
"""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "oc_careers.parquet"

QUERY = f"""
WITH split_oc AS (
    SELECT
        trim(unnest(string_split(oc_name, ';'))) AS oc_name,
        team, season, week, points_scored
    FROM read_parquet('{DATA_DIR}/team_games.parquet')
    WHERE oc_name IS NOT NULL
)
SELECT
    oc_name,
    MIN(season) AS first_season,
    MAX(season) AS last_season,
    COUNT(DISTINCT season) AS seasons_coached,
    COUNT(DISTINCT team) AS teams_coached,
    string_agg(DISTINCT team, ', ' ORDER BY team) AS teams,
    COUNT(*) AS career_games,
    SUM(points_scored) AS career_points,
    AVG(points_scored) AS career_ppg
FROM split_oc
GROUP BY oc_name
ORDER BY career_ppg DESC
"""


def build_oc_careers_table():
    con = duckdb.connect(":memory:")
    df = con.execute(QUERY).fetchdf()
    con.close()

    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} OC career rows to {OUT_PATH}")
    return df


if __name__ == "__main__":
    build_oc_careers_table()
