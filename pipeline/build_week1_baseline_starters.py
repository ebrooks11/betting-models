"""
Derives a "healthy baseline" starting lineup per team from Week 1 play-by-play
— who was actually on the field doing the position's job in the FIRST HALF
of their Week 1 game, before a season's worth of injuries could set in.

Why not just use data/depth_chart.parquet (ESPN's depth chart)?
---------------------------------------------------------------------
That table is a LIVE, injury-adjusted depth chart, not a fixed healthy
baseline — confirmed directly: Arizona's RB slot has Jeremiyah Love at
depth_rank 1 and James Conner (Arizona's real lead back entering the
season, since hurt) down at depth_rank 4, tagged IR. It's exactly the kind
of downgrade this file exists to help notice, not something to compare
against as if it were the baseline.

Why not full-game Week 1 stats?
--------------------------------------
A starter hurt mid-game (or in a blowout, pulled in the second half) would
have his stats diluted by whoever replaced him, understating his role. The
first half is a reasonable proxy for "who the team actually planned to
start," without needing the current season's personnel/participation data
(unavailable — see fetch_current_season_data.py's _fetch_pbp docstring),
which the normal depth-chart-via-snap-counts approach this project uses
elsewhere depends on.

Why not every position?
---------------------------
pbp.parquet only names the player directly INVOLVED in a play's outcome —
passer, rusher, receiver. There's no "who was on the field" list for plays
where a player didn't touch the ball (impossible without personnel data),
so offensive line, defensive line, linebacker, and defensive back starters
can't be derived this way at all. This file only covers QB/RB/WR/TE — the
offensive skill positions pbp can actually answer for. Everything else has
no row here, deliberately, rather than a guess dressed up as data.

Offense/defense only, no special teams
-------------------------------------------
Deliberately excludes kicker/punter/returner entirely, and restricts every
count below to actual scrimmage plays (play_type "pass"/"run", plus
"no_play" for a play negated by a penalty and "qb_kneel"/"qb_spike" where
relevant) — not just whichever rows happen to have a given id column
filled in. That distinction matters: a trick play like a fake punt can
leave passer_player_id set on a play_type="punt" row, which would
otherwise leak a special-teams snap into an offensive skill player's
count.

Method, per team, per position:
  - QB: pass attempts (passer_player_id)
  - RB: rush attempts (rusher_player_id), excluding any play where the
    rusher was also a passer on some other play in the half (catches QB
    scrambles/kneels without needing a separately-identified "starting
    QB" first) and any qb_kneel/qb_spike play
  - WR / TE: targets (receiver_player_id), split by the player's listed
    position (data/ids.parquet) since pbp doesn't tag a receiver's
    position itself
Ranked by count within (team, position); ties broken by player_id for
determinism. Rows with count 0 aren't included — pbp simply has no signal
for a player who didn't do any of this in the half.

Run locally once Week 1 pbp is available (reads
data/raw/_current_season/pbp.parquet if present, else fetches fresh):
    python3 pipeline/build_week1_baseline_starters.py

Not wired into fetch_current_season_data.py's weekly refresh — Week 1 pbp
doesn't change after the fact, so this only needs to run once per season,
not every week.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "week1_baseline_starters.parquet"
SCRATCH_PBP_PATH = DATA_DIR / "raw" / "_current_season" / "pbp.parquet"


def _load_week1_pbp(season: int) -> pd.DataFrame:
    if SCRATCH_PBP_PATH.exists():
        df = pd.read_parquet(SCRATCH_PBP_PATH)
        df = df[(df.season == season) & (df.week == 1)]
        if not df.empty:
            return df
    import nfl_data_py as nfl
    df = nfl.import_pbp_data([season], downcast=True, include_participation=False)
    return df[df.week == 1]


def _position_lookup() -> dict:
    ids = pd.read_parquet(DATA_DIR / "ids.parquet", columns=["gsis_id", "position"])
    ids = ids.dropna(subset=["gsis_id"]).drop_duplicates(subset="gsis_id")
    return dict(zip(ids["gsis_id"], ids["position"]))


def _rank_rows(df: pd.DataFrame, id_col: str, name_col: str, position: str, metric: str, season: int, team_col: str = "posteam") -> list[dict]:
    counts = (
        df.dropna(subset=[id_col])
        .groupby([team_col, id_col, name_col])
        .size()
        .reset_index(name="count")
    )
    rows = []
    for team, g in counts.groupby(team_col):
        g = g.sort_values(["count", id_col], ascending=[False, True])
        for depth_rank, r in enumerate(g.itertuples(), start=1):
            rows.append({
                "season": season, "team": team, "position": position, "depth_rank": depth_rank,
                "player": getattr(r, name_col), "player_id": getattr(r, id_col),
                "metric": metric, "count": int(r.count),
            })
    return rows


_SCRIMMAGE_PLAY_TYPES = ["pass", "run", "no_play", "qb_kneel", "qb_spike"]


def build(season: int) -> pd.DataFrame:
    pbp = _load_week1_pbp(season)
    half1 = pbp[pbp.qtr.isin([1, 2]) & pbp.posteam.notna()]
    # Scrimmage plays only — not special teams (kickoff/punt/field_goal/
    # extra_point). Filtering by play_type here, not just by which id
    # columns happen to be non-null, matters for a case like a fake punt:
    # that leaves passer_player_id set on a play_type="punt" row, which
    # would otherwise leak a special-teams snap into an offensive skill
    # player's count.
    half1 = half1[half1.play_type.isin(_SCRIMMAGE_PLAY_TYPES)]

    rows = []

    rows += _rank_rows(half1, "passer_player_id", "passer_player_name", "QB", "pass_attempts", season)

    passer_ids_by_team = half1.dropna(subset=["passer_player_id"]).groupby("posteam")["passer_player_id"].apply(set)
    rush = half1[(half1.qb_kneel != 1) & (half1.qb_spike != 1)].dropna(subset=["rusher_player_id"]).copy()
    rush = rush[~rush.apply(lambda r: r["rusher_player_id"] in passer_ids_by_team.get(r["posteam"], set()), axis=1)]
    rows += _rank_rows(rush, "rusher_player_id", "rusher_player_name", "RB", "rush_attempts", season)

    pos_lookup = _position_lookup()
    targets = half1.dropna(subset=["receiver_player_id"]).copy()
    targets["_position"] = targets["receiver_player_id"].map(pos_lookup)
    rows += _rank_rows(targets[targets._position == "WR"], "receiver_player_id", "receiver_player_name", "WR", "targets", season)
    rows += _rank_rows(targets[targets._position == "TE"], "receiver_player_id", "receiver_player_name", "TE", "targets", season)

    return pd.DataFrame(rows).sort_values(["team", "position", "depth_rank"]).reset_index(drop=True)


def main():
    from datetime import date
    season = date.today().year if date.today().month >= 3 else date.today().year - 1
    df = build(season)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows ({df['team'].nunique()} teams) to {OUT_PATH}")
    print(df[(df.position == "RB") & (df.depth_rank == 1)].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
