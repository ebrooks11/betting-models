"""
Backtest: how often has roster_rating's implied spread picked the side
that actually covered, across every played game this season?

See pipeline/team_rating.py's own "Backtest: roster_rating against the
closing spread" docstring section for the headline finding and the known
depth-chart-snapshot limitation that applies to every number this script
produces — not reproduced in full here, read that first.

Methodology
--------------
For each played REG-season game with a closing spread
(schedules.parquet's spread_line) and a roster_rating available for both
teams that week:

  model_margin = home team's roster_rating - away team's roster_rating
                 (home's edge, in the same calibrated-to-real-scoring
                 "points" unit roster_rating already targets — see
                 team_rating.py's "Two-stage design" section)
  edge         = model_margin - spread_line
                 (how much more/less home-favorable the model is than
                 the market's own closing line)
  pick         = "home" if edge > 0, "away" if edge < 0, else no pick
                 (edge exactly 0 — the model and market agree exactly,
                 nothing to grade)
  actual       = whichever side's home_margin (home_score - away_score)
                 beat spread_line — same grading
                 generate_matchups_table.py's _team_game_log already uses
                 for recent_games.covered (see that function's own
                 comments for the sign-convention caveat)

A push (actual == "push") or a zero-edge tie is excluded from the graded
count rather than counted as wrong — same "don't penalize an ungradeable
case" discipline the rest of this project already uses for covered/
total_result elsewhere, not a new rule invented for this script.

Run locally:
    python3 pipeline/backtest_roster_rating.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.team_rating import team_ratings_by_week

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def backtest(season: int) -> pd.DataFrame:
    """One row per played, spread-graded game this season: the model's
    ATS pick (home/away/None) vs. the side that actually covered
    (home/away/push)."""
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    sched = schedules[(schedules.season == season) & (schedules.game_type == "REG")]
    weeks = sorted(sched["week"].unique().tolist())
    ratings, _healthy = team_ratings_by_week(season, weeks)

    played = sched[sched.home_score.notna() & sched.away_score.notna() & sched.spread_line.notna()]

    rows = []
    for r in played.itertuples():
        home_rating = ratings.get((r.home_team, r.week))
        away_rating = ratings.get((r.away_team, r.week))
        if home_rating is None or away_rating is None:
            continue
        model_margin = home_rating - away_rating
        home_margin = r.home_score - r.away_score
        edge = model_margin - r.spread_line
        cover_diff = home_margin - r.spread_line
        actual = "push" if cover_diff == 0 else ("home" if cover_diff > 0 else "away")
        pick = None if edge == 0 else ("home" if edge > 0 else "away")
        rows.append({
            "week": r.week, "home_team": r.home_team, "away_team": r.away_team,
            "spread_line": r.spread_line, "model_margin": round(model_margin, 1),
            "edge": round(edge, 1), "pick": pick, "actual": actual,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    graded = df[(df.actual != "push") & df["pick"].notna()]
    correct = int((graded["pick"] == graded["actual"]).sum())
    total = len(graded)

    print(f"{len(df)} played games with a spread and ratings available for both teams")
    print(f"{len(df) - total} excluded (push or zero-edge tie)")
    print(f"{correct}/{total} correct against the spread ({correct / total * 100:.1f}%)")
    print()

    by_week = graded.groupby("week").apply(
        lambda g: pd.Series({"n": len(g), "correct": int((g.pick == g.actual).sum())}),
        include_groups=False,
    )
    by_week["pct"] = (by_week["correct"] / by_week["n"] * 100).round(1)
    print(by_week.to_string())


if __name__ == "__main__":
    summarize(backtest(season=2026))
