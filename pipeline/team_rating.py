"""
Roster Rating — a per-team, per-week number meant to read like a points
projection: "how many points would this team's current starting lineup be
expected to score against a league-average opponent, given the Madden
overall ratings of its (possibly injury-adjusted) starters." Consumed by
generate_matchups_table.py, which attaches it to each team's summary as
roster_rating, right after preseason_win_total.

This file documents every design decision inline, since the whole point
of a hand-built weighting system like this is that the weights ARE the
model — if the reasoning isn't written down next to the numbers, the next
person (or a future me) has no way to tell a deliberate choice from an
arbitrary one.

------------------------------------------------------------------------
THE TWO-STAGE DESIGN: raw talent score, then calibration to real points
------------------------------------------------------------------------
Stage 1 (RAW SCORE): every starter's Madden overall (0-100) is scaled to
a 0-1 fraction and multiplied by a fixed per-position weight (POSITION_
WEIGHTS below) — a 100-overall quarterback contributes the QB weight in
full, a 50-overall quarterback contributes half of it. Summing this
across all ~23 offensive/defensive starting slots gives a RAW score with
no inherent unit — it's "how much roster talent, by this system's own
accounting," not points.

Stage 2 (CALIBRATION): raw scores are linearly rescaled so the result
reads as an actual points number, anchored to real NFL scoring history
rather than to this model's own arbitrary raw-score range:
  - the league's actual average points-per-team-per-game over the most
    recent 5 completed seasons becomes the rating an exactly-average-raw
    team gets (see AVG_TEAM_POINTS_PER_GAME / _league_avg_ppg below)
  - the single highest-raw-score team in the league is pinned to that
    average plus CALIBRATION_SPREAD (9) points
  - the single lowest-raw-score team is pinned to that average minus
    CALIBRATION_SPREAD points
  - every other team falls on the straight line between its own side's
    extreme and the average (see _fit_calibration/_apply_calibration)

Why 9 — this was a number the project owner proposed, checked against
real scoring data rather than taken on faith. Team-game points_scored
from data/team_games.parquet, seasons 2021-2025:
  - league-wide average: 22.5 points/team/game
  - averaged across the 5 seasons individually, the single best-scoring
    team that year beat the league average by 8.3 points, and the
    single worst-scoring team trailed it by 7.5 points
  - taking the single most extreme team-SEASON across the whole 5-year
    span instead (not averaged per-year): the best was +10.7, the worst
    was -8.6
So the real spread between a league's best and worst scoring offense in
a given year is consistently in the 7.5-10.7 range depending on exactly
how it's measured. 9 lands squarely inside that range — not the largest
defensible number, not the smallest, a reasonable round compromise. This
model's spread describes roster TALENT rather than realized scoring (no
regression toward the mean from game-to-game variance, since there's no
variance here at all — a talent score doesn't have "bad luck" weeks), so
if anything this model's edge-to-edge spread could reasonably run a
little wider than the realized-scoring spread above; 9 is a conservative
pick given that, not an aggressive one.

------------------------------------------------------------------------
Which players count, and how they're identified (data sources)
------------------------------------------------------------------------
Starters come from data/depth_chart.parquet (ESPN's own stated depth
chart — ONE value per (team, position, slot_index), see fetch_depth_chart.py).
Known limitation, stated plainly: that file is a single current-week
snapshot, not an accumulating weekly history the way injury_reports.parquet
is — so there is no way to know who ESPN listed as starting QB in, say,
Week 2 after the fact. Every week's rating in this file is therefore
built from the SAME current depth chart (who's listed as starting now),
not a week-specific one. What DOES vary week to week is (1) which Madden
ratings snapshot applies (see _madden_iteration_for_week) and (2) which
of those depth-chart starters were actually unavailable that specific
week per data/injury_reports.parquet's real per-week history — so a
Week 2 rating still reflects "this roster, as currently understood, with
that week's actual injury situation applied to it," which is the most
historically-accurate reconstruction the available data supports, just
not a perfect one (a since-released/traded/IR'd player who isn't on
today's depth chart at all is invisible to every week's rating, including
weeks when they may have actually started).

Offense/defense only, no special teams — matches this project's existing
convention (see build_week1_baseline_starters.py's own "Offense/defense
only" section) and the user's own framing of this feature.

------------------------------------------------------------------------
Position weights — the actual numbers, and why
------------------------------------------------------------------------
Anchored first to two explicit examples the project owner gave (QB=10,
a mid-tier skill position like WR=5), then to NFL positional-value
research (market/draft-capital-based rankings — see the "Positional
Value in the NFL" research consulted while designing this: QB is
~2.7x the value of an average non-QB starter; edge rusher is the
clear #2 position, narrowly ahead of wide receiver; offensive tackle
and interior defensive line follow closely behind those two; running
back and off-ball linebacker are consistently the most devalued
starting positions in modern cap/draft markets), then filled in with
ordinary football judgment for positions the research didn't address
directly (interior OL, safety, slot corner, fullback).

POSITION_WEIGHTS values (at a 100 Madden overall, this is the full
point contribution; scales linearly down to 0 at a 0 overall):
  QB   10.0  — the clear outlier, matches both the explicit example and
               the "~2.7x an average starter" research finding
  WR    5.0  — the project owner's own example; 3 starting WR slots per
               team, so a true WR corps can out-weigh any other single
               position group
  LDE/
  RDE   5.5  — edge rushers, research's narrow #2 behind QB (just above
               WR) — true pass-rushing ends in both 4-3 and 3-4 share
               this weight (see the 3-4 OLB note below for why the
               *other* edge-ish position in a 3-4 is handled differently)
  LT/RT 4.5  — tackles, research's next tier down from edge/WR
  LCB/
  RCB   4.5  — outside corner; not covered directly by the research
               consulted, but shutdown corners command some of the
               largest non-QB second contracts in the league, so this
               sits with tackles rather than down with safety
  LDT/
  RDT   4.0  — 4-3 defensive tackle (interior DL), research's tier
               alongside tackle, just under edge
  NT    8.0  — the 3-4's single nose tackle anchors the ENTIRE interior
               line alone, where a 4-3 splits that job across two DTs
               (4.0 + 4.0 = 8.0) — weighted so a team's interior-line
               talent budget is scheme-neutral: a 3-4 team isn't
               penalized for running a 1-man line unless its nose
               tackle is actually weaker than the other team's two DTs
               combined (see "scheme neutrality" below)
  NB    3.5  — slot corner; a full-time starter in modern 3-WR
               sub-packages but still a half-step behind an outside
               corner by convention
  RB    3.0  — explicitly one of the most devalued starting positions
               in current NFL cap/draft markets ("running back
               devaluation" is a well-established theme), set below
               tackle/edge/corner and roughly level with TE
  TE    3.0  — level with RB; valuable but a clear step behind the
               WR corps
  LG/RG/
  C     2.5  — interior offensive line; valuable but a step behind
               tackle, who faces the harder one-on-one pass-rush
               matchups on the edge
  FS/SS 2.5  — safety; set level with interior OL rather than with
               corner, matching how safety typically trails corner in
               cap-market research even though both cover
  WLB/SLB 2.5 — see "3-4 OLB" note directly below
  MLB   1.0  — off-ball linebacker, the project owner's own example of
               a low-weight position, and consistently one of the two
               most devalued starting spots (with RB) in cap-market
               research
  LILB/
  RILB  0.5  — half of MLB's weight each, so the 3-4's two inside
               backers (0.5 + 0.5 = 1.0) sum to the SAME total as the
               4-3's single Mike linebacker — scheme neutrality again
  FB    0.5  — the project owner's own low-weight example; only 13/32
               teams even list a starting fullback, so this also only
               ever applies to a minority of teams

3-4 OLB note (WLB/SLB): a 3-4 defense's weak-side/strong-side
linebackers are frequently its primary EDGE rushers (functionally close
to a 4-3's defensive end), while a 4-3's WLB/SLB are conventional
off-ball linebackers closer to MLB in role. This is a real scheme
difference this model does NOT attempt to detect or weight differently
— WLB/SLB get one fixed weight (2.5) regardless of scheme, deliberately
set above MLB's pure-off-ball weight (1.0) but below a true primary edge
(5.5) as a single blended compromise. Flagged explicitly as a
simplification, not an oversight.

Scheme neutrality: by design, a team's defensive-line weight total
(LDE+RDE+LDT+RDT = 19.0, vs LDE+RDE+NT = 19.0) and linebacker weight
total (MLB = 1.0, vs LILB+RILB = 1.0) are identical whether the team
runs a 4-3 or a 3-4 — so this model doesn't systematically favor either
scheme; a team's rating differs based on its players' Madden ratings,
not which scheme label ESPN's depth chart happens to use. Confirmed
directly: every 4-3 and 3-4 team's maximum-possible defensive raw score
(all starters at 100 overall) sums to the same 42.5.

------------------------------------------------------------------------
Injury substitution
------------------------------------------------------------------------
Normally, the depth_rank == 1 player at each (team, position, slot_index)
is used as-is — this model does NOT second-guess ESPN's own stated
starter even if a backup happens to have a higher Madden rating; a
healthy, agreed-upon starter is trusted by default.

A starter is treated as "unavailable" this week — triggering substitution
— under either of two conditions:
  1. They're flagged as not playing per that week's own
     data/injury_reports.parquet row (reusing generate_matchups_table.py's
     _is_inactive logic: confirmed game-day-inactive, or a status other
     than Questionable/No designation — see that file's "Injury ordering"
     docstring section for the full reasoning), OR
  2. They have no usable Madden rating at all for that week (most often a
     name-format mismatch between ESPN's depth chart and EA's roster,
     e.g. a nickname — see generate_matchups_table.py's "Madden rating on
     injury rows" section for a documented real example of this same
     class of miss). Extending the substitution rule to also cover this
     case (not just real injuries) was a deliberate choice: without it, a
     pure text-matching failure would silently zero out that slot's whole
     contribution instead of falling back to a usable teammate.

When triggered, the system looks at every OTHER player listed behind that
starter at the SAME (team, position, slot_index) — not other slots of the
same position (e.g. a hurt WR1 does not pull from the WR2/WR3 depth
charts, just WR1's own backups) — and picks whichever of them has the
single HIGHEST Madden rating for that week, per the project owner's own
instruction ("the highest rated team member behind them slots in"). This
is intentionally a ONE-LEVEL substitution: the chosen backup's own
availability isn't itself re-checked. If literally no player at that slot
has a usable rating (starter or any backup), that slot contributes 0 for
that week rather than raising an error — rare, and worth monitoring, but
not fatal to the rest of the computation.

------------------------------------------------------------------------
Which Madden snapshot applies to which NFL week
------------------------------------------------------------------------
Per EA's own convention (confirmed via the ratings site's own "Weekly
Updates" filter, and already documented in fetch_madden_ratings.py):
"Week N Ratings" means the ratings in effect AFTER week N happened and
BEFORE week N+1 — i.e. they're the right snapshot to use for week N+1's
game. So: NFL week 1 uses "1-base" (Launch/preseason) ratings, since no
in-season snapshot exists yet; NFL week W (W >= 2) uses
"madden-ratings-week-{W-1}". Once a requested week is beyond the latest
snapshot actually scraped (this is a manual process — see
fetch_madden_ratings.py — not something that updates itself every week),
this falls back to whatever the latest available snapshot is, same
"most recent wins" rule generate_matchups_table.py's own
_madden_iteration_rank already uses. That means every future week's
rating is frozen at the most recently scraped ratings until the next
manual Madden scrape, which is an accepted, documented staleness, not a
bug.
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# See this file's docstring "Position weights" section for the reasoning
# behind every one of these numbers.
POSITION_WEIGHTS = {
    # Offense
    "QB": 10.0,
    "WR": 5.0,
    "RB": 3.0,
    "TE": 3.0,
    "LT": 4.5, "RT": 4.5,
    "LG": 2.5, "RG": 2.5, "C": 2.5,
    "FB": 0.5,
    # Defense
    "LDE": 5.5, "RDE": 5.5,
    "LDT": 4.0, "RDT": 4.0,
    "NT": 8.0,
    "LCB": 4.5, "RCB": 4.5,
    "NB": 3.5,
    "FS": 2.5, "SS": 2.5,
    "WLB": 2.5, "SLB": 2.5,
    "MLB": 1.0,
    "LILB": 0.5, "RILB": 0.5,
}

CALIBRATION_SPREAD = 9.0  # see docstring's "Why 9" section
CALIBRATION_SEASON_WINDOW = 5  # "the last 5 years" per the project owner's request

# Duplicated (not imported) from generate_matchups_table.py deliberately —
# that file imports FROM this module (team_ratings_by_week), so importing
# back the other way would be circular. Small, stable helpers like this
# one and _INACTIVE_STATUSES are cheap enough to keep in sync by hand;
# see fetch_madden_ratings.py/_parse_madden_backfill.py for the same
# accepted duplication pattern elsewhere in this codebase.
_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\Z")


def _norm_player_name(name: str) -> str:
    return _SUFFIX_RE.sub("", name.replace(".", "").strip().lower())


_INACTIVE_STATUSES = {
    "Out",
    "Doubtful",
    "Inactive (healthy/other)",
    "Injured Reserve",
    "IR - Designated to Return",
    "PUP",
    "NFI",
    "Suspended",
}

_MADDEN_WEEK_RE = re.compile(r"week-(\d+)\Z")


def _madden_iteration_rank(iteration: str) -> int:
    m = _MADDEN_WEEK_RE.search(iteration)
    return int(m.group(1)) if m else -1


def _madden_iteration_for_week(week: int, available_iterations: set) -> str:
    """See docstring's "Which Madden snapshot applies" section."""
    wanted = "1-base" if week <= 1 else f"madden-ratings-week-{week - 1}"
    if wanted in available_iterations:
        return wanted
    return max(available_iterations, key=_madden_iteration_rank)


def _league_avg_ppg(team_games: pd.DataFrame, current_season: int) -> float:
    """League-wide average points scored per team per game, over the
    CALIBRATION_SEASON_WINDOW most recently COMPLETED seasons (i.e. not
    including the current, possibly in-progress, season) — "use the last
    5 years to find that number" per the project owner's request,
    computed fresh each run rather than frozen as a constant so it rolls
    forward automatically as seasons complete."""
    seasons = range(current_season - CALIBRATION_SEASON_WINDOW, current_season)
    df = team_games[team_games.season.isin(seasons) & team_games.points_scored.notna()]
    return float(df["points_scored"].mean())


def _starter_slots(depth_chart: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, position, slot_index) this model scores —
    offense/defense only (side != 'ST'), and only positions actually in
    POSITION_WEIGHTS (this naturally excludes anything not covered, like
    a long-snapper that somehow leaked into a non-ST group)."""
    df = depth_chart[depth_chart.side.isin(["OFF", "DEF"]) & depth_chart.position.isin(POSITION_WEIGHTS)]
    return df


def _madden_overall_lookup(madden: pd.DataFrame) -> dict:
    """(iteration, team, normalized player name) -> overall rating."""
    out = {}
    for r in madden.itertuples():
        full_name = f"{r.first_name} {r.last_name}"
        out[(r.madden_iteration, r.team, _norm_player_name(full_name))] = int(r.overall)
    return out


def _unavailable_players(injuries: pd.DataFrame, season: int, week: int) -> set:
    """Set of (team, normalized player name) not expected to play in a
    given week, per that week's actual injury_reports.parquet rows —
    reuses generate_matchups_table.py's _is_inactive definition (a status
    other than Questionable/No designation, or confirmed game-day-inactive)."""
    wk = injuries[(injuries.season == season) & (injuries.week == week)]
    out = set()
    for r in wk.itertuples():
        if r.game_day_inactive or r.status in _INACTIVE_STATUSES:
            out.add((r.team, _norm_player_name(r.player)))
    return out


def _resolve_slot(candidates: pd.DataFrame, team: str, iteration: str, madden_lookup: dict, unavailable: set):
    """candidates: depth_chart rows for one (team, position, slot_index),
    sorted by depth_rank. Returns the Madden overall (float, 0-100) that
    should count for this slot this week, or None if nobody usable was
    found — see docstring's "Injury substitution" section."""
    starter = candidates.iloc[0]
    starter_key = (team, _norm_player_name(starter.player))
    starter_rating = madden_lookup.get((iteration, *starter_key))

    starter_ok = starter_key not in unavailable and starter_rating is not None
    if starter_ok:
        return starter_rating

    # Substitute: highest-Madden-rated player among the rest of this
    # exact slot's depth chart (not cross-slot) — one level only, their
    # own availability isn't re-checked (see docstring).
    best = None
    for _, c in candidates.iloc[1:].iterrows():
        rating = madden_lookup.get((iteration, team, _norm_player_name(c.player)))
        if rating is not None and (best is None or rating > best):
            best = rating
    return best


def _raw_score_for_team_week(team: str, iteration: str, slots: pd.DataFrame, madden_lookup: dict, unavailable: set) -> float:
    total = 0.0
    for (position, slot_index), group in slots[slots.team == team].groupby(["position", "slot_index"]):
        group = group.sort_values("depth_rank")
        rating = _resolve_slot(group, team, iteration, madden_lookup, unavailable)
        if rating is not None:
            total += (rating / 100.0) * POSITION_WEIGHTS[position]
    return total


def _fit_calibration(raw_by_team: dict, avg_ppg: float) -> dict:
    """See docstring's "Two-stage design" section — a piecewise-linear
    fit: the average-raw team maps to avg_ppg exactly, the single
    highest-raw team maps to avg_ppg + CALIBRATION_SPREAD exactly, the
    single lowest-raw team maps to avg_ppg - CALIBRATION_SPREAD exactly,
    and everything else falls on the straight line between its own
    side's extreme and the average (upper half and lower half can have
    different slopes, since real talent distributions aren't symmetric
    around their own mean)."""
    values = list(raw_by_team.values())
    raw_avg = sum(values) / len(values)
    raw_max = max(values)
    raw_min = min(values)
    return {"avg_ppg": avg_ppg, "raw_avg": raw_avg, "raw_max": raw_max, "raw_min": raw_min}


def _apply_calibration(raw: float, cal: dict) -> float:
    avg_ppg, raw_avg, raw_max, raw_min = cal["avg_ppg"], cal["raw_avg"], cal["raw_max"], cal["raw_min"]
    if raw >= raw_avg:
        span = raw_max - raw_avg
        frac = (raw - raw_avg) / span if span > 0 else 0.0
        return avg_ppg + CALIBRATION_SPREAD * frac
    else:
        span = raw_avg - raw_min
        frac = (raw_avg - raw) / span if span > 0 else 0.0
        return avg_ppg - CALIBRATION_SPREAD * frac


def team_ratings_by_week(season: int, weeks: list) -> dict:
    """Returns {(team, week): rating} for every team and every week in
    `weeks`, rounded to 1 decimal. Returns {} entirely if any required
    source is missing or has no rows for this season (e.g. depth_chart.parquet
    hasn't been scraped yet, or madden_ratings.parquet has no data for a
    new season) — this stat is additive, not load-bearing, so the rest of
    matchups.json should build fine without it."""
    depth_chart_path = DATA_DIR / "depth_chart.parquet"
    madden_path = DATA_DIR / "madden_ratings.parquet"
    team_games_path = DATA_DIR / "team_games.parquet"
    injuries_path = DATA_DIR / "injury_reports.parquet"
    if not (depth_chart_path.exists() and madden_path.exists() and team_games_path.exists() and injuries_path.exists()):
        return {}

    depth_chart = pd.read_parquet(depth_chart_path)
    depth_chart = depth_chart[depth_chart.season == season]
    madden = pd.read_parquet(madden_path)
    madden = madden[madden.season == season]
    team_games = pd.read_parquet(team_games_path)
    injuries = pd.read_parquet(injuries_path)
    injuries = injuries[injuries.season == season]

    if depth_chart.empty or madden.empty:
        return {}

    slots = _starter_slots(depth_chart)
    madden_lookup = _madden_overall_lookup(madden)
    available_iterations = set(madden["madden_iteration"].unique())
    all_teams = sorted(slots["team"].unique())
    avg_ppg = _league_avg_ppg(team_games, season)

    # Calibration anchor: each team's full-strength (no injury
    # substitution) raw score using the latest available Madden
    # snapshot — a stable, week-independent reference so the avg/+9/-9
    # scale itself doesn't drift around week to week (see docstring).
    latest_iteration = max(available_iterations, key=_madden_iteration_rank)
    raw_full_strength = {
        team: _raw_score_for_team_week(team, latest_iteration, slots, madden_lookup, unavailable=set())
        for team in all_teams
    }
    calibration = _fit_calibration(raw_full_strength, avg_ppg)

    ratings = {}
    for week in weeks:
        iteration = _madden_iteration_for_week(week, available_iterations)
        unavailable = _unavailable_players(injuries, season, week)
        for team in all_teams:
            raw = _raw_score_for_team_week(team, iteration, slots, madden_lookup, unavailable)
            ratings[(team, week)] = round(_apply_calibration(raw, calibration), 1)
    return ratings
