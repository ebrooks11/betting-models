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
Position weights — grounded in 2026 NFL franchise tag values
------------------------------------------------------------------------
SUPERSEDES an earlier version of this file that hand-tuned these weights
from positional-value research and the project owner's own examples
(QB=10, WR=5, MLB/FB=0.5-1). Replaced on the project owner's explicit
instruction to anchor them to real contract data instead: the 2026 NFL
franchise tag amount for each position, confirmed directly against two
independent sources (NBC Sports' report on the NFL's own announcement,
and Over the Cap's published figures) —

  FRANCHISE_TAG_2026 ($ millions, non-exclusive tag, one year):
    QB  43.895   WR  27.298   DT  27.127   LB  26.865   OL  25.773
    DE  24.434   CB  21.161   S   20.149   TE  15.045   RB  14.293

The franchise tag is a clean, externally-verifiable number for exactly
the thing this model wants — "what does the market pay for a top-5
player at this position" — since by rule it's defined as the average of
the top 5 cap hits at the position (or 120% of the player's own prior
salary, if greater), recalculated by the league every year. Using it
instead of hand-tuned numbers means this table updates by replacing ten
numbers with next year's tag values, not by re-litigating judgment calls.

POSITION_WEIGHTS = 10.0 * (that position's tag value / the QB tag value)
— QB is kept as the anchor at exactly 10.0 (preserving the scale of the
original design) and every other position is a direct proportional
share of QB's tag value, not a re-guessed number:
  QB   10.00  (43.895 / 43.895)
  WR    6.22  (27.298 / 43.895)
  DT    6.18  (27.127 / 43.895)  — interior defensive line
  LB    6.12  (26.865 / 43.895)  — see "LB tag quirk" below
  OL    5.87  (25.773 / 43.895)  — every offensive line spot
  DE    5.57  (24.434 / 43.895)  — edge
  CB    4.82  (21.161 / 43.895)  — every cornerback spot
  S     4.59  (20.149 / 43.895)
  TE    3.43  (15.045 / 43.895)
  RB    3.26  (14.293 / 43.895)  — FB uses this too, see below

Coarser than the position labels this model otherwise uses — mapped
onto POSITION_WEIGHTS as follows:
  QB -> QB
  WR -> WR (all 3 starting slots)
  RB, FB -> RB — the CBA's tag rules don't give fullback its own
    category at all; a tagged FB would be tagged as a running back, so
    this model values the two identically rather than inventing an
    unsupported fullback discount
  TE -> TE
  LT, RT, LG, RG, C -> OL — the tag doesn't distinguish tackle from
    guard from center, so neither does this model anymore (the earlier
    hand-tuned version gave tackle a premium over guard/center; that
    distinction isn't in the data, so it's gone)
  LDE, RDE -> DE
  LDT, RDT -> DT (one slot each, see "scheme-neutral slot splitting" below)
  NT -> DT x2 (one slot covering the same two-gap job, see below)
  LCB, RCB, NB -> CB — the tag doesn't distinguish outside corner from
    slot/nickel either; the earlier version's nickel discount is gone
    for the same reason as the tackle/guard distinction above
  FS, SS -> S
  MLB -> LB (one slot, see below)
  LILB, RILB -> LB / 2 each (two slots splitting the same job, see below)
  WLB, SLB -> LB (one slot each, see below)

LB tag quirk (worth stating plainly): the NFL's own "linebacker" tag
category is known to run high because it includes premium EDGE-rushing
3-4 outside linebackers (pass rushers who are nominally listed as LB,
not DE) alongside ordinary off-ball run-and-cover linebackers — which is
exactly why LB's tag value (26.865) sits almost as high as DT (27.127)
and well above DE's own (24.434). This model does NOT try to separate
those two very different real-world linebacker jobs back out — every
linebacker slot (MLB, LILB/RILB, WLB, SLB) draws from the same LB tag
figure, full stop. That's a direct, transparent consequence of using the
league's own tag categories as given rather than re-imposing a manual
split the data doesn't support; it does mean a true off-ball Mike
linebacker is valued more generously here than positional-value research
alone would suggest, and that's an accepted tradeoff of being data-driven
rather than hand-tuned.

Scheme-neutral slot splitting (NOT a valuation choice — a structural
one): the tag gives one value per position CATEGORY, but this model
needs a value per SLOT, and some categories map to a different number of
slots depending on a team's scheme (a 4-3 splits its interior line
across 2 DTs where a 3-4 uses 1 NT; a 4-3 uses 1 MLB where a 3-4 uses 2
inside linebackers). Dividing a shared job's value across however many
bodies a given scheme uses for it (NT = 2x the per-DT-slot value, since
one nose tackle alone occupies the same two interior gaps a 4-3 splits
across two tackles; LILB/RILB = half the per-LB-slot value each, since
two insiders split the same single "middle of the defense" job a 4-3
gives to one Mike) keeps a team's defensive-line and linebacker talent
BUDGET identical regardless of which scheme label ESPN's depth chart
happens to use for it — this is about fairly counting personnel, not
about re-deciding what a position is worth, so it's kept even though the
underlying per-position weights are now externally sourced. WLB and SLB
are NOT split this way in either scheme — both 4-3 and 3-4 defenses
field exactly one WLB and one SLB as standalone slots, so each gets the
full per-slot LB weight in both cases. Confirmed directly: every 4-3 and
3-4 team's maximum-possible defensive-line-plus-linebacker raw score
(all starters at 100 overall) sums to the same 41.85 (DE 11.13 + DT/NT
12.36 + LB 18.36), and its full defensive total (adding CB/NB/S, neither
of which differs by scheme) sums to the same 65.50 either way.

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

------------------------------------------------------------------------
Offensive Line Rating (ol_ratings_by_week) — a second, narrower stat
------------------------------------------------------------------------
Same starter-resolution machinery as roster_rating (current depth chart,
that week's own injury substitution via _resolve_slot, that week's own
Madden snapshot via _madden_iteration_for_week) but scoped to just the 5
offensive-line slots (LT, LG, C, RG, RT) and reported directly on a 0-100
scale — NOT run through roster_rating's points-calibration stage, since
the project owner asked for this one on "a 100 point scale" as a
standalone O-line quality score, not a points projection. A line of five
100-overall players scores exactly 100; the formula is a straight
weighted average, no rescaling against real NFL data the way
roster_rating's avg/+9/-9 calibration is.

Per-slot weights, and why they're NOT from FRANCHISE_TAG_2026
--------------------------------------------------------------------
The project owner asked for this to be weighted by franchise tag salary,
same as roster_rating — but the actual rule book doesn't cooperate here:
the NFL's franchise tag has exactly one blended "Offensive Line" category
(see FRANCHISE_TAG_2026 above), with no separate tackle/guard/center
split at all. Using that single blended number would make every one of
the 5 OL slots worth the same, which defeats the entire point of a
request for POSITION-weighted O-line grading (a left tackle and a guard
are not interchangeable).

So OL_SUBPOSITION_TOP5_APY substitutes the closest real-data equivalent:
each sub-position's own top-5 highest-APY contracts in the league right
now (average annual value; confirmed across multiple sources per
position), using the EXACT SAME METHODOLOGY the official tag itself uses
("average of the top 5 salaries at the position") — just computed
per-sub-position instead of blended, since the league doesn't publish
that split itself.
  Tackle (T): Tunsil 30.1, Sewell 30.0, Slater 28.7, Wirfs 28.6,
              Williams 27.5 -> avg 28.98
  Guard  (G): Skoronski 25.0, Bergeron 24.0, Tyler Smith 24.0,
              Trey Smith 23.5, Lindstrom 20.5 -> avg 23.40
  Center (C): Linderbaum 27.0, Humphrey 18.0, Jurgens 17.0,
              McGovern 13.0, McCoy 12.0 -> avg 17.40
Worth stating plainly since it's a little surprising: this shows guard
now paying ABOVE center in today's market (23.40 vs 17.40) — several
guard contracts reset that market in 2026 (Tyler Smith, then Skoronski).
Real, current data, not a transcription error.

Per-SLOT weights (not per-sub-position) are each sub-position's share of
the 5-slot "value budget" — 2 tackles + 2 guards + 1 center, weighted by
OL_SUBPOSITION_TOP5_APY — chosen so the 5 weights sum to exactly 1.0,
which is what makes an all-100-overall line score exactly 100:
  LT = RT = 28.98 / (2*28.98 + 2*23.40 + 17.40) = 0.2372 each
  LG = RG = 23.40 / (same total 122.16)          = 0.1915 each
  C                                               = 0.1424
OL Rating = 100 * (0.2372*LT_ovr/100 + 0.1915*LG_ovr/100 + 0.1424*C_ovr/100
                    + 0.1915*RG_ovr/100 + 0.2372*RT_ovr/100), i.e. just the
weighted-average overall across the 5 slots — computed in code from
OL_SUBPOSITION_TOP5_APY (not hardcoded weights), same pattern as
POSITION_WEIGHTS being derived from FRANCHISE_TAG_2026 above.
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 2026 NFL non-exclusive franchise tag values ($ millions) — confirmed
# directly against two independent sources (NBC Sports' report on the
# league's own announcement, and Over the Cap's published figures). See
# this file's docstring "Position weights" section for the full
# reasoning behind using these at all, the QB=10.0 anchor, and every
# slot-mapping/splitting decision below.
FRANCHISE_TAG_2026 = {
    "QB": 43.895, "WR": 27.298, "DT": 27.127, "LB": 26.865, "OL": 25.773,
    "DE": 24.434, "CB": 21.161, "S": 20.149, "TE": 15.045, "RB": 14.293,
}
_QB_ANCHOR = 10.0
_TAG_WEIGHT = {pos: _QB_ANCHOR * val / FRANCHISE_TAG_2026["QB"] for pos, val in FRANCHISE_TAG_2026.items()}

POSITION_WEIGHTS = {
    # Offense
    "QB": _TAG_WEIGHT["QB"],
    "WR": _TAG_WEIGHT["WR"],
    "RB": _TAG_WEIGHT["RB"],
    "FB": _TAG_WEIGHT["RB"],  # no separate FB tag category — CBA tags a fullback as a running back
    "TE": _TAG_WEIGHT["TE"],
    "LT": _TAG_WEIGHT["OL"], "RT": _TAG_WEIGHT["OL"],
    "LG": _TAG_WEIGHT["OL"], "RG": _TAG_WEIGHT["OL"], "C": _TAG_WEIGHT["OL"],
    # Defense
    "LDE": _TAG_WEIGHT["DE"], "RDE": _TAG_WEIGHT["DE"],
    "LDT": _TAG_WEIGHT["DT"], "RDT": _TAG_WEIGHT["DT"],
    "NT": _TAG_WEIGHT["DT"] * 2,  # one slot covering the same two-gap job a 4-3 splits across 2 DTs
    "LCB": _TAG_WEIGHT["CB"], "RCB": _TAG_WEIGHT["CB"], "NB": _TAG_WEIGHT["CB"],
    "FS": _TAG_WEIGHT["S"], "SS": _TAG_WEIGHT["S"],
    "WLB": _TAG_WEIGHT["LB"], "SLB": _TAG_WEIGHT["LB"],  # standalone slots in both schemes, not split
    "MLB": _TAG_WEIGHT["LB"],
    "LILB": _TAG_WEIGHT["LB"] / 2, "RILB": _TAG_WEIGHT["LB"] / 2,  # split the same job a 4-3 gives one MLB
}

CALIBRATION_SPREAD = 9.0  # see docstring's "Why 9" section
CALIBRATION_SEASON_WINDOW = 5  # "the last 5 years" per the project owner's request

# Top-5 highest-APY ($ millions) contracts by offensive-line SUB-position,
# right now — see docstring's "Offensive Line Rating" section for why
# this (not FRANCHISE_TAG_2026's single blended "OL" figure) is used, and
# for the sourcing of each individual contract value.
OL_SUBPOSITION_TOP5_APY = {
    "T": (30.1 + 30.0 + 28.7 + 28.6 + 27.5) / 5,   # Tunsil, Sewell, Slater, Wirfs, Williams
    "G": (25.0 + 24.0 + 24.0 + 23.5 + 20.5) / 5,   # Skoronski, Bergeron, Ty.Smith, Tr.Smith, Lindstrom
    "C": (27.0 + 18.0 + 17.0 + 13.0 + 12.0) / 5,   # Linderbaum, Humphrey, Jurgens, McGovern, McCoy
}


def _ol_slot_weights() -> dict:
    """Per-slot weights for LT/RT/LG/RG/C that sum to exactly 1.0 — see
    docstring's "Offensive Line Rating" section for the derivation."""
    t, g, c = OL_SUBPOSITION_TOP5_APY["T"], OL_SUBPOSITION_TOP5_APY["G"], OL_SUBPOSITION_TOP5_APY["C"]
    total = 2 * t + 2 * g + c
    return {"LT": t / total, "RT": t / total, "LG": g / total, "RG": g / total, "C": c / total}


OL_SLOT_WEIGHTS = _ol_slot_weights()

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


def _ol_rating_for_team_week(team: str, iteration: str, ol_slots: pd.DataFrame, madden_lookup: dict, unavailable: set) -> float:
    total = 0.0
    for (position, slot_index), group in ol_slots[ol_slots.team == team].groupby(["position", "slot_index"]):
        group = group.sort_values("depth_rank")
        rating = _resolve_slot(group, team, iteration, madden_lookup, unavailable)
        if rating is not None:
            total += OL_SLOT_WEIGHTS[position] * rating
    return round(total, 1)


def ol_ratings_by_week(season: int, weeks: list) -> dict:
    """Returns {(team, week): rating} for the Offensive Line Rating — see
    docstring's "Offensive Line Rating" section. Same starter/injury-
    substitution machinery as team_ratings_by_week, scoped to just the 5
    OL slots, no points-calibration stage (already a natural 0-100
    scale). Returns {} if a required source is missing, same
    additive-not-load-bearing contract as team_ratings_by_week."""
    depth_chart_path = DATA_DIR / "depth_chart.parquet"
    madden_path = DATA_DIR / "madden_ratings.parquet"
    injuries_path = DATA_DIR / "injury_reports.parquet"
    if not (depth_chart_path.exists() and madden_path.exists() and injuries_path.exists()):
        return {}

    depth_chart = pd.read_parquet(depth_chart_path)
    depth_chart = depth_chart[depth_chart.season == season]
    madden = pd.read_parquet(madden_path)
    madden = madden[madden.season == season]
    injuries = pd.read_parquet(injuries_path)
    injuries = injuries[injuries.season == season]

    if depth_chart.empty or madden.empty:
        return {}

    ol_slots = depth_chart[depth_chart.position.isin(OL_SLOT_WEIGHTS)]
    madden_lookup = _madden_overall_lookup(madden)
    available_iterations = set(madden["madden_iteration"].unique())
    all_teams = sorted(ol_slots["team"].unique())

    ratings = {}
    for week in weeks:
        iteration = _madden_iteration_for_week(week, available_iterations)
        unavailable = _unavailable_players(injuries, season, week)
        for team in all_teams:
            ratings[(team, week)] = _ol_rating_for_team_week(team, iteration, ol_slots, madden_lookup, unavailable)
    return ratings
