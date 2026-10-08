"""
Generate docs/data/matchups.json for docs/matchups.html, built from
data/schedules.parquet (the games list itself, and — across its full
multi-season history, not just the current season — each team's recent
game log), data/injury_reports.parquet, data/team_games.parquet,
data/game_lines.parquet, data/win_totals.parquet, and
data/snap_counts.parquet — deliberately never data/pbp.parquet directly
(see "Starting QB" below for why). A full matchup view, not just injuries:
each game shows the current line (and, once played, the final score),
and each team shows season-to-date point differential/EPA per
play/success rate plus its head coach, both coordinators, starting QB,
and last-5-games form, with injuries as one section among several.

Games list comes from schedules.parquet, not injury_reports.parquet
------------------------------------------------------------------------
An earlier version of this script took its list of games/weeks from
injury_reports.parquet, which happens to carry game/date/kickoff metadata
too. That meant the page's week/game coverage was bounded by whatever
weeks the injury-report automation had touched — a team with no injury
news that week, or a week the Drive folder hadn't been synced for yet,
would simply be missing from the page. schedules.parquet is the real,
authoritative source for "what games exist and when" (built fresh each
week by pipeline/fetch_current_season_data.py, covering the full season
including games not yet played — see that script and
build_team_games_table.py for how it's kept current), so every other
source here (injuries, team stats, lines) is now joined onto that games
list rather than the other way around. A game with no injury data yet, or
no line yet, still shows up with whatever it does have.

Not the {n, columns, data} columnar shape used by generate_game_table.py and
its descendants (generate_players_table.py, generate_coordinators_table.py,
generate_team_games_table.py) — that convention fits a sortable flat table,
and a matchups page is a list of per-game cards, not a table. Shape here:

{
  "season": 2026,
  "weeks": [1, 2, 3],
  "current_week": 3,
  "games_by_week": {
    "3": [
      {
        "game_date": "...", "kickoff_et": "...",
        "away_team": "ATL", "home_team": "GB",
        "lines": {"bookmaker": "draftkings", "home_moneyline": ..., ...} | null,
        "final": {"home_score": 24, "away_score": 17} | null,
        "away_team_summary": { ...see _team_summary_for_season/_coordinators_with_fallback...,
                                "recent_games": [ {season, week, opponent, home_away,
                                                    team_score, opp_score, result,
                                                    covered, total_result}, ... ] },
        "home_team_summary": { ... },
        "away_injuries": [ {player, pos, status, injury, practice: [d1,d2,d3],
                             game_day_inactive, expected_return, notes,
                             snap_share, likely_starter}, ... ],
        "home_injuries": [ ... ]
      }, ...
    ],
    "2": [...], "3": [...]
  }
}

current_week = the earliest week in schedules.parquet with at least one
game not yet played (null home_score), falling back to the season's final
week once everything's complete — this is the page's default selection,
"upcoming" in practice. weeks is every week in the full season schedule
(1-18 for a normal REG season), not just weeks that happen to have
injury/line data yet, so the week picker lets you browse the whole season
even before injuries or lines exist for a given week.

Injury ordering
------------------
Within each team's injury list, rows are ordered by (1) confirmed
game-day-inactive first — regardless of status/type, since "not playing
today" is the single most actionable fact for a fantasy manager reading
this ahead of kickoff — then (2) _STATUS_RANK (the actual weekly
game-status designations first — Out down to No designation — then
healthy scratches, then season-long reserve-list entries), then (3)
player name. This ordering is a display judgment call made here, not
baked into injury_reports.parquet itself, so it's easy to change later
without rebuilding the underlying table.

Starter emphasis (snap_share, likely_starter)
------------------------------------------------
Each injury row also carries snap_share — the player's recent share of
their unit's snaps (see _snap_shares_before_week's docstring for the
exact window/fallback logic) — and likely_starter, a simple threshold on
that value. Neither comes from a real depth chart (data/depth_charts.parquet
stops at 2024 and isn't kept current by this pipeline); they're inferred
from data/snap_counts.parquet, which PFR does publish within a day or two
of each 2026 game. This is a proxy, not ground truth — a committee
backfield can have no player clear the threshold, and a player who
hasn't taken a single snap in a long time (new signing, far outside
either season's window) just comes back with snap_share: null — but it's
real usage data, not a guess, and directly answers what a fantasy manager
actually wants to know about an injury: was this player actually
playing.

Team season summary
-----------------------
point_differential, points_per_game, offensive_points_per_game,
epa_per_play, success_rate, and early_down_success_rate are each averaged
across the team's team_games.parquet rows for the current season, but
only games strictly before the week being shown — not every game played
so far regardless of week. Computing one "as of right now" snapshot and
reusing it on every week's card would leak future results into past
weeks' cards (a Week 1 preview would end up showing stats that include
Weeks 2 and 3, which hadn't happened yet) — recomputed per week instead,
matching how a broadcast previews a game with "season entering tonight"
stats, not "season including tonight and afterward." games_played is
included alongside the rates so a 1-game and a 10-game average aren't
read as equally confident.

The matchups page's compare table is offense-only throughout (point_differential
and points_per_game are kept here as general-purpose facts since other
code may still want them, but aren't surfaced in the compare table — see
build_team_games_table.py's "Offensive points" docstring section for why
points_scored/points_allowed mix in defensive/special-teams scoring and
offensive_points_per_game is the one actually shown instead).

preseason_win_total is the one exception to this before-this-week
recomputation — it's a single value set before the season starts (see
pipeline/fetch_win_totals.py) and never changes, so
it's just looked up for the season, not averaged or windowed. The
before-this-week cutoff applies to
starting_qb below, for the same reason.

Coordinators with fallback
------------------------------
hc_name/oc_name/dc_name come from team_games.parquet for the current
season — but early in a season, Wikipedia's own season article often
hasn't had its staff section filled in yet (confirmed directly: every
2026 team-season scrape returned "no staff table found" as of this
writing, even though the season articles themselves already exist), so
those columns are usually still null for the current season this early.
Rather than showing a hole, each of the three fields independently falls
back to the most recent earlier season that has a non-null value for that
team, and reports which season it actually came from (coordinators_as_of)
so the page can be honest that it might be stale (an offseason hire could
have replaced that name and Wikipedia just hasn't caught up here yet).

Starting QB
--------------
Read from team_games.parquet's own starting_qb column (whoever led that
team in pass attempts in that specific game — see that table's docstring
for the full definition/reasoning), taking the value from each team's
most recent game before the one being shown. Deliberately the most
recent single game's starter, not a cumulative-attempts leader across
the season — a cumulative total is skewed toward whoever started earlier
weeks even after being benched, injured, or replaced, so a team that
changed starters recently would keep showing the old one for the rest of
the season. Checked directly against a real case: Atlanta's cumulative
attempts leader through week 2 was Cooper Rush, but Michael Penix Jr.
took over the following week — this proxy can't know that a change is
happening in the very game being previewed (that data doesn't exist
until after that game is played), but it does correctly drop a stale
starter the moment a more recent game shows otherwise, which cumulative
attempts would not.

Deliberately not reading pbp.parquet directly in this file (unlike an
earlier version of this script): pbp.parquet is the one genuinely large
file in this pipeline, and team_games.parquet already has a per-game
starting_qb column for exactly this reason — see that table's docstring,
"Building from a partial pbp.parquet." Every consumer of per-game QB
info, including this one, should go through team_games.parquet instead of
needing pbp.parquet themselves.

Final score (final)
-----------------------
Set once a game's home_score/away_score are both non-null in
schedules.parquet (i.e. the game has been played), null before kickoff.
Shown alongside the pregame line so a past week's card reads as "here's
what was expected, here's what happened," not just a bare line that
quietly stopped mattering once the game ended.

Recent form (recent_games)
-------------------------------
Each team's last RECENT_GAMES_WINDOW (5) played REG-season games before
the one being shown — see _team_game_log and _recent_games_before_week.
Reaches back across the season boundary when
the current season doesn't have 5 games yet (same "before this week"
discipline as everything else on this page — only games before the one
being previewed, so a Week 2 card never shows Week 3's result — but with
a wider lookback than the single-season stats above, matching how a
broadcast's "last 5" graphic works): a Week 2 preview shows last
season's final 4 games plus this season's Week 1, not just one game.

Each game also carries covered (did THIS team beat the closing spread —
True/False/"push"/null) and total_result ("over"/"under"/"push"/null),
both graded against schedules.parquet's own spread_line/total_line —
nflverse's historical closing lines, present for essentially every
past game (unlike data/game_lines.parquet, this project's own
accumulated snapshot history, which only goes back to when that scraper
started running). Note the sign convention difference: spread_line is
the expected HOME margin (positive = home favored), the opposite of
game_lines.parquet's home_spread (positive = home underdog, the normal
sportsbook-board convention used in the Lines row above) — confirmed
directly against a shared game (2026 week 4 CLE/PIT: spread_line -2.5
here vs. home_spread +2.5 there, an exact negation). See the comments
in _team_game_log for the cover/total formulas.

Game lines
-------------
data/game_lines.parquet accumulates the full snapshot history (every
bookmaker, every day) — see pipeline/fetch_game_lines_from_sheet.py. This
file reduces that to "the current line" via that module's
current_lines() (preferred bookmaker, latest snapshot per market), then
looks it up by (home_team, away_team). Null when no line exists for that
matchup — most commonly because the game has already kicked off (books
pull lines once a game starts) or the odds Sheet simply hasn't been
snapshotted for that far-out a game yet; either way, the page should
render fine without it.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data" / "matchups.json"

# Same normalization convention as build_team_games_table.py /
# build_injury_reports_table.py — not expected to matter for the current
# season specifically (nfl_data_py's current-season schedules already use
# canonical codes), but applied for consistency in case this is ever run
# against an older season.
_TEAM_NORM = {
    "OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA",
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA",
}


def _norm_team(code):
    return _TEAM_NORM.get(code, code)


def _fmt_kickoff_et(gametime) -> str:
    """schedules.parquet's gametime is 24-hour "HH:MM" ET — reformat to
    the "H:MM AM/PM" style the rest of this page already uses."""
    if gametime is None or (isinstance(gametime, float) and pd.isna(gametime)):
        return None
    hour, minute = gametime.split(":")
    hour = int(hour)
    period = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12}:{minute} {period}"


_STATUS_RANK = {
    "Out": 0,
    "Doubtful": 1,
    "Questionable": 2,
    "No designation": 3,
    "Inactive (healthy/other)": 4,
    "Injured Reserve": 5,
    "IR - Designated to Return": 5,
    "PUP": 6,
    "NFI": 6,
    "Suspended": 7,
}

_COORD_FIELDS = ["hc_name", "oc_name", "dc_name"]


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if pd.isna(v):
        return None
    return v


def _round(v, ndigits=3):
    v = _clean(v)
    return round(float(v), ndigits) if v is not None else None


_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\Z")


def _norm_player_name(name: str) -> str:
    """injury_reports.parquet and snap_counts.parquet don't always agree on
    "A.J. Terrell" vs "AJ Terrell" or "Mack Wilson" vs "Mack Wilson Sr." —
    strip periods/suffixes so the two sources' names line up for the
    starter-emphasis join below. Confirmed directly: without this, ~60%
    of 2026 injury rows matched a snap_counts row by (team, player); with
    it, that's somewhat higher, and the (team, ...) half of the key is
    still what actually prevents misattributing one player's snaps to a
    different person who happens to share a name (see SNAP_SHARE_THRESHOLD
    below — two different active 2026 players are both named "Justin
    Jefferson," one a Browns LB and one a Vikings WR)."""
    return _SUFFIX_RE.sub("", name.replace(".", "").strip().lower())


# A player counts as "likely starting if healthy" when they've been playing
# the bulk of their unit's snaps — not tied to a specific depth-chart slot
# (which isn't in any of this pipeline's data), just overall usage. 0.55 is
# a judgment call: high enough to exclude true committee/rotational pieces,
# low enough to still catch every-down players who come off the field in
# some packages (e.g. a early-down runner in a pass-heavy offense).
SNAP_SHARE_STARTER_THRESHOLD = 0.55
# How many of a player's most recent active (nonzero-snap) games to average
# over — recent-form, not a season-long average, so a role that changed
# recently (new starter winning a job, a committee resolving) is reflected
# quickly rather than smoothed out by weeks-old data.
SNAP_SHARE_WINDOW = 3


def _snap_shares_before_week(snap_counts: pd.DataFrame, season: int, week: int) -> dict:
    """(team, normalized player name) -> recent snap share (0-1), used to
    flag injuries to players who'd likely be starting if healthy.

    Falls back to the player's most recent active games from the prior
    season when they haven't played a single snap yet this season before
    the week in question — the exact case that matters most here: a
    player hurt in week 1 (or before the season even started) has no
    current-season snap data to show they were a starter, but is also
    the player whose absence is most worth calling out. Confirmed on a
    real case: Arizona's James Conner played 3 total snaps in 2026 before
    being out the rest of the way, which this fallback catches via his
    clear starter-level 2025 usage; a player who's never been more than a
    backup in either season just correctly comes back with no match.
    This fallback can't follow a team change across the offseason (keyed
    on the player's *current* team), which is the right failure mode —
    crediting a new team with a snap share earned on a different roster
    would be misleading, not just imprecise.
    """
    snaps = snap_counts.copy()
    snaps["active"] = (snaps["offense_snaps"].fillna(0) + snaps["defense_snaps"].fillna(0) + snaps["st_snaps"].fillna(0)) > 0
    snaps["share"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps["norm_player"] = snaps["player"].map(_norm_player_name)

    current = snaps[(snaps.season == season) & (snaps.week < week) & snaps.active]
    prior = snaps[(snaps.season == season - 1) & snaps.active]

    out = {}
    for source in (prior, current):  # current season applied last, so it wins where both exist
        for (team, player), g in source.groupby(["team", "norm_player"]):
            recent = g.sort_values("week", ascending=False).head(SNAP_SHARE_WINDOW)
            out[(team, player)] = recent["share"].mean()
    return out


def _player_row(r, snap_shares):
    snap_share = snap_shares.get((r["team"], _norm_player_name(r["player"])))
    return {
        "player": r["player"],
        "pos": _clean(r["pos"]),
        "status": r["status"],
        "injury": _clean(r["injury"]),
        "practice": [_clean(r["practice_day_1"]), _clean(r["practice_day_2"]), _clean(r["practice_day_3"])],
        "game_day_inactive": bool(r["game_day_inactive"]),
        "expected_return": _clean(r["expected_return"]),
        "notes": _clean(r["notes"]),
        "snap_share": _round(snap_share, 2),
        "likely_starter": bool(snap_share is not None and snap_share >= SNAP_SHARE_STARTER_THRESHOLD),
    }


def _team_injuries(df, season, week, team, snap_shares):
    # Keyed by (season, week, team) only, not also by the injury sheet's
    # own "AWAY @ HOME" game string — the games list itself now comes from
    # schedules.parquet (see build()), a different source, so matching on
    # team identity alone is both simpler and more robust than requiring
    # the two sources' game-string formatting to agree.
    rows = df[(df.season == season) & (df.week == week) & (df.team == team)].copy()
    rows["_inactive_rank"] = ~rows["game_day_inactive"]
    rows["_rank"] = rows["status"].map(_STATUS_RANK).fillna(9)
    rows = rows.sort_values(["_inactive_rank", "_rank", "player"])
    return [_player_row(r, snap_shares) for _, r in rows.iterrows()]


def _team_summary_before_week(team_games: pd.DataFrame, season: int, week: int) -> dict:
    season_df = team_games[(team_games.season == season) & (team_games.week < week)]
    out = {}
    for team, g in season_df.groupby("team"):
        out[team] = {
            "games_played": int(len(g)),
            # point_differential/points_per_game are team-level (they
            # include defensive/special-teams scoring via points_scored/
            # points_allowed) — kept here as general-purpose facts for
            # anything else that wants them, but NOT shown on the matchups
            # page's compare table, which is offense-only throughout (see
            # offensive_points_per_game below and build_team_games_table.py's
            # "Offensive points" docstring section for why).
            "point_differential": _round((g["points_scored"] - g["points_allowed"]).mean(), 1),
            "points_per_game": _round(g["points_scored"].mean(), 1),
            "offensive_points_per_game": _round(g["offensive_points"].mean(), 1),
            "epa_per_play": _round(g["epa_per_play"].mean()),
            "success_rate": _round(g["success_rate"].mean()),
            "early_down_success_rate": _round(g["early_down_success_rate"].mean()),
        }
    return out


def _coordinators_with_fallback(team_games: pd.DataFrame, season: int) -> dict:
    by_team_season = (
        team_games[["team", "season"] + _COORD_FIELDS]
        .drop_duplicates(subset=["team", "season"])
        .sort_values(["team", "season"], ascending=[True, False])
    )
    out = {}
    for team, g in by_team_season.groupby("team"):
        g = g[g.season <= season]
        entry = {}
        for field in _COORD_FIELDS:
            non_null = g[g[field].notna()]
            if non_null.empty:
                entry[field] = None
                entry[f"{field}_as_of"] = None
            else:
                row = non_null.iloc[0]
                entry[field] = row[field]
                entry[f"{field}_as_of"] = int(row["season"])
        out[team] = entry
    return out


def _starting_qbs_before_week(team_games: pd.DataFrame, season: int, week: int) -> dict:
    df = team_games[
        (team_games.season == season) & (team_games.week < week) & team_games.starting_qb.notna()
    ]
    if df.empty:
        return {}
    # Most recent game per team, then that game's starter — see module
    # docstring's "Starting QB" section for why this beats using whoever
    # led cumulative attempts across the season so far.
    idx = df.groupby("team")["week"].idxmax()
    latest = df.loc[idx]
    return dict(zip(latest["team"], latest["starting_qb"]))


def _game_lines_lookup(path: Path) -> dict:
    # Running this file directly (python3 pipeline/generate_matchups_table.py,
    # as every caller of this does) puts pipeline/ itself on sys.path, not
    # the repo root — add the repo root explicitly so "pipeline" is
    # importable as a package (same fix as fetch_injury_reports_from_drive.py).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.fetch_game_lines_from_sheet import current_lines

    if not path.exists():
        return {}
    df = current_lines(pd.read_parquet(path))
    if df.empty:
        return {}
    lookup = {}
    for _, r in df.iterrows():
        lookup[(r["home_team"], r["away_team"])] = {
            "bookmaker": r["bookmaker"],
            "home_moneyline": _clean(r["home_moneyline"]),
            "away_moneyline": _clean(r["away_moneyline"]),
            "home_spread": _clean(r["home_spread"]),
            "home_spread_price": _clean(r["home_spread_price"]),
            "away_spread": _clean(r["away_spread"]),
            "away_spread_price": _clean(r["away_spread_price"]),
            "total_point": _clean(r["total_point"]),
            "over_price": _clean(r["over_price"]),
            "under_price": _clean(r["under_price"]),
        }
    return lookup


def _win_totals_lookup(path: Path, season: int) -> dict:
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["season", "team", "win_total"])
    df = df[df.season == season]
    return dict(zip(df["team"], df["win_total"]))


RECENT_GAMES_WINDOW = 5


def _team_game_log(schedules_all: pd.DataFrame) -> pd.DataFrame:
    """Long-format log, one row per (team, played REG game), from both
    teams' perspective — the input to each team's "last 5 games" strip.
    Spans every season in schedules.parquet, not just the current one, so
    early in a season (not yet 5 games played) the window still reaches
    back into the prior season rather than showing fewer than 5 — the
    same "always fill the window" choice _snap_shares_before_week makes
    for the same reason. Team codes are normalized (see _TEAM_NORM) since
    this reaches back far enough in some teams' histories to hit old
    codes like OAK/SD/STL that the current season never uses."""
    df = schedules_all[(schedules_all.game_type == "REG") & schedules_all.home_score.notna()].copy()
    df["home_team"] = df["home_team"].map(_norm_team)
    df["away_team"] = df["away_team"].map(_norm_team)

    # schedules.parquet's spread_line is nflverse's own convention — the
    # expected HOME margin (positive = home favored), the opposite sign of
    # the "home team's own spread as shown on a sportsbook board" convention
    # used elsewhere on this page (data/game_lines.parquet's home_spread,
    # positive = home underdog). Confirmed directly against a real game
    # already cross-checked against DraftKings odds: 2026 week 4 CLE
    # (home) vs PIT, spread_line -2.5 vs game_lines.parquet's home_spread
    # +2.5 for the same game — exact negation. home_margin = home_score -
    # away_score; home covers if home_margin > spread_line (i.e. beat the
    # expected margin), not the "+ home_spread > 0" formula betResults()
    # in matchups.html uses for the live board-convention spread.
    home_margin = df["home_score"] - df["away_score"]
    cover_diff = home_margin - df["spread_line"]
    df["_cover_side"] = None
    df.loc[df.spread_line.notna() & (cover_diff > 0), "_cover_side"] = "home"
    df.loc[df.spread_line.notna() & (cover_diff < 0), "_cover_side"] = "away"
    df.loc[df.spread_line.notna() & (cover_diff == 0), "_cover_side"] = "push"

    actual_total = df["home_score"] + df["away_score"]
    df["_total_result"] = None
    df.loc[df.total_line.notna() & (actual_total > df["total_line"]), "_total_result"] = "over"
    df.loc[df.total_line.notna() & (actual_total < df["total_line"]), "_total_result"] = "under"
    df.loc[df.total_line.notna() & (actual_total == df["total_line"]), "_total_result"] = "push"

    home = df.rename(columns={"home_team": "team", "away_team": "opponent", "home_score": "team_score", "away_score": "opp_score"})
    home["home_away"] = "home"
    away = df.rename(columns={"away_team": "team", "home_team": "opponent", "away_score": "team_score", "home_score": "opp_score"})
    away["home_away"] = "away"

    cols = ["season", "week", "team", "opponent", "team_score", "opp_score", "home_away", "_cover_side", "_total_result"]
    log = pd.concat([home[cols], away[cols]], ignore_index=True)
    log["result"] = "T"
    log.loc[log.team_score > log.opp_score, "result"] = "W"
    log.loc[log.team_score < log.opp_score, "result"] = "L"
    # covered: whether THIS team (not just the home side) covered — "push"
    # either way when cover_side is "push". Built as an object column from
    # the start (not a bool column later overwritten with "push"/None) to
    # avoid a dtype-cast warning.
    log["covered"] = pd.Series([None] * len(log), dtype=object)
    log.loc[log["_cover_side"] == log["home_away"], "covered"] = True
    log.loc[(log["_cover_side"].notna()) & (log["_cover_side"] != log["home_away"]) & (log["_cover_side"] != "push"), "covered"] = False
    log.loc[log["_cover_side"] == "push", "covered"] = "push"
    log = log.rename(columns={"_total_result": "total_result"}).drop(columns=["_cover_side"])
    return log


def _recent_games_before_week(log: pd.DataFrame, season: int, week: int, team: str) -> list:
    g = log[(log.team == team) & ((log.season < season) | ((log.season == season) & (log.week < week)))]
    g = g.sort_values(["season", "week"], ascending=False).head(RECENT_GAMES_WINDOW)
    # Most-recent-first here; matchups.html reverses this for display
    # (oldest-to-newest, left-to-right — the usual "form guide" reading
    # order) so the ordering choice lives in one place, not both.
    return [
        {
            "season": int(r.season), "week": int(r.week), "opponent": r.opponent,
            "home_away": r.home_away, "team_score": int(r.team_score), "opp_score": int(r.opp_score),
            "result": r.result, "covered": _clean(r.covered), "total_result": _clean(r.total_result),
        }
        for r in g.itertuples()
    ]


def _team_summary_entry(team, summaries, coordinators, starting_qbs, win_totals, recent_games):
    entry = dict(summaries.get(team, {
        "games_played": 0, "point_differential": None, "points_per_game": None,
        "offensive_points_per_game": None, "epa_per_play": None,
        "success_rate": None, "early_down_success_rate": None,
    }))
    entry.update(coordinators.get(team, {f: None for f in _COORD_FIELDS} | {f"{f}_as_of": None for f in _COORD_FIELDS}))
    entry["starting_qb"] = starting_qbs.get(team)
    entry["preseason_win_total"] = _clean(win_totals.get(team))
    entry["recent_games"] = recent_games.get(team, [])
    return entry


def build():
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    season = int(schedules["season"].max())
    sched = schedules[(schedules.season == season) & (schedules.game_type == "REG")].copy()
    sched["away_team"] = sched["away_team"].map(_norm_team)
    sched["home_team"] = sched["home_team"].map(_norm_team)

    inj = pd.read_parquet(DATA_DIR / "injury_reports.parquet")
    inj = inj[inj.season == season]

    team_games = pd.read_parquet(DATA_DIR / "team_games.parquet")
    coordinators = _coordinators_with_fallback(team_games, season)
    lines_lookup = _game_lines_lookup(DATA_DIR / "game_lines.parquet")
    win_totals = _win_totals_lookup(DATA_DIR / "win_totals.parquet", season)
    snap_counts = pd.read_parquet(DATA_DIR / "snap_counts.parquet", columns=[
        "season", "week", "team", "player", "offense_snaps", "offense_pct", "defense_snaps", "defense_pct", "st_snaps",
    ])
    game_log = _team_game_log(schedules)
    all_teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))

    weeks = sorted(sched["week"].unique().tolist())
    # "Upcoming" = the earliest week with at least one game not yet
    # played (home_score still null) — falls back to the final week once
    # the whole season is complete.
    unplayed = sched[sched["home_score"].isna()]
    current_week = int(unplayed["week"].min()) if not unplayed.empty else int(sched["week"].max())

    games_by_week = {}
    for week in weeks:
        summaries = _team_summary_before_week(team_games, season, week)
        starting_qbs = _starting_qbs_before_week(team_games, season, week)
        snap_shares = _snap_shares_before_week(snap_counts, season, week)
        recent_games = {team: _recent_games_before_week(game_log, season, week, team) for team in all_teams}

        wk_games = sched[sched.week == week].sort_values(["gameday", "gametime"])
        week_games = []
        for _, g in wk_games.iterrows():
            away, home = g["away_team"], g["home_team"]
            played = pd.notna(g["home_score"]) and pd.notna(g["away_score"])
            week_games.append({
                "game_date": g["gameday"],
                "kickoff_et": _fmt_kickoff_et(g["gametime"]),
                "away_team": away,
                "home_team": home,
                "lines": lines_lookup.get((home, away)),
                "final": {"home_score": int(g["home_score"]), "away_score": int(g["away_score"])} if played else None,
                "away_team_summary": _team_summary_entry(away, summaries, coordinators, starting_qbs, win_totals, recent_games),
                "home_team_summary": _team_summary_entry(home, summaries, coordinators, starting_qbs, win_totals, recent_games),
                "away_injuries": _team_injuries(inj, season, week, away, snap_shares),
                "home_injuries": _team_injuries(inj, season, week, home, snap_shares),
            })
        games_by_week[str(week)] = week_games

    out = {
        "season": season,
        "weeks": weeks,
        "current_week": current_week,
        "games_by_week": games_by_week,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    n_games = sum(len(v) for v in games_by_week.values())
    n_with_lines = sum(1 for wk in games_by_week.values() for g in wk if g["lines"])
    print(f"Wrote {n_games} games ({n_with_lines} with lines) across {len(weeks)} weeks -> {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build()
