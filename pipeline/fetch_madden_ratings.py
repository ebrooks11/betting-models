"""
Madden NFL player ratings — data/madden_ratings.parquet.

AUTOMATED as of 2026-11-19, via .github/workflows/madden_ratings_refresh.yml
(weekly). This supersedes an earlier version of this file that was pure
manual-process documentation with no executable fetch code at all, on the
belief that ea.com/games/madden-nfl/ratings couldn't be scraped by a plain
script. That belief was based on testing the wrong thing:

  - drop-api.ea.com/rating/madden-nfl (the JSON API the ratings PAGE itself
    calls client-side) DOES silently return zero results to a plain scripted
    request with identical URL/params/headers, including from inside the
    same browser session right after a working request — confirmed
    directly, consistent with bot-fingerprint-based blocking (Akamai or
    similar).
  - But the ratings PAGE at https://www.ea.com/games/madden-nfl/ratings
    itself is ordinary server-rendered HTML with no such protection —
    confirmed directly with a plain `requests.get()` (no browser, no JS, no
    cookies, a generic desktop-Chrome User-Agent string): a 200 response
    containing the full real `__NEXT_DATA__` payload, same data a browser
    session sees. The fix was simply: scrape the page's own SSR'd HTML, not
    the API it happens to also call.

How the automated fetch works (scrape_all/build below)
-----------------------------------------------------------
For each of the 32 ids in EA_TEAM_ID_TO_ABBR, GET
`https://www.ea.com/games/madden-nfl/ratings?team={id}` (no `iteration`
param — omitting it means "whatever's current," which is what a weekly
cron wants), extract the `<script id="__NEXT_DATA__">` blob via regex, and
read `.props.pageProps.ratingDetails.items` from the parsed JSON — the
same per-player shape building this project's rows below as the one-time
manual scrapes already used (id, firstName, lastName, position.id,
overallRating, age, jerseyNum, yearsPro, archetype.label, iteration.id,
and every `stats.<name>.value`).

Every team's players should agree on the SAME `iteration.id` for a single
run (confirmed in practice) — scrape_all() raises if they don't, since
that would mean a partial/inconsistent scrape (e.g. EA rolled over to a
new week mid-run) rather than something to silently paper over.
build() is a no-op (prints and returns, doesn't touch the parquet) if the
scraped iteration is already present in data/madden_ratings.parquet —
this file accumulates one snapshot per iteration, never overwrites, and
re-running against an iteration already captured should never duplicate
rows (this also makes a manually re-triggered or slightly-early workflow
run harmless).

Historical backfill (manual, browser-driven) — still sometimes needed
-----------------------------------------------------------------------------
The automated path above only ever fetches CURRENT. If the weekly workflow
ever misses a week outright (an outage, a failed run nobody noticed), the
PAST iteration is still recoverable afterward — EA does NOT delete old
iterations, confirmed directly: the ratings page's own "Weekly Updates"
filter (Filter & Sort -> Weekly Updates in the UI) lists every past
iteration, and a fresh request to
`https://www.ea.com/games/madden-nfl/ratings?team={id}&iteration={iteration_id}`
(iteration ids like "madden-ratings-week-7", or "1-base" for launch)
correctly returns that specific week's roster — e.g. team=3 with
`&iteration=madden-ratings-week-7` returned 56 players all tagged
iteration.id "madden-ratings-week-7", distinct from the 57 players
returned for the same team with no iteration param at all (current).
Requesting an iteration EA hasn't published yet returns 0 items rather
than an error — a clean "not available yet" signal.
`d.props.pageProps.ratingsFilters.iterations` (array of {id, label}) on
any loaded page lists every iteration EA currently has published, if you
need to confirm what's actually available before picking iteration ids.

Backfilling is still most practically done via a real browser (Claude in
Chrome or similar), not this file's plain-requests path — plain requests
against `&iteration=...` URLs have NOT been separately verified the way
the no-iteration "current" path has been (the automated path above is
only proven for current; if it turns out EA treats the iteration-
parameterized URL differently for a bare scripted request, that's still
untested). Useful browser-driven technique for scraping many teams in one
sitting without hitting the ~51,274-character single-tool-call truncation
limit a browser-automation tool imposes (not an EA limit): per team, write
that team's full mapped roster into `localStorage` (keyed by team, or by
`${team}__${iteration}` when scraping multiple iterations in one pass —
localStorage persists across same-origin navigations), returning only a
tiny `{team, count}` summary per call instead of the full roster; after
every team, one final call reads every localStorage entry, combines them,
and triggers a browser download of the combined JSON for a script to pick
up from disk. See git history (pipeline/_parse_madden_scrape.py and
pipeline/_parse_madden_week5_6.py) for one-off parsers built this way —
each tied to that session's specific captured files, not directly
rerunnable, but the technique carries over.

Why this is worth tracking at all
-----------------------------------------
Confirmed directly: the per-player JSON includes a full Madden attribute
breakdown (~54 named ratings — speed, awareness, catching, manCoverage,
etc., not just the headline overall), and EA's own `iteration` field
("madden-ratings-week-4") confirms these do get updated during the
season, which is the whole reason to track them by week rather than once.

Schema (data/madden_ratings.parquet)
------------------------------------------
One row per (season, team, ea_player_id) per scrape. Accumulates one
snapshot per scrape — like data/injury_reports.parquet and
data/depth_chart.parquet — rather than being overwritten.

  season            — this project's season year (not Madden's own "Week N"
                       label, which is tracked separately in madden_iteration
                       since it isn't guaranteed to line up with the real
                       NFL week the data was pulled)
  madden_iteration  — EA's own version label for this snapshot, e.g.
                       "madden-ratings-week-4", straight from the API
  scraped_date      — the date this snapshot was actually pulled
  team              — this project's nflverse team abbreviation (mapped
                       from EA's own numeric team id via EA_TEAM_ID_TO_ABBR)
  ea_player_id      — EA's own numeric player id (not an nflverse/gsis id —
                       no crosswalk exists between the two; match players
                       across tables by (team, name) if ever needed, same
                       normalization caveats as everywhere else in this
                       project that does that)
  first_name, last_name, position, overall, age, jersey_num, years_pro,
  archetype
  stat_<name>       — one column per Madden attribute (stat_speed,
                       stat_awareness, stat_catching, stat_manCoverage, ...),
                       flattened from the API's stats.<name>.value
"""

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "madden_ratings.parquet"

# EA's own numeric team id -> this project's nflverse team abbreviation —
# read off the ratings page's own __NEXT_DATA__ blob
# (props.pageProps.ratingsFilters.teamGroups). Confirmed stable across the
# whole scrape (every team found, no id collisions).
EA_TEAM_ID_TO_ABBR = {
    3: "BUF", 12: "MIA", 22: "NE", 18: "NYJ",
    2: "CIN", 5: "CLE", 25: "BAL", 29: "PIT",
    10: "IND", 17: "JAX", 30: "TEN", 32: "HOU",
    4: "DEN", 8: "LAC", 9: "KC", 23: "LV",
    11: "DAL", 13: "PHI", 16: "NYG", 26: "WAS",
    1: "CHI", 19: "DET", 20: "GB", 31: "MIN",
    6: "TB", 14: "ATL", 21: "CAR", 27: "NO",
    7: "ARI", 15: "SF", 24: "LA", 28: "SEA",
}

_RATINGS_URL = "https://www.ea.com/games/madden-nfl/ratings"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def _fetch_team_items(team_id: int, session: requests.Session) -> list:
    """One team's current roster (list of EA's own per-player dicts), or
    [] if the page loaded but the expected JSON wasn't found — callers
    should treat that as a skip, not a fatal error, same convention as
    fetch_depth_chart.py's scrape_team_depth_chart."""
    resp = session.get(_RATINGS_URL, params={"team": team_id}, timeout=20)
    resp.raise_for_status()
    match = _NEXT_DATA_RE.search(resp.text)
    if not match:
        return []
    data = json.loads(match.group(1))
    return data.get("props", {}).get("pageProps", {}).get("ratingDetails", {}).get("items", [])


def _flatten_player(p: dict, team_abbr: str, season: int, scraped_date: str) -> dict:
    row = {
        "season": season,
        "madden_iteration": (p.get("iteration") or {}).get("id"),
        "scraped_date": scraped_date,
        "team": team_abbr,
        "ea_player_id": p.get("id"),
        "first_name": p.get("firstName"),
        "last_name": p.get("lastName"),
        "position": (p.get("position") or {}).get("id"),
        "overall": p.get("overallRating"),
        "age": p.get("age"),
        "jersey_num": p.get("jerseyNum"),
        "years_pro": p.get("yearsPro"),
        "archetype": (p.get("archetype") or {}).get("label"),
    }
    for stat_name, stat_val in (p.get("stats") or {}).items():
        row[f"stat_{stat_name}"] = stat_val.get("value") if isinstance(stat_val, dict) else stat_val
    return row


def scrape_all(season: int) -> pd.DataFrame:
    """Fetch the CURRENT iteration for all 32 teams. Raises RuntimeError if
    teams disagree on which iteration is current (a sign of a partial/
    inconsistent scrape, e.g. EA rolled over to a new week mid-run) rather
    than silently mixing two iterations into one supposed snapshot."""
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    scraped_date = date.today().isoformat()
    rows = []
    failed_teams = []
    for team_id, abbr in EA_TEAM_ID_TO_ABBR.items():
        try:
            items = _fetch_team_items(team_id, session)
        except Exception as e:
            print(f"  {abbr}: ERROR - {e}")
            failed_teams.append(abbr)
            continue
        if not items:
            print(f"  {abbr}: no items returned")
            failed_teams.append(abbr)
            continue
        rows.extend(_flatten_player(p, abbr, season, scraped_date) for p in items)

    if failed_teams:
        print(f"  WARNING: {len(failed_teams)}/{len(EA_TEAM_ID_TO_ABBR)} team(s) failed: {failed_teams}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    iterations = df["madden_iteration"].unique()
    if len(iterations) > 1:
        raise RuntimeError(f"Teams disagree on current Madden iteration: {sorted(iterations)} — aborting without writing")
    return df


def build():
    """Fetch current ratings and append as a new snapshot to
    data/madden_ratings.parquet — a no-op (existing file untouched) if
    this iteration is already present, or if nothing was scraped at all."""
    season = date.today().year if date.today().month >= 3 else date.today().year - 1
    print(f"Fetching current Madden ratings ({season})...")
    df = scrape_all(season)
    if df.empty:
        print("No rows scraped — leaving data/madden_ratings.parquet as-is")
        return

    iteration = df["madden_iteration"].iloc[0]
    print(f"  {len(df)} rows across {df['team'].nunique()} teams, iteration={iteration}")

    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    if not existing.empty and iteration in existing["madden_iteration"].unique():
        print(f"  {iteration} is already in data/madden_ratings.parquet ({(existing.madden_iteration == iteration).sum()} rows) — nothing new to add")
        return

    combined = pd.concat([existing, df], ignore_index=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(combined)} total rows ({len(df)} new, iteration={iteration}) to {OUT_PATH}")


if __name__ == "__main__":
    build()
