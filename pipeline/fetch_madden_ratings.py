"""
Madden NFL player ratings — data/madden_ratings.parquet.

This is NOT an automated fetch script like every other pipeline/fetch_*.py
file. It can't be: ea.com/games/madden-nfl/ratings is backed by a real API
(drop-api.ea.com/rating/madden-nfl) that answers a genuine browser but
silently returns zero results to a plain scripted request with identical
URL/params/headers — confirmed directly, including from inside the same
browser session right after a working request. That's consistent with
bot-fingerprint-based blocking (Akamai or similar), not something fixable by
getting headers right. Per explicit decision, this stays a manual,
browser-driven process rather than adding real headless-browser automation
(Playwright etc.) as a new CI dependency for one data source.

How to refresh this data (manual steps)
--------------------------------------------
0. CORRECTION, confirmed directly (2026-11-19): an earlier version of this
   docstring claimed EA only ever serves the CURRENT iteration and that
   historical weeks are permanently unrecoverable once EA moves past them.
   That was wrong — it was based on testing the raw drop-api.ea.com bulk
   endpoint with an iteration param (which does come back empty), not the
   ratings PAGE itself. The page's own "Weekly Updates" filter
   (Filter & Sort -> Weekly Updates in the UI) proves every past iteration
   is still browsable, and a fresh navigation to
   `https://www.ea.com/games/madden-nfl/ratings?team={id}&iteration={iteration_id}`
   (iteration ids like "madden-ratings-week-7", or "1-base" for launch)
   correctly returns that specific week's roster in __NEXT_DATA__ — e.g.
   team=3&iteration=madden-ratings-week-7 returned 56 players all tagged
   iteration.id "madden-ratings-week-7", distinct from the 57 players
   returned for the same team with no iteration param (i.e. "current").
   Requesting an iteration EA hasn't published yet (e.g. one past the
   current week) returns 0 items rather than an error — a clean "not
   available yet" signal, not proof the mechanism doesn't work. So: past
   weeks ARE recoverable after the fact; only genuinely future-relative-
   to-EA's-own-publishing weeks aren't, and only until EA publishes them.
1. Open https://www.ea.com/games/madden-nfl/ratings?team={id} (append
   &iteration={iteration_id} to pull a specific past week instead of
   current) for each of the 32 ids in EA_TEAM_ID_TO_ABBR below, as actual
   page navigations (not fetch() calls after the fact — a scripted fetch
   from an already-loaded page gets blocked after the first one, even
   same-session; only a fresh SSR'd page load reliably works). Checking
   `d.props.pageProps.ratingsFilters.iterations` (an array of {id, label})
   on any loaded page lists every iteration EA currently has published,
   if you need to confirm what's available before picking iteration ids.
2. On each load, the page's own server-rendered data is already sitting in
   `document.getElementById('__NEXT_DATA__')` — read
   `JSON.parse(...).props.pageProps.ratingDetails.items` from it. No
   additional fetch() needed.
3. IMPORTANT: a single JS tool-call result over ~51,274 characters gets
   silently truncated (not an EA limit — a limit in the tool used to drive
   the browser), which produces invalid JSON. A full team's roster with
   full per-player stats routinely exceeds that, and JSON.stringify-ing
   the whole thing in one call hits this every time. Two ways around it,
   both confirmed working:
     (a) Split each team's `items` into chunks of ~30 players
         (`items.slice(0,30)`, `items.slice(30,60)`, `items.slice(60,90)`
         — 3 chunks covers every real roster size seen, ~46-90 players;
         a smaller team's trailing chunks just come back as empty arrays,
         which is fine) and return each chunk as a plain array (NOT
         JSON.stringify'd — let the tool's own pretty-printer render the
         object) in its own call.
     (b) For scraping all 32 teams in one sitting, (a) is still ~3 calls x
         32 teams = ~96 round trips of a few-KB chunk each, which adds up.
         A faster approach used successfully for a full-league scrape:
         per team, write that team's full mapped roster into
         `localStorage` (keyed by team, e.g. `localStorage.setItem(team,
         JSON.stringify(items.map(fmt)))`) — localStorage persists across
         same-origin navigations, so it survives moving on to the next
         team's page load — returning only a tiny `{team, count}` summary
         per call instead of the full roster. After all 32 teams, one
         final call reads every team's localStorage entry, combines them,
         and triggers a browser download of the combined JSON (a Blob +
         temporary `<a download>` click), which lands in ~/Downloads and
         can be picked up directly from there. This cuts the round-trip
         count roughly in half and keeps each individual tool-call result
         tiny regardless of roster size.
4. Per player, pull: id, firstName, lastName, position.id, overallRating,
   age, jerseyNum, yearsPro, archetype.label, iteration.id, and every
   `stats.<name>.value` (NOT `.diff` — that's redundant once you're
   accumulating your own snapshots over time; see below).
5. Parse the saved results and write data/madden_ratings.parquet. See
   pipeline/_parse_madden_scrape.py for the parsing logic used the first
   time (tied to that session's specific tool-result file paths, so not
   directly rerunnable, but the parsing approach carries over).

Why this is worth the manual effort
-----------------------------------------
Confirmed directly: the per-player JSON includes a full Madden attribute
breakdown (~54 named ratings — speed, awareness, catching, manCoverage,
etc., not just the headline overall), and EA's own `iteration` field
("madden-ratings-week-4") confirms these do get updated during the season,
which is the whole reason to track them by week rather than once.

Schema (data/madden_ratings.parquet)
------------------------------------------
One row per (season, team, ea_player_id) per scrape. Accumulates one
snapshot per scrape — like data/injury_reports.parquet and
data/depth_chart.parquet — rather than being overwritten. Each scrape is
tagged with whatever madden_iteration it actually pulled (not
necessarily "whatever's current" — see step 0 above: past iterations can
be pulled explicitly via the `iteration` URL param, so a single sitting
can backfill multiple weeks, not just add the latest one).

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

CHUNK_SIZE = 15  # players per JS call — see step 3 above for why

# Paste into the browser tool per team page load, once per 15-player slice
# (start, end): e.g. JS_CHUNK_SNIPPET.format(start=0, end=15).
JS_CHUNK_SNIPPET = """
(() => {{
  const d = JSON.parse(document.getElementById('__NEXT_DATA__').textContent);
  const items = d.props.pageProps.ratingDetails.items;
  function fmt(p) {{
    return {{
      id: p.id, first: p.firstName, last: p.lastName,
      pos: p.position ? p.position.id : null,
      overall: p.overallRating, age: p.age, jersey: p.jerseyNum,
      yearsPro: p.yearsPro,
      archetype: p.archetype ? p.archetype.label : null,
      iteration: p.iteration ? p.iteration.id : null,
      stats: Object.fromEntries(Object.entries(p.stats).map(([k, v]) => [k, v.value])),
    }};
  }}
  return items.slice({start}, {end}).map(fmt);
}})();
"""
