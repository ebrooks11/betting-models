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
1. Open https://www.ea.com/games/madden-nfl/ratings?team={id} for each of
   the 32 ids in EA_TEAM_ID_TO_ABBR below, as actual page navigations (not
   fetch() calls after the fact — a scripted fetch from an already-loaded
   page gets blocked after the first one, even same-session; only a fresh
   SSR'd page load reliably works).
2. On each load, the page's own server-rendered data is already sitting in
   `document.getElementById('__NEXT_DATA__')` — read
   `JSON.parse(...).props.pageProps.ratingDetails.items` from it. No
   additional fetch() needed.
3. IMPORTANT: a single JS tool-call result over ~51,274 characters gets
   silently truncated (not an EA limit — a limit in the tool used to drive
   the browser), which produces invalid JSON. A full team's roster with
   full per-player stats routinely exceeds that. Split each team's `items`
   into slices of 15 players (`items.slice(0,15)`, `items.slice(15,30)`,
   ... through `items.slice(75,90)` — 6 slices covers every real roster
   size seen, ~46-90 players; a smaller team's trailing slices just come
   back as empty arrays, which is fine) and read each slice as its own
   call.
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
data/depth_chart.parquet — rather than being overwritten, since EA only
ever shows "current," not a queryable history (historical `iteration`
values like "madden-ratings-week-1" were tried directly and consistently
came back empty from the bulk list endpoint — see note above about what
does and doesn't work).

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
