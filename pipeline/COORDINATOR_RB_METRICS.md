# Coordinator table: RB-focused metrics

Reference for the RB/run-game columns being added to `data/offensive_coordinators.parquet`
via `pipeline/build_offensive_coordinators_table.py`. One row per (team, season) — these
are team/OC-level stats, not tied to a single player, except where noted (Primary/Secondary RB).

Investigated against the actual data before writing this — two of the requested metrics
have real gaps, noted below rather than silently approximated.

## Data sources and their floors

| Source | What it adds here | Season floor |
|---|---|---|
| `pbp.parquet` | run rate, rush EPA, explosive rate, red-zone rush rates, formation rates/EPA/success | 2006+ (formation-based metrics: **2016+ only** — `offense_formation` is NULL for every play before 2016, confirmed by direct query) |
| `ftn.parquet` | screen rate | **2022+ only** (FTN charting doesn't exist before 2022) |
| `seasonal_pfr.parquet` | yards before contact per rush | **2018+ only** (existing PFR floor, same as the RB table) |
| `seasonal_rosters.parquet` | RB position lookup, used to filter pbp rush/target plays to RB-position players | 2006+ |

Rows outside a metric's floor get `NULL` for that column, same convention as every other
partial-coverage column already in this table (e.g. `qb_cpoe` before 2016).

## Metrics

### Run Rate
`team rush attempts / (team rush attempts + team pass attempts)`, all rushers/passers
(not RB-specific — this is a play-calling tendency stat). Excludes plays negated by
penalty in the same way the position tables already do (`sack=0`, `two_point_attempt=0`
excluded from both counts, consistent with existing convention).

### Screen Rate
`screen passes / total pass attempts`, from FTN's `is_screen_pass` flag, joined to pbp on
`(game_id, play_id)` = `(nflverse_game_id, nflverse_play_id)` — verified this join works
correctly against real screen plays. **2022+ only.**

### Route Participation by RBs — not derivable as asked
No source we have charts "ran a route" vs. "stayed in to block" (FTN's charting doesn't
include this either — checked its full column list). This is genuinely a gap; the closest
computable substitute is **RB Target Share** (`RB targets / team pass attempts`) — measures
how often a RB is the actual target, not how often one releases into a route without being
thrown to. Building this as `rb_target_share` under a name that doesn't overclaim what it
measures, rather than mislabeling it "route participation."

### Yards Before Contact per Rush
`SUM(pfr rushing yards before contact) / SUM(pfr rush attempts)`, summed across every
player tagged `pos='RB'` on that team-season in `seasonal_pfr` (not just the primary
back). **2018+ only.**

### Rush EPA
`AVG(epa)` on pbp rows where `rush_attempt=1`, `two_point_attempt=0`, and the rusher is
RB-position that season (via `seasonal_rosters`) — i.e. QB scrambles/sneaks and
trick-play rushes by other positions are excluded. All seasons.

### Primary RB / Secondary RB / Primary RB Rush % / Secondary RB Rush %
- Primary RB: already exists (`rb_name`, by carries). Secondary RB: the #2 RB by carries
  that team-season (new — `row_number() ... = 2` off the same window used to pick primary).
- Rush % for each = that back's carries / **team total rush attempts (all positions)** —
  chose this denominator (rather than RB-only carries) since "share of the run game" reads
  more naturally against the whole backfield+trick-play rushing pool. Flagging this as a
  judgment call in case a different denominator is wanted.

### Formation rates / EPA / success rate
Per-formation breakdown using pbp's `offense_formation` (2016+ only). Observed formation
values in our data: `SHOTGUN`, `SINGLEBACK`, `UNDER CENTER`, `I_FORM`, `EMPTY`, `PISTOL`,
`JUMBO`, `WILDCAT` (8 formations, confirmed by direct query — hardcoding these rather than
a dynamic pivot since the set is small and has been stable across 2016-2025).
- Formation rate: `plays in formation X / all offensive plays`
- Formation EPA: `AVG(epa)` on plays in formation X
- Formation success rate: `AVG(success)` on plays in formation X — using pbp's own
  `success` column directly (nflverse's standard down/distance-based success definition)
  rather than recomputing it.

This is 8 formations × 3 stats = 24 columns, on top of everything else — a real jump in
width. Flagging in case a smaller subset (e.g. just SHOTGUN/SINGLEBACK/I_FORM/EMPTY, the
four with real sample size) is preferred over all 8.

### Explosive run rate
`rush attempts (by RB) gaining >= 10 yards / total RB rush attempts`. 10 yards is a
common convention for a "explosive run" (as opposed to 15+ used by some analysts) —
flagging the threshold choice explicitly since it's a judgment call, not a standard everyone
agrees on.

### Rush rate inside the 20 / inside the 10 / inside the 5
`team rush attempts / (team rush attempts + team pass attempts)` restricted to plays where
`yardline_100 <= 20` / `<= 10` / `<= 5` respectively (distance to the opponent's end zone —
fully populated in pbp, no gaps). Team-level play-calling tendency, not RB-filtered, same
reasoning as Run Rate above.
