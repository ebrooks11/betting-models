# Feature Log

Each feature is tagged with its **perspective**:
- `posteam` — computed from the offensive team's view (primary team on offense)
- `defteam` — computed from the defensive team's view (primary team on defense)
- `game` — game-level context, not tied to a specific side

**Naming convention:**
- Features computed from `defteam` perspective are prefixed with `def_`
- Features computed from `posteam` perspective have no prefix (or use `off_` for EPA splits)
- When used as an opponent feature in a model, the dataset builder adds the `opp_` prefix

---

## EPA & Efficiency

| Feature | Perspective | Description |
|---------|-------------|-------------|
| `off_epa_per_play` | posteam | Raw offensive EPA per play (game-level) |
| `def_epa_per_play` | defteam | Raw defensive EPA per play allowed (game-level) |
| `off_epa_rolling` | posteam | 3-game rolling avg offensive EPA per play |
| `def_epa_rolling` | defteam | 3-game rolling avg defensive EPA per play allowed |
| `off_epa_no_to_rolling` | posteam | Offensive EPA per play excluding turnover plays |
| `def_epa_no_to_rolling` | defteam | Defensive EPA per play allowed excluding turnover plays |
| `off_epa_early_down_rolling` | posteam | Offensive EPA on 1st & 2nd down |
| `def_epa_early_down_rolling` | defteam | Defensive EPA allowed on 1st & 2nd down |
| `off_epa_first_down_rolling` | posteam | Offensive EPA on 1st down only |
| `def_epa_first_down_rolling` | defteam | Defensive EPA allowed on 1st down only |
| `off_epa_iter_adj` | posteam | Season-to-date offensive EPA adjusted iteratively for opponent defensive quality (5 iterations) |
| `def_epa_iter_adj` | defteam | Season-to-date defensive EPA adjusted iteratively for opponent offensive quality (5 iterations) |
| `rush_ypc_iter_adj` | posteam | Season-to-date rushing yards per carry adjusted iteratively for opponent defensive quality (5 iterations) |
| `def_rush_ypc_iter_adj` | defteam | Season-to-date rushing yards per carry allowed adjusted iteratively for opponent offensive quality (5 iterations) |
| `rush_epa_iter_adj` | posteam | Season-to-date rush EPA per play adjusted iteratively for opponent defensive quality (5 iterations) |
| `def_rush_epa_iter_adj` | defteam | Season-to-date rush EPA allowed per play adjusted iteratively for opponent offensive quality (5 iterations) |
| `off_11_epa_iter_adj` | posteam | Season-to-date 11-personnel EPA adjusted iteratively for opponent 11-personnel defense quality (5 iterations) |
| `def_vs_11_epa_iter_adj` | defteam | Season-to-date 11-personnel EPA allowed adjusted iteratively for opponent 11-personnel offense quality (5 iterations) |
| `off_12_epa_iter_adj` | posteam | Season-to-date 12-personnel EPA adjusted iteratively for opponent 12-personnel defense quality (5 iterations) |
| `def_vs_12_epa_iter_adj` | defteam | Season-to-date 12-personnel EPA allowed adjusted iteratively for opponent 12-personnel offense quality (5 iterations) |
| `points_scored_iter_adj` | posteam | Season-to-date points scored adjusted iteratively for opponent defensive quality (5 iterations) |
| `points_allowed_iter_adj` | defteam | Season-to-date points allowed adjusted iteratively for opponent offensive quality (5 iterations) |

## Passing

| Feature | Perspective | Description |
|---------|-------------|-------------|
| `cpoe_rolling` | posteam | Completion % over expected — QB accuracy relative to throw difficulty |
| `ypa_rolling` | posteam | Yards per pass attempt |
| `adot_rolling` | posteam | Average depth of target (how far downfield throws go) |
| `pass_attempts_pg_rolling` | posteam | Pass attempts per game |
| `completions_pg_rolling` | posteam | Completions per game |
| `qb_hit_rate_rolling` | posteam | How often the primary team's QB gets hit per pass play (OL quality) |
| `sack_rate_rolling` | posteam | How often the primary team's QB gets sacked per pass play (OL quality) |
| `def_qb_hit_rate_rolling` | defteam | How often this team's defense generates a QB hit per pass play (pass rush quality) |
| `def_sack_rate_rolling` | defteam | How often this team's defense generates a sack per pass play (pass rush quality) |

## Personnel & Formation

| Feature | Perspective | Description |
|---------|-------------|-------------|
| `off_11_epa_rolling` | posteam | Offensive EPA per play in 11 personnel (1 RB, 1 TE, 3 WR) |
| `off_12_epa_rolling` | posteam | Offensive EPA per play in 12 personnel (1 RB, 2 TE, 2 WR) |
| `off_11_epa_early_down_rolling` | posteam | 11 personnel EPA on 1st & 2nd down |
| `off_12_epa_early_down_rolling` | posteam | 12 personnel EPA on 1st & 2nd down |
| `off_11_rate_rolling` | posteam | % of plays run in 11 personnel |
| `off_12_rate_rolling` | posteam | % of plays run in 12 personnel |
| `off_21_rate_rolling` | posteam | % of plays run in 21 personnel |
| `def_vs_11_epa_rolling` | defteam | Defensive EPA allowed vs 11 personnel |
| `def_vs_12_epa_rolling` | defteam | Defensive EPA allowed vs 12 personnel |
| `def_vs_21_epa_rolling` | defteam | Defensive EPA allowed vs 21 personnel |
| `def_nickel_rate_rolling` | defteam | % of plays the defense runs nickel coverage |
| `def_base_rate_rolling` | defteam | % of plays the defense runs base coverage |
| `def_dime_rate_rolling` | defteam | % of plays the defense runs dime coverage |
| `off_vs_nickel_epa_rolling` | posteam | Offensive EPA vs nickel defense |
| `off_vs_base_epa_rolling` | posteam | Offensive EPA vs base defense |

## Rushing

| Feature | Perspective | Description |
|---------|-------------|-------------|
| `rush_yards_pg_rolling` | posteam | Rushing yards per game (non-QB runs) |
| `rush_ypc_rolling` | posteam | Rushing yards per carry (non-QB runs) |
| `rush_epa_rolling` | posteam | EPA per designed run play |
| `rush_first_down_rate_rolling` | posteam | % of designed runs that gain a first down |
| `rush_explosive_rate_rolling` | posteam | % of designed runs gaining 10+ yards |
| `qb_rush_yards_pg_rolling` | posteam | QB scramble yards per game |
| `stuff_rate_rolling` | posteam | % of the primary team's runs stopped at or behind the line of scrimmage |
| `def_rush_yards_pg_rolling` | defteam | Rushing yards allowed per game (designed runs only) |
| `def_rush_ypc_rolling` | defteam | Rushing yards allowed per carry (designed runs only) |
| `def_rush_epa_rolling` | defteam | EPA per designed run allowed |
| `def_rush_first_down_rate_rolling` | defteam | % of designed runs against this defense that gain a first down |
| `def_rush_explosive_rate_rolling` | defteam | % of designed runs against this defense gaining 10+ yards |
| `def_tfl_rate_rolling` | defteam | How often this team's defense generates a tackle for loss per rush attempt faced |

## Scoring & Turnover

| Feature | Perspective | Description |
|---------|-------------|-------------|
| `points_scored_rolling` | posteam | Avg points scored over last 3 games |
| `points_allowed_rolling` | defteam | Avg points allowed over last 3 games |
| `turnovers_committed_rolling` | posteam | Avg turnovers committed per game (INTs + fumbles lost) |
| `turnovers_forced_rolling` | defteam | Avg turnovers forced per game |

## Pace & Situational

| Feature | Perspective | Description |
|---------|-------------|-------------|
| `plays_per_game_rolling` | posteam | Offensive plays per game |
| `top_rolling` | posteam | Time of possession per game (seconds) |
| `first_downs_pg_rolling` | posteam | First downs per game |
| `first_down_rate_rolling` | posteam | % of plays that result in a first down |
| `third_down_rate_rolling` | posteam | % of plays that reach 3rd down (lower = better) |
| `fourth_down_attempt_rate_rolling` | posteam | % of 4th downs attempted (excluding punts/FGs) |
| `explosive_plays_pg_rolling` | posteam | Explosive plays (20+ yards) per game |

## QB

| Feature | Perspective | Description |
|---------|-------------|-------------|
| `qbr_rolling` | posteam | ESPN Total QBR, 3-game rolling avg |

## Game Context

| Feature | Perspective | Description |
|---------|-------------|-------------|
| `is_home` | game | 1 if primary team is home, 0 if away |
| `rest_days` | game | Days since last game for primary team |
| `rest_advantage` | game | Primary team rest days minus opponent rest days |
