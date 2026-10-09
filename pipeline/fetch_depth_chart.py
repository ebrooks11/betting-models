"""
Scrapes each NFL team's current depth chart from ESPN
(espn.com/nfl/team/depth/_/name/{abbr}) — who's starting at each position
slot, and who's behind them. This is a DIFFERENT, independent source from
this project's usual "who started" signal (data/snap_counts.parquet's
per-game snap shares, used by generate_matchups_table.py's starter-emphasis
feature) — deliberately so: ESPN's depth chart reflects the team's own
stated Week-N plan (including a player who's never taken a snap yet, e.g.
heading into the season or coming off injury), where snap counts can only
ever describe a game that's already been played.

ESPN's team page embeds its data as a JSON blob assigned to
window['__espnfitt__'] inside a <script> tag — scraped from that directly
rather than the rendered HTML, same reasoning as fetch_coordinators.py
preferring a template's data-mw JSON over rendered markup: stable across
ESPN's own page-rendering changes. page.content.depth.dethTeamGroups (sic
— that's ESPN's own key, not a typo introduced here) is a fixed-length,
fixed-order list of exactly 3 groups for every team, confirmed directly
across all 32: [0] offense, [1] defense, [2] special teams. Each group has
rows: [position_label, player_1, player_2, ...] with players already in
depth order — player_1 (depth_rank 1) is the starter.

Run locally (blocked in remote containers, like fetch_win_totals.py):
    python3 pipeline/fetch_depth_chart.py
"""

import json
import re

import requests

# nflverse team abbreviation -> ESPN's own abbreviation, only where they
# differ (everything else is just .lower()).
ESPN_ABBR_OVERRIDES = {
    "LA": "lar",
    "WAS": "wsh",
}

ALL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN",
    "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

SIDES = ["OFF", "DEF", "ST"]  # dethTeamGroups' fixed index order

_JSON_BLOB_RE = re.compile(r"window\['__espnfitt__'\]\s*=\s*(\{.*?\});\s*</script>", re.S)


def _espn_abbr(abbr: str) -> str:
    return ESPN_ABBR_OVERRIDES.get(abbr, abbr.lower())


def scrape_team_depth_chart(abbr: str, season: int, session: requests.Session) -> list[dict]:
    """Fetch one team's current ESPN depth chart. Returns an empty list
    (with a printed reason) if the page or the expected JSON isn't found —
    callers should treat that as a skip, not a fatal error.
    """
    url = f"https://www.espn.com/nfl/team/depth/_/name/{_espn_abbr(abbr)}"
    resp = session.get(url, timeout=15)
    if resp.status_code == 404:
        print(f"  {abbr}: page not found ({url})")
        return []
    resp.raise_for_status()

    match = _JSON_BLOB_RE.search(resp.text)
    if not match:
        print(f"  {abbr}: __espnfitt__ JSON blob not found")
        return []
    data = json.loads(match.group(1))

    try:
        groups = data["page"]["content"]["depth"]["dethTeamGroups"]
    except KeyError:
        print(f"  {abbr}: no depth chart groups in page JSON")
        return []
    if len(groups) != len(SIDES):
        print(f"  {abbr}: expected {len(SIDES)} depth chart groups, got {len(groups)} — skipping")
        return []

    rows = []
    for side, group in zip(SIDES, groups):
        formation = group.get("name")
        for slot_index, row in enumerate(group["rows"]):
            position = row[0]
            for depth_rank, player in enumerate(row[1:], start=1):
                if not player:
                    continue
                rows.append({
                    "season": season,
                    "team": abbr,
                    "side": side,
                    "formation": formation,
                    "position": position,
                    "slot_index": slot_index,
                    "depth_rank": depth_rank,
                    "player": player.get("displayName") or player.get("name"),
                    "espn_player_id": player.get("uid"),
                    "injury_status": "/".join(player.get("injuries") or []) or None,
                })
    return rows


def scrape_all(season: int) -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research scraper)"})
    rows = []
    for team in ALL_TEAMS:
        try:
            rows.extend(scrape_team_depth_chart(team, season, session))
        except Exception as e:
            print(f"  {season} {team}: ERROR - {e}")
    return rows


def main():
    import pandas as pd
    from datetime import date

    season = date.today().year if date.today().month >= 3 else date.today().year - 1
    rows = scrape_all(season)
    df = pd.DataFrame(rows)
    print(f"{len(df)} rows across {df['team'].nunique()} teams")
    starters = df[df.depth_rank == 1]
    print(starters[starters.side == "OFF"].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
