"""
Fetch weekly QBR data from ESPN for 2024 and 2025 seasons.
Run this locally and upload the output to data/qbr_2024_2025.parquet.

Usage:
    python scripts/fetch_qbr_2024_2025.py
"""

import time
import urllib.request
import urllib.error
import json
import pandas as pd

SEASONS = [2024, 2025]
WEEKS = range(1, 23)  # covers regular season + any extra weeks
SEASON_TYPE = 2       # 2 = regular season


def fetch_qbr_week(season, week):
    url = (
        f"https://site.web.api.espn.com/apis/fitt/v3/sports/football/nfl/qbr"
        f"?qbrType=weeks&seasontype={SEASON_TYPE}&isqualified=false"
        f"&season={season}&week={week}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code}: {url}")
        return []

    athletes = data.get("athletes", [])
    if not athletes:
        return []

    # Stats are in categories[0]["totals"] as an ordered list matching these names
    STAT_NAMES = ["qbr_total", "pts_added", "qb_plays", "epa_total", "pass",
                  "run", "exp_sack", "penalty", "qbr_raw", "sack"]

    rows = []
    for entry in athletes:
        a = entry.get("athlete", {})
        game = entry.get("game", {})
        opp = game.get("teamOpponent", {})
        cats = entry.get("categories", [])
        totals = cats[0]["totals"] if cats else []
        stats = dict(zip(STAT_NAMES, totals))

        rows.append({
            "season": season,
            "season_type": "Regular",
            "game_id": game.get("id"),
            "game_week": week,
            "week_text": game.get("weekText"),
            "week_num": game.get("weekNumber", week),
            "team_abb": a.get("teamShortName"),
            "player_id": a.get("id"),
            "name_short": a.get("shortName"),
            "name_first": a.get("firstName"),
            "name_last": a.get("lastName"),
            "name_display": a.get("displayName"),
            "headshot_href": (a.get("headshot") or {}).get("href"),
            "team": a.get("teamName"),
            "opp_id": opp.get("id"),
            "opp_abb": opp.get("abbreviation"),
            "opp_team": opp.get("displayName"),
            "opp_name": opp.get("name"),
            "rank": entry.get("rank"),
            "qualified": entry.get("qualified"),
            **{k: float(v) if v is not None else None for k, v in stats.items()},
        })
    return rows


def main():
    all_rows = []
    for season in SEASONS:
        for week in WEEKS:
            rows = fetch_qbr_week(season, week)
            if rows:
                print(f"  {season} week {week:2d}: {len(rows)} QBs")
                all_rows.extend(rows)
            else:
                print(f"  {season} week {week:2d}: no data (season may not have reached this week)")
            time.sleep(0.3)  # be polite

    if not all_rows:
        print("No data fetched — check your internet connection or the ESPN endpoint.")
        return

    df = pd.DataFrame(all_rows)
    out = "data/qbr_2024_2025.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")
    print(df.groupby(["season", "week_num"]).size().reset_index(name="qbs").to_string(index=False))


if __name__ == "__main__":
    main()
