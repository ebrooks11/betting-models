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
        f"?season={season}&seasontype={SEASON_TYPE}&week={week}&limit=100"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code}: {url}")
        return []

    # Debug: print top-level keys and first 500 chars on first call
    if season == SEASONS[0] and week == 1:
        print(f"    DEBUG keys: {list(data.keys())}")
        print(f"    DEBUG raw[:500]: {raw[:500]}")

    athletes = data.get("athletes", [])
    if not athletes:
        return []

    rows = []
    for entry in athletes:
        a = entry.get("athlete", {})
        team = entry.get("team", {})
        cats = entry.get("categories", [])

        # ESPN returns categories in a fixed order — grab by label to be safe
        stat_map = {}
        for cat in cats:
            label = cat.get("name", "")
            totals = cat.get("totals", [None])
            stat_map[label] = totals[0] if totals else None

        rows.append({
            "season": season,
            "season_type": "Regular",
            "game_week": f"Week {week}",
            "week_num": week,
            "team_abb": team.get("abbreviation"),
            "player_id": a.get("id"),
            "name_short": a.get("shortName"),
            "name_first": a.get("firstName"),
            "name_last": a.get("lastName"),
            "name_display": a.get("displayName"),
            "qbr_total": stat_map.get("qbr"),
            "pts_added": stat_map.get("ptsAdded"),
            "qb_plays": stat_map.get("plays"),
            "epa_total": stat_map.get("epa"),
            "pass": stat_map.get("pass"),
            "run": stat_map.get("run"),
            "exp_sack": stat_map.get("expSack"),
            "penalty": stat_map.get("penalty"),
            "qbr_raw": stat_map.get("qbrRaw"),
            "sack": stat_map.get("sack"),
            "qualified": entry.get("qualified"),
            "rank": entry.get("rank"),
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
