"""
Scrapes historical NFL preseason win totals from sportsoddshistory.com.
Outputs exports/win_totals.csv with one row per team per season.

Run locally (blocked in remote containers):
    python3 pipeline/fetch_win_totals.py

Columns:
    season       — NFL season year (e.g. 2024 = the 2024 season)
    team         — nflverse team abbreviation (e.g. "KC", "SF")
    win_total    — preseason over/under line
    over_odds    — juice on the over
    under_odds   — juice on the under
    actual_wins  — regular season wins (from the page, for reference)
    result       — "Over", "Under", or "Push"

Table columns on site: Team | Win Total | Over Odds | Under Odds | Week bet settled | Actual Wins | Result
"""

import time
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

# sportsoddshistory.com team name → nflverse abbreviation
TEAM_MAP = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Oakland Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "St. Louis Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}

SEASONS = range(2016, 2026)
BASE_URL = "https://www.sportsoddshistory.com/nfl-win/?y={year}&sa=nfl&t=win&o=t"


def _num(text):
    """Parse a numeric value from a string; returns None if not parseable."""
    text = text.strip()
    if not text or text in ("-", "N/A", ""):
        return None
    m = re.search(r"-?\d+\.?\d*", text)
    return float(m.group()) if m else None


def scrape_season(year, session):
    url = BASE_URL.format(year=year)
    resp = session.get(url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        print(f"  {year}: no tables found")
        return []

    rows_out = []
    for table in tables:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not any("team" in h for h in headers):
            continue

        # Map header name → column index
        col = {h: i for i, h in enumerate(headers)}

        # Locate columns by keyword
        def find_col(*keywords):
            for kw in keywords:
                for h, i in col.items():
                    if kw in h:
                        return i
            return None

        win_idx   = find_col("win total", "total")
        over_idx  = find_col("over odds", "over")
        under_idx = find_col("under odds", "under")
        wins_idx  = find_col("actualwins", "actual wins", "actual")
        result_idx = find_col("result")

        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 2:
                continue

            team_raw = cells[0]
            abbr = TEAM_MAP.get(team_raw) or next(
                (v for k, v in TEAM_MAP.items() if team_raw in k or k in team_raw),
                None,
            )
            if abbr is None:
                print(f"  {year}: unrecognized team '{team_raw}' — skipping")
                continue

            def cell(idx):
                return cells[idx] if idx is not None and idx < len(cells) else None

            rows_out.append({
                "season":      year,
                "team":        abbr,
                "win_total":   _num(cell(win_idx))   if win_idx   is not None else None,
                "over_odds":   _num(cell(over_idx))  if over_idx  is not None else None,
                "under_odds":  _num(cell(under_idx)) if under_idx is not None else None,
                "actual_wins": _num(cell(wins_idx))  if wins_idx  is not None else None,
                "result":      cell(result_idx),
            })
        break  # first matching table is enough

    return rows_out


def main():
    out_path = "exports/win_totals.csv"
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research scraper)"})

    all_rows = []
    for year in SEASONS:
        print(f"Fetching {year}...")
        try:
            rows = scrape_season(year, session)
            print(f"  {year}: {len(rows)} teams")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  {year}: ERROR — {e}")
        time.sleep(1.0)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["season", "team"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
