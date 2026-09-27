"""Pull the current game lines (spread, total, moneyline) from the "Game
Odds" tab of the "Odds Snapshot" Google Sheet that pipeline/fetch_odds_snapshot.py
writes to, and write one row per game to data/game_lines.parquet for
docs/matchups.html to show alongside injuries and team stats.

Run locally to test (uses the same .env credentials as fetch_odds_snapshot.py):
    python3 pipeline/fetch_game_lines_from_sheet.py

Not a time series
--------------------
fetch_odds_snapshot.py's whole point is accumulating a snapshot per day
so line movement is visible in the Sheet itself. This script deliberately
throws that history away and keeps only the single latest snapshot per
(event, bookmaker, market, outcome) — the matchups page wants "what's the
line right now," not a movement chart. If a movement view is wanted
later, it should read the Sheet's full history directly rather than this
file, which is a lossy "current state" projection of it on purpose.

Reference bookmaker
-----------------------
The Odds API returns the same market from ~10 different books, which
usually disagree slightly. Rather than averaging across books (hiding
which number came from where) or showing all of them (too much for a
matchup card), one bookmaker is picked per event: DraftKings if present,
else whichever book has that event's most recent snapshot. This is a
judgment call for legibility, not a claim that DraftKings is the "best"
line — easy to change to a different book or a median-of-books approach
later.

Team-name mapping
---------------------
The odds Sheet stores full team names ("Buffalo Bills"), not the
abbreviations every other table in this pipeline uses ("BUF") — mapped
via data/team_desc.parquet's team_name -> team_abbr, the same source of
truth the rest of this pipeline would use for that mapping if it needed
it elsewhere.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "game_lines.parquet"

GAME_ODDS_SHEET = "Game Odds"
PREFERRED_BOOKMAKER = "draftkings"


def _load_local_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass


def _open_sheet():
    import gspread
    from google.oauth2.service_account import Credentials

    sa_json = os.environ.get("GCP_SA_KEY_JSON")
    sa_path = os.environ.get("GCP_SA_KEY_PATH")
    if sa_json:
        info = json.loads(sa_json)
    elif sa_path:
        info = json.loads(Path(sa_path).read_text())
    else:
        sys.exit("Neither GCP_SA_KEY_JSON nor GCP_SA_KEY_PATH is set (see pipeline/ODDS_SNAPSHOT_SETUP.md)")

    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        sys.exit("GOOGLE_SHEET_ID not set (see pipeline/ODDS_SNAPSHOT_SETUP.md)")

    creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)


def _team_name_map() -> dict:
    # team_desc.parquet has two rows for "Los Angeles Rams" — team_abbr
    # 'LA' and 'LAR' both map to the same franchise (an nflverse source
    # quirk, not a typo). Since dict(zip(...)) keeps whichever row comes
    # last, that would otherwise silently resolve to 'LAR', while every
    # other table in this pipeline normalizes the Rams to 'LA' — patched
    # explicitly here to match that convention.
    desc = pd.read_parquet(DATA_DIR / "team_desc.parquet")
    name_map = dict(zip(desc["team_name"], desc["team_abbr"]))
    name_map["Los Angeles Rams"] = "LA"
    return name_map


def _pick_bookmaker(event_df: pd.DataFrame) -> str:
    books = event_df["bookmaker"].unique()
    if PREFERRED_BOOKMAKER in books:
        return PREFERRED_BOOKMAKER
    latest = event_df.sort_values("snapshot_datetime_utc").iloc[-1]
    return latest["bookmaker"]


def build():
    _load_local_env()
    sheet = _open_sheet()
    ws = sheet.worksheet(GAME_ODDS_SHEET)
    records = ws.get_all_records()
    if not records:
        print("Game Odds tab is empty — nothing to build.")
        return

    df = pd.DataFrame.from_records(records)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["point"] = pd.to_numeric(df["point"], errors="coerce")

    name_map = _team_name_map()
    df["home_abbr"] = df["home_team"].map(name_map)
    df["away_abbr"] = df["away_team"].map(name_map)
    unmapped = df[df["home_abbr"].isna() | df["away_abbr"].isna()]
    if not unmapped.empty:
        print(f"  Warning: {unmapped['event_id'].nunique()} event(s) have an unmapped team name, skipped:")
        for name in set(unmapped["home_team"]) | set(unmapped["away_team"]):
            if name not in name_map:
                print(f"    '{name}'")
    df = df.dropna(subset=["home_abbr", "away_abbr"])

    rows = []
    for event_id, event_df in df.groupby("event_id"):
        book = _pick_bookmaker(event_df)
        book_df = event_df[event_df["bookmaker"] == book]
        # Latest snapshot per (market, outcome) within this event+bookmaker.
        book_df = book_df.sort_values("snapshot_datetime_utc").drop_duplicates(
            subset=["market", "outcome"], keep="last"
        )

        first = event_df.iloc[0]
        home, away = first["home_abbr"], first["away_abbr"]

        def _price(market, outcome):
            m = book_df[(book_df["market"] == market) & (book_df["outcome"] == first[outcome])]
            return m["price"].iloc[0] if not m.empty else None

        def _point(market, outcome):
            m = book_df[(book_df["market"] == market) & (book_df["outcome"] == first[outcome])]
            return m["point"].iloc[0] if not m.empty else None

        over = book_df[(book_df["market"] == "totals") & (book_df["outcome"] == "Over")]
        under = book_df[(book_df["market"] == "totals") & (book_df["outcome"] == "Under")]

        rows.append({
            "event_id": event_id,
            "commence_time": first["commence_time"],
            "home_team": home,
            "away_team": away,
            "bookmaker": book,
            "home_moneyline": _price("h2h", "home_team"),
            "away_moneyline": _price("h2h", "away_team"),
            "home_spread": _point("spreads", "home_team"),
            "home_spread_price": _price("spreads", "home_team"),
            "away_spread": _point("spreads", "away_team"),
            "away_spread_price": _price("spreads", "away_team"),
            "total_point": over["point"].iloc[0] if not over.empty else None,
            "over_price": over["price"].iloc[0] if not over.empty else None,
            "under_price": under["price"].iloc[0] if not under.empty else None,
            "snapshot_datetime_utc": book_df["snapshot_datetime_utc"].max(),
        })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(out)} games' current lines to {OUT_PATH}")
    return out


if __name__ == "__main__":
    build()
