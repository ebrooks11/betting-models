"""Pull game-odds history from the "Game Odds" tab of the "Odds Snapshot"
Google Sheet (written by pipeline/fetch_odds_snapshot.py) and accumulate it
into data/game_lines.parquet — a durable, growing historical record of
every (event, bookmaker, market, outcome) snapshot ever pulled, not just
"what's the line right now."

Run locally to test (uses the same .env credentials as fetch_odds_snapshot.py):
    python3 pipeline/fetch_game_lines_from_sheet.py

This accumulates, it doesn't overwrite
------------------------------------------
data/game_lines.parquet is a tracked exception to this pipeline's usual
"data/ is gitignored, always regenerable" rule (see .gitignore) —
specifically so line-movement history (the whole point of
fetch_odds_snapshot.py accumulating daily snapshots in the Sheet in the
first place) actually lives in this repo's own data layer, not only in
an external Google Sheet nothing else here can see. Every run appends
whatever (event_id, bookmaker, market, outcome, snapshot_datetime_utc)
combinations aren't already present — pulling the Sheet's full contents
every time is harmless since it's deduplicated against what's already
stored, not re-appended.

Picking "the current line" is a separate, later step
----------------------------------------------------------
This file intentionally keeps every snapshot at the Sheet's own
granularity (every bookmaker, not reduced to one) — current_lines() below
reduces that down to one row per game (preferred bookmaker, latest
snapshot per market/outcome) for pipeline/generate_matchups_table.py to
show "the line right now." Anything that wants line movement over time
instead should read data/game_lines.parquet directly rather than going
through current_lines().

Reference bookmaker (used by current_lines(), not by the accumulation above)
---------------------------------------------------------------------------------
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

_DEDUP_KEY = ["event_id", "bookmaker", "market", "outcome", "snapshot_datetime_utc"]


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


def _fetch_sheet_df() -> pd.DataFrame:
    sheet = _open_sheet()
    ws = sheet.worksheet(GAME_ODDS_SHEET)
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["point"] = pd.to_numeric(df["point"], errors="coerce")

    name_map = _team_name_map()
    df["home_team"] = df["home_team"].map(name_map)
    df["away_team"] = df["away_team"].map(name_map)
    unmapped = df[df["home_team"].isna() | df["away_team"].isna()]
    if not unmapped.empty:
        print(f"  Warning: {unmapped['event_id'].nunique()} event(s) have an unmapped team name, skipped")
    df = df.dropna(subset=["home_team", "away_team"])

    # outcome is "Over"/"Under" for totals, but a full team name for h2h/
    # spreads (e.g. "Kansas City Chiefs", not "KC") — map those too so
    # outcome is consistently in the same abbreviation space as
    # home_team/away_team; anything not a recognized team name (Over/
    # Under) passes through unchanged.
    df["outcome"] = df["outcome"].map(lambda v: name_map.get(v, v))

    return df[[
        "event_id", "snapshot_date", "snapshot_datetime_utc", "commence_time",
        "home_team", "away_team", "bookmaker", "market", "outcome", "price", "point",
    ]]


def build() -> pd.DataFrame:
    """Accumulate: append any (event, bookmaker, market, outcome, snapshot)
    combination not already stored. Returns the full accumulated table."""
    _load_local_env()
    fresh = _fetch_sheet_df()
    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()

    if fresh.empty:
        print("Game Odds tab is empty — nothing new to accumulate.")
        return existing

    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset=_DEDUP_KEY, keep="last")
    combined = combined.sort_values(["event_id", "bookmaker", "market", "outcome", "snapshot_datetime_utc"]).reset_index(drop=True)

    new_count = len(combined) - len(existing)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"Accumulated {new_count:,} new snapshot rows ({len(combined):,} total) in {OUT_PATH}")
    return combined


def _pick_bookmaker(event_df: pd.DataFrame) -> str:
    books = event_df["bookmaker"].unique()
    if PREFERRED_BOOKMAKER in books:
        return PREFERRED_BOOKMAKER
    latest = event_df.sort_values("snapshot_datetime_utc").iloc[-1]
    return latest["bookmaker"]


def current_lines(df: pd.DataFrame = None) -> pd.DataFrame:
    """Reduce the full accumulated history to one row per game: preferred
    bookmaker, latest snapshot per market/outcome. See this module's
    docstring for why this reduction lives separately from accumulation."""
    if df is None:
        df = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    if df.empty:
        return df

    rows = []
    for event_id, event_df in df.groupby("event_id"):
        book = _pick_bookmaker(event_df)
        book_df = event_df[event_df["bookmaker"] == book]
        book_df = book_df.sort_values("snapshot_datetime_utc").drop_duplicates(subset=["market", "outcome"], keep="last")

        first = event_df.iloc[0]
        home, away = first["home_team"], first["away_team"]

        def _price(market, team_col):
            m = book_df[(book_df["market"] == market) & (book_df["outcome"] == first[team_col])]
            return m["price"].iloc[0] if not m.empty else None

        def _point(market, team_col):
            m = book_df[(book_df["market"] == market) & (book_df["outcome"] == first[team_col])]
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

    return pd.DataFrame(rows)


if __name__ == "__main__":
    build()
