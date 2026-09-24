"""Fetch a same-day snapshot of NFL odds (game lines + player props) from
The Odds API and append it to a Google Sheet. Designed to run daily via
GitHub Actions (.github/workflows/odds_snapshot.yml) — each run appends
new rows rather than overwriting, so the Sheet accumulates a time series
per (event, bookmaker, market, outcome): the lookahead line the week
before, the Tuesday-morning line, and every day's movement into kickoff.

Run locally to test:
    python3 pipeline/fetch_odds_snapshot.py

Setup (one-time, done outside this repo — see ODDS_SNAPSHOT_SETUP.md in
this directory for the full walkthrough):
    1. An API key from https://the-odds-api.com/
    2. A Google Cloud service account with Sheets API access, shared as
       Editor on the target Google Sheet
    3. Three secrets, named identically for local testing (.env) and for
       the GitHub Actions workflow (repo secrets):
       - ODDS_API_KEY       — the-odds-api.com API key
       - GCP_SA_KEY_JSON     — the full service-account JSON key, as one string
       - GOOGLE_SHEET_ID     — the target spreadsheet's ID (from its URL)

Why this can't reuse this session's Google Drive access
----------------------------------------------------------
The Drive/Sheets access available inside a Claude conversation is tied to
that conversation's OAuth session — it doesn't exist for a GitHub Actions
runner. A cron job needs its own standing credential: a GCP service
account is the standard way to grant a headless process its own
Sheets-writing identity, independent of any human's login.

Endpoints used
---------------
- Game odds (h2h/spreads/totals), one call for the whole sport:
    GET /v4/sports/americanfootball_nfl/odds
  This is NOT where player props live — The Odds API only returns props
  from the *per-event* odds endpoint, one call per game:
    GET /v4/sports/americanfootball_nfl/events/{event_id}/odds?markets=...
  So fetching props costs one API call per game, not one for the whole
  week — see PROP_LOOKAHEAD_DAYS below for how this script limits that.

Quota management
------------------
The Odds API bills per (market x region) requested, whether or not a book
has actually posted that market yet — a props request for a game three
weeks out still costs credits even though nothing will come back (books
don't post player props that early). PROP_LOOKAHEAD_DAYS restricts prop
fetches to games starting soon, and PROP_MARKETS below is a curated
subset, not every market the API offers — both are here to keep a daily
cron job from burning through a monthly credit quota. Tune both to your
plan's limits (check your remaining quota via the x-requests-remaining
response header, logged below on every run).

Sheet schema (tidy/long format, not wide)
--------------------------------------------
Two worksheets, "Game Odds" and "Player Props", each one row per
(snapshot, event, bookmaker, market, outcome) — long format so a new
snapshot is just appended rows, and so Sheets' own filter/pivot tools (or
a future pipeline/build_odds_snapshots_table.py, if this ever gets pulled
back into data/*.parquet) can slice by any of those dimensions without
reshaping. Game Odds adds a `point` column (the spread/total line) that's
null for h2h; Player Props adds `player` (join key back to this
repo's other player-name columns is a separate, not-yet-solved problem —
The Odds API's names aren't guaranteed to match nflverse's gsis_id-linked
names exactly).
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"
REGIONS = "us"
ODDS_FORMAT = "american"

GAME_MARKETS = "h2h,spreads,totals"

# Curated subset of The Odds API's player-prop markets — not exhaustive,
# tune to taste/quota. Full list: https://the-odds-api.com/sports-odds-data/betting-markets.html
PROP_MARKETS = [
    "player_pass_yds", "player_pass_tds", "player_pass_interceptions",
    "player_rush_yds", "player_rush_attempts",
    "player_reception_yds", "player_receptions",
    "player_anytime_td",
]

# Only fetch props for games starting within this many days — see the
# module docstring's "Quota management" section.
PROP_LOOKAHEAD_DAYS = 10

GAME_ODDS_SHEET = "Game Odds"
PLAYER_PROPS_SHEET = "Player Props"

GAME_ODDS_HEADER = [
    "snapshot_date", "snapshot_datetime_utc", "event_id", "commence_time",
    "home_team", "away_team", "bookmaker", "market", "outcome", "price", "point",
]
PLAYER_PROPS_HEADER = [
    "snapshot_date", "snapshot_datetime_utc", "event_id", "commence_time",
    "home_team", "away_team", "bookmaker", "market", "player", "outcome", "price", "point",
]


def _load_local_env():
    """Load .env for local testing (no-op if python-dotenv isn't
    installed or there's no .env file — GitHub Actions sets real env vars
    directly and doesn't need this)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass


def _get_odds_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("ODDS_API_KEY not set (see pipeline/ODDS_SNAPSHOT_SETUP.md)")
    return key


def fetch_game_odds(api_key: str) -> list[dict]:
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
        params={"apiKey": api_key, "regions": REGIONS, "markets": GAME_MARKETS, "oddsFormat": ODDS_FORMAT},
        timeout=30,
    )
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    print(f"  Game odds: {len(resp.json())} events fetched (API credits remaining: {remaining})")
    return resp.json()


def fetch_player_props(api_key: str, events: list[dict]) -> list[dict]:
    """One API call per event within PROP_LOOKAHEAD_DAYS — see module
    docstring for why this isn't a single bulk call."""
    now = datetime.now(timezone.utc)
    results = []
    for event in events:
        commence = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        days_out = (commence - now).days
        if days_out > PROP_LOOKAHEAD_DAYS:
            continue
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events/{event['id']}/odds",
            params={"apiKey": api_key, "regions": REGIONS, "markets": ",".join(PROP_MARKETS), "oddsFormat": ODDS_FORMAT},
            timeout=30,
        )
        if resp.status_code == 422:
            # No bookmaker has posted any of these markets for this event yet.
            continue
        resp.raise_for_status()
        results.append(resp.json())
    remaining = resp.headers.get("x-requests-remaining") if events else None
    print(f"  Player props: fetched for {len(results)} of {len(events)} events (API credits remaining: {remaining})")
    return results


def _rows_from_game_odds(events: list[dict], snapshot_date: str, snapshot_dt: str) -> list[list]:
    rows = []
    for event in events:
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append([
                        snapshot_date, snapshot_dt, event["id"], event["commence_time"],
                        event["home_team"], event["away_team"], bm["key"], market["key"],
                        outcome["name"], outcome.get("price"), outcome.get("point"),
                    ])
    return rows


def _rows_from_player_props(events: list[dict], snapshot_date: str, snapshot_dt: str) -> list[list]:
    rows = []
    for event in events:
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append([
                        snapshot_date, snapshot_dt, event["id"], event["commence_time"],
                        event["home_team"], event["away_team"], bm["key"], market["key"],
                        outcome.get("description"), outcome["name"], outcome.get("price"), outcome.get("point"),
                    ])
    return rows


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

    creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)


def _append_rows(spreadsheet, tab_name: str, header: list[str], rows: list[list]):
    try:
        ws = spreadsheet.worksheet(tab_name)
    except Exception:
        ws = spreadsheet.add_worksheet(title=tab_name, rows=1, cols=len(header))
        ws.append_row(header)
    if ws.row_count == 0 or not ws.get("A1"):
        ws.append_row(header)
    if rows:
        ws.append_rows(rows, value_input_option="USER_ENTERED")
    print(f"  Appended {len(rows)} rows to '{tab_name}'")


def main():
    _load_local_env()
    api_key = _get_odds_api_key()

    now = datetime.now(timezone.utc)
    snapshot_date = now.strftime("%Y-%m-%d")
    snapshot_dt = now.isoformat()

    print(f"Fetching odds snapshot for {snapshot_date}...")
    events = fetch_game_odds(api_key)
    prop_events = fetch_player_props(api_key, events)

    game_rows = _rows_from_game_odds(events, snapshot_date, snapshot_dt)
    prop_rows = _rows_from_player_props(prop_events, snapshot_date, snapshot_dt)

    spreadsheet = _open_sheet()
    _append_rows(spreadsheet, GAME_ODDS_SHEET, GAME_ODDS_HEADER, game_rows)
    _append_rows(spreadsheet, PLAYER_PROPS_SHEET, PLAYER_PROPS_HEADER, prop_rows)

    print("Done.")


if __name__ == "__main__":
    main()
