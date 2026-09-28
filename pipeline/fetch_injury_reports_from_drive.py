"""Pull the latest per-week injury-report sheet from the user's "NFL
Injury Reports 2026" Google Drive folder, save it as a raw CSV, and
rebuild data/injury_reports.parquet + docs/data/matchups.json from it.
Designed to run on a schedule via GitHub Actions
(.github/workflows/injury_reports_sync.yml), which commits only
docs/data/matchups.json back to the repo — data/ (the raw CSVs and the
parquet) is gitignored repo-wide, same as every other pipeline script's
output, so the static site actually serves the update but the
intermediate files aren't tracked.

Run locally to test:
    python3 pipeline/fetch_injury_reports_from_drive.py

Setup (one-time — see INJURY_REPORTS_SYNC_SETUP.md for the full
walkthrough): reuses the same GCP service account already set up for
pipeline/fetch_odds_snapshot.py (same GCP_SA_KEY_JSON / GCP_SA_KEY_PATH
secret — no new credential needed), but that account additionally needs:
    1. The Google Drive API enabled on the same GCP project (the odds
       script only ever needed the Sheets API)
    2. Viewer access shared on the "NFL Injury Reports 2026" folder
       itself (sharing the folder covers every sheet inside it,
       including ones the scheduled task creates after this is set up)

Why this can't just reuse a Claude conversation's Drive access
------------------------------------------------------------------
Same reasoning as fetch_odds_snapshot.py: a GitHub Actions runner has no
human logged in, so it needs its own standing credential (the service
account) rather than borrowing a conversation's OAuth session.

How "the latest sheet for a week" is determined
----------------------------------------------------
The user's scheduled task replaces the sheet for a given week with a
brand-new file (new Drive file ID, new "(updated ...)" timestamp in the
title) each time it rebuilds that week, rather than editing one file in
place — this was confirmed by watching the Week 3 file change ID between
an early estimate and the post-game final report during development.
So: list every spreadsheet in the folder, parse (season, week) out of
each title via regex, and within each (season, week) group keep only the
file with the newest Drive `modifiedTime` — never assume a previously
seen file ID is still the current one.

Change detection
------------------
Downloaded CSV content is compared byte-for-byte against whatever's
already saved at data/raw/injury_reports/{season}_week{NN}.csv, and the
parquet/JSON rebuild is skipped if nothing changed. Locally, where data/
persists between runs, this makes a repeat run with no new sheet content
a fast no-op. In GitHub Actions this comparison is moot every single
run — data/ is gitignored (see above), so each run starts from a fresh
checkout with no prior CSVs to compare against, and every week is always
"changed" on the first pass. The rebuild happens every run regardless
(cheap: a handful of small CSVs), and the workflow's own `git diff` right
before committing is what actually prevents a no-op commit when the
final matchups.json content hasn't meaningfully changed.
"""

import json
import os
import re
import sys
from pathlib import Path

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.service_account import Credentials

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "injury_reports"

# "NFL Injury Reports 2026" folder — see INJURY_REPORTS_SYNC_SETUP.md.
# Not a secret (a folder ID alone grants no access), so this is a plain
# constant rather than plumbed through another GitHub secret.
INJURY_REPORTS_FOLDER_ID = "1Rt1Dw0Fo25BgdR_iEWfLAkIV51krM2hh"

_TITLE_RE = re.compile(r"(\d{4}) Week (\d{2})")
_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _load_local_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass


def _drive_session() -> AuthorizedSession:
    sa_json = os.environ.get("GCP_SA_KEY_JSON")
    sa_path = os.environ.get("GCP_SA_KEY_PATH")
    if sa_json:
        info = json.loads(sa_json)
    elif sa_path:
        info = json.loads(Path(sa_path).read_text())
    else:
        sys.exit("Neither GCP_SA_KEY_JSON nor GCP_SA_KEY_PATH is set (see pipeline/INJURY_REPORTS_SYNC_SETUP.md)")
    creds = Credentials.from_service_account_info(info, scopes=_SCOPES)
    return AuthorizedSession(creds)


def _latest_sheet_per_week(session: AuthorizedSession) -> dict:
    resp = session.get(
        "https://www.googleapis.com/drive/v3/files",
        params={
            "q": (
                f"'{INJURY_REPORTS_FOLDER_ID}' in parents "
                "and mimeType = 'application/vnd.google-apps.spreadsheet' "
                "and trashed = false"
            ),
            "fields": "files(id, name, modifiedTime)",
            "pageSize": 100,
        },
        timeout=30,
    )
    resp.raise_for_status()

    latest = {}
    for f in resp.json().get("files", []):
        m = _TITLE_RE.search(f["name"])
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))  # (season, week)
        if key not in latest or f["modifiedTime"] > latest[key]["modifiedTime"]:
            latest[key] = f
    return latest


def _download_as_csv(session: AuthorizedSession, file_id: str) -> bytes:
    resp = session.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
        params={"mimeType": "text/csv"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


def sync() -> bool:
    """Returns True if anything changed (and the parquet/JSON were rebuilt)."""
    _load_local_env()
    session = _drive_session()

    latest = _latest_sheet_per_week(session)
    if not latest:
        sys.exit(f"No injury-report sheets found in folder {INJURY_REPORTS_FOLDER_ID} — check folder ID/sharing.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    changed_weeks = []
    for (season, week), f in sorted(latest.items()):
        raw_path = RAW_DIR / f"{season}_week{week:02d}.csv"
        content = _download_as_csv(session, f["id"])
        if raw_path.exists() and raw_path.read_bytes() == content:
            print(f"  Week {week}: unchanged ({f['name']})")
            continue
        raw_path.write_bytes(content)
        changed_weeks.append(week)
        print(f"  Week {week}: updated from '{f['name']}'")

    if not changed_weeks:
        print("No injury-report sheets changed since last sync.")
        return False

    # Run directly (python3 pipeline/fetch_injury_reports_from_drive.py,
    # as both this file's docstring and the GitHub Actions workflow do),
    # Python puts this script's own directory on sys.path, not the repo
    # root — so "pipeline" as a package isn't otherwise importable from
    # here. Add the repo root explicitly rather than requiring callers to
    # invoke this via `python3 -m pipeline.fetch_injury_reports_from_drive`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.build_injury_reports_table import build_injury_reports_table
    from pipeline.fetch_game_lines_from_sheet import build as build_game_lines
    from pipeline.generate_matchups_table import build as build_matchups_json

    build_injury_reports_table()
    # Refresh game lines here too (not just in the weekly team-stats job)
    # so matchups.json always reflects a current line whenever it's being
    # regenerated for any reason, not only on the odds workflow's own
    # schedule. Cheap — one Sheet read.
    try:
        build_game_lines()
    except SystemExit as e:
        print(f"  Game lines refresh skipped: {e}")
    build_matchups_json()
    return True


if __name__ == "__main__":
    sync()
