# Injury reports sync setup

One-time setup for `pipeline/fetch_injury_reports_from_drive.py` and
`.github/workflows/injury_reports_sync.yml`. This reuses the same GCP
service account created for `pipeline/fetch_odds_snapshot.py` (see
`ODDS_SNAPSHOT_SETUP.md`) — **do that setup first** if you haven't. No
new secret is needed; this just grants that same service account two
more things.

## 1. Enable the Google Drive API

The odds script only ever needed the Sheets API. Reading a folder's
contents needs the Drive API too:

1. Go to https://console.cloud.google.com/ and open the same project you
   used for the odds setup.
2. **APIs & Services → Library** → search **Google Drive API** → **Enable**.

(Tested during development: skipping this produces a `403
accessNotConfigured` error naming the exact project and a direct
activation link — if you see that, this step is what's missing.)

## 2. Share the injury-reports folder with the service account

1. In Google Drive, open the **"NFL Injury Reports 2026"** folder (the
   one your scheduled task writes weekly sheets into).
2. **Share** it with the service account's email (the `client_email`
   field in its JSON key — same one you shared the odds Sheet with).
   **Viewer** is enough; this script only reads.
3. Sharing the folder itself (not each sheet individually) means new
   weekly sheets the scheduled task creates later are automatically
   visible too — nothing to re-share each week.

## 3. Test locally

Your existing `.env` from the odds setup already has `GCP_SA_KEY_JSON` (or
`GCP_SA_KEY_PATH`) — no new local secret needed.

```bash
pip install -r requirements.txt
python3 pipeline/fetch_injury_reports_from_drive.py
```

First run should print one line per week found in the folder ("updated"
or "unchanged"), then rebuild `data/injury_reports.parquet` and
`docs/data/matchups.json` if anything changed.

## 4. Verify the workflow

No new GitHub secret needed here either — `.github/workflows/injury_reports_sync.yml`
reuses the `GCP_SA_KEY_JSON` repo secret already set up for the odds
workflow. In this repo's **Actions** tab, select "Injury reports sync" →
**Run workflow** to trigger it manually and confirm it commits the
refreshed files (only if something actually changed — an unchanged week
produces no commit, by design).

## How "latest sheet for a week" is picked

Your scheduled task replaces a week's sheet with a brand-new file (new
Drive file ID, new "(updated ...)" timestamp) each time it rebuilds that
week, rather than editing one in place. The script lists every
spreadsheet in the folder, groups them by the `(season, week)` parsed out
of each title, and keeps only the one with the newest Drive
`modifiedTime` per group — so it always finds the current version even
though the file ID keeps changing.

## Known caveats

Same two as the odds workflow (see `ODDS_SNAPSHOT_SETUP.md`) — scheduled
runs aren't perfectly on-time, and GitHub disables a schedule after 60
days of repo inactivity. One more specific to this workflow: it commits
directly to whatever branch the workflow runs on, with no PR/review step
— fine for generated data files, but worth knowing if branch protection
rules would otherwise block a direct push.
