# Odds snapshot setup

One-time setup for `pipeline/fetch_odds_snapshot.py` and
`.github/workflows/odds_snapshot.yml`. All of this happens outside this
repo (the-odds-api.com dashboard, Google Cloud Console, GitHub's web UI),
so it's manual — nothing here can be scripted from inside the repo.

## 1. Get an API key

Sign up at https://the-odds-api.com/ and copy your API key from the
dashboard. Check which plan tier you're on and its monthly credit quota —
`fetch_odds_snapshot.py`'s module docstring explains how it tries to stay
within that quota (curated prop markets, a lookahead window), but you may
still want to tune `PROP_MARKETS`/`PROP_LOOKAHEAD_DAYS` in that file to
fit your plan.

## 2. Create a Google Cloud service account

This is what lets a GitHub Actions runner — with no human logged in —
write to a Google Sheet on its own.

1. Go to https://console.cloud.google.com/ and create a project (or reuse
   one you already have).
2. **APIs & Services → Library** → enable the **Google Sheets API**.
3. **APIs & Services → Credentials → Create Credentials → Service account**.
   Give it any name (e.g. "odds-snapshot-writer"). No roles needed at the
   project level — access is granted per-sheet in step 4.
4. Open the new service account → **Keys → Add key → Create new key →
   JSON**. This downloads a `.json` file — this is the only copy Google
   gives you, so save it somewhere safe. This file is a credential, not
   just an identifier: treat it like a password.

## 3. Share the target Google Sheet with the service account

1. Create (or open) the Google Sheet you want the snapshots written to.
2. Click **Share**, and share it with the service account's email —
   it's the `client_email` field inside the JSON key, formatted like
   `odds-snapshot-writer@your-project.iam.gserviceaccount.com`. Give it
   **Editor** access.
3. Copy the sheet's ID out of its URL:
   `https://docs.google.com/spreadsheets/d/THIS_PART/edit`

The script creates its own "Game Odds" and "Player Props" tabs (with
headers) the first time it runs, so the sheet itself can start empty.

## 4. Store the credentials locally, for testing

```bash
cp .env.example .env
```

Fill in `.env` (already git-ignored, never commit it):
- `ODDS_API_KEY` — from step 1
- `GOOGLE_SHEET_ID` — from step 3
- Either `GCP_SA_KEY_JSON` (paste the entire downloaded JSON file's
  contents as one line) or `GCP_SA_KEY_PATH` (a local path to the JSON
  file itself, e.g. `secrets/gcp_service_account.json` — the `secrets/`
  directory is also git-ignored, so this is a safe place to leave the raw
  file if you'd rather not paste its contents inline)

Then test with:
```bash
pip install -r requirements.txt
python3 pipeline/fetch_odds_snapshot.py
```

## 5. Store the credentials in GitHub, for the scheduled workflow

In this repo on GitHub: **Settings → Secrets and variables → Actions →
New repository secret**. Add three secrets, matching the workflow's env
block in `.github/workflows/odds_snapshot.yml`:

| Secret name       | Value                                              |
|--------------------|-----------------------------------------------------|
| `ODDS_API_KEY`     | same as your local `.env`                          |
| `GCP_SA_KEY_JSON`  | the full JSON key file's contents, pasted as one secret |
| `GOOGLE_SHEET_ID`  | same as your local `.env`                          |

GitHub secrets are encrypted at rest and never shown again after saving —
if you need to change one later you're replacing it, not editing it.

## 6. Verify the workflow

The schedule (`.github/workflows/odds_snapshot.yml`) runs daily at 13:00
UTC, but you don't have to wait for that to confirm it works: in this
repo's **Actions** tab, select "Daily odds snapshot" → **Run workflow**
to trigger it by hand (this is what `workflow_dispatch` in the workflow
file enables). Check the run's logs for the API-credits-remaining lines
the script prints, and check the Sheet for new rows.

## Known caveats

- **Scheduled workflows aren't perfectly on-time.** GitHub queues cron
  triggers and can delay them under load — don't rely on this for
  anything that needs to run at a precise minute.
- **GitHub disables a scheduled workflow after 60 days with no repo
  activity** (any push/commit resets the clock). If the daily runs
  silently stop, this is the first thing to check.
- **Player-prop odds availability is inherently sparse for games far
  out** — sportsbooks don't post props until close to kickoff, so early
  snapshots may show game odds with few or no prop rows for that week's
  games. That's expected, not a bug.
