# Refreshing injury reports

How to get an updated week's injury report from the "NFL Injury Reports
2026" Google Drive folder (populated by the user's own scheduled task)
into `docs/matchups.html`. See `pipeline/build_injury_reports_table.py`
and `pipeline/generate_matchups_table.py`'s docstrings for how the data
is shaped once it's in the repo — this doc is only about getting it here.

## Why this isn't automatic

The site is static (`docs/*.html` reading pre-generated `docs/data/*.json`
files) — nothing on the page fetches the Google Sheet directly. Someone
has to pull each week's sheet in and rebuild the JSON. Right now that
"someone" has to be a Claude conversation with Google Drive access: that
access is tied to the conversation's own OAuth session, not something a
plain script can use. Steps 1-3 below are the part that requires this;
steps 4-6 are ordinary Python anyone can run once the raw CSV exists.

(This is the same gap discussed for automating away — either extending
the GitHub Actions + service-account pattern already built for
`pipeline/fetch_odds_snapshot.py`, or a scheduled Claude task. Neither is
built yet, so for now this checklist is the process.)

## Steps

1. **Find the current sheet.** Search the "NFL Injury Reports 2026" Drive
   folder for the target week. Filenames look like
   `2026 Week 03 – NFL Injury Report (updated ...)` — the scheduled task
   replaces the file (new file ID, new timestamp) each time it rebuilds a
   week rather than editing one in place, so always grab whichever file
   for that week has the most recent "updated" timestamp, not an ID used
   in a previous refresh.

2. **Download it as CSV**, using the Drive connector's
   `download_file_content` with `exportMimeType: text/csv`. Do not use
   `read_file_content` for this — it silently truncates anything past a
   few hundred rows and will produce an incomplete week without any
   obvious error.

3. **Decode and save it** to
   `data/raw/injury_reports/{season}_week{NN}.csv` (e.g. `2026_week03.csv`),
   overwriting the existing file for that week if one's already there.

4. **Rebuild the parquet table:**
   ```bash
   python3 pipeline/build_injury_reports_table.py
   ```
   Re-parses every CSV in `data/raw/injury_reports/` into
   `data/injury_reports.parquet`.

5. **Regenerate the page's JSON:**
   ```bash
   python3 pipeline/generate_matchups_table.py
   ```
   This is what `docs/matchups.html` actually reads — step 4 alone
   doesn't touch the page.

6. **Reload the matchups page.** Serving `docs/` locally, that's it. If
   the site is deployed (e.g. GitHub Pages), commit and push
   `data/raw/injury_reports/*.csv`, `data/injury_reports.parquet`, and
   `docs/data/matchups.json` — the live site won't update until those are
   pushed.

## Known gotcha

If the matchups page shows stale-looking data (a game's report frozen at
an early, mostly-empty state — "Inactives not yet posted", every player
"No designation"), it almost always means the underlying sheet has been
rebuilt since the last time steps 1-3 were run, not a bug in the build
scripts. Re-run the checklist for that week before looking anywhere else.
