"""
One-off helper: parses the raw tool-result JSON files saved from the manual
browser-batch scrape of ea.com/games/madden-nfl/ratings (see
pipeline/fetch_madden_ratings.py's docstring for why this has to be a manual,
browser-driven process rather than a normal requests-based fetch script) and
builds data/madden_ratings.parquet from them.

Each team's roster was fetched as 6 separate javascript_tool calls (15-player
slices: 0-15, 15-30, ..., 75-90) rather than one, because a single result
over ~51,274 characters gets silently truncated by the tool itself — hit
this directly: an initial full-roster-per-call attempt produced invalid,
truncated JSON for every team. A smaller roster's trailing slices just come
back as empty arrays, which is harmless.

Not meant to be reusable as-is next time — the tool-result file paths below
are specific to this one scrape session. Kept as a record of the parsing
logic (which IS reusable) for whoever runs the next manual scrape.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("/Users/evanbrooks/.claude/projects/-Users-evanbrooks-Documents-development-betting-models/37fb18c6-9570-487b-9678-9dd71bc4731e/tool-results")

# (result file, [EA team ids in the order they were navigated to within that batch])
# Each team id corresponds to exactly 6 consecutive javascript_tool entries
# in that file (the 15-player chunks).
BATCHES = [
    ("toolu_01QiDS5VYFqEh7gMiyNrKE2M.json", [3]),
    ("toolu_01563pqMfGaTUazgcEn59F2A.json", [12, 22]),
    ("toolu_01NwZR9cxATsFtTGzdaPcqMX.json", [18, 2]),
    ("toolu_016FLpZxpBbSGQ9g3gjmtpY8.json", [5, 25]),
    ("toolu_01RaJs5LVRmcwbMpwcVnoYQq.json", [29, 10]),
    ("toolu_01C3PP92Rps7BsA2LNMMH33D.json", [17, 30]),
    ("toolu_015Tc5ZhjmkzCzcQXXsewrag.json", [32, 4]),
    ("toolu_01H8fjmGrCdi5YMYjZSDnAga.json", [8, 9]),
    ("toolu_01VySd4x6iUKspTBUgWq3Dck.json", [23, 11]),
    ("toolu_011ajq4Ff5sxanGSGhBw9ZLr.json", [13, 16]),
    ("toolu_01NvADMLZqsH5RMZJr51gwMT.json", [26, 1]),
    ("toolu_01PRNN7Cnwrhg8d5ajFyBHQ5.json", [19, 20]),
    ("toolu_01NRUPYYYzFDPnbXTy4zUyK6.json", [31, 6]),
    ("toolu_011gdbT4KcRxrKCCqMxgQuHU.json", [14, 21, 27]),
    ("toolu_016JYbgLzJk6h3hvADq8k2RR.json", [7, 15, 24, 28]),
]
CHUNKS_PER_TEAM = 6

# EA's own numeric team id -> this project's nflverse team abbreviation —
# read off data/props/pageProps/ratingsFilters/teamGroups in the page's own
# __NEXT_DATA__ blob.
EA_TEAM_ID_TO_ABBR = {
    3: "BUF", 12: "MIA", 22: "NE", 18: "NYJ",
    2: "CIN", 5: "CLE", 25: "BAL", 29: "PIT",
    10: "IND", 17: "JAX", 30: "TEN", 32: "HOU",
    4: "DEN", 8: "LAC", 9: "KC", 23: "LV",
    11: "DAL", 13: "PHI", 16: "NYG", 26: "WAS",
    1: "CHI", 19: "DET", 20: "GB", 31: "MIN",
    6: "TB", 14: "ATL", 21: "CAR", 27: "NO",
    7: "ARI", 15: "SF", 24: "LA", 28: "SEA",
}

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "madden_ratings.parquet"


def _extract_js_results(result_path: Path) -> list[list[dict]]:
    """Returns the parsed JSON payload of each javascript_tool entry, in order."""
    entries = json.loads(result_path.read_text())
    out = []
    for e in entries:
        text = e.get("text", "")
        if text.startswith("[javascript_tool:javascript_exec] "):
            payload = text[len("[javascript_tool:javascript_exec] "):]
            out.append(json.loads(payload))
    return out


def build() -> pd.DataFrame:
    season = date.today().year if date.today().month >= 3 else date.today().year - 1
    scraped_date = date.today().isoformat()

    rows = []
    for filename, team_ids in BATCHES:
        js_results = _extract_js_results(RESULTS_DIR / filename)
        expected = len(team_ids) * CHUNKS_PER_TEAM
        assert len(js_results) == expected, f"{filename}: expected {expected} chunk results ({len(team_ids)} teams x {CHUNKS_PER_TEAM}), got {len(js_results)}"
        for i, team_id in enumerate(team_ids):
            abbr = EA_TEAM_ID_TO_ABBR[team_id]
            chunks = js_results[i * CHUNKS_PER_TEAM:(i + 1) * CHUNKS_PER_TEAM]
            # BUF's very first chunk (a dry-run call before the chunking
            # scheme was finalized) came back as {"c0": [...], "n": 56}
            # instead of a plain list like every other chunk.
            chunks = [c["c0"] if isinstance(c, dict) else c for c in chunks]
            players = [p for chunk in chunks for p in chunk]
            for p in players:
                row = {
                    "season": season,
                    "madden_iteration": p.get("iteration"),
                    "scraped_date": scraped_date,
                    "team": abbr,
                    "ea_player_id": p["id"],
                    "first_name": p["first"],
                    "last_name": p["last"],
                    "position": p["pos"],
                    "overall": p["overall"],
                    "age": p["age"],
                    "jersey_num": p["jersey"],
                    "years_pro": p["yearsPro"],
                    "archetype": p["archetype"],
                }
                for stat_name, value in p["stats"].items():
                    row[f"stat_{stat_name}"] = value
                rows.append(row)

    df = pd.DataFrame(rows)
    dupes = df[df.duplicated(subset=["team", "ea_player_id"], keep=False)]
    if not dupes.empty:
        print(f"WARNING: {len(dupes)} duplicate (team, ea_player_id) rows:")
        print(dupes[["team", "ea_player_id", "first_name", "last_name"]])
    return df


if __name__ == "__main__":
    df = build()
    print(f"{len(df)} rows, {df['team'].nunique()} teams, iteration(s): {df['madden_iteration'].unique()}")
    missing_teams = set(EA_TEAM_ID_TO_ABBR.values()) - set(df["team"].unique())
    if missing_teams:
        print(f"WARNING: missing teams: {missing_teams}")
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
