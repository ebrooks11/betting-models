"""
One-off helper: parses the raw tool-result JSON files saved from the manual
browser-batch scrape of ea.com/games/madden-nfl/ratings for madden-ratings-week-5
and madden-ratings-week-6 (the two snapshots that became available between
the original Launch+Week1-4 backfill and 2026-10-24) and appends them to
data/madden_ratings.parquet. Same chunking/parsing approach as
_parse_madden_backfill.py — see that file's docstring for why each team's
roster was split into 6 separate 15-player javascript_tool calls.

Not meant to be reusable as-is next time — the tool-result file paths
below are specific to this one scrape session.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("/Users/evanbrooks/.claude/projects/-Users-evanbrooks-Documents-development-betting-models/37fb18c6-9570-487b-9678-9dd71bc4731e/tool-results")

BATCHES_BY_ITERATION = {
    "madden-ratings-week-5": [
        ("toolu_01RbBhodT1ieHMRqXq4j4Twv.json", [3, 12, 22, 18]),
        ("toolu_01Y6m1i9qr5REoZkMDt9jtPF.json", [2, 5, 25, 29]),
        ("toolu_01NGgnpsucyjGV3UVNCDwEKB.json", [10, 17, 30, 32]),
        ("toolu_015v7wWPVwskAzhrRXX8bVL8.json", [4, 8, 9, 23]),
        ("toolu_01TWHUiFyrte7eAVpsbdZZBp.json", [11, 13, 16, 26]),
        ("toolu_015yGuMEmagaEa3xxegx2mzP.json", [1, 19, 20, 31]),
        ("toolu_014fjicAAfWXukkJk5ZLkLKc.json", [6, 14, 21, 27]),
        ("toolu_015gskZ8UUTxsJjtEAFA9v6W.json", [7, 15, 24, 28]),
    ],
    "madden-ratings-week-6": [
        ("toolu_017tUT4u1sFQWzhSfTpGgnmc.json", [3, 12, 22, 18]),
        ("toolu_01QUWGp6rdmRAoqWiME6pWkb.json", [2, 5, 25, 29]),
        ("toolu_01QWGFRkytPuuqKLQ9FRN519.json", [10, 17, 30, 32]),
        ("toolu_01FwqQw6DEzXCszxxQMtC5bk.json", [4, 8, 9, 23]),
        ("toolu_012fpnfxfXJs2sNJohhVX9hc.json", [11, 13, 16, 26]),
        ("toolu_018gsvbvsMTtrFJQvGwyUbks.json", [1, 19, 20, 31]),
        ("toolu_01Aew331YuRh9tJjiLdyEJKW.json", [6, 14, 21, 27]),
        ("toolu_01H9xvYHatwHNHhGR2kdovoF.json", [7, 15, 24, 28]),
    ],
}
CHUNKS_PER_TEAM = 6

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
    for iteration_id, batches in BATCHES_BY_ITERATION.items():
        teams_seen = set()
        for filename, team_ids in batches:
            js_results = _extract_js_results(RESULTS_DIR / filename)
            expected = len(team_ids) * CHUNKS_PER_TEAM
            assert len(js_results) == expected, f"{iteration_id}/{filename}: expected {expected} chunk results ({len(team_ids)} teams x {CHUNKS_PER_TEAM}), got {len(js_results)}"
            for i, team_id in enumerate(team_ids):
                abbr = EA_TEAM_ID_TO_ABBR[team_id]
                teams_seen.add(abbr)
                chunks = js_results[i * CHUNKS_PER_TEAM:(i + 1) * CHUNKS_PER_TEAM]
                players = [p for chunk in chunks for p in chunk]
                for p in players:
                    row = {
                        "season": season,
                        "madden_iteration": p.get("iteration") or iteration_id,
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
        missing = set(EA_TEAM_ID_TO_ABBR.values()) - teams_seen
        assert not missing, f"{iteration_id}: missing teams {missing}"

    df = pd.DataFrame(rows)
    dupes = df[df.duplicated(subset=["madden_iteration", "team", "ea_player_id"], keep=False)]
    if not dupes.empty:
        print(f"WARNING: {len(dupes)} duplicate (iteration, team, ea_player_id) rows:")
        print(dupes[["madden_iteration", "team", "ea_player_id", "first_name", "last_name"]])
    return df


if __name__ == "__main__":
    new_data = build()
    print(f"New: {len(new_data)} rows across iterations {sorted(new_data['madden_iteration'].unique())}")
    for it in sorted(new_data["madden_iteration"].unique()):
        sub = new_data[new_data["madden_iteration"] == it]
        print(f"  {it}: {len(sub)} rows, {sub['team'].nunique()} teams")

    existing = pd.read_parquet(OUT_PATH)
    combined = pd.concat([existing, new_data], ignore_index=True)

    dupes = combined[combined.duplicated(subset=["madden_iteration", "team", "ea_player_id"], keep=False)]
    assert dupes.empty, f"Found {len(dupes)} duplicate rows after merge with existing data"

    combined.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(combined)} total rows ({sorted(combined['madden_iteration'].unique())}) to {OUT_PATH}")
