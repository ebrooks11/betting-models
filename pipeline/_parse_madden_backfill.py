"""
One-off helper: parses the raw tool-result JSON files saved from the manual
browser-batch backfill scrape of ea.com/games/madden-nfl/ratings for the
historical iterations (Launch Ratings + Weeks 1-3, as opposed to the
week-4/current snapshot already in data/madden_ratings.parquet from
pipeline/_parse_madden_scrape.py) and appends them to that file.

Same chunking/parsing approach as _parse_madden_scrape.py (see that file's
docstring for why each team's roster was split into 6 separate 15-player
javascript_tool calls). Not meant to be reusable as-is next time — the
tool-result file paths below are specific to this one backfill session.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("/Users/evanbrooks/.claude/projects/-Users-evanbrooks-Documents-development-betting-models/37fb18c6-9570-487b-9678-9dd71bc4731e/tool-results")

# (result file, [EA team ids in the order they were navigated to within that batch])
BATCHES_BY_ITERATION = {
    "1-base": [
        ("toolu_01QjLwnqebpe1kt9YuDrFkZp.json", [3, 12, 22]),
        ("toolu_01N4XXMsai6J2tHtayM5UpcJ.json", [18, 2, 5]),
        ("toolu_01GJjZenZM61UAKnobkVf8Zr.json", [25, 29, 10, 17]),
        ("toolu_014YjBjD1BgV9RdafG4aDrBR.json", [30, 32, 4, 8]),
        ("toolu_01FxeRto1nZM2BzZwcdevdzt.json", [9, 23, 11, 13]),
        ("toolu_01PJF7KvPhW5sF2BmPSipSBs.json", [16, 26, 19]),
        ("toolu_01BRFKsG2vzSrsP1wgpUJd7M.json", [20, 31, 6, 14]),
        ("toolu_013CPh7fUyk6zRPirKEqUe6H.json", [21, 27, 7, 15]),
        ("toolu_014AAp3Hq7HKHAncr1V2xuYv.json", [24, 28]),
        ("toolu_014PaVNbMqYCjawTgLvWFPuS.json", [1]),
    ],
    "madden-ratings-week-1": [
        ("toolu_018vjFZSdrs5svhGBbsAJh9M.json", [3, 12, 22, 18]),
        ("toolu_01AHMxJ7bHuWECgZFAyoSTgG.json", [2, 5, 25, 29]),
        ("toolu_01BqGND2Qk8i4yMDSrzgq2zb.json", [10, 17, 30, 32]),
        ("toolu_015knKdB1QRyoq4iSBqeYPR8.json", [4, 8, 9, 23]),
        ("toolu_01RNtwPcTDhwR1WJBDdb5fLA.json", [11, 13, 16, 26]),
        ("toolu_01EX6h5wXG23Gsz4kXRdBZvr.json", [1, 19, 20, 31]),
        ("toolu_019dHMLKg9fbLGWsE3xqTPYV.json", [6, 14, 21, 27]),
        ("toolu_01XAp2YU6ZQBM8N48EsiJQUw.json", [7, 15, 24, 28]),
    ],
    "madden-ratings-week-2": [
        ("toolu_01WrdXuMMLMatFHaCVRYeeBT.json", [3, 12, 22, 18]),
        ("toolu_01ALqA9srGWw7XXpUxH4nEu6.json", [2, 5, 25, 29]),
        ("toolu_01WAfYHkRV15LAwmR9JuuZkv.json", [10, 17, 30, 32]),
        ("toolu_013ua74cQRHwhU18rQAMSJ7q.json", [4, 8, 9, 23]),
        ("toolu_01UZxkAed761DZo5uspmq4NH.json", [11, 13, 16, 26]),
        ("toolu_01RgCPvW991yHLnHYdNGF46g.json", [1, 19, 20, 31]),
        ("toolu_01GyCbTPN6CW45zGM7PKgdgL.json", [6, 14, 21, 27]),
        ("toolu_01CLpFcbqh4hnsHEks6Hf261.json", [7, 15, 24, 28]),
    ],
    "madden-ratings-week-3": [
        ("toolu_01Qe5L65xwVxk5rcaUMzx9Yr.json", [3, 12, 22, 18]),
        ("toolu_01F354r33FNSGqd44uRBHUV3.json", [2, 5, 25, 29]),
        ("toolu_01UUKzXTKnWwB3dw8DKESmUb.json", [10, 17, 30, 32]),
        ("toolu_01ATsBDXyKNNyZFtH2egzm3M.json", [4, 8, 9, 23]),
        ("toolu_017EbcJTP1UAGbuEibBmsafS.json", [11, 13, 16, 26]),
        ("toolu_016Jeh2AfLc4zME8RXSV6dkw.json", [1, 19, 20, 31]),
        ("toolu_01MEEpYY5vRZKTGNyxF4DyRb.json", [6, 14, 21, 27]),
        ("toolu_01QhrMy6GdWYVGSM11qDT9oD.json", [7, 15, 24, 28]),
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
    backfill = build()
    print(f"Backfill: {len(backfill)} rows across iterations {sorted(backfill['madden_iteration'].unique())}")
    for it in sorted(backfill["madden_iteration"].unique()):
        sub = backfill[backfill["madden_iteration"] == it]
        print(f"  {it}: {len(sub)} rows, {sub['team'].nunique()} teams")

    existing = pd.read_parquet(OUT_PATH)
    combined = pd.concat([existing, backfill], ignore_index=True)

    dupes = combined[combined.duplicated(subset=["madden_iteration", "team", "ea_player_id"], keep=False)]
    assert dupes.empty, f"Found {len(dupes)} duplicate rows after merge with existing data"

    combined.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(combined)} total rows ({sorted(combined['madden_iteration'].unique())}) to {OUT_PATH}")
