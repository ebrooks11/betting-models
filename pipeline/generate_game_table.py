"""
Generate docs/data/games.json — columnar format for efficient browser loading.
One entry per column (array of all row values), consumed by docs/explore.html.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data_loader import get_schedule_data
from src.model_utils import TEAM_MAP, load_features, PLAYOFF_TYPES

SEASONS = range(2016, 2026)
OUT_PATH = Path("docs/data/games.json")

FEATURES = [
    ("off_epa_rolling",              "Off EPA"),
    ("off_11_epa_rolling",           "Off EPA 11"),
    ("off_12_epa_rolling",           "Off EPA 12"),
    ("off_epa_early_down_rolling",   "Early Down EPA"),
    ("off_epa_first_down_rolling",   "1st Down EPA"),
    ("def_epa_rolling",              "Def EPA"),
    ("def_vs_11_epa_rolling",        "Def vs 11"),
    ("def_vs_12_epa_rolling",        "Def vs 12"),
    ("qb_off_11_epa_rolling",        "QB EPA 11"),
    ("qb_off_12_epa_rolling",        "QB EPA 12"),
    ("qbr_rolling",                  "QBR"),
    ("cpoe_rolling",                 "CPOE"),
    ("point_diff_rolling",           "Pt Diff"),
    ("turnovers_committed_rolling",  "TOs Committed"),
    ("to_diff_rolling",              "TO Diff"),
    ("first_down_rate_rolling",      "1st Down Rate"),
    ("top_rolling",                  "Time of Poss"),
    ("rest_days",                    "Rest Days"),
]

FEAT_COLS   = [f for f, _ in FEATURES]
FEAT_LABELS = {f: lbl for f, lbl in FEATURES}


def _r(v, n=2):
    """Round float or return None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), n)


def build():
    print("Loading features...")
    features = load_features()

    print("Loading schedules...")
    schedules = get_schedule_data(list(SEASONS))

    game_types = ["REG"] + PLAYOFF_TYPES
    games = schedules[schedules["game_type"].isin(game_types)].copy()
    games["home_team"] = games["home_team"].replace(TEAM_MAP)
    games["away_team"] = games["away_team"].replace(TEAM_MAP)
    games = games[games["week"] >= 4]

    final_wk = games["season"].map(lambda s: 18 if s >= 2021 else 17)
    reg = games["game_type"] == "REG"
    games = games[~reg | (games["week"] < final_wk)]

    available = [c for c in FEAT_COLS if c in features.columns]
    missing   = [c for c in FEAT_COLS if c not in features.columns]
    if missing:
        print(f"  Warning: missing features: {missing}")

    team_feats = features[["season", "week", "team"] + available].copy()

    merged = games.merge(
        team_feats.rename(columns={c: f"h_{c}" for c in available}),
        left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"], how="left",
    ).drop(columns=["team"])
    merged = merged.merge(
        team_feats.rename(columns={c: f"a_{c}" for c in available}),
        left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"], how="left",
    ).drop(columns=["team"])

    # Build columnar arrays
    cols = {
        "season":    [], "week": [], "game_type": [],
        "home":      [], "away": [],
        "spread":    [], "margin": [], "covered": [],
    }
    for c in available:
        cols[f"h_{c}"] = []
        cols[f"a_{c}"] = []
        cols[f"d_{c}"] = []

    for _, g in merged.iterrows():
        spread = g.get("spread_line")
        hs     = g.get("home_score")
        as_    = g.get("away_score")

        has_result = pd.notna(hs) and pd.notna(as_)
        has_spread = pd.notna(spread)
        margin     = (float(hs) - float(as_)) if has_result else None
        covered    = None
        if has_result and has_spread:
            diff = margin - float(spread)
            covered = True if diff > 0 else (False if diff < 0 else None)

        cols["season"].append(int(g["season"]))
        cols["week"].append(int(g["week"]))
        cols["game_type"].append(str(g["game_type"]))
        cols["home"].append(str(g["home_team"]))
        cols["away"].append(str(g["away_team"]))
        cols["spread"].append(_r(spread))
        cols["margin"].append(_r(margin))
        cols["covered"].append(covered)

        for c in available:
            hv = _r(g.get(f"h_{c}"))
            av = _r(g.get(f"a_{c}"))
            dv = _r(hv - av) if hv is not None and av is not None else None
            cols[f"h_{c}"].append(hv)
            cols[f"a_{c}"].append(av)
            cols[f"d_{c}"].append(dv)

    # Column metadata for the UI
    meta = [
        {"key": "season",    "label": "Season",  "type": "int",   "group": "game"},
        {"key": "week",      "label": "Week",     "type": "int",   "group": "game"},
        {"key": "game_type", "label": "Type",     "type": "str",   "group": "game"},
        {"key": "home",      "label": "Home",     "type": "str",   "group": "game"},
        {"key": "away",      "label": "Away",     "type": "str",   "group": "game"},
        {"key": "spread",    "label": "Spread",   "type": "float", "group": "game"},
        {"key": "margin",    "label": "Margin",   "type": "float", "group": "game"},
        {"key": "covered",   "label": "Covered",  "type": "bool",  "group": "game"},
    ]
    for c in available:
        lbl = FEAT_LABELS.get(c, c)
        meta.append({"key": f"d_{c}", "label": f"Δ {lbl}",  "type": "float", "group": "diff", "feat": c})
        meta.append({"key": f"h_{c}", "label": f"H {lbl}",  "type": "float", "group": "home", "feat": c})
        meta.append({"key": f"a_{c}", "label": f"A {lbl}",  "type": "float", "group": "away", "feat": c})

    n = len(cols["season"])
    out = {"n": n, "columns": meta, "data": cols}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {n} games → {OUT_PATH}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build()
