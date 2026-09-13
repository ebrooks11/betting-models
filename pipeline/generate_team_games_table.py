"""
Generate docs/data/team_games.json — columnar format for the "Game Log" view
in docs/coordinators.html, built from data/team_games.parquet.

Same {n, columns, data} columnar shape as generate_game_table.py's
games.json — see that file for the convention this follows. game_id is
dropped from the output: it's a join key for this pipeline, not a column
a fantasy-manager reader of the game log needs to see.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data" / "team_games.json"

# Same abbreviation convention as generate_coordinators_table.py.
ABBR = {
    "epa": "EPA", "oc": "OC", "pct": "%",
    "rate": "Rate", "rates": "Rates",
    "rush": "Rush", "pass": "Pass",
    "success": "Succ", "explosive": "Expl",
    "neutral": "Neutral", "script": "Script",
    "personnel": "Pers",
}

LABEL_OVERRIDES = {
    "oc_name": "OC",
    "hc_name": "HC",
    "is_home": "Home",
    "points_scored": "Pts For",
    "points_allowed": "Pts Against",
    "epa_per_play": "EPA/Play",
    "personnel_known_plays": "Pers. Plays (n)",
    "plays_20plus": "20+ Plays",
    "plays_40plus": "40+ Plays",
}


def _humanize(key: str) -> str:
    if key in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[key]
    words = []
    for part in key.split("_"):
        low = part.lower()
        words.append(ABBR[low] if low in ABBR else part.capitalize())
    return " ".join(words)


def _col_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "str"


def _cell(v, col_type):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    if col_type == "float":
        return round(float(v), 3)
    if col_type == "int":
        return int(v)
    if col_type == "bool":
        return bool(v)
    return str(v)


def build():
    df = pd.read_parquet(DATA_DIR / "team_games.parquet").drop(columns=["game_id"])

    columns_meta = []
    cols = {}
    for key in df.columns:
        ctype = _col_type(df[key])
        columns_meta.append({"key": key, "label": _humanize(key), "type": ctype})
        cols[key] = [_cell(v, ctype) for v in df[key]]

    out = {"n": len(df), "columns": columns_meta, "data": cols}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {len(df):,} team-game rows, {len(df.columns)} cols -> {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build()
