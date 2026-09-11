"""
Generate docs/data/coordinators.json — columnar format for docs/coordinators.html,
built from data/offensive_coordinators.parquet.

This is a standalone generator (not folded into generate_players_table.py)
because the coordinators page is expected to diverge significantly from the
player-position pages over time — different filters, different stats,
possibly defensive coordinators later. Keeping it separate now avoids
coupling two scripts that are meant to evolve independently.

Same {n, columns, data} columnar shape as generate_game_table.py's
games.json — see that file for the convention this follows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data" / "coordinators.json"

# Preserve known abbreviations/acronyms, and shorten common long words, when
# humanizing snake_case column names into display labels.
ABBR = {
    "epa": "EPA", "id": "ID", "td": "TD", "tds": "TDs", "pct": "%",
    "yac": "YAC", "oc": "OC", "2pt": "2PT",
    "qb": "QB", "rb": "RB", "wr": "WR", "te": "TE", "cpoe": "CPOE",
    "years": "Yrs", "year": "Yr",
    "passing": "Pass", "rushing": "Rush", "receiving": "Rec", "receptions": "Rec",
    "attempts": "Att",
    "interceptions": "Int",
    "yards": "Yds",
    "total": "Tot",
    "points": "Pts",
    "fantasy": "Fan",
    "targets": "Tgt", "target": "Tgt",
    "rush": "Rush", "share": "Shr",
    "formation": "Fmn", "success": "Succ",
    "ybc": "YBC", "explosive": "Expl",
}

LABEL_OVERRIDES = {
    "oc_name": "OC",
    "qb_fantasy_points_per_game": "QB PPG",
    "rb_fantasy_points_per_game": "RB PPG",
    "wr_fantasy_points_per_game": "WR PPG",
    "te_fantasy_points_per_game": "TE PPG",
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
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
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
    if col_type == "date":
        return str(v)[:10]
    return str(v)


def build():
    df = pd.read_parquet(DATA_DIR / "offensive_coordinators.parquet")

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
    print(f"Wrote {len(df):,} OC-season rows, {len(df.columns)} cols -> {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build()
