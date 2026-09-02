"""
Generate docs/data/players_{qb,rb,wr,te}.json — columnar format for the
browser player explorer (docs/players.html), one file per position table
in data/*.parquet (quarterbacks/running_backs/wide_receivers/tight_ends).

Same {n, columns, data} columnar shape as generate_game_table.py's
games.json — see that file for the convention this follows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "data"

TABLES = [
    ("qb", "quarterbacks.parquet"),
    ("rb", "running_backs.parquet"),
    ("wr", "wide_receivers.parquet"),
    ("te", "tight_ends.parquet"),
]

# Per-position column trims for the UI (JSON only omits these — the
# underlying parquet tables are untouched).
EXCLUDE_COLS = {
    "qb": {
        "gsis_id", "football_name", "jersey_number", "birth_date", "college",
        "avg_air_yards_to_sticks", "avg_time_to_los",
        "percent_attempts_gte_eight_defenders", "scrambles",
        # everything after oc_name (background/draft/combine/contract)
        "draft_round", "draft_pick", "draft_year",
        "forty", "bench", "vertical", "broad_jump", "cone", "shuttle",
        "combine_height", "combine_weight",
        "contract_apy", "contract_year_signed", "contract_years",
    },
}

# Preserve known abbreviations/acronyms when humanizing snake_case column
# names into display labels; anything else is just title-cased.
ABBR = {
    "epa": "EPA", "qbr": "QBR", "pfr": "PFR", "ppr": "PPR", "apy": "APY",
    "id": "ID", "td": "TD", "tds": "TDs", "pct": "%", "yac": "YAC",
    "oc": "OC", "2pt": "2PT", "pacr": "PACR", "gsis": "GSIS", "los": "LOS",
}


def _humanize(key: str) -> str:
    words = []
    for part in key.split("_"):
        low = part.lower()
        if low in ABBR:
            words.append(ABBR[low])
        else:
            words.append(part.capitalize())
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


def build_one(position: str, filename: str):
    df = pd.read_parquet(DATA_DIR / filename)

    exclude = EXCLUDE_COLS.get(position, set())
    if exclude:
        df = df.drop(columns=[c for c in exclude if c in df.columns])

    columns_meta = []
    cols = {}
    for key in df.columns:
        ctype = _col_type(df[key])
        columns_meta.append({"key": key, "label": _humanize(key), "type": ctype})
        cols[key] = [_cell(v, ctype) for v in df[key]]

    out = {"n": len(df), "columns": columns_meta, "data": cols}
    out_path = OUT_DIR / f"players_{position}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"  {position}: {len(df):,} rows, {len(df.columns)} cols -> {out_path} ({size_kb:.0f} KB)")


def build():
    for position, filename in TABLES:
        build_one(position, filename)


if __name__ == "__main__":
    build()
