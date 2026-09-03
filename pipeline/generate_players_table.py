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
        # ESPN QBR data — not published for 2024/2025, blank for those rows
        "qbr_total", "qbr_raw", "pts_added", "qbr_epa_total",
        "qbr_pass_epa", "qbr_run_epa", "qb_plays", "qbr_qualified",
        # everything after oc_name (background/draft/combine/contract)
        "draft_round", "draft_pick", "draft_year",
        "forty", "bench", "vertical", "broad_jump", "cone", "shuttle",
        "combine_height", "combine_weight",
        "contract_apy", "contract_year_signed", "contract_years",
    },
}

# Preserve known abbreviations/acronyms, and shorten common long words, when
# humanizing snake_case column names into display labels — the table has a
# lot of columns and long headers make it unusably wide. Anything not in
# this map is just title-cased as-is.
ABBR = {
    "epa": "EPA", "qbr": "QBR", "pfr": "PFR", "ppr": "PPR", "apy": "APY",
    "id": "ID", "td": "TD", "tds": "TDs", "pct": "%", "yac": "YAC",
    "oc": "OC", "2pt": "2PT", "pacr": "PACR", "gsis": "GSIS", "los": "LOS",
    "years": "Yrs", "year": "Yr",
    "passing": "Pass", "rushing": "Rush", "receiving": "Rec", "receptions": "Rec",
    "completions": "Comp", "completion": "Comp",
    "attempts": "Att",
    "interceptions": "Int",
    "yards": "Yds",
    "fumbles": "Fum",
    "first": "1st",
    "downs": "Dwn",
    "conversions": "Conv",
    "differential": "Diff",
    "percentage": "%", "percent": "%",
    "expected": "Exp", "expectation": "Exp",
    "above": "Abv",
    "before": "Bfr", "after": "Aft",
    "broken": "Brkn", "tackles": "Tkl", "tackle": "Tkl",
    "target": "Tgt", "targets": "Tgt",
    "intended": "Intnd",
    "share": "Shr",
    "cushion": "Cush",
    "separation": "Sep",
    "blitzed": "Blitz",
    "signed": "Sgnd",
    "aggressiveness": "Aggr",
    "total": "Tot",
    "points": "Pts",
    "with": "w/",
    "pa": "PA",    # play-action
    "rpo": "RPO",  # run-pass option
    "combine": "Cmb", "height": "Ht", "weight": "Wt",
    "round": "Rnd",
    "efficiency": "Eff",
    "contact": "Cntct",
    "depth": "Dpth",
}

# Full-key overrides for labels that don't read well built word-by-word.
LABEL_OVERRIDES = {
    "percent_attempts_gte_eight_defenders": "% vs 8+ Box",
    "player_name": "Player",
    "oc_name": "OC",
    "pa_pass_att": "PA Att",
    "pa_pass_yards": "PA Yds",
    "rpo_pass_att": "RPO Att",
    "rpo_pass_yards": "RPO Yds",
}


def _humanize(key: str) -> str:
    if key in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[key]
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
