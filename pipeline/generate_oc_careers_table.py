"""
Generate docs/data/oc_careers.json — columnar format for the "Career" view
in docs/coordinators.html, built from data/oc_careers.parquet.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data" / "oc_careers.json"

ABBR = {
    "ppg": "PPG",
}

LABEL_OVERRIDES = {
    "oc_name": "OC",
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
    df = pd.read_parquet(DATA_DIR / "oc_careers.parquet")

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
    print(f"Wrote {len(df):,} OC career rows, {len(df.columns)} cols -> {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build()
