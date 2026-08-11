"""Export the feature matrix to CSV for manual analysis in Google Sheets."""

import pandas as pd
from pathlib import Path

from config import SEASONS, ALL_FEATURES, MIN_WEEK
from src.data_loader import get_pbp_data, get_schedule_data
from src.feature_engineering import build_feature_matrix

OUTPUT_DIR = Path("exports")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading data...")
pbp = get_pbp_data(SEASONS)
schedules = get_schedule_data(SEASONS)

print("Building feature matrix...")
df = build_feature_matrix(pbp, schedules)

# Keep only week 4+ rows (same filter as model training)
df = df[df["week"] >= MIN_WEEK].copy()

# Select the columns most useful for manual modeling
cols = [
    "season", "week", "team", "opponent",
    "is_home", "spread_line", "score", "opponent_score",
    "rest_days", "rest_advantage",
] + ALL_FEATURES + ["margin"]

# margin = score - opponent_score (home perspective when is_home=1)
df["margin"] = df["score"] - df["opponent_score"]

available = [c for c in cols if c in df.columns]
export_df = df[available].sort_values(["season", "week", "team"]).reset_index(drop=True)

out_path = OUTPUT_DIR / "nfl_features.csv"
export_df.to_csv(out_path, index=False)

print(f"\nExported {len(export_df):,} rows x {len(export_df.columns)} columns")
print(f"Saved to: {out_path}")
print(f"\nColumns: {list(export_df.columns)}")
print(f"\nSample (first 3 rows):")
print(export_df.head(3).to_string())
