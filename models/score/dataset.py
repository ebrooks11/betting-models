"""
Build a team-game dataset for score prediction.

One row per team per game. Each row contains:
  - This team's offensive rolling features
  - The opponent's defensive rolling features
  - is_home flag
  - Target: points_scored
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.data_loader import get_schedule_data
from src.model_utils import load_features, PLAYOFF_TYPES, TEAM_MAP

TRAIN_SEASONS = range(2016, 2023)
TEST_SEASONS  = range(2023, 2026)


def build_score_dataset(schedules: pd.DataFrame, features: pd.DataFrame,
                        off_cols: list[str], def_cols: list[str],
                        include_playoffs: bool = True) -> pd.DataFrame:
    """
    Return one row per team per game with offensive and defensive features.

    Parameters
    ----------
    off_cols : feature columns representing the team's own offense
    def_cols : feature columns representing the opponent's defense
    """
    game_types = ["REG"] + (PLAYOFF_TYPES if include_playoffs else [])
    games = schedules[schedules["game_type"].isin(game_types)].copy()
    games["home_team"] = games["home_team"].replace(TEAM_MAP)
    games["away_team"] = games["away_team"].replace(TEAM_MAP)

    # Drop weeks 1-3 and final regular-season week (same filters as margin model)
    games = games[games["week"] >= 4]
    final_wk = games["season"].map(lambda s: 18 if s >= 2021 else 17)
    reg = games["game_type"] == "REG"
    games = games[~reg | (games["week"] < final_wk)]
    games = games.dropna(subset=["home_score", "away_score", "spread_line"])

    all_feat_cols = list(set(off_cols + def_cols))
    feat = features[["season", "week", "team"] + all_feat_cols].copy()

    # Build home team rows
    home = games[["season", "week", "game_type", "home_team", "away_team",
                  "home_score", "spread_line"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent",
                                "home_score": "points_scored"})
    home["is_home"] = 1

    # Build away team rows
    away = games[["season", "week", "game_type", "away_team", "home_team",
                  "away_score", "spread_line"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent",
                                "away_score": "points_scored"})
    away["is_home"] = 0
    # Flip spread sign for away perspective (spread_line is from home team's view)
    away["spread_line"] = -away["spread_line"]

    rows = pd.concat([home, away], ignore_index=True)

    # Merge this team's offensive features (keep column names as-is)
    rows = rows.merge(
        feat.rename(columns={"team": "team"}),
        on=["season", "week", "team"], how="left"
    )

    # Merge opponent's defensive features (prefix with opp_)
    def_rename = {c: f"opp_{c}" for c in def_cols}
    rows = rows.merge(
        feat[["season", "week", "team"] + def_cols]
            .rename(columns={**def_rename, "team": "opponent"}),
        on=["season", "week", "opponent"], how="left"
    )

    all_model_cols = off_cols + [f"opp_{c}" for c in def_cols]
    rows = rows.dropna(subset=all_model_cols + ["points_scored"])
    return rows.reset_index(drop=True)
