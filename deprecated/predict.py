"""Generate predictions for upcoming NFL games."""

import pandas as pd
import numpy as np

from config import ALL_FEATURES, SEASONS, ROLLING_WINDOW
from src.data_loader import get_pbp_data, get_schedule_data
from src.feature_engineering import build_feature_matrix
from src.model import load_model


def get_upcoming_games(schedules: pd.DataFrame) -> pd.DataFrame:
    """Find games that haven't been played yet."""
    games = schedules[schedules["game_type"] == "REG"].copy()
    upcoming = games[games["home_score"].isna()]
    return upcoming


def build_current_team_stats(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Get the most recent rolling stats for each team."""
    team_stats = {}
    for team, group in df.groupby("team"):
        latest = group.sort_values(["season", "week"]).iloc[-1]
        team_stats[team] = latest
    return team_stats


def predict_upcoming(
    df: pd.DataFrame, upcoming: pd.DataFrame
) -> pd.DataFrame:
    """Generate predictions for upcoming games."""
    model, scaler = load_model()
    team_stats = build_current_team_stats(df)

    predictions = []
    for _, game in upcoming.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        if home not in team_stats or away not in team_stats:
            continue

        home_features = _build_prediction_row(team_stats[home], team_stats[away], is_home=1, game=game)
        away_features = _build_prediction_row(team_stats[away], team_stats[home], is_home=0, game=game)

        if home_features is None or away_features is None:
            continue

        home_X = scaler.transform([home_features])
        away_X = scaler.transform([away_features])

        home_pred = model.predict(home_X)[0]
        away_pred = model.predict(away_X)[0]

        pred_spread = home_pred - away_pred
        pred_total = home_pred + away_pred

        predictions.append({
            "week": game["week"],
            "away_team": away,
            "home_team": home,
            "away_pred": round(away_pred, 1),
            "home_pred": round(home_pred, 1),
            "pred_spread": round(pred_spread, 1),
            "pred_total": round(pred_total, 1),
            "book_spread": game.get("spread_line", None),
            "book_total": game.get("total_line", None),
        })

    result = pd.DataFrame(predictions)

    if "book_spread" in result.columns and result["book_spread"].notna().any():
        result["spread_edge"] = round(result["pred_spread"] - result["book_spread"], 1)
    if "book_total" in result.columns and result["book_total"].notna().any():
        result["total_edge"] = round(result["pred_total"] - result["book_total"], 1)

    return result


def _build_prediction_row(
    team_latest: pd.Series, opp_latest: pd.Series, is_home: int, game: pd.Series
) -> list[float] | None:
    """Build a feature vector for a single team in an upcoming game."""
    row = []
    for feat in ALL_FEATURES:
        if feat == "is_home":
            row.append(is_home)
        elif feat == "rest_advantage":
            team_rest = team_latest.get("rest_days", 7)
            opp_rest = opp_latest.get("rest_days", 7)
            row.append(team_rest - opp_rest)
        elif feat == "win_streak":
            row.append(team_latest.get("win_streak", 0))
        elif feat == "week":
            row.append(game["week"])
        elif feat.startswith("opp_"):
            opp_feat = feat[4:]
            val = opp_latest.get(opp_feat, np.nan)
            if pd.isna(val):
                return None
            row.append(val)
        else:
            val = team_latest.get(feat, np.nan)
            if pd.isna(val):
                return None
            row.append(val)
    return row


def display_predictions(preds: pd.DataFrame):
    """Pretty-print predictions."""
    if preds.empty:
        print("No upcoming games found.")
        return

    print(f"\n{'='*70}")
    print(f"NFL Predictions — Week {preds['week'].iloc[0]}")
    print(f"{'='*70}")

    for _, row in preds.iterrows():
        print(f"\n{row['away_team']} @ {row['home_team']}")
        print(f"  Predicted score: {row['away_team']} {row['away_pred']}  "
              f"{row['home_team']} {row['home_pred']}")
        print(f"  Predicted spread: {row['home_team']} {row['pred_spread']:+.1f}")
        print(f"  Predicted total:  {row['pred_total']:.1f}")

        if "book_spread" in row and pd.notna(row.get("book_spread")):
            print(f"  Book spread:      {row['book_spread']:+.1f}  "
                  f"(edge: {row.get('spread_edge', 0):+.1f})")
        if "book_total" in row and pd.notna(row.get("book_total")):
            print(f"  Book total:       {row['book_total']:.1f}  "
                  f"(edge: {row.get('total_edge', 0):+.1f})")


if __name__ == "__main__":
    print("Loading data...")
    pbp = get_pbp_data(SEASONS)
    schedules = get_schedule_data(SEASONS)

    print("Building features...")
    df = build_feature_matrix(pbp, schedules)

    upcoming = get_upcoming_games(schedules)
    if upcoming.empty:
        print("\nNo upcoming games in the schedule data.")
        print("Try refreshing: python -m src.data_loader --refresh")
        print("Or run 'python -m src.model' to evaluate on historical data.")
    else:
        print(f"Found {len(upcoming)} upcoming games.")
        preds = predict_upcoming(df, upcoming)
        display_predictions(preds)
