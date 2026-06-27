"""Train, evaluate, and save the NFL prediction model."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import (
    ALL_FEATURES,
    RIDGE_ALPHA,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def train_model(
    df: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    alpha: float = RIDGE_ALPHA,
) -> tuple[Ridge, StandardScaler]:
    """Train a Ridge regression model on the given data."""
    X = df[features].values
    y = df["score"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)

    return model, scaler


def evaluate_predictions(
    df: pd.DataFrame, predictions: np.ndarray
) -> dict:
    """Compute evaluation metrics from predicted team scores."""
    df = df.copy()
    df["predicted_score"] = predictions

    # MAE on individual team scores
    mae = np.mean(np.abs(df["score"] - df["predicted_score"]))

    games = _build_game_pairs(df)

    return _compute_metrics(games, mae)


def _build_game_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Pair home and away rows into single game rows."""
    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    home = home.rename(columns={
        "team": "home_team",
        "opponent": "away_team",
        "predicted_score": "home_pred",
        "score": "home_actual",
    })[["season", "week", "home_team", "away_team", "home_pred",
        "home_actual", "spread_line", "total_line"]]

    away = away.rename(columns={
        "team": "away_team",
        "predicted_score": "away_pred",
        "score": "away_actual",
    })[["season", "week", "away_team", "away_pred", "away_actual"]]

    games = home.merge(away, on=["season", "week", "away_team"])
    return games


def _compute_metrics(games: pd.DataFrame, mae: float) -> dict:
    """Calculate ATS and O/U accuracy from game pairs."""
    games = games.copy()

    # All spreads from home perspective: positive = home wins/favored
    games["pred_margin"] = games["home_pred"] - games["away_pred"]
    games["actual_margin"] = games["home_actual"] - games["away_actual"]
    # nflverse spread_line: positive = home favored by that many points
    games["book_margin"] = games["spread_line"]

    # ATS: does our model pick the correct side of the book's spread?
    # Home covers if actual margin exceeds the book's expected margin
    games["home_covers"] = games["actual_margin"] > games["book_margin"]
    games["model_picks_home"] = games["pred_margin"] > games["book_margin"]

    # Exclude pushes
    non_push_ats = games[games["actual_margin"] != games["book_margin"]]
    ats_correct = (non_push_ats["model_picks_home"] == non_push_ats["home_covers"]).sum()
    ats_total = len(non_push_ats)
    ats_pct = ats_correct / ats_total if ats_total > 0 else 0

    # O/U accuracy
    games["pred_total"] = games["home_pred"] + games["away_pred"]
    games["actual_total"] = games["home_actual"] + games["away_actual"]
    games["pred_over"] = games["pred_total"] > games["total_line"]
    games["actual_over"] = games["actual_total"] > games["total_line"]
    non_push_ou = games[games["actual_total"] != games["total_line"]]
    ou_correct = (non_push_ou["pred_over"] == non_push_ou["actual_over"]).sum()
    ou_total = len(non_push_ou)
    ou_pct = ou_correct / ou_total if ou_total > 0 else 0

    # Edge picks: games where our model disagrees with the book's favorite
    games["model_edge"] = games["pred_margin"] - games["book_margin"]
    games["model_disagrees"] = (
        (games["model_picks_home"] == True) & (games["book_margin"] < 0)
    ) | (
        (games["model_picks_home"] == False) & (games["book_margin"] > 0)
    )
    disagree_games = games[(games["model_disagrees"]) & (games["actual_margin"] != games["book_margin"])]
    edge_correct = (disagree_games["model_picks_home"] == disagree_games["home_covers"]).sum()
    edge_total = len(disagree_games)
    edge_pct = edge_correct / edge_total if edge_total > 0 else 0

    return {
        "mae": round(mae, 2),
        "ats_record": f"{ats_correct}-{ats_total - ats_correct}",
        "ats_pct": round(ats_pct * 100, 1),
        "edge_record": f"{edge_correct}-{edge_total - edge_correct}",
        "edge_pct": round(edge_pct * 100, 1),
        "edge_games": edge_total,
        "ou_record": f"{ou_correct}-{ou_total - ou_correct}",
        "ou_pct": round(ou_pct * 100, 1),
        "total_games": len(games),
    }


def walk_forward_evaluate(df: pd.DataFrame) -> dict:
    """Walk-forward validation: train on prior seasons, test on each validation season."""
    all_results = []
    all_test_dfs = []

    for val_season in VALIDATION_SEASONS:
        train_df = df[df["season"] < val_season]
        test_df = df[df["season"] == val_season]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        model, scaler = train_model(train_df)
        X_test = scaler.transform(test_df[ALL_FEATURES].values)
        preds = model.predict(X_test)

        test_df = test_df.copy()
        test_df["predicted_score"] = preds
        all_test_dfs.append(test_df)

        season_metrics = evaluate_predictions(test_df, preds)
        season_metrics["season"] = val_season
        all_results.append(season_metrics)

        print(f"  {val_season}: MAE={season_metrics['mae']}, "
              f"ATS={season_metrics['ats_pct']}% ({season_metrics['ats_record']}), "
              f"Edge={season_metrics['edge_pct']}% ({season_metrics['edge_record']} in {season_metrics['edge_games']} contrarian picks), "
              f"O/U={season_metrics['ou_pct']}% ({season_metrics['ou_record']})")

    # Overall metrics across all validation seasons
    combined = pd.concat(all_test_dfs, ignore_index=True)
    overall = evaluate_predictions(combined, combined["predicted_score"].values)
    overall["season"] = "overall"
    all_results.append(overall)

    print(f"\n  Overall: MAE={overall['mae']}, "
          f"ATS={overall['ats_pct']}% ({overall['ats_record']}), "
          f"Edge={overall['edge_pct']}% ({overall['edge_record']} in {overall['edge_games']} contrarian picks), "
          f"O/U={overall['ou_pct']}% ({overall['ou_record']})")

    return {"per_season": all_results[:-1], "overall": all_results[-1]}


def save_model(model: Ridge, scaler: StandardScaler) -> Path:
    """Save trained model and scaler to disk."""
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "ridge_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"Model saved to {path}")
    return path


def load_model() -> tuple[Ridge, StandardScaler]:
    """Load trained model and scaler from disk."""
    path = MODEL_DIR / "ridge_model.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]


def print_feature_importance(model: Ridge, features: list[str] = ALL_FEATURES):
    """Print model coefficients ranked by absolute value."""
    coefs = pd.Series(model.coef_, index=features)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    print("\nFeature importance (Ridge coefficients):")
    for feat, coef in coefs.items():
        direction = "+" if coef > 0 else "-"
        print(f"  {direction} {feat}: {coef:.3f}")


if __name__ == "__main__":
    from config import SEASONS
    from src.data_loader import get_pbp_data, get_schedule_data
    from src.feature_engineering import build_feature_matrix

    print("Loading data...")
    pbp = get_pbp_data(SEASONS)
    schedules = get_schedule_data(SEASONS)

    print("Building features...")
    df = build_feature_matrix(pbp, schedules)

    print(f"\nWalk-forward evaluation ({VALIDATION_SEASONS[0]}-{VALIDATION_SEASONS[-1]}):")
    results = walk_forward_evaluate(df)

    # Train final model on all data and save
    print("\nTraining final model on all data...")
    model, scaler = train_model(df)
    print_feature_importance(model)
    save_model(model, scaler)
