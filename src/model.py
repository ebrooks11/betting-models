"""Train, evaluate, and save the NFL prediction model."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import (
    GAME_FEATURES,
    MIN_WEEK,
    RIDGE_ALPHA,
    VALIDATION_SEASONS,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def train_model(
    df: pd.DataFrame,
    features: list[str] = GAME_FEATURES,
    alpha: float = RIDGE_ALPHA,
) -> tuple[Ridge, StandardScaler]:
    """Train a Ridge regression model predicting home score margin."""
    X = df[features].values
    y = df["margin"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)

    return model, scaler


def train_xgb_model(
    df: pd.DataFrame,
    features: list[str] = GAME_FEATURES,
) -> tuple[XGBRegressor, None]:
    """Train an XGBoost regression model predicting home score margin."""
    X = df[features].values
    y = df["margin"].values

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    return model, None


def evaluate_predictions(df: pd.DataFrame, predictions: np.ndarray) -> dict:
    """Compute ATS accuracy from predicted margins on game-level rows."""
    df = df.copy()
    df["pred_margin"] = predictions

    # spread_line positive = home favored; home covers if actual margin > spread
    non_push = df[df["margin"] != df["spread_line"]]
    home_covers = non_push["margin"] > non_push["spread_line"]
    model_picks_home = non_push["pred_margin"] > non_push["spread_line"]

    ats_correct = (model_picks_home == home_covers).sum()
    ats_total = len(non_push)
    ats_pct = ats_correct / ats_total if ats_total > 0 else 0

    mae = np.mean(np.abs(df["margin"] - df["pred_margin"]))

    return {
        "mae": round(mae, 2),
        "ats_record": f"{ats_correct}-{ats_total - ats_correct}",
        "ats_pct": round(ats_pct * 100, 1),
        "total_games": len(df),
    }


def _run_walk_forward(df: pd.DataFrame, use_xgb: bool = False) -> dict:
    """Walk-forward validation for one algorithm on game-level rows."""
    all_results = []
    all_test_dfs = []

    for val_season in VALIDATION_SEASONS:
        train_df = df[(df["season"] < val_season) & (df["week"] >= MIN_WEEK)]
        test_df = df[(df["season"] == val_season) & (df["week"] >= MIN_WEEK)]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        if use_xgb:
            model, _ = train_xgb_model(train_df)
            X_test = test_df[GAME_FEATURES].values
        else:
            model, scaler = train_model(train_df)
            X_test = scaler.transform(test_df[GAME_FEATURES].values)

        preds = model.predict(X_test)

        test_df = test_df.copy()
        test_df["pred_margin"] = preds
        all_test_dfs.append(test_df)

        season_metrics = evaluate_predictions(test_df, preds)
        season_metrics["season"] = val_season
        all_results.append(season_metrics)

    combined = pd.concat(all_test_dfs, ignore_index=True)
    overall = evaluate_predictions(combined, combined["pred_margin"].values)
    overall["season"] = "overall"
    all_results.append(overall)

    return {"per_season": all_results[:-1], "overall": all_results[-1]}


def walk_forward_evaluate(df: pd.DataFrame) -> dict:
    """Walk-forward validation comparing Ridge and XGBoost side-by-side."""
    print("\n  Season       Ridge ATS%    XGB ATS%    Ridge MAE   XGB MAE")
    print("  " + "-" * 60)

    ridge_results = _run_walk_forward(df, use_xgb=False)
    xgb_results = _run_walk_forward(df, use_xgb=True)

    ridge_by_season = {r["season"]: r for r in ridge_results["per_season"]}
    xgb_by_season = {r["season"]: r for r in xgb_results["per_season"]}

    for season in VALIDATION_SEASONS:
        r = ridge_by_season.get(season, {})
        x = xgb_by_season.get(season, {})
        print(
            f"  {season}         {r.get('ats_pct', 'N/A'):>5}%        {x.get('ats_pct', 'N/A'):>5}%"
            f"       {r.get('mae', 'N/A'):>5}      {x.get('mae', 'N/A'):>5}"
        )

    ro = ridge_results["overall"]
    xo = xgb_results["overall"]
    print("  " + "-" * 60)
    print(
        f"  Overall       {ro['ats_pct']:>5}%        {xo['ats_pct']:>5}%"
        f"       {ro['mae']:>5}      {xo['mae']:>5}"
    )
    print(f"\n  Ridge ATS record: {ro['ats_record']}  |  XGB ATS record: {xo['ats_record']}")

    return {"ridge": ridge_results, "xgb": xgb_results}


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


def print_feature_importance(model: Ridge, features: list[str] = GAME_FEATURES):
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
    from src.feature_engineering import build_feature_matrix, build_game_matrix

    print("Loading data...")
    pbp = get_pbp_data(SEASONS)
    schedules = get_schedule_data(SEASONS)

    print("Building features...")
    team_df = build_feature_matrix(pbp, schedules)
    df = build_game_matrix(team_df)
    print(f"Game matrix: {len(df):,} games")

    print(f"\nWalk-forward evaluation ({VALIDATION_SEASONS[0]}-{VALIDATION_SEASONS[-1]}):")
    results = walk_forward_evaluate(df)

    print("\nTraining final Ridge model on all data...")
    model, scaler = train_model(df[df["week"] >= MIN_WEEK])
    print_feature_importance(model)
    save_model(model, scaler)
