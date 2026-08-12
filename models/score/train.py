"""
Train and evaluate the score prediction model.

Usage:
    python models/score/train.py               # walk-forward CV
    python models/score/train.py --verbose     # per-fold breakdown
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.data_loader import get_schedule_data
from src.model_utils import load_features
from models.score.dataset import build_score_dataset, TRAIN_SEASONS, TEST_SEASONS

CV_TRAIN_START = 2016
CV_TEST_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]

# Offensive features (this team's rolling stats)
OFF_COLS = ["points_scored_rolling", "qbr_rolling", "plays_per_game_rolling"]

# Defensive features (opponent's rolling stats)
DEF_COLS = ["points_allowed_rolling"]

# After dataset build: off cols keep their name, def cols get "opp_" prefix
FEATURE_COLS = OFF_COLS + [f"opp_{c}" for c in DEF_COLS] + ["is_home"]


def evaluate_scores(actual: pd.Series, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    bias = (predicted - actual.values).mean()
    pred_std = predicted.std()
    actual_std = actual.std()
    return dict(mae=mae, bias=bias, pred_std=pred_std, actual_std=actual_std, n=len(actual))


def run_cv(df: pd.DataFrame, verbose: bool = False):
    print(f"\nFeatures: {FEATURE_COLS}")
    print(f"\n{'─'*72}")
    print(f"  {'Season':<8} {'N':>6} {'MAE':>7} {'Bias':>7} {'Pred σ':>8} {'Act σ':>8}")
    print(f"{'─'*72}")

    all_actual = []
    all_predicted = []

    for test_season in CV_TEST_SEASONS:
        train = df[df["season"].isin(range(CV_TRAIN_START, test_season))]
        test  = df[df["season"] == test_season].copy()

        if len(train) < 100 or len(test) < 20:
            continue

        m = LinearRegression()
        m.fit(train[FEATURE_COLS], train["points_scored"])
        preds = m.predict(test[FEATURE_COLS])

        metrics = evaluate_scores(test["points_scored"], preds)
        all_actual.extend(test["points_scored"].tolist())
        all_predicted.extend(preds.tolist())

        print(f"  {test_season:<8} {metrics['n']:>6} {metrics['mae']:>7.2f} "
              f"{metrics['bias']:>+7.2f} {metrics['pred_std']:>8.2f} {metrics['actual_std']:>8.2f}")

        if verbose:
            # Show score distribution
            print(f"           pred range: [{preds.min():.1f}, {preds.max():.1f}]  "
                  f"actual range: [{test['points_scored'].min():.1f}, {test['points_scored'].max():.1f}]")

    print(f"{'─'*72}")
    overall = evaluate_scores(pd.Series(all_actual), np.array(all_predicted))
    print(f"  {'OVERALL':<8} {overall['n']:>6} {overall['mae']:>7.2f} "
          f"{overall['bias']:>+7.2f} {overall['pred_std']:>8.2f} {overall['actual_std']:>8.2f}")

    print(f"\n  Columns: MAE = mean absolute error (points), Bias = avg over/under-prediction,")
    print(f"           Pred σ = std dev of predictions, Act σ = std dev of actual scores")

    # Naive baseline: predict league average every game
    naive_pred = np.full(len(all_actual), np.mean(all_actual))
    naive_mae = mean_absolute_error(all_actual, naive_pred)
    print(f"\n  Naive baseline (always predict mean): MAE = {naive_mae:.2f}")
    print(f"  Model improvement over baseline:      {naive_mae - overall['mae']:+.2f} pts")

    # Show final model coefficients (trained on all TRAIN_SEASONS)
    full_train = df[df["season"].isin(TRAIN_SEASONS)]
    m_final = LinearRegression()
    m_final.fit(full_train[FEATURE_COLS], full_train["points_scored"])
    print(f"\n  Final model (trained on {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)}):")
    for col, coef in zip(FEATURE_COLS, m_final.coef_):
        print(f"    {col}: {coef:+.4f}")
    print(f"    intercept: {m_final.intercept_:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading data...")
    features  = load_features()
    schedules = get_schedule_data(list(range(CV_TRAIN_START, max(CV_TEST_SEASONS) + 1)))

    print("Building dataset...")
    df = build_score_dataset(schedules, features, OFF_COLS, DEF_COLS, include_playoffs=True)
    print(f"  {len(df):,} team-game rows  ({len(df)//2:,} games)")
    print(f"  Score range: {df['points_scored'].min():.0f} – {df['points_scored'].max():.0f} pts")
    print(f"  Mean score: {df['points_scored'].mean():.1f} pts")

    run_cv(df, verbose=args.verbose)


if __name__ == "__main__":
    main()
