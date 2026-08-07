"""Generate predictions JSON for the UI. Run this before the season each week.

Usage:
    python generate_predictions.py                          # generate all weeks
    python generate_predictions.py --season 2025 --week 12 # single week
    python generate_predictions.py --out docs/data/predictions.json  # legacy single-file
"""

import argparse
from src.predictions_generator import generate, generate_all, generate_week, _train_models, load_features
from src.data_loader import get_schedule_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Legacy: write single predictions.json")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--out-dir", default="docs/data")
    args = parser.parse_args()

    if args.out:
        generate(out_path=args.out)
    elif args.season and args.week:
        from config import SEASONS as ALL_SEASONS
        schedules = get_schedule_data(ALL_SEASONS)
        features = load_features()
        print("Training models...")
        trained = _train_models(schedules, features)
        generate_week(args.season, args.week, schedules, features, trained, args.out_dir)
        print(f"Wrote {args.season} week {args.week} → {args.out_dir}/{args.season}/week_{args.week}.json")
    else:
        generate_all(out_dir=args.out_dir)
