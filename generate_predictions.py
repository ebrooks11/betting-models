"""Generate predictions JSON for the UI. Run this before the season each week.

Usage:
    python generate_predictions.py
    python generate_predictions.py --out docs/data/predictions.json
"""

import argparse
from src.predictions_generator import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/data/predictions.json")
    args = parser.parse_args()
    generate(out_path=args.out)
