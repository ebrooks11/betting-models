"""
Walk-forward cross-validation for NFL margin prediction models.

For each test season in CV_TEST_SEASONS, trains on all prior seasons
and evaluates on that single season. Reports per-fold and averaged
ATS accuracy and units so we can see which models are consistently
good vs. lucky in a single window.

Usage:
    python cross_validate.py                   # top models from run_all_models
    python cross_validate.py --all             # every model in ALL_MODELS
    python cross_validate.py --min-seasons 3   # only show models with N+ valid folds
"""

import argparse
from collections import defaultdict

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import get_schedule_data
from src.model_utils import load_features, build_dataset, ats_accuracy, TRAIN_SEASONS
from run_all_models import ALL_MODELS

# Walk-forward folds: each entry is the test season; training = all prior seasons
# Starting from 2020 so each fold has at least 4 training seasons
CV_TEST_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
CV_TRAIN_START  = 2016  # earliest season to include in training

# Focused model list (top performers + interesting variants)
FOCUS_MODELS = [
    ("off_11/12 + qb_off + def + cpoe",           ["off_11_epa_rolling","off_12_epa_rolling","qb_off_11_epa_rolling","qb_off_12_epa_rolling","def_epa_rolling","cpoe_rolling"]),
    ("off_11/12 + qb_off + def + pd + qbr + def_vs_12", ["off_11_epa_rolling","off_12_epa_rolling","qb_off_11_epa_rolling","qb_off_12_epa_rolling","def_epa_rolling","point_diff_rolling","qbr_rolling","def_vs_12_epa_rolling"]),
    ("off_11/12 + qb_off + def + pd",             ["off_11_epa_rolling","off_12_epa_rolling","qb_off_11_epa_rolling","qb_off_12_epa_rolling","def_epa_rolling","point_diff_rolling"]),
    ("qb_off + def + pd",                          ["qb_off_11_epa_rolling","qb_off_12_epa_rolling","def_epa_rolling","point_diff_rolling"]),
    ("off_11/12 + qb_off + pd + qbr + TO",        ["off_11_epa_rolling","off_12_epa_rolling","qb_off_11_epa_rolling","qb_off_12_epa_rolling","point_diff_rolling","qbr_rolling","turnovers_committed_rolling"]),
    ("off_11/12 + qb_off + def + pd + qbr",       ["off_11_epa_rolling","off_12_epa_rolling","qb_off_11_epa_rolling","qb_off_12_epa_rolling","def_epa_rolling","point_diff_rolling","qbr_rolling"]),
    ("off_11/12 + qb_off + def + cpoe + pd",      ["off_11_epa_rolling","off_12_epa_rolling","qb_off_11_epa_rolling","qb_off_12_epa_rolling","def_epa_rolling","cpoe_rolling","point_diff_rolling"]),
    ("off_11/12 + def_vs_11/12 + pd + qbr",       ["off_11_epa_rolling","off_12_epa_rolling","def_vs_11_epa_rolling","def_vs_12_epa_rolling","point_diff_rolling","qbr_rolling"]),
]


def _units(w, l):
    return w / 1.1 - l


def cv_model(df, feat_cols, min_seasons=2):
    """
    Run walk-forward CV for a single feature set.
    Returns list of per-fold dicts, or empty list if insufficient data.
    """
    all_cols = feat_cols + [f"opp_{c}" for c in feat_cols]
    all_cols = [c for c in all_cols if c in df.columns]
    if len(all_cols) < 2:
        return []

    folds = []
    for test_season in CV_TEST_SEASONS:
        train_seasons = range(CV_TRAIN_START, test_season)
        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"] == test_season].copy()

        if len(train) < 50 or len(test) < 10:
            continue

        m = LinearRegression()
        m.fit(train[all_cols], train["margin"])
        test["predicted_margin"] = m.predict(test[all_cols])

        ats, n, push   = ats_accuracy(test)
        e3,  n3, push3 = ats_accuracy(test, min_edge=3.0)

        covered = test["margin"] - test["spread_line"]
        pick    = test["predicted_margin"] - test["spread_line"]
        push_m  = covered == 0
        hit_m   = (pick > 0) == (covered > 0)
        wins    = int(hit_m[~push_m].sum())
        losses  = int((~hit_m)[~push_m].sum())

        e3_mask = pick.abs() >= 3.0
        hit3    = hit_m[e3_mask & ~push_m]
        w3      = int(hit3.sum())
        l3      = int((~hit3).sum())

        folds.append(dict(
            season=test_season,
            train_n=len(train), test_n=int(n),
            ats=ats, wins=wins, losses=losses, units=_units(wins, losses),
            e3=e3, n3=int(n3), w3=w3, l3=l3, units3=_units(w3, l3),
        ))

    return folds


def summarise(folds):
    """Average metrics across folds."""
    if not folds:
        return None
    keys = ["ats", "units", "e3", "units3", "n3"]
    out = {k: sum(f[k] for f in folds) / len(folds) for k in keys}
    out["n_folds"]    = len(folds)
    out["total_wins"] = sum(f["wins"] for f in folds)
    out["total_losses"] = sum(f["losses"] for f in folds)
    out["total_w3"]   = sum(f["w3"] for f in folds)
    out["total_l3"]   = sum(f["l3"] for f in folds)
    out["total_units"]  = _units(out["total_wins"], out["total_losses"])
    out["total_units3"] = _units(out["total_w3"], out["total_l3"])
    return out


def print_fold_table(label, folds):
    header = f"  {'Season':<8} {'Train N':>8} {'Test N':>7} {'ATS':>7} {'Units':>7} | {'E>3 ATS':>8} {'E>3 N':>6} {'E>3 u':>7}"
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print('─'*70)
    print(header)
    for f in folds:
        print(f"  {f['season']:<8} {f['train_n']:>8} {f['test_n']:>7} "
              f"{f['ats']:>7.1%} {f['units']:>+7.1f} | "
              f"{f['e3']:>8.1%} {f['n3']:>6} {f['units3']:>+7.1f}")


def print_summary_table(results):
    """Print a ranked summary across all models."""
    # Sort by avg ATS
    ranked = sorted(results, key=lambda r: r["summary"]["ats"], reverse=True)

    print(f"\n{'═'*90}")
    print(f"  WALK-FORWARD CV SUMMARY  (test seasons: {', '.join(str(s) for s in CV_TEST_SEASONS)})")
    print(f"{'═'*90}")
    print(f"  {'Model':<44} {'Folds':>5} {'Avg ATS':>8} {'Total u':>8} | {'Avg E>3':>8} {'Tot E>3u':>9}")
    print(f"  {'─'*44} {'─'*5} {'─'*8} {'─'*8}   {'─'*8} {'─'*9}")

    for r in ranked:
        s = r["summary"]
        print(f"  {r['label']:<44} {s['n_folds']:>5} {s['ats']:>8.1%} {s['total_units']:>+8.1f} | "
              f"{s['e3']:>8.1%} {s['total_units3']:>+9.1f}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",          action="store_true", help="Run all models in ALL_MODELS")
    parser.add_argument("--verbose",      action="store_true", help="Print per-fold breakdown for each model")
    parser.add_argument("--min-seasons",  type=int, default=2, help="Require at least N valid folds (default 2)")
    args = parser.parse_args()

    print("Loading data…")
    features  = load_features()
    all_seasons = list(range(CV_TRAIN_START, max(CV_TEST_SEASONS) + 1))
    schedules = get_schedule_data(all_seasons)

    models = ALL_MODELS if args.all else FOCUS_MODELS

    print(f"\nRunning walk-forward CV on {len(models)} models "
          f"(test seasons: {', '.join(str(s) for s in CV_TEST_SEASONS)})…\n")

    all_results = []
    seen = set()

    for label, feat_cols in models:
        key = tuple(sorted(feat_cols))
        if key in seen:
            continue
        seen.add(key)

        try:
            df = build_dataset(schedules, features, feat_cols, include_playoffs=True)
        except Exception as e:
            print(f"  SKIP {label}: {e}")
            continue

        folds = cv_model(df, feat_cols)
        if len(folds) < args.min_seasons:
            print(f"  SKIP {label}: only {len(folds)} valid folds")
            continue

        summary = summarise(folds)
        all_results.append(dict(label=label, features=feat_cols, folds=folds, summary=summary))

        if args.verbose:
            print_fold_table(label, folds)
        else:
            s = summary
            print(f"  {label}: avg {s['ats']:.1%} / {s['total_units']:+.1f}u total "
                  f"| E>3: {s['e3']:.1%} / {s['total_units3']:+.1f}u  ({s['n_folds']} folds)")

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
