"""
Run all model experiments, update the README leaderboard, and regenerate UI JSON files.

Usage:
    python run_all_models.py                  # run all models + update readme + regenerate UI
    python run_all_models.py --eval-only      # run models and update readme, skip UI regeneration
    python run_all_models.py --ui-only        # skip model eval, just regenerate UI from current best model
"""

import sys
import argparse
import re
import textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import SEASONS
from src.data_loader import get_schedule_data
from src.model_utils import load_features, build_dataset, ats_accuracy

TRAIN_SEASONS = range(2016, 2023)
TEST_SEASONS  = range(2023, 2026)

# ── All models to evaluate ────────────────────────────────────────────────────
# Each entry: (human_label, feature_cols)
ALL_MODELS = [
    # Combined off_11/12 + qb_off family
    ("off_11/12 + qb_off + def + pd + qbr + def_vs_12", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling", "def_vs_12_epa_rolling",
    ]),
    ("off_11/12 + qb_off + def + pd + qbr", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
    ]),
    ("off_11/12 + qb_off + def + pd + qbr + cpoe + def_vs_12", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
        "cpoe_rolling", "def_vs_12_epa_rolling",
    ]),
    ("off_11/12 + qb_off + def + pd + qbr + cpoe", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling", "cpoe_rolling",
    ]),
    ("off_11/12 + qb_off + def + pd + def_vs_12", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "def_vs_12_epa_rolling",
    ]),
    ("off_11/12 + qb_off + def + pd", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling",
    ]),
    ("off_11/12 + qb_off + def + cpoe", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "cpoe_rolling",
    ]),
    ("off_11/12 + qb_off + pd + qbr + def_vs_12", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "point_diff_rolling", "qbr_rolling", "def_vs_12_epa_rolling",
    ]),
    ("off_11/12 + qb_off + pd + qbr + TO", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "point_diff_rolling", "qbr_rolling", "turnovers_committed_rolling",
    ]),
    ("off_11/12 + qb_off + pd + qbr", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "point_diff_rolling", "qbr_rolling",
    ]),
    # w5 variants
    ("off_11/12 (w5) + qb_off + pd + qbr", [
        "off_11_epa_rolling_w5", "off_12_epa_rolling_w5",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "point_diff_rolling", "qbr_rolling",
    ]),
    # QB-only family
    ("qb_off + def + pd + qbr + cpoe", [
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling", "cpoe_rolling",
    ]),
    ("qb_off + def + pd + qbr", [
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
    ]),
    ("qb_off + def + pd", [
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling",
    ]),
    ("qb_off + pd + qbr", [
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "point_diff_rolling", "qbr_rolling",
    ]),
    ("qb_off (w5) + pd + qbr", [
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "point_diff_rolling", "qbr_rolling",
    ]),
    ("qb_weighted_raw + pd + qbr", [
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "qb_epa_rolling", "point_diff_rolling", "qbr_rolling",
    ]),
    ("qb_first_down + pd + qbr", [
        "qb_epa_rolling", "def_epa_first_down_rolling",
        "point_diff_rolling", "qbr_rolling",
    ]),
    ("pd + qb_cpoe + qbr", [
        "point_diff_rolling", "qb_cpoe_rolling", "qbr_rolling",
    ]),
    # Off formation (no qb_off) family
    ("off_11/12 + def + pd + qbr", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
    ]),
    ("off_11/12 + def + pd + qbr + cpoe", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling", "cpoe_rolling",
    ]),
    ("off_11/12 + pd + qbr", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "point_diff_rolling", "qbr_rolling",
    ]),
    ("off_11/12 + def", [
        "off_11_epa_rolling", "off_12_epa_rolling", "def_epa_rolling",
    ]),
    ("off_11/12 + def_vs_11/12 + pd + qbr", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "def_vs_11_epa_rolling", "def_vs_12_epa_rolling",
        "point_diff_rolling", "qbr_rolling",
    ]),
    ("off_11/12 + def_vs_11/12 + pd + qbr (formation)", [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "def_vs_11_epa_rolling", "def_vs_12_epa_rolling",
        "off_epa_first_down_rolling", "point_diff_rolling", "qbr_rolling",
    ]),
    # EPA baseline family
    ("off + def", [
        "off_epa_rolling", "def_epa_rolling",
    ]),
    ("off + def + cpoe", [
        "off_epa_rolling", "def_epa_rolling", "cpoe_rolling",
    ]),
    ("off + def + qbr", [
        "off_epa_rolling", "def_epa_rolling", "qbr_rolling",
    ]),
    ("off + def + rest", [
        "off_epa_rolling", "def_epa_rolling", "rest_advantage",
    ]),
    ("off + def + pace", [
        "off_epa_rolling", "def_epa_rolling", "plays_per_game_rolling",
    ]),
    ("off + def + TOP", [
        "off_epa_rolling", "def_epa_rolling", "top_rolling",
    ]),
    ("off + def + pace + TOP", [
        "off_epa_rolling", "def_epa_rolling", "plays_per_game_rolling", "top_rolling",
    ]),
    ("off + def + no_TO", [
        "off_epa_no_to_rolling", "def_epa_no_to_rolling",
    ]),
    ("off + def + early_down", [
        "off_epa_early_down_rolling", "def_epa_early_down_rolling",
    ]),
    ("off + def + first_down", [
        "off_epa_rolling", "def_epa_rolling",
        "off_epa_first_down_rolling", "def_epa_first_down_rolling",
    ]),
    ("off + def + first_down + pace", [
        "off_epa_rolling", "def_epa_rolling",
        "off_epa_first_down_rolling", "def_epa_first_down_rolling",
        "plays_per_game_rolling",
    ]),
    ("off + def + off_first_down + pace", [
        "off_epa_rolling", "def_epa_rolling",
        "off_epa_first_down_rolling", "plays_per_game_rolling",
    ]),
    ("off + def + first_down + pace + qbr", [
        "off_epa_rolling", "def_epa_rolling",
        "off_epa_first_down_rolling", "def_epa_first_down_rolling",
        "plays_per_game_rolling", "qbr_rolling",
    ]),
    ("off + def + qbr + first_down + pace", [
        "off_epa_rolling", "def_epa_rolling", "qbr_rolling",
        "off_epa_first_down_rolling", "plays_per_game_rolling",
    ]),
    ("off + def + sack_rate", [
        "off_epa_rolling", "def_epa_rolling", "sack_rate_rolling",
    ]),
    ("off + def + qb_hit_rate", [
        "off_epa_rolling", "def_epa_rolling", "qb_hit_rate_rolling",
    ]),
    ("off + def + stuff_rate", [
        "off_epa_rolling", "def_epa_rolling", "stuff_rate_rolling",
    ]),
    ("pd + cpoe + qbr", [
        "point_diff_rolling", "cpoe_rolling", "qbr_rolling",
    ]),
]


def _units(w, l):
    return w / 1.1 - l


def run_all_models(features, schedules):
    """Train and evaluate every model. Returns list of result dicts sorted by overall ATS."""
    results = []
    seen_features = set()

    for label, feat_cols in ALL_MODELS:
        key = tuple(sorted(feat_cols))
        if key in seen_features:
            continue
        seen_features.add(key)

        try:
            df = build_dataset(schedules, features, feat_cols, include_playoffs=True)
        except Exception as e:
            print(f"  SKIP {label}: {e}")
            continue

        train = df[df["season"].isin(TRAIN_SEASONS)]
        test  = df[df["season"].isin(TEST_SEASONS)]
        if len(train) < 50 or len(test) < 20:
            print(f"  SKIP {label}: insufficient data (train={len(train)}, test={len(test)})")
            continue

        all_cols = feat_cols + [f"opp_{c}" for c in feat_cols]
        # Only keep cols that exist
        all_cols = [c for c in all_cols if c in df.columns]
        if len(all_cols) < 2:
            print(f"  SKIP {label}: missing feature columns")
            continue

        m = LinearRegression()
        m.fit(train[all_cols], train["margin"])
        test = test.copy()
        test["predicted_margin"] = m.predict(test[all_cols])

        ats, n, _   = ats_accuracy(test)
        e3,  n3, _  = ats_accuracy(test, min_edge=3.0)

        covered  = test["margin"] - test["spread_line"]
        pick     = test["predicted_margin"] - test["spread_line"]
        push_m   = covered == 0
        hit_m    = (pick > 0) == (covered > 0)
        wins     = int(hit_m[~push_m].sum())
        losses   = int((~hit_m)[~push_m].sum())

        e3_mask  = pick.abs() >= 3.0
        hit3     = hit_m[e3_mask & ~push_m]
        w3       = int(hit3.sum())
        l3       = int((~hit3).sum())

        results.append(dict(
            label=label,
            features=feat_cols,
            n=int(n), ats=ats, wins=wins, losses=losses, units=_units(wins, losses),
            n3=int(n3), e3=e3, w3=w3, l3=l3, units3=_units(w3, l3),
        ))
        print(f"  {label}: {ats:.1%} n={n} {_units(wins,losses):+.1f}u | E>3: {e3:.1%} n3={n3} {_units(w3,l3):+.1f}u")

    return results


def _feat_sub(feat_cols):
    return "<sub>`" + "`, `".join(feat_cols) + "`</sub>"


def build_leaderboard_tables(results):
    """Return (top10_ats_table, top10_e3_table, best_model_key) markdown strings."""

    by_ats = sorted(results, key=lambda r: (r["ats"], r["units"]), reverse=True)
    by_e3  = sorted(results, key=lambda r: (r["e3"],  r["units3"]), reverse=True)

    def table_rows(ranked):
        rows = []
        for i, r in enumerate(ranked[:10], 1):
            bold_ats   = "**" if i == 1 else ""
            bold_units = "**" if i == 1 else ""
            bold_e3    = "**" if r["e3"] == ranked[0]["e3"] else ""
            bold_u3    = "**" if r["units3"] == ranked[0]["units3"] else ""
            rows.append(
                f"| {i} | {r['label']}<br>{_feat_sub(r['features'])} "
                f"| {r['n']} | {bold_ats}{r['ats']:.1%}{bold_ats} "
                f"| {bold_units}{r['units']:+.1f}{bold_units} "
                f"| {bold_e3}{r['e3']:.1%}{bold_e3} "
                f"| {r['n3']} "
                f"| {bold_u3}{r['units3']:+.1f}{bold_u3} |"
            )
        return "\n".join(rows)

    header = (
        "| Rank | Model | N | ATS | Units | E>3 ATS | E>3 N | E>3 Units |\n"
        "|------|-------|---|-----|-------|---------|-------|-----------|"
    )

    ats_table = header + "\n" + table_rows(by_ats)
    e3_table  = header + "\n" + table_rows(by_e3)

    best     = by_ats[0]
    best_e3  = by_e3[0]
    return ats_table, e3_table, best, best_e3


def update_readme(ats_table, e3_table, best):
    readme = Path("README.md")
    text = readme.read_text()

    # Replace the two leaderboard blocks between their headings and the Notable note
    ats_section = (
        "### Top 10 by Overall ATS\n\n"
        "Includes playoff games (WC, DIV, CON, SB). Train 2016–2022, test 2023–2025.\n\n"
        + ats_table
    )
    e3_section = (
        "### Top 10 by E>3 ATS (selective high-confidence bets)\n\n"
        + e3_table
    )

    notable = (
        f"\n\n**Notable:** Overall #1 is `{best['label']}` at "
        f"{best['ats']:.1%} / {best['units']:+.1f}u "
        f"(E>3: {best['e3']:.1%} / {best['units3']:+.1f}u, n={best['n3']})."
    )

    # Replace ATS table block
    text = re.sub(
        r"### Top 10 by Overall ATS.*?(?=### Top 10 by E>3 ATS)",
        ats_section + "\n\n",
        text,
        flags=re.DOTALL,
    )

    # Replace E>3 table block (up to but not including the next ### heading or end of Notable note)
    text = re.sub(
        r"### Top 10 by E>3 ATS \(selective high-confidence bets\).*?(?=\n### |\Z)",
        e3_section + notable + "\n\n",
        text,
        flags=re.DOTALL,
    )

    readme.write_text(text)
    print(f"\nREADME updated. Best model: {best['label']} — {best['ats']:.1%} / {best['units']:+.1f}u")


def regenerate_ui(best, best_e3):
    """Patch predictions_generator MODELS with the two best models and regenerate all JSONs."""
    import src.predictions_generator as pg
    from src.predictions_generator import generate_all

    pg.MODELS = {
        "model_overall": {
            "label": best["label"],
            "features": best["features"],
        },
        "model_e3": {
            "label": best_e3["label"],
            "features": best_e3["features"],
        },
    }

    print("\nRegenerating UI JSON files...")
    generate_all(out_dir="docs/data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="Run models and update README only; skip UI regeneration")
    parser.add_argument("--ui-only",   action="store_true", help="Skip model eval; regenerate UI with current best model")
    args = parser.parse_args()

    if args.ui_only:
        print("Regenerating UI only (skipping model evaluation)...")
        from src.predictions_generator import generate_all
        generate_all(out_dir="docs/data")
        return

    print("Loading data...")
    features  = load_features()
    schedules = get_schedule_data(SEASONS)

    print(f"\nEvaluating {len(ALL_MODELS)} models...\n")
    results = run_all_models(features, schedules)

    if not results:
        print("No results — check feature availability.")
        return

    ats_table, e3_table, best, best_e3 = build_leaderboard_tables(results)
    update_readme(ats_table, e3_table, best)

    if not args.eval_only:
        regenerate_ui(best, best_e3)
        print("\nDone. Commit docs/data/ and README.md to publish.")
    else:
        print("\nDone. README updated. Run without --eval-only to also regenerate UI JSON files.")


if __name__ == "__main__":
    main()
