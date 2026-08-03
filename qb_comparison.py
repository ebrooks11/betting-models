"""Compare top-10 team-level models vs QB-feature equivalents vs combined (team+QB)."""

from sklearn.linear_model import LinearRegression
from src.data_loader import get_schedule_data
from src.model_utils import load_features, build_dataset, ats_accuracy
from config import SEASONS

TRAIN_2016 = range(2016, 2022)
TRAIN_2006 = range(2006, 2022)
TEST = range(2022, 2026)

features = load_features()
schedules = get_schedule_data(SEASONS)


def run_model(feature_cols, train_seasons=TRAIN_2016, label=""):
    df = build_dataset(schedules, features, feature_cols)
    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"].isin(TEST)]
    all_cols = feature_cols + [f"opp_{c}" for c in feature_cols]
    m = LinearRegression()
    m.fit(train[all_cols], train["margin"])
    test = test.copy()
    test["predicted_margin"] = m.predict(test[all_cols])

    ats, n, push = ats_accuracy(test)
    wins = (test["predicted_margin"] - test["spread_line"] > 0) == (test["margin"] - test["spread_line"] > 0)
    wins = wins[~((test["margin"] - test["spread_line"]) == 0)]
    n_wins = wins.sum()
    n_losses = (~wins).sum()
    units = n_wins * (1 / 1.1) - n_losses

    e3, n3, p3 = ats_accuracy(test, min_edge=3.0)
    mask3 = (test["predicted_margin"] - test["spread_line"]).abs() >= 3.0
    t3 = test[mask3]
    wins3 = (t3["predicted_margin"] - t3["spread_line"] > 0) == (t3["margin"] - t3["spread_line"] > 0)
    wins3 = wins3[~((t3["margin"] - t3["spread_line"]) == 0)]
    u3 = wins3.sum() * (1 / 1.1) - (~wins3).sum()

    return {"label": label, "n": n, "ats": ats, "units": units, "e3_ats": e3, "e3_n": n3, "e3_units": u3}


# --- Top-10 team-level model definitions ---
TEAM_MODELS = [
    ("weighted+raw + pd + qbr",
     ["off_11_weighted_epa_rolling", "off_12_weighted_epa_rolling", "off_epa_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("off_11/12 + pd + qbr",
     ["off_11_epa_rolling", "off_12_epa_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("off_11/12 + def + pd + qbr",
     ["off_11_epa_rolling", "off_12_epa_rolling", "def_epa_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("off_11/12 + def + pd + qbr + cpoe",
     ["off_11_epa_rolling", "off_12_epa_rolling", "def_epa_rolling", "point_diff_rolling", "qbr_rolling", "cpoe_rolling"],
     TRAIN_2016),
    ("off_11/12 + def + cpoe",
     ["off_11_epa_rolling", "off_12_epa_rolling", "def_epa_rolling", "cpoe_rolling"],
     TRAIN_2016),
    ("off_11/12_w5 + pd + qbr",
     ["off_11_epa_rolling_w5", "off_12_epa_rolling_w5", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("off_11/12 + def + pd",
     ["off_11_epa_rolling", "off_12_epa_rolling", "def_epa_rolling", "point_diff_rolling"],
     TRAIN_2016),
    ("pd + cpoe + qbr",
     ["point_diff_rolling", "cpoe_rolling", "qbr_rolling"],
     TRAIN_2006),
    ("off_11/12 + def",
     ["off_11_epa_rolling", "off_12_epa_rolling", "def_epa_rolling"],
     TRAIN_2016),
    ("first_down + pd + qbr",
     ["off_epa_first_down_rolling", "def_epa_first_down_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
]

# --- QB-feature equivalents (swap team off_11/12 → qb_off_11/12, team cpoe/qbr → qb_cpoe/qbr) ---
QB_MODELS = [
    ("QB: weighted+raw + pd + qbr",
     ["qb_off_11_epa_rolling", "qb_off_12_epa_rolling", "qb_epa_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("QB: off_11/12 + pd + qbr",
     ["qb_off_11_epa_rolling", "qb_off_12_epa_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("QB: off_11/12 + def + pd + qbr",
     ["qb_off_11_epa_rolling", "qb_off_12_epa_rolling", "def_epa_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("QB: off_11/12 + def + pd + qbr + cpoe",
     ["qb_off_11_epa_rolling", "qb_off_12_epa_rolling", "def_epa_rolling", "point_diff_rolling", "qbr_rolling", "qb_cpoe_rolling"],
     TRAIN_2016),
    ("QB: off_11/12 + def + cpoe",
     ["qb_off_11_epa_rolling", "qb_off_12_epa_rolling", "def_epa_rolling", "qb_cpoe_rolling"],
     TRAIN_2016),
    ("QB: off_11/12_w5 + pd + qbr",
     ["qb_off_11_epa_rolling", "qb_off_12_epa_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("QB: off_11/12 + def + pd",
     ["qb_off_11_epa_rolling", "qb_off_12_epa_rolling", "def_epa_rolling", "point_diff_rolling"],
     TRAIN_2016),
    ("QB: pd + qb_cpoe + qbr",
     ["point_diff_rolling", "qb_cpoe_rolling", "qbr_rolling"],
     TRAIN_2006),
    ("QB: off_11/12 + def",
     ["qb_off_11_epa_rolling", "qb_off_12_epa_rolling", "def_epa_rolling"],
     TRAIN_2016),
    ("QB: first_down + pd + qbr",
     ["qb_epa_rolling", "def_epa_first_down_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
]

# --- Combined (team + QB) models: add QB offensive columns alongside team columns ---
COMBINED_MODELS = [
    ("COMB: weighted+raw + qb_off + pd + qbr",
     ["off_11_weighted_epa_rolling", "off_12_weighted_epa_rolling", "off_epa_rolling",
      "qb_off_11_epa_rolling", "qb_off_12_epa_rolling", "qb_epa_rolling",
      "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("COMB: off_11/12 + qb_off_11/12 + pd + qbr",
     ["off_11_epa_rolling", "off_12_epa_rolling",
      "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
      "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("COMB: off_11/12 + qb_off + def + pd + qbr",
     ["off_11_epa_rolling", "off_12_epa_rolling",
      "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
      "def_epa_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("COMB: off_11/12 + qb_off + def + pd + qbr + cpoe",
     ["off_11_epa_rolling", "off_12_epa_rolling",
      "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
      "def_epa_rolling", "point_diff_rolling", "qbr_rolling", "cpoe_rolling"],
     TRAIN_2016),
    ("COMB: off_11/12 + qb_off + def + cpoe",
     ["off_11_epa_rolling", "off_12_epa_rolling",
      "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
      "def_epa_rolling", "cpoe_rolling"],
     TRAIN_2016),
    ("COMB: off_11/12_w5 + qb_off + pd + qbr",
     ["off_11_epa_rolling_w5", "off_12_epa_rolling_w5",
      "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
      "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
    ("COMB: off_11/12 + qb_off + def + pd",
     ["off_11_epa_rolling", "off_12_epa_rolling",
      "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
      "def_epa_rolling", "point_diff_rolling"],
     TRAIN_2016),
    ("COMB: pd + cpoe + qb_cpoe + qbr",
     ["point_diff_rolling", "cpoe_rolling", "qb_cpoe_rolling", "qbr_rolling"],
     TRAIN_2006),
    ("COMB: off_11/12 + qb_off + def",
     ["off_11_epa_rolling", "off_12_epa_rolling",
      "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
      "def_epa_rolling"],
     TRAIN_2016),
    ("COMB: first_down + qb_epa + pd + qbr",
     ["off_epa_first_down_rolling", "def_epa_first_down_rolling",
      "qb_epa_rolling", "point_diff_rolling", "qbr_rolling"],
     TRAIN_2016),
]

print("Running team-level models...")
team_results = [run_model(f, t, l) for l, f, t in TEAM_MODELS]

print("Running QB-feature models...")
qb_results = [run_model(f, t, l) for l, f, t in QB_MODELS]

print("Running combined (team+QB) models...")
comb_results = [run_model(f, t, l) for l, f, t in COMBINED_MODELS]

# Print comparison table
HDR = f"{'Model':<48} {'N':>5} {'ATS':>7} {'Units':>8} {'E>3 ATS':>9} {'E>3 N':>7} {'E>3 Units':>10}"
SEP = "-" * len(HDR)

for title, results in [
    ("TEAM-LEVEL MODELS (top 10)", team_results),
    ("QB-FEATURE EQUIVALENTS", qb_results),
    ("COMBINED (team + QB features)", comb_results),
]:
    print("\n" + "=" * len(HDR))
    print(title)
    print(SEP)
    print(HDR)
    print(SEP)
    for r in results:
        print(f"{r['label']:<48} {r['n']:>5} {r['ats']:>7.1%} {r['units']:>+8.1f} {r['e3_ats']:>9.1%} {r['e3_n']:>7} {r['e3_units']:>+10.1f}")

print("\n" + "=" * len(HDR))
print("DELTA vs TEAM BASELINE")
print(SEP)
print(f"{'Model':<34} {'src':>6} {'ΔATS':>7} {'ΔUnits':>8} {'ΔE>3 ATS':>10} {'ΔE>3 Units':>12}")
print(SEP)
for t, q, c in zip(team_results, qb_results, comb_results):
    short = t['label'][:34]
    print(f"{short:<34} {'QB':>6} {q['ats']-t['ats']:>+7.1%} {q['units']-t['units']:>+8.1f} {q['e3_ats']-t['e3_ats']:>+10.1%} {q['e3_units']-t['e3_units']:>+12.1f}")
    print(f"{'':<34} {'COMB':>6} {c['ats']-t['ats']:>+7.1%} {c['units']-t['units']:>+8.1f} {c['e3_ats']-t['e3_ats']:>+10.1%} {c['e3_units']-t['e3_units']:>+12.1f}")
    print(SEP)
