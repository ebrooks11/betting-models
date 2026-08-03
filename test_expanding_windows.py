"""
Compare w3 rolling vs season-to-date expanding window for top models.
Tests 3 models side by side.
"""

from sklearn.linear_model import LinearRegression
from src.data_loader import get_schedule_data
from src.model_utils import load_features, build_dataset, evaluate
from config import SEASONS

TRAIN_SEASONS = range(2016, 2022)
TEST_SEASONS = range(2022, 2026)

features = load_features()
schedules = get_schedule_data(SEASONS)

MODELS = {
    "off11/12+qboff+def+pd+qbr (w3)": [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
    ],
    "off11/12+qboff+def+pd+qbr (expanding)": [
        "off_11_epa_expanding", "off_12_epa_expanding",
        "qb_off_11_epa_expanding", "qb_off_12_epa_expanding",
        "def_epa_expanding", "point_diff_expanding", "qbr_expanding",
    ],
    "off11/12+qboff+pd+qbr+TO (w3)": [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "point_diff_rolling", "qbr_rolling", "turnovers_committed_rolling",
    ],
    "off11/12+qboff+pd+qbr+TO (expanding)": [
        "off_11_epa_expanding", "off_12_epa_expanding",
        "qb_off_11_epa_expanding", "qb_off_12_epa_expanding",
        "point_diff_expanding", "qbr_expanding", "turnovers_committed_expanding",
    ],
    "off11/12+qboff+def+pd+qbr+cpoe+def_vs12 (w3)": [
        "off_11_epa_rolling", "off_12_epa_rolling",
        "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
        "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
        "cpoe_rolling", "def_vs_12_epa_rolling",
    ],
    "off11/12+qboff+def+pd+qbr+cpoe+def_vs12 (expanding)": [
        "off_11_epa_expanding", "off_12_epa_expanding",
        "qb_off_11_epa_expanding", "qb_off_12_epa_expanding",
        "def_epa_expanding", "point_diff_expanding", "qbr_expanding",
        "cpoe_expanding", "def_vs_12_epa_expanding",
    ],
}

for name, feat_cols in MODELS.items():
    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    df = build_dataset(schedules, features, feat_cols, include_playoffs=True)
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    all_cols = feat_cols + [f"opp_{c}" for c in feat_cols]
    model = LinearRegression()
    model.fit(train[all_cols], train["margin"])
    train, test = train.copy(), test.copy()
    train["predicted_margin"] = model.predict(train[all_cols])
    test["predicted_margin"] = model.predict(test[all_cols])
    evaluate(train, test, model, feat_cols)
