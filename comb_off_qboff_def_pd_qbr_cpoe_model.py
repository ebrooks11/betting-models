"""
Combined team + QB offensive formation EPA + defensive EPA + point differential + QBR + CPOE.
Train 2016-2021, test 2022-2025.
Rank 4 overall ATS: 54.8% (n=513), +23.5u. Rank 2 E>3: 57.0% (n=256), +22.7u.
Best model appearing in both top-5 leaderboards.
"""

from sklearn.linear_model import LinearRegression
from src.data_loader import get_schedule_data
from src.model_utils import load_features, build_dataset, evaluate
from config import SEASONS

TRAIN_SEASONS = range(2016, 2022)
TEST_SEASONS = range(2022, 2026)

FEATURE_COLS = [
    "off_11_epa_rolling",
    "off_12_epa_rolling",
    "qb_off_11_epa_rolling",
    "qb_off_12_epa_rolling",
    "def_epa_rolling",
    "point_diff_rolling",
    "qbr_rolling",
    "cpoe_rolling",
]

features = load_features()
schedules = get_schedule_data(SEASONS)
df = build_dataset(schedules, features, FEATURE_COLS, include_playoffs=True)

train = df[df["season"].isin(TRAIN_SEASONS)]
test = df[df["season"].isin(TEST_SEASONS)]
print(f"\nTrain: {len(train)} games ({train['season'].min()}-{train['season'].max()})")
print(f"Test:  {len(test)} games ({test['season'].min()}-{test['season'].max()})")

all_cols = FEATURE_COLS + [f"opp_{c}" for c in FEATURE_COLS]
model = LinearRegression()
model.fit(train[all_cols], train["margin"])
train, test = train.copy(), test.copy()
train["predicted_margin"] = model.predict(train[all_cols])
test["predicted_margin"] = model.predict(test[all_cols])
evaluate(train, test, model, FEATURE_COLS)
