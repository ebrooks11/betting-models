"""
Combined off_11/12 + qb_off + pd + qbr + turnovers_committed_rolling.
Train 2016-2021, test 2022-2025.
ATS: 55.4% (n=513), +27.3u. E>3: 58.3% (n=259), +29.9u.
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
    "point_diff_rolling",
    "qbr_rolling",
    "turnovers_committed_rolling",
]

features = load_features()
schedules = get_schedule_data(SEASONS)
df = build_dataset(schedules, features, FEATURE_COLS)

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
