"""
EPA + rest advantage model: 3-game rolling EPA plus rest advantage.
Train 2015-2021, test 2022-2024.
"""

from sklearn.linear_model import LinearRegression

from src.data_loader import get_schedule_data
from src.model_utils import (
    TRAIN_SEASONS, TEST_SEASONS,
    load_features, build_dataset, evaluate,
)
from config import SEASONS

FEATURE_COLS = ["off_epa_rolling", "def_epa_rolling", "rest_advantage"]

print("Loading data...")
features = load_features()
schedules = get_schedule_data(SEASONS)

print("Building dataset...")
df = build_dataset(schedules, features, FEATURE_COLS)

train = df[df["season"].isin(TRAIN_SEASONS)]
test = df[df["season"].isin(TEST_SEASONS)]

print(f"\nTrain: {len(train)} games ({train['season'].min()}-{train['season'].max()})")
print(f"Test:  {len(test)} games ({test['season'].min()}-{test['season'].max()})")

all_cols = FEATURE_COLS + [f"opp_{c}" for c in FEATURE_COLS]
model = LinearRegression()
model.fit(train[all_cols], train["margin"])

train = train.copy()
test = test.copy()
train["predicted_margin"] = model.predict(train[all_cols])
test["predicted_margin"] = model.predict(test[all_cols])

evaluate(train, test, model, FEATURE_COLS)
