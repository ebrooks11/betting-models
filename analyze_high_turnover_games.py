"""Analyze whether high-turnover games (one team with 4+ turnovers) are
disproportionately wrong predictions. Uses the top overall ATS model as reference.
"""

from sklearn.linear_model import LinearRegression
from src.data_loader import get_pbp_data, get_schedule_data
from src.model_utils import load_features, build_dataset, ats_accuracy, TEAM_MAP
from config import SEASONS

TRAIN_SEASONS = range(2016, 2022)
TEST_SEASONS = range(2022, 2026)
TURNOVER_THRESHOLD = 4  # one team commits this many or more

FEATURE_COLS = [
    "off_11_epa_rolling",
    "off_12_epa_rolling",
    "qb_off_11_epa_rolling",
    "qb_off_12_epa_rolling",
    "def_epa_rolling",
    "point_diff_rolling",
    "qbr_rolling",
]

# --- Build model dataset ---
features = load_features()
schedules = get_schedule_data(SEASONS)
df = build_dataset(schedules, features, FEATURE_COLS)

all_cols = FEATURE_COLS + [f"opp_{c}" for c in FEATURE_COLS]
train = df[df["season"].isin(TRAIN_SEASONS)].copy()
test  = df[df["season"].isin(TEST_SEASONS)].copy()

model = LinearRegression()
model.fit(train[all_cols], train["margin"])
train["predicted_margin"] = model.predict(train[all_cols])
test["predicted_margin"]  = model.predict(test[all_cols])

# --- Compute turnovers per team per game from PBP ---
print("Loading PBP data for turnover counts...")
pbp = get_pbp_data(SEASONS)[["season", "week", "posteam", "interception", "fumble_lost"]]
pbp = pbp[pbp["posteam"].notna()]

turnovers = (
    pbp.groupby(["season", "week", "posteam"])[["interception", "fumble_lost"]]
    .sum()
    .reset_index()
)
turnovers["turnovers"] = turnovers["interception"] + turnovers["fumble_lost"]
turnovers["posteam"] = turnovers["posteam"].replace(TEAM_MAP)

high_to = turnovers[turnovers["turnovers"] >= TURNOVER_THRESHOLD][["season", "week", "posteam"]].copy()
high_to_set = set(zip(high_to["season"], high_to["week"], high_to["posteam"]))

def flag_high_to(row):
    home_high = (row["season"], row["week"], row["team"]) in high_to_set
    away_high = (row["season"], row["week"], row["opponent"]) in high_to_set
    return home_high or away_high

for split_name, split in [("Train", train), ("Test", test)]:
    split = split.copy()
    split["high_to"] = split.apply(flag_high_to, axis=1)

    normal = split[~split["high_to"]]
    flagged = split[split["high_to"]]

    n_ats,    n_n,    n_push    = ats_accuracy(normal)
    f_ats,    f_n,    f_push    = ats_accuracy(flagged)
    n_e3,     n_n3,   n_p3      = ats_accuracy(normal,  min_edge=3.0)
    f_e3,     f_n3,   f_p3      = ats_accuracy(flagged, min_edge=3.0)

    print(f"\n{'='*55}")
    print(f"{split_name}  ({len(split)} total games, {len(flagged)} high-TO flagged)")
    print(f"{'='*55}")
    print(f"{'':30s} {'Normal':>10} {'High-TO':>10}")
    print(f"{'Games (non-push)':30s} {n_n:>10} {f_n:>10}")
    print(f"{'ATS accuracy':30s} {n_ats:>9.1%} {f_ats:>9.1%}")
    print(f"{'E>3 games':30s} {n_n3:>10} {f_n3:>10}")
    print(f"{'E>3 ATS accuracy':30s} {n_e3:>9.1%} {f_e3:>9.1%}")

    # Distribution of turnover counts in flagged games
    flagged2 = split[split["high_to"]].copy()
    flagged2["home_to"] = flagged2.apply(
        lambda r: turnovers.query("season==@r.season and week==@r.week and posteam==@r.team")["turnovers"].values[0]
        if len(turnovers.query("season==@r.season and week==@r.week and posteam==@r.team")) else 0, axis=1
    )
    flagged2["away_to"] = flagged2.apply(
        lambda r: turnovers.query("season==@r.season and week==@r.week and posteam==@r.opponent")["turnovers"].values[0]
        if len(turnovers.query("season==@r.season and week==@r.week and posteam==@r.opponent")) else 0, axis=1
    )
    flagged2["max_to"] = flagged2[["home_to", "away_to"]].max(axis=1)
    print(f"\nTurnover distribution (games with max team TOs ≥ {TURNOVER_THRESHOLD}):")
    print(flagged2["max_to"].value_counts().sort_index().to_string())
