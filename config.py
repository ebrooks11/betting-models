SEASONS = list(range(2010, 2026))
TRAIN_SEASONS = list(range(2010, 2020))
VALIDATION_SEASONS = list(range(2020, 2024))

ROLLING_WINDOW = 5
MIN_WEEK = 4

OFFENSIVE_FEATURES = [
    "off_epa_per_play",
]

DEFENSIVE_FEATURES = [
    "def_epa_per_play",
]

OPPONENT_FEATURES = [
    "opp_off_epa_per_play",
    "opp_def_epa_per_play",
]

CONTEXTUAL_FEATURES = [
    "rest_advantage",
]

ALL_FEATURES = OFFENSIVE_FEATURES + DEFENSIVE_FEATURES + OPPONENT_FEATURES + CONTEXTUAL_FEATURES

# Game-level features (one row per game, home perspective, no is_home flag needed)
GAME_FEATURES = [f for f in ALL_FEATURES if f != "is_home"]

PERSONNEL_FEATURES = ["home_qb", "away_qb", "home_coach", "away_coach", "rest_advantage"]

RIDGE_ALPHA = 1.0
