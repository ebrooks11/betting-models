SEASONS = list(range(2015, 2025))
TRAIN_SEASONS = list(range(2015, 2022))
VALIDATION_SEASONS = list(range(2022, 2025))

ROLLING_WINDOW = 5
MIN_WEEK = 4

OFFENSIVE_FEATURES = [
    "off_epa_per_play",
    "off_points_per_game",
]

DEFENSIVE_FEATURES = [
    "def_epa_per_play",
    "def_points_per_game",
]

OPPONENT_FEATURES = []

CONTEXTUAL_FEATURES = [
    "is_home",
    "rest_advantage",
    "win_streak",
    "week",
]

ALL_FEATURES = OFFENSIVE_FEATURES + DEFENSIVE_FEATURES + OPPONENT_FEATURES + CONTEXTUAL_FEATURES

RIDGE_ALPHA = 1.0
