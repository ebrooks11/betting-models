SEASONS = list(range(2015, 2025))
TRAIN_SEASONS = list(range(2015, 2022))
VALIDATION_SEASONS = list(range(2022, 2025))

ROLLING_WINDOW = 5
MIN_WEEK = 4

OFFENSIVE_FEATURES = [
    "off_success_rate",
    "off_points_per_game",
]

DEFENSIVE_FEATURES = [
    "def_success_rate",
    "def_points_per_game",
]

OPPONENT_FEATURES = [
    "opp_off_success_rate",
    "opp_off_points_per_game",
    "opp_def_success_rate",
    "opp_def_points_per_game",
]

CONTEXTUAL_FEATURES = [
    "is_home",
    "rest_advantage",
    "win_streak",
    "week",
]

ALL_FEATURES = OFFENSIVE_FEATURES + DEFENSIVE_FEATURES + OPPONENT_FEATURES + CONTEXTUAL_FEATURES

RIDGE_ALPHA = 1.0
