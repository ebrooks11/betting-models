SEASONS = list(range(2015, 2025))
TRAIN_SEASONS = list(range(2015, 2020))
VALIDATION_SEASONS = list(range(2020, 2025))

ROLLING_WINDOW = 5

OFFENSIVE_FEATURES = [
    "off_epa_per_play",
    "off_points_per_game",
    "off_turnovers_per_game",
]

DEFENSIVE_FEATURES = [
    "def_epa_per_play",
    "def_points_per_game",
    "def_takeaways_per_game",
]

OPPONENT_FEATURES = [
    "opp_off_epa_per_play",
    "opp_off_points_per_game",
    "opp_off_turnovers_per_game",
    "opp_def_epa_per_play",
    "opp_def_points_per_game",
    "opp_def_takeaways_per_game",
]

CONTEXTUAL_FEATURES = [
    "is_home",
    "rest_days",
    "win_streak",
    "week",
]

ALL_FEATURES = OFFENSIVE_FEATURES + DEFENSIVE_FEATURES + OPPONENT_FEATURES + CONTEXTUAL_FEATURES

RIDGE_ALPHA = 1.0
