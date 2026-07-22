SEASONS = list(range(2015, 2025))
TRAIN_SEASONS = list(range(2015, 2022))
VALIDATION_SEASONS = list(range(2022, 2025))

ROLLING_WINDOW = 5
MIN_WEEK = 4

OFFENSIVE_FEATURES = [
    "off_epa_per_play",
    "off_epa_per_play_q3",
    "off_success_rate_q3",
    "off_points_per_game",
]

DEFENSIVE_FEATURES = [
    "def_epa_per_play",
    "def_epa_per_play_q3",
    "def_success_rate_q3",
    "def_points_per_game",
]

OPPONENT_FEATURES = [
    "opp_off_epa_per_play",
    "opp_off_epa_per_play_q3",
    "opp_off_success_rate_q3",
    "opp_off_points_per_game",
    "opp_def_epa_per_play",
    "opp_def_epa_per_play_q3",
    "opp_def_success_rate_q3",
    "opp_def_points_per_game",
]

CONTEXTUAL_FEATURES = [
    "is_home",
    "rest_advantage",
]

ALL_FEATURES = OFFENSIVE_FEATURES + DEFENSIVE_FEATURES + OPPONENT_FEATURES + CONTEXTUAL_FEATURES

# Game-level features (one row per game, home perspective, no is_home flag needed)
GAME_FEATURES = [f for f in ALL_FEATURES if f != "is_home"]

PERSONNEL_FEATURES = ["home_qb", "away_qb", "home_coach", "away_coach", "rest_advantage"]

RIDGE_ALPHA = 1.0
