"""Tests for feature engineering with synthetic data."""

import pandas as pd
import numpy as np
import pytest

from src.feature_engineering import compute_game_stats, build_rolling_features, add_scores_and_context


def make_synthetic_pbp(n_weeks=6, season=2023):
    """Create minimal play-by-play data for two teams over n weeks."""
    rows = []
    teams = [("KC", "LV"), ("LV", "KC")]

    for week in range(1, n_weeks + 1):
        home, away = teams[week % 2]
        for play_num in range(20):
            for posteam, defteam in [(home, away), (away, home)]:
                rows.append({
                    "season": season,
                    "week": week,
                    "play_id": play_num,
                    "play_type": "pass" if play_num % 2 == 0 else "run",
                    "posteam": posteam,
                    "defteam": defteam,
                    "epa": np.random.normal(0, 1),
                    "success": int(np.random.random() > 0.5),
                    "yards_gained": np.random.normal(5, 3),
                    "interception": int(np.random.random() > 0.95),
                    "fumble_lost": int(np.random.random() > 0.95),
                    "down": np.random.choice([1, 2, 3, 4]),
                    "third_down_converted": int(np.random.random() > 0.6),
                })
    return pd.DataFrame(rows)


def make_synthetic_schedules(n_weeks=6, season=2023):
    """Create minimal schedule data for two teams."""
    rows = []
    teams = [("KC", "LV"), ("LV", "KC")]

    for week in range(1, n_weeks + 1):
        home, away = teams[week % 2]
        rows.append({
            "season": season,
            "week": week,
            "game_type": "REG",
            "home_team": home,
            "away_team": away,
            "home_score": np.random.randint(14, 35),
            "away_score": np.random.randint(14, 35),
            "spread_line": np.random.uniform(-7, 7),
            "total_line": np.random.uniform(40, 55),
            "gameday": f"{season}-09-{7 * week + 3:02d}",
        })
    return pd.DataFrame(rows)


class TestComputeGameStats:
    def test_returns_both_teams(self):
        pbp = make_synthetic_pbp()
        stats = compute_game_stats(pbp)
        assert set(stats["team"].unique()) == {"KC", "LV"}

    def test_has_offensive_columns(self):
        pbp = make_synthetic_pbp()
        stats = compute_game_stats(pbp)
        assert "off_epa_per_play" in stats.columns
        assert "off_success_rate" in stats.columns
        assert "off_third_down_rate" in stats.columns

    def test_has_defensive_columns(self):
        pbp = make_synthetic_pbp()
        stats = compute_game_stats(pbp)
        assert "def_epa_per_play" in stats.columns
        assert "def_success_rate" in stats.columns
        assert "def_takeaways" in stats.columns

    def test_one_row_per_team_per_week(self):
        pbp = make_synthetic_pbp(n_weeks=4)
        stats = compute_game_stats(pbp)
        counts = stats.groupby(["season", "week", "team"]).size()
        assert (counts == 1).all()


class TestAddScoresAndContext:
    def test_home_away_flag(self):
        pbp = make_synthetic_pbp()
        schedules = make_synthetic_schedules()
        stats = compute_game_stats(pbp)
        df = add_scores_and_context(stats, schedules)
        assert set(df["is_home"].unique()) == {0, 1}

    def test_has_rest_days(self):
        pbp = make_synthetic_pbp()
        schedules = make_synthetic_schedules()
        stats = compute_game_stats(pbp)
        df = add_scores_and_context(stats, schedules)
        assert "rest_days" in df.columns
        assert df["rest_days"].notna().all()

    def test_has_win_streak(self):
        pbp = make_synthetic_pbp()
        schedules = make_synthetic_schedules()
        stats = compute_game_stats(pbp)
        df = add_scores_and_context(stats, schedules)
        assert "win_streak" in df.columns


class TestBuildRollingFeatures:
    def test_no_data_leakage(self):
        """First game of each team should use NaN (no prior data), not its own stats."""
        pbp = make_synthetic_pbp(n_weeks=6)
        schedules = make_synthetic_schedules(n_weeks=6)
        stats = compute_game_stats(pbp)
        df = add_scores_and_context(stats, schedules)
        df = build_rolling_features(df, window=3)

        for team in df["team"].unique():
            team_df = df[df["team"] == team].sort_values(["season", "week"])
            first_game = team_df.iloc[0]
            assert pd.isna(first_game["off_points_per_game"])

    def test_rolling_values_exist(self):
        pbp = make_synthetic_pbp(n_weeks=6)
        schedules = make_synthetic_schedules(n_weeks=6)
        stats = compute_game_stats(pbp)
        df = add_scores_and_context(stats, schedules)
        df = build_rolling_features(df, window=3)

        # After a few games, rolling values should be populated
        for team in df["team"].unique():
            team_df = df[df["team"] == team].sort_values(["season", "week"])
            later_games = team_df.iloc[2:]
            assert later_games["off_epa_per_play"].notna().all()
