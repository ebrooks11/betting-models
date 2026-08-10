"""Generate predictions for upcoming NFL games using the top 3 models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

from src.data_loader import get_schedule_data
from src.model_utils import TEAM_MAP, load_features, build_dataset

TRAIN_SEASONS = range(2016, 2022)
TEST_SEASONS = range(2022, 2026)

MODELS = {
    "off+def+pd+qbr+def_vs12": {
        "label": "Model",
        "features": [
            "off_11_epa_rolling", "off_12_epa_rolling",
            "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
            "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
            "def_vs_12_epa_rolling",
        ],
    },
    "off+def+pd+qbr+cpoe+def_vs12": {
        "label": "E>3 Model",
        "features": [
            "off_11_epa_rolling", "off_12_epa_rolling",
            "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
            "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
            "cpoe_rolling", "def_vs_12_epa_rolling",
        ],
    },
}

# All stats to surface in the UI per team
DISPLAY_STATS = [
    ("off_11_epa_rolling",          "Off EPA 11",                   "epa"),
    ("off_12_epa_rolling",          "Off EPA 12",                   "epa"),
    ("off_epa_rolling",             "Overall OFF EPA",              "epa"),
    ("def_epa_rolling",             "EPA Allowed",                  "epa_inv"),
    ("def_vs_12_epa_rolling",       "Def vs 12",                    "epa_inv"),
    ("qbr_rolling",                 "QBR",                          "qbr"),
    ("point_diff_rolling",          "Pt. Diff.",                    "pts"),
    ("turnovers_committed_rolling", "TOs",                          "to"),
    ("off_11_rate_rolling",         "11 Rate",                      "rate"),
    ("off_12_rate_rolling",         "12 Rate",                      "rate"),
    ("wins",                        "Wins",                         "int"),
    ("losses",                      "Losses",                       "int"),
    ("power_wins",                  "Power Wins",                   "int"),
    ("power_losses",                "Power Losses",                 "int"),
]


def _train_models(
    schedules: pd.DataFrame, features: pd.DataFrame
) -> dict[str, tuple[LinearRegression, list[str]]]:
    """Train all three models on TRAIN_SEASONS."""
    trained = {}
    for key, cfg in MODELS.items():
        feat_cols = cfg["features"]
        df = build_dataset(schedules, features, feat_cols, include_playoffs=True)
        train = df[df["season"].isin(TRAIN_SEASONS)]
        all_cols = feat_cols + [f"opp_{c}" for c in feat_cols]
        model = LinearRegression()
        model.fit(train[all_cols], train["margin"])
        trained[key] = (model, feat_cols)
    return trained


def _latest_team_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent row per team (their current rolling stats)."""
    feat = features.copy()
    feat["team"] = feat["team"].replace(TEAM_MAP)
    return (
        feat.sort_values(["team", "season", "week"])
        .groupby("team")
        .last()
        .reset_index()
        .set_index("team")
    )


def _team_features_at_week(features: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Return per-team features as of a specific (season, week).

    For any NaN feature, falls back to the most recent non-NaN value within
    the same team/season so that games with missing QB data (injury, etc.)
    still produce predictions using the last known stats.
    """
    season_feat = features[features["season"] == season].copy()
    season_feat["team"] = season_feat["team"].replace(TEAM_MAP)

    # Get the exact week row
    week_feat = season_feat[season_feat["week"] == week].set_index("team")

    # Build a forward-fill fallback: last known value per team up to this week
    prior = (
        season_feat[season_feat["week"] <= week]
        .sort_values("week")
        .groupby("team")
        .last()
    )

    # Fill NaNs in the target week using the fallback
    week_feat = week_feat.combine_first(prior)
    return week_feat


def _predict_game(
    home_row: pd.Series,
    away_row: pd.Series,
    trained_models: dict,
) -> dict:
    """Return predictions from all 3 models for a single game."""
    OPTIONAL_ZERO = {"off_12_epa_rolling", "qb_off_12_epa_rolling", "def_vs_12_epa_rolling"}
    preds = {}
    for key, (model, feat_cols) in trained_models.items():
        home_vals = [0.0 if (np.isnan(home_row.get(c, np.nan)) and c in OPTIONAL_ZERO) else home_row.get(c, np.nan) for c in feat_cols]
        away_vals = [0.0 if (np.isnan(away_row.get(c, np.nan)) and c in OPTIONAL_ZERO) else away_row.get(c, np.nan) for c in feat_cols]
        if any(np.isnan(v) for v in home_vals + away_vals):
            preds[key] = None
            continue
        X = np.array(home_vals + away_vals).reshape(1, -1)
        pred = float(model.predict(X)[0])
        preds[key] = round(pred, 1)
    return preds


def _team_stats_dict(row: pd.Series) -> dict:
    out = {}
    for col, label, fmt in DISPLAY_STATS:
        val = row.get(col)
        out[col] = {
            "label": label,
            "value": round(float(val), 3) if pd.notna(val) else None,
            "format": fmt,
        }
    return out


def _build_game_entry(g: pd.Series, home_row: pd.Series, away_row: pd.Series,
                      trained_models: dict, is_completed: bool = False) -> dict:
    """Build the JSON structure for a single game."""
    home = str(g["home_team"])
    away = str(g["away_team"])

    raw_preds = _predict_game(home_row, away_row, trained_models)

    model_preds = {}
    valid_preds = []
    for key, pred in raw_preds.items():
        spread = g.get("spread_line")
        if pred is not None:
            pick = home if pred > (float(spread) if pd.notna(spread) else 0) else away
        else:
            pick = None
        # Edge = how much the pick covers the spread (always positive when pick is correct).
        # predicted_margin is home-team perspective, spread_line is home-team perspective.
        # When pick=home: edge = pred - spread (positive = home covers by that much)
        # When pick=away: edge = spread - pred (positive = away covers by that much)
        if pred is not None and pd.notna(spread):
            spread_f = float(spread)
            edge = round(pred - spread_f, 1) if pick == home else round(spread_f - pred, 1)
        else:
            edge = None
        model_result = None
        if is_completed and pick is not None and pd.notna(g.get("home_score")):
            margin = float(g["home_score"]) - float(g["away_score"])
            spread_val = float(spread) if pd.notna(spread) else 0.0
            home_covered = margin > spread_val
            model_correct = (pick == home) == home_covered
            model_result = bool(model_correct)
        model_preds[key] = {
            "label": MODELS[key]["label"],
            "predicted_margin": pred,
            "edge": edge,
            "pick": pick,
            "correct": model_result,
        }
        if pred is not None:
            valid_preds.append(pred)

    avg_pred = round(float(np.mean(valid_preds)), 1) if valid_preds else None
    picks = [v["pick"] for v in model_preds.values() if v["pick"]]
    consensus = len(picks) > 0

    spread_line = float(g["spread_line"]) if pd.notna(g.get("spread_line")) else None
    total_line = float(g["total_line"]) if pd.notna(g.get("total_line")) else None

    result = None
    if is_completed and pd.notna(g.get("home_score")):
        home_score = int(g["home_score"])
        away_score = int(g["away_score"])
        margin = home_score - away_score
        home_covered = bool(margin > spread_line) if spread_line is not None else None
        result = {
            "home_score": home_score,
            "away_score": away_score,
            "margin": int(margin),
            "home_covered": home_covered,
        }

    entry = {
        "away_team": away,
        "home_team": home,
        "gameday": str(g.get("gameday", "")),
        "game_type": str(g.get("game_type", "REG")),
        "spread_line": spread_line,
        "total_line": total_line,
        "predictions": {
            "models": model_preds,
            "avg_predicted_margin": avg_pred,
            "consensus": consensus,
            "consensus_pick": picks[0] if consensus else None,
        },
        "home_qb": str(home_row.get("starting_qb", "") or ""),
        "away_qb": str(away_row.get("starting_qb", "") or ""),
        "home_stats": _team_stats_dict(home_row),
        "away_stats": _team_stats_dict(away_row),
        "result": result,
    }
    return entry


def generate_week(
    season: int,
    week: int,
    schedules: pd.DataFrame,
    features: pd.DataFrame,
    trained_models: dict,
    out_dir: str = "docs/data",
) -> dict:
    """Generate predictions for a specific season/week and write to docs/data/{season}/week_{week}.json."""
    sched = schedules[schedules["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    sched["home_team"] = sched["home_team"].replace(TEAM_MAP)
    sched["away_team"] = sched["away_team"].replace(TEAM_MAP)

    game_pool = sched[(sched["season"] == season) & (sched["week"] == week)]
    if game_pool.empty:
        return {}

    is_completed = game_pool["home_score"].notna().all()
    game_type = game_pool["game_type"].iloc[0]
    if game_type == "REG":
        label = f"Week {week} · {season} Season"
    else:
        label = f"{game_type} · {season} Season"

    # For historical/completed weeks, use features at that specific week
    # For future weeks, use latest available features
    team_map = _team_features_at_week(features, season, week)

    games_out = []
    for _, g in game_pool.iterrows():
        home = str(g["home_team"])
        away = str(g["away_team"])
        if home not in team_map.index or away not in team_map.index:
            continue
        entry = _build_game_entry(g, team_map.loc[home], team_map.loc[away],
                                   trained_models, is_completed=is_completed)
        games_out.append(entry)

    games_out.sort(key=lambda g: (
        not g["predictions"]["consensus"],
        -abs(g["predictions"]["avg_predicted_margin"] or 0),
    ))

    payload = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "week_label": label,
        "season": season,
        "week": week,
        "completed": is_completed,
        "demo_mode": False,
        "games": games_out,
    }

    out_path = Path(out_dir) / str(season) / f"week_{week}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, cls=_NumpyEncoder)

    return payload


def generate_index(available: list[dict], out_dir: str = "docs/data") -> None:
    """Write docs/data/index.json listing available seasons and weeks."""
    # Group by season
    from collections import defaultdict
    by_season: dict[int, list[int]] = defaultdict(list)
    for item in available:
        by_season[item["season"]].append(item["week"])

    seasons = []
    for s in sorted(by_season.keys(), reverse=True):
        weeks = sorted(by_season[s])
        seasons.append({"season": s, "weeks": weeks})

    default = available[-1] if available else None

    payload = {
        "seasons": seasons,
        "default": default,
    }
    out_path = Path(out_dir) / "index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, cls=_NumpyEncoder)
    print(f"Wrote index.json with {len(seasons)} seasons")


def generate_all(out_dir: str = "docs/data") -> None:
    """Generate per-week JSONs for all test seasons plus the current/upcoming week."""
    from config import SEASONS as ALL_SEASONS

    schedules = get_schedule_data(ALL_SEASONS)
    features = load_features()

    print("Training models...")
    trained = _train_models(schedules, features)

    sched = schedules[schedules["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    sched["home_team"] = sched["home_team"].replace(TEAM_MAP)
    sched["away_team"] = sched["away_team"].replace(TEAM_MAP)

    available = []

    # Generate all completed weeks in test seasons
    completed = sched[sched["home_score"].notna()]
    for season in sorted(TEST_SEASONS):
        season_games = completed[completed["season"] == season]
        weeks = sorted(season_games["week"].unique())
        for week in weeks:
            week = int(week)
            print(f"  Generating {season} week {week}...")
            payload = generate_week(season, week, schedules, features, trained, out_dir)
            has_any_pred = any(
                list(g["predictions"]["models"].values())[0]["predicted_margin"] is not None
                for g in payload.get("games", [])
                if g.get("predictions", {}).get("models")
            )
            if payload.get("games") and has_any_pred and week >= 4:
                available.append({"season": season, "week": week})

    # Also generate the upcoming week (or last completed REG week as demo)
    upcoming = sched[sched["home_score"].isna()].sort_values(["season", "week"])
    if not upcoming.empty:
        next_season = int(upcoming["season"].min())
        next_week = int(upcoming[upcoming["season"] == next_season]["week"].min())
        print(f"  Generating upcoming: {next_season} week {next_week}...")
        # For upcoming, use latest features
        team_map = _latest_team_features(features)
        game_pool = upcoming[(upcoming["season"] == next_season) & (upcoming["week"] == next_week)]
        label = f"Week {next_week} · {next_season} Season"
        games_out = []
        for _, g in game_pool.iterrows():
            home = str(g["home_team"])
            away = str(g["away_team"])
            if home not in team_map.index or away not in team_map.index:
                continue
            entry = _build_game_entry(g, team_map.loc[home], team_map.loc[away], trained, is_completed=False)
            games_out.append(entry)
        games_out.sort(key=lambda g: (
            not g["predictions"]["consensus"],
            -abs(g["predictions"]["avg_predicted_margin"] or 0),
        ))
        payload = {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "week_label": label,
            "season": next_season,
            "week": next_week,
            "completed": False,
            "demo_mode": False,
            "games": games_out,
        }
        out_path = Path(out_dir) / str(next_season) / f"week_{next_week}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, cls=_NumpyEncoder)
        if games_out:
            available.append({"season": next_season, "week": next_week})

    generate_index(available, out_dir)
    print(f"Done. Generated {len(available)} week files.")


def generate(seasons: list[int] | None = None, out_path: str = "docs/data/predictions.json") -> dict:
    """
    Train models and generate predictions for upcoming games.
    Falls back to the most recently completed week if no upcoming games exist.
    This is kept for backwards compatibility; prefer generate_all() for the full UI.
    """
    from config import SEASONS as ALL_SEASONS

    schedules = get_schedule_data(ALL_SEASONS)
    features = load_features()

    print("Training models...")
    trained = _train_models(schedules, features)

    team_latest = _latest_team_features(features)
    team_map = team_latest

    sched = schedules[schedules["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    sched["home_team"] = sched["home_team"].replace(TEAM_MAP)
    sched["away_team"] = sched["away_team"].replace(TEAM_MAP)

    upcoming = sched[sched["home_score"].isna()].sort_values(["season", "week"])

    demo_mode = upcoming.empty
    if demo_mode:
        completed = sched[sched["home_score"].notna() & (sched["game_type"] == "REG")]
        last_season = completed["season"].max()
        last_week = completed[completed["season"] == last_season]["week"].max()
        game_pool = completed[
            (completed["season"] == last_season) & (completed["week"] == last_week)
        ]
        label = f"Week {last_week} · {last_season} Season (completed — demo)"
    else:
        next_season = upcoming["season"].min()
        next_week = upcoming[upcoming["season"] == next_season]["week"].min()
        game_pool = upcoming[
            (upcoming["season"] == next_season) & (upcoming["week"] == next_week)
        ]
        label = f"Week {next_week} · {next_season} Season"

    games_out = []
    for _, g in game_pool.iterrows():
        home = str(g["home_team"])
        away = str(g["away_team"])
        if home not in team_map.index or away not in team_map.index:
            continue
        entry = _build_game_entry(g, team_map.loc[home], team_map.loc[away], trained)
        games_out.append(entry)

    games_out.sort(key=lambda g: (
        not g["predictions"]["consensus"],
        -abs(g["predictions"]["avg_predicted_margin"] or 0),
    ))

    payload = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "week_label": label,
        "demo_mode": demo_mode,
        "games": games_out,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, cls=_NumpyEncoder)

    print(f"Wrote {len(games_out)} games → {out_path}")
    return payload
