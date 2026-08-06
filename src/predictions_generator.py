"""Generate predictions for upcoming NFL games using the top 3 models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import get_schedule_data
from src.model_utils import TEAM_MAP, load_features, build_dataset

TRAIN_SEASONS = range(2016, 2022)

MODELS = {
    "off+def+pd+qbr": {
        "label": "OFF/DEF + PD + QBR",
        "features": [
            "off_11_epa_rolling", "off_12_epa_rolling",
            "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
            "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
        ],
    },
    "off+def+pd+qbr+cpoe+vs12": {
        "label": "OFF/DEF + PD + QBR + CPOE + DEF_VS12",
        "features": [
            "off_11_epa_rolling", "off_12_epa_rolling",
            "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
            "def_epa_rolling", "point_diff_rolling", "qbr_rolling",
            "cpoe_rolling", "def_vs_12_epa_rolling",
        ],
    },
    "off+pd+qbr+TO": {
        "label": "OFF + PD + QBR + TO",
        "features": [
            "off_11_epa_rolling", "off_12_epa_rolling",
            "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
            "point_diff_rolling", "qbr_rolling",
            "turnovers_committed_rolling",
        ],
    },
}

# All stats to surface in the UI per team
DISPLAY_STATS = [
    ("off_11_epa_rolling",          "OFF EPA (11 personnel)",       "epa"),
    ("off_12_epa_rolling",          "OFF EPA (12 personnel)",       "epa"),
    ("qb_off_11_epa_rolling",       "QB EPA (11 personnel)",        "epa"),
    ("qb_off_12_epa_rolling",       "QB EPA (12 personnel)",        "epa"),
    ("off_epa_rolling",             "Overall OFF EPA",              "epa"),
    ("def_epa_rolling",             "DEF EPA allowed",              "epa_inv"),
    ("def_vs_12_epa_rolling",       "DEF vs 12 personnel",         "epa_inv"),
    ("qbr_rolling",                 "QBR",                          "qbr"),
    ("cpoe_rolling",                "CPOE",                         "pct"),
    ("point_diff_rolling",          "Point Differential",           "pts"),
    ("turnovers_committed_rolling", "Turnovers Committed",          "to"),
    ("off_11_rate_rolling",         "11 Personnel Rate",            "rate"),
    ("off_12_rate_rolling",         "12 Personnel Rate",            "rate"),
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
    )


def _predict_game(
    home_row: pd.Series,
    away_row: pd.Series,
    trained_models: dict,
) -> dict:
    """Return predictions from all 3 models for a single game."""
    preds = {}
    for key, (model, feat_cols) in trained_models.items():
        home_vals = [home_row.get(c, np.nan) for c in feat_cols]
        away_vals = [away_row.get(c, np.nan) for c in feat_cols]
        if any(np.isnan(v) for v in home_vals + away_vals):
            preds[key] = None
            continue
        # Feature vector: home_feats + opp_feats (same layout as training)
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


def generate(seasons: list[int] | None = None, out_path: str = "docs/data/predictions.json") -> dict:
    """
    Train models and generate predictions for upcoming games.
    Falls back to the most recently completed week if no upcoming games exist.
    """
    from config import SEASONS as ALL_SEASONS

    schedules = get_schedule_data(ALL_SEASONS)
    features = load_features()

    print("Training models...")
    trained = _train_models(schedules, features)

    team_latest = _latest_team_features(features)
    team_map = team_latest.set_index("team")

    # Find upcoming (unplayed) games
    sched = schedules[schedules["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    sched["home_team"] = sched["home_team"].replace(TEAM_MAP)
    sched["away_team"] = sched["away_team"].replace(TEAM_MAP)

    upcoming = sched[sched["home_score"].isna()].sort_values(["season", "week"])

    demo_mode = upcoming.empty
    if demo_mode:
        # Fall back to last completed regular season week for a meaningful demo
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

        home_row = team_map.loc[home]
        away_row = team_map.loc[away]

        raw_preds = _predict_game(home_row, away_row, trained)

        # Build per-model output
        model_preds = {}
        valid_preds = []
        for key, pred in raw_preds.items():
            spread = g.get("spread_line")
            edge = round(pred - float(spread), 1) if pred is not None and pd.notna(spread) else None
            if pred is not None:
                pick = home if pred > (float(spread) if pd.notna(spread) else 0) else away
            else:
                pick = None
            model_preds[key] = {
                "label": MODELS[key]["label"],
                "predicted_margin": pred,
                "edge": edge,
                "pick": pick,
            }
            if pred is not None:
                valid_preds.append(pred)

        avg_pred = round(float(np.mean(valid_preds)), 1) if valid_preds else None
        picks = [v["pick"] for v in model_preds.values() if v["pick"]]
        consensus = len(set(picks)) == 1 and len(picks) == 3

        spread_line = float(g["spread_line"]) if pd.notna(g.get("spread_line")) else None
        total_line = float(g["total_line"]) if pd.notna(g.get("total_line")) else None

        games_out.append({
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
            "home_stats": _team_stats_dict(home_row),
            "away_stats": _team_stats_dict(away_row),
        })

    # Sort by consensus first, then by abs edge
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
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(games_out)} games → {out_path}")
    return payload
