# Score Model Results

Walk-forward CV: train on all seasons before test year, test on each season 2020–2025.
One row per team per game. Target: `points_scored`. Model: OLS linear regression.
All rolling features use a 3-game window shifted by 1 (no leakage).

Opponent features are prefixed `opp_` by the dataset builder.
`is_home` is included in every model.

---

## Rankings (by overall MAE, ascending)

| Rank | MAE | Pred σ | Primary Features | Opponent Features |
|------|-----|--------|-----------------|-------------------|
| 1 | **7.6006** | 3.01 | `points_scored_rolling`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `off_epa_iter_adj` | `points_allowed_rolling`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling`, `def_epa_iter_adj` |
| 2 | 7.6192 | 3.04 | same + `cpoe_rolling` | same + `def_sack_rate_rolling` |
| 3 | 7.6213 | 3.10 | `off_epa_iter_adj`, `def_epa_iter_adj` | `off_epa_iter_adj`, `def_epa_iter_adj` |
| 4 | 7.6517 | 2.74 | `points_scored_rolling`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `rush_epa_rolling` | `points_allowed_rolling`, `def_tfl_rate_rolling` |
| 5 | 7.6518 | 2.74 | `points_scored_rolling`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling` | `points_allowed_rolling`, `def_tfl_rate_rolling` |
| 6 | 7.6523 | 2.74 | same as rank 5 | `points_allowed_rolling`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling` |
| 7 | 7.6671 | 2.75 | same as rank 5 | `points_allowed_rolling`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling`, `def_sack_rate_rolling` |
| 8 | 7.6691 | 2.77 | same as rank 5 + `cpoe_rolling` | `points_allowed_rolling`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling`, `def_sack_rate_rolling` |
| 9 | 7.7167 | 2.38 | `points_scored_rolling`, `points_allowed_rolling` | — |
| 10 | 7.7207 | 2.47 | `points_scored_rolling`, `points_allowed_rolling`, `turnovers_committed_rolling`, `top_rolling` | `turnovers_committed_rolling`, `top_rolling` |
| 11 | 7.7325 | 2.45 | `points_scored_rolling`, `qbr_rolling`, `plays_per_game_rolling`, `turnovers_committed_rolling`, `top_rolling` | `turnovers_committed_rolling`, `top_rolling` |

---

## Notes

- **Naive baseline** (always predict league mean): MAE ≈ 7.93
- **Pred σ vs Act σ**: actual score std dev is ~10; model predictions cluster around 2.7–3.1. This is a known limitation of rolling-average linear regression — it regresses heavily to the mean.
- **Iter adj features**: `off_epa_iter_adj` and `def_epa_iter_adj` are season-to-date EPA values adjusted iteratively (5 rounds) for opponent quality. Rank 3 shows these 4 features alone nearly match the best 13-feature model — strong signal-to-noise ratio.
- **def_tfl_rate** and **def_sack_rate** are computed from the `defteam` perspective (see `docs/feature_log.md`). `sack_rate` (no prefix) is `posteam` perspective (OL quality).
- Results computed with the current pipeline state. Earlier session results (~7.37 MAE) cannot be reproduced due to pipeline changes mid-session.
