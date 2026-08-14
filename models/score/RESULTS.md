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
| **1** | **7.5663** | **2.98** | `points_scored_iter_adj`, `off_epa_iter_adj`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `cpoe_rolling` | `points_allowed_iter_adj`, `def_epa_iter_adj` |
| 2 | 7.5682 | 3.00 | same + `off_11_epa_rolling` | same |
| 3 | 7.5705 | 2.99 | same as rank 2 + `ypa_rolling` | same |
| 4 | 7.5747 | 2.97 | `points_scored_iter_adj`, `off_epa_iter_adj`, `ypa_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling` | `points_allowed_iter_adj`, `def_epa_iter_adj` |
| 5 | 7.5757 | 2.98 | `points_scored_iter_adj`, `off_epa_iter_adj`, `ypa_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling` | `points_allowed_iter_adj`, `def_epa_iter_adj`, `def_qb_hit_rate_rolling` |
| 6 | 7.5842 | 3.00 | `points_scored_iter_adj`, `off_epa_iter_adj`, `ypa_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling` | `points_allowed_iter_adj`, `def_epa_iter_adj`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling` |
| 7 | 7.5846 | 3.01 | `points_scored_iter_adj`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `off_epa_iter_adj` | `points_allowed_iter_adj`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling`, `def_epa_iter_adj` |
| 8 | 7.6006 | 3.01 | `points_scored_rolling`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `off_epa_iter_adj` | `points_allowed_rolling`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling`, `def_epa_iter_adj` |
| 9 | 7.6192 | 3.04 | same as rank 8 + `cpoe_rolling` | same + `def_sack_rate_rolling` |
| 10 | 7.6213 | 3.10 | `off_epa_iter_adj`, `def_epa_iter_adj` | `off_epa_iter_adj`, `def_epa_iter_adj` |
| 11 | 7.6517 | 2.74 | `points_scored_rolling`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `rush_epa_rolling` | `points_allowed_rolling`, `def_tfl_rate_rolling` |
| 12 | 7.6518 | 2.74 | `points_scored_rolling`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling` | `points_allowed_rolling`, `def_tfl_rate_rolling` |
| 13 | 7.6523 | 2.74 | same as rank 12 | `points_allowed_rolling`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling` |
| 14 | 7.6671 | 2.75 | same as rank 12 | `points_allowed_rolling`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling`, `def_sack_rate_rolling` |
| 15 | 7.6691 | 2.77 | same as rank 12 + `cpoe_rolling` | `points_allowed_rolling`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling`, `def_sack_rate_rolling` |
| 16 | 7.7167 | 2.38 | `points_scored_rolling`, `points_allowed_rolling` | — |
| 17 | 7.7207 | 2.47 | `points_scored_rolling`, `points_allowed_rolling`, `turnovers_committed_rolling`, `top_rolling` | `turnovers_committed_rolling`, `top_rolling` |
| 18 | 7.7325 | 2.45 | `points_scored_rolling`, `qbr_rolling`, `plays_per_game_rolling`, `turnovers_committed_rolling`, `top_rolling` | `turnovers_committed_rolling`, `top_rolling` |

---

## Notes

- **Naive baseline** (always predict league mean): MAE ≈ 7.93
- **Pred σ vs Act σ**: actual score std dev is ~10; model predictions cluster around 2.7–3.1. This is a known limitation of rolling-average linear regression — it regresses heavily to the mean.
- **Iter adj features**: `off_epa_iter_adj` and `def_epa_iter_adj` are season-to-date EPA values adjusted iteratively (5 rounds) for opponent quality. Similarly for `points_scored_iter_adj`/`points_allowed_iter_adj`. Key insight: using **iter_adj versions of points** instead of rolling averages consistently beats rolling-only models.
- **Key discovery (ranks 1–6)**: Replacing `points_scored_rolling`/`points_allowed_rolling` with their iterative opponent-adjusted counterparts (`points_scored_iter_adj`/`points_allowed_iter_adj`) improved rank-1 MAE from 7.6006 → 7.5663. The opponent-quality adjustment strips out schedule difficulty, giving a cleaner signal.
- **def_tfl_rate** and **def_sack_rate** are computed from the `defteam` perspective (see `docs/feature_log.md`). `sack_rate` (no prefix) is `posteam` perspective (OL quality).
- Results computed with the current pipeline state. Earlier session results (~7.37 MAE) cannot be reproduced due to pipeline changes mid-session.
