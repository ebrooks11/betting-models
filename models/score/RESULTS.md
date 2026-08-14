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
| **1** | **7.5355** | **3.25** | `points_scored_iter_adj`, `off_epa_iter_adj`, `def_epa_iter_adj`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `cpoe_rolling` | `points_allowed_iter_adj`, `off_epa_iter_adj`, `cpoe_rolling` |
| 2 | 7.5363 | 3.25 | same | `points_allowed_iter_adj`, `off_epa_iter_adj` |
| 3 | 7.5368 | 3.24 | same | `points_allowed_iter_adj`, `def_epa_iter_adj`, `off_epa_iter_adj` |
| 4 | 7.5663 | 2.98 | `points_scored_iter_adj`, `off_epa_iter_adj`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `cpoe_rolling` | `points_allowed_iter_adj`, `def_epa_iter_adj` |
| 5 | 7.5682 | 3.00 | same as rank 4 + `off_11_epa_rolling` | same |
| 6 | 7.5705 | 2.99 | same as rank 5 + `ypa_rolling` | same |
| 7 | 7.5747 | 2.97 | `points_scored_iter_adj`, `off_epa_iter_adj`, `ypa_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling` | `points_allowed_iter_adj`, `def_epa_iter_adj` |
| 8 | 7.5757 | 2.98 | same as rank 7 | same + `def_qb_hit_rate_rolling` |
| 9 | 7.5842 | 3.00 | same as rank 7 | same + `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling` |
| 10 | 7.5846 | 3.01 | same as rank 7 + `first_downs_pg_rolling` | same + `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling` |
| 11 | 7.6006 | 3.01 | `points_scored_rolling`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `off_epa_iter_adj` | `points_allowed_rolling`, `def_tfl_rate_rolling`, `def_qb_hit_rate_rolling`, `def_epa_iter_adj` |
| 12 | 7.6192 | 3.04 | same as rank 11 + `cpoe_rolling` | same + `def_sack_rate_rolling` |
| 13 | 7.6213 | 3.10 | `off_epa_iter_adj`, `def_epa_iter_adj` | `off_epa_iter_adj`, `def_epa_iter_adj` |
| 14 | 7.6517 | 2.74 | `points_scored_rolling`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling`, `rush_epa_rolling` | `points_allowed_rolling`, `def_tfl_rate_rolling` |
| 15 | 7.6518 | 2.74 | `points_scored_rolling`, `ypa_rolling`, `first_downs_pg_rolling`, `third_down_rate_rolling`, `qb_rush_yards_pg_rolling`, `off_11_epa_rolling`, `off_12_epa_rolling`, `rush_ypc_rolling` | `points_allowed_rolling`, `def_tfl_rate_rolling` |
| 16 | 7.6523 | 2.74 | same as rank 15 | same + `def_qb_hit_rate_rolling` |
| 17 | 7.6671 | 2.75 | same as rank 15 | same + `def_qb_hit_rate_rolling`, `def_sack_rate_rolling` |
| 18 | 7.6691 | 2.77 | same as rank 15 + `cpoe_rolling` | same + `def_qb_hit_rate_rolling`, `def_sack_rate_rolling` |
| 19 | 7.7167 | 2.38 | `points_scored_rolling`, `points_allowed_rolling` | — |
| 20 | 7.7207 | 2.47 | `points_scored_rolling`, `points_allowed_rolling`, `turnovers_committed_rolling`, `top_rolling` | `turnovers_committed_rolling`, `top_rolling` |
| 21 | 7.7325 | 2.45 | `points_scored_rolling`, `qbr_rolling`, `plays_per_game_rolling`, `turnovers_committed_rolling`, `top_rolling` | `turnovers_committed_rolling`, `top_rolling` |

---

## Notes

- **Naive baseline** (always predict league mean): MAE ≈ 7.93
- **Pred σ vs Act σ**: actual score std dev is ~10; model predictions cluster around 2.7–3.25. Higher σ = less mean-regression. Top models now reach 3.24–3.25.
- **Key discovery (ranks 1–3)**: Adding `def_epa_iter_adj` as a *primary* feature and `off_epa_iter_adj` as an *opponent* feature — i.e., giving each row the opponent's offensive quality and the primary team's own defensive quality — drove a jump from 7.5663 → 7.5355. The intuition: a team's own defense affects scoring through field position and game script, and the opponent's offense sets the ceiling on what the primary team's defense will face.
- **Key discovery (ranks 4–6)**: Replacing `points_scored_rolling`/`points_allowed_rolling` with their `iter_adj` counterparts improved rank-1 from 7.6006 → 7.5663.
- **Rank 13**: `off/def_epa_iter_adj` only (4 features) nearly matches rank 11 with 13 features — strong signal-to-noise.
- **def_tfl_rate** and **def_sack_rate** are computed from the `defteam` perspective (see `docs/feature_log.md`). `sack_rate` (no prefix) is `posteam` perspective (OL quality).
- Results computed with the current pipeline state. Earlier session results (~7.37 MAE) cannot be reproduced due to pipeline changes mid-session.
