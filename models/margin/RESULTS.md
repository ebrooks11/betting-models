# Margin Model Results

Walk-forward CV: train on 2016–2022, test on 2023–2025.
One row per game (home team perspective). Target: `home_score − away_score`.
Model: OLS linear regression. All rolling features use a 3-game window shifted by 1 (no leakage).
Opponent features are prefixed `opp_` by the dataset builder.

Break-even at standard −110 juice is **52.38%**.
Units at −110 juice: `wins / 1.1 − losses`.
**E>3** = games where `|predicted_margin − spread_line| ≥ 3` (high-confidence picks).

Includes playoff games (WC, DIV, CON, SB). Weeks 1–3 and final regular-season week excluded.

---

## Top 10 by Overall ATS

| Rank | Model | N | ATS | Units | E>3 ATS | E>3 N | E>3 Units |
|------|-------|---|-----|-------|---------|-------|-----------|
| 1 | off_11/12 + qb_off + def + cpoe<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `cpoe_rolling`</sub> | 492 | **53.7%** | **+12.0** | 52.0% | 275 | -2.0 |
| 2 | off_11/12 + qb_off + def + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 488 | 53.3% | +8.4 | 53.8% | 253 | +6.6 |
| 3 | off_11/12 + qb_off + def + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 488 | 53.1% | +6.5 | 52.9% | 255 | +2.7 |
| 4 | off_11/12 + qb_off + def + pd<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`</sub> | 489 | 53.0% | +5.5 | 54.2% | 251 | +8.6 |
| 5 | off_11/12 + def_vs_11/12 + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `def_vs_11_epa_rolling`, `def_vs_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 504 | 52.8% | +3.8 | 51.0% | 259 | -7.0 |
| 6 | off_11/12 + qb_off + pd + qbr + TO<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `turnovers_committed_rolling`</sub> | 488 | 52.7% | +2.6 | 54.8% | 261 | +12.0 |
| 7 | off_11/12 + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 504 | 52.6% | +1.9 | 50.0% | 258 | -11.7 |
| 8 | off_11/12 + qb_off + def + pd + qbr + cpoe<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `cpoe_rolling`</sub> | 488 | 52.5% | +0.7 | 53.3% | 242 | +4.3 |
| 9 | off_11/12 + qb_off + def + pd + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `def_vs_12_epa_rolling`</sub> | 489 | 52.4% | -0.3 | 54.2% | 240 | +8.2 |
| 10 | off_11/12 + def + pd + qbr<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 504 | 52.2% | -1.9 | 50.2% | 265 | -11.1 |

---

## Top 10 by E>3 ATS (high-confidence picks)

| Rank | Model | N | ATS | Units | E>3 ATS | E>3 N | E>3 Units |
|------|-------|---|-----|-------|---------|-------|-----------|
| 1 | qb_off + def + pd<br><sub>`qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`</sub> | 489 | 51.5% | -7.9 | **55.8%** | 260 | **+16.8** |
| 2 | off_11/12 + qb_off + pd + qbr + TO<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `turnovers_committed_rolling`</sub> | 488 | 52.7% | +2.6 | 54.8% | 261 | +12.0 |
| 3 | off_11/12 + qb_off + def + pd + qbr + cpoe + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `cpoe_rolling`, `def_vs_12_epa_rolling`</sub> | 488 | 52.0% | -3.1 | 54.8% | 241 | +11.0 |
| 4 | off_11/12 + qb_off + def + pd<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`</sub> | 489 | 53.0% | +5.5 | 54.2% | 251 | +8.6 |
| 5 | off_11/12 + qb_off + def + pd + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `def_vs_12_epa_rolling`</sub> | 489 | 52.4% | -0.3 | 54.2% | 240 | +8.2 |
| 6 | off_11/12 + qb_off + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 488 | 51.4% | -8.8 | 54.2% | 253 | +8.5 |
| 7 | qb_weighted_raw + pd + qbr<br><sub>`qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `qb_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 488 | 50.6% | -16.5 | 54.2% | 253 | +8.5 |
| 8 | qb_first_down + pd + qbr<br><sub>`qb_epa_rolling`, `def_epa_first_down_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 507 | 50.7% | -16.4 | 54.0% | 261 | +8.2 |
| 9 | off_11/12 + qb_off + def + pd + qbr + def_vs_12<br><sub>`off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `point_diff_rolling`, `qbr_rolling`, `def_vs_12_epa_rolling`</sub> | 488 | 53.3% | +8.4 | 53.8% | 253 | +6.6 |
| 10 | off_11/12 (w5) + qb_off + pd + qbr<br><sub>`off_11_epa_rolling_w5`, `off_12_epa_rolling_w5`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `point_diff_rolling`, `qbr_rolling`</sub> | 488 | 51.2% | -10.7 | 53.4% | 232 | +4.7 |

---

## ATS by Week (best model, test 2023–2025)

Best model: `off_11_epa_rolling`, `off_12_epa_rolling`, `qb_off_11_epa_rolling`, `qb_off_12_epa_rolling`, `def_epa_rolling`, `cpoe_rolling`

| Weeks | ATS | n | E>3 ATS | E>3 n |
|-------|-----|---|---------|-------|
| Wk 4–6 | 53.3% | 137 | 51.4% | 74 |
| Wk 7–9 | 50.4% | 139 | 51.6% | 64 |
| Wk 10–12 | 55.5% | 146 | 52.3% | 65 |
| Wk 13–15 | 55.8% | 154 | 53.6% | 84 |
| Wk 16–17 | **61.3%** | 111 | 50.8% | 59 |

---

## Notes

- Model improves meaningfully mid-to-late season — weeks 10–17 are substantially stronger than weeks 4–9
- Weeks 16–17 hit 61.3% — rolling averages are most informative once teams have 10+ games of data
- Offensive formation EPA (11/12 personnel) is the strongest signal — captures scheme efficiency beyond raw EPA
- 3-game rolling window consistently beats expanding or 5-game windows for overall ATS
- OLS linear regression — Ridge regularization was tested and showed no improvement
- Spread is never used as a feature — model is market-independent
