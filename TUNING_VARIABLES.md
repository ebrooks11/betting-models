# Tuning Variables

Variables we can manipulate to find the most accurate model.

## Data Scope
- **Number of training seasons** — currently 10 (2015–2024); more data vs. more relevance
- **Game types included** — regular season only vs. including playoffs
- **Play filters** — exclude garbage time, exclude penalties, etc.

## Feature Engineering
- **Rolling window size** — currently 5 games; smaller = more reactive, larger = more stable
- **Which features to include/exclude** — drop noisy features, add new ones
- **Feature weighting** — equal weight vs. exponential decay (weight recent games more)
- **Cross-season carryover** — reset stats each season vs. carry over from prior year
- **Opponent adjustments** — raw stats vs. opponent-adjusted (e.g., EPA relative to that defense's strength)

## Model Algorithm
- **Algorithm type** — Ridge regression, Lasso, XGBoost, Random Forest, etc.
- **Regularization strength** — Ridge alpha (currently 1.0)
- **Algorithm-specific hyperparameters** — tree depth, learning rate, number of estimators, etc.

## Target Variable
- **What to predict** — team score (current plan) vs. point differential vs. win probability
- **Target transform** — raw points vs. log-scaled

## Validation Strategy
- **Walk-forward split point** — how many seasons to train vs. validate
- **Retrain frequency** — once per season vs. rolling retrain as new weeks come in

## Prediction Thresholds
- **Minimum edge to flag a bet** — e.g., only bet when predicted spread differs from the book by 3+ points
- **Confidence filters** — only bet games where feature data is complete (no bye-week gaps)

## Priority Order

Highest-impact levers to experiment with first:

1. Rolling window size
2. Which features to include
3. Algorithm type
4. Opponent adjustments
5. Cross-season carryover
