## Dataset and baseline (Person 1)

**Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (`split` config), English tweets
labelled with 6 emotions: sadness, joy, love, anger, fear, surprise.

**Why this dataset:** it is genuinely imbalanced (so accuracy alone is misleading and macro F1 matters), the texts are
short (fast CPU inference with flan-t5-base), and it is small enough to run in one day.

**Splits (`python -m src.data_prep`):**
- Train: stratified random subsample of the original train split, `TRAIN_SIZE = 4000`.
- Validation: stratified subsample of the original validation split, `VAL_SIZE = 1000`.
- Test: stratified subsample of the original test split, `TEST_SIZE = 1000`.
- Stratification keeps the original class proportions, so the imbalance is preserved on purpose. Seed = 42.
- Cleaning: strip whitespace, drop empty and duplicate texts, drop any val/test text that also appears in train.
- Output: `data/processed/{train,val,test}.csv` (columns `text,label,label_name`) and
  `data/processed/class_distribution.json`.

**Class distribution (fill in from the real `class_distribution.json` after running; do not type numbers from memory):**

| Class | Train count | Train % | Test count | Test % |
|---|---|---|---|---|
| sadness | 1166 | 29.17 | 290 | 29.06 |
| joy | 1341 | 33.55 | 346 | 34.67 |
| love | 325 | 8.13 | 79 | 7.92 |
| anger | 539 | 13.49 | 138 | 13.83 |
| fear | 484 | 12.11 | 112 | 11.22 |
| surprise | 142 | 3.55 | 33 | 3.31 |

Imbalance ratio (majority / minority, train): `9.44` (`class_distribution.json` -> `train.imbalance_ratio`).

**Baseline (`python -m src.baseline`):** TF-IDF (1-2 grams, `min_df=2`, sublinear tf) + Logistic Regression with
`class_weight=None`. Trained on train, evaluated once on test. Metrics (accuracy, macro F1, per-class F1, confusion
matrix) come from the shared `src/metrics.py` and are written to `results/baseline_metrics.json`.

**Baseline results (fill in from `results/baseline_metrics.json`):** accuracy `0.6754` (67.54%), macro F1 `0.4675` (46.75%).

**Limitations:**
- Train is subsampled to 4k rows, so absolute scores are lower than full-data results in the literature.
- Labels in this dataset were produced by hashtag distant supervision, so they are noisy; the ceiling is below 100%.
- Rare classes (surprise, love) have few test examples, so their per-class F1 has high variance.
- Single seed, single split: no confidence intervals.
