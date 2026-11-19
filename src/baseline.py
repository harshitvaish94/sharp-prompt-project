"""Person 1: TF-IDF + Logistic Regression baseline (plain, class_weight=None).

Run:  python -m src.baseline
Writes results/baseline_metrics.json
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.metrics import save_metrics  # noqa: E402


def load_split(path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data_prep` first.")
    return pd.read_csv(path)


def build_model() -> object:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=1000, class_weight=None, random_state=C.SEED),
    )


def main():
    train, test = load_split(C.TRAIN_CSV), load_split(C.TEST_CSV)
    model = build_model().fit(train["text"], train["label"])
    pred = model.predict(test["text"])
    out = save_metrics(C.BASELINE_JSON, "TF-IDF + LogReg (baseline)", test["label"], pred, C.LABELS,
                       n_train=int(len(train)))
    print(f"accuracy={out['accuracy']:.4f}  macro_f1={out['macro_f1']:.4f}  (n_test={out['n_test']})")
    for k, v in out["per_class_f1"].items():
        print(f"   {k:9s} F1={v:.3f}")
    print(f"Wrote {C.BASELINE_JSON}")


if __name__ == "__main__":
    main()
