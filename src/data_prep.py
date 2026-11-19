"""Person 1: load dair-ai/emotion, stratified-subsample, write CSVs + class distribution.

Run:  python -m src.data_prep
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def load_raw() -> dict:
    """Return {"train": df, "validation": df, "test": df} with columns text,label."""
    from datasets import load_dataset

    ds = load_dataset(C.DATASET_NAME, "split", cache_dir=str(C.RAW_DIR))
    return {k: ds[k].to_pandas()[["text", "label"]] for k in ("train", "validation", "test")}


def stratified_subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep n rows while preserving class proportions (imbalance is kept on purpose)."""
    if n >= len(df):
        return df.copy()
    sub, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=C.SEED)
    return sub


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset="text")
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(lambda i: C.LABELS[i])
    return df[["text", "label", "label_name"]].reset_index(drop=True)


def distribution(df: pd.DataFrame) -> dict:
    counts = df["label_name"].value_counts().reindex(C.LABELS, fill_value=0)
    total = int(counts.sum())
    nonzero = counts[counts > 0]
    return {
        "n": total,
        "counts": {k: int(v) for k, v in counts.items()},
        "percent": {k: round(100.0 * int(v) / total, 2) for k, v in counts.items()},
        "imbalance_ratio": round(float(nonzero.max() / nonzero.min()), 2),
        "majority_class": str(counts.idxmax()),
        "minority_class": str(nonzero.idxmin()),
    }


def build(raw: dict) -> dict:
    """Subsample/clean the raw splits, write CSVs and class_distribution.json."""
    splits = {
        "train": finalize(stratified_subsample(raw["train"], C.TRAIN_SIZE)),
        "val": finalize(stratified_subsample(raw["validation"], C.VAL_SIZE)),
        "test": finalize(stratified_subsample(raw["test"], C.TEST_SIZE)),
    }
    # leakage check: no test/val text may appear in train
    train_texts = set(splits["train"]["text"])
    for name in ("val", "test"):
        splits[name] = splits[name][~splits[name]["text"].isin(train_texts)].reset_index(drop=True)

    splits["train"].to_csv(C.TRAIN_CSV, index=False)
    splits["val"].to_csv(C.VAL_CSV, index=False)
    splits["test"].to_csv(C.TEST_CSV, index=False)

    dist = {name: distribution(df) for name, df in splits.items()}
    with open(C.DIST_JSON, "w") as f:
        json.dump(dist, f, indent=2)
    return dist


def main():
    raw = load_raw()
    dist = build(raw)
    for name, d in dist.items():
        print(f"\n[{name}] n={d['n']}  imbalance ratio={d['imbalance_ratio']} "
              f"({d['majority_class']} / {d['minority_class']})")
        for k in C.LABELS:
            print(f"   {k:9s} {d['counts'][k]:5d}  {d['percent'][k]:5.2f}%")
    print(f"\nWrote {C.TRAIN_CSV.name}, {C.VAL_CSV.name}, {C.TEST_CSV.name}, {C.DIST_JSON.name} -> {C.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
