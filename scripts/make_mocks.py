"""DEV ONLY: write mock JSONs (marked "_mock": true) for files that do not exist yet.
Never overwrites real files. Remove with scripts/delete_mocks.py before final commit."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

L = list(C.LABELS)
PROC = Path(C.DATA_DIR) / "processed"
RES = Path(C.RESULTS_DIR)


def split(n, counts):
    pct = {k: round(100 * v / n, 2) for k, v in counts.items()}
    return {"n": n, "counts": counts, "percent": pct,
            "imbalance_ratio": round(max(counts.values()) / min(counts.values()), 2),
            "majority_class": max(counts, key=counts.get),
            "minority_class": min(counts, key=counts.get)}


def write(path, obj):
    if path.exists():
        print("skip (exists):", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    obj["_mock"] = True
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("wrote mock:", path)


dist = {"train": split(4000, dict(zip(L, [1160, 1330, 330, 540, 480, 160]))),
        "val": split(1000, dict(zip(L, [290, 340, 80, 130, 120, 40]))),
        "test": split(1000, dict(zip(L, [290, 340, 80, 130, 120, 40])))}
write(PROC / "class_distribution.json", dist)

n = len(L)
cm = [[0] * n for _ in range(n)]
for i in range(n):
    cm[i][i] = 10 * (i + 5)
    cm[i][(i + 1) % n] = 7
write(RES / "baseline_metrics.json",
      {"accuracy": 0.0, "macro_f1": 0.0, "per_class_f1": {k: 0.0 for k in L},
       "confusion_matrix": cm, "method": "mock", "n_test": 1000, "n_train": 4000})
