"""Delete every JSON under data/processed and results that is marked "_mock": true."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

for folder in (Path(C.DATA_DIR) / "processed", Path(C.RESULTS_DIR)):
    for p in folder.glob("*.json"):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("_mock"):
                p.unlink()
                print("deleted mock:", p)
        except (json.JSONDecodeError, AttributeError):
            pass
