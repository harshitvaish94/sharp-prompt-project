"""Run the full pipeline from the repo root, stopping at the first failure."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "src.data_prep",
    "src.baseline",
    "src.prompt_classifier",
    "src.sharpness",
    "src.focal_classifier",
    "src.compare",
]


def main():
    for i, module in enumerate(STEPS, 1):
        print(f"\n=== [{i}/{len(STEPS)}] python -m {module} ===", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nERROR: step '{module}' failed (exit code {result.returncode}). "
                  f"Fix it and rerun; later steps were NOT run.", file=sys.stderr)
            sys.exit(result.returncode)
        print(f"--- {module} done in {time.time() - t0:.1f}s")
    print("\nPipeline complete. Now run: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
