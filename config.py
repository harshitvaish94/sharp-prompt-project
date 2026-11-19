"""Shared constants. Written by Person 1, then FROZEN. Do not edit without telling the team."""
from pathlib import Path

SEED = 42

# dair-ai/emotion label ids, index = label id
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
DIST_JSON = PROCESSED_DIR / "class_distribution.json"

BASELINE_JSON = RESULTS_DIR / "baseline_metrics.json"

DATASET_NAME = "dair-ai/emotion"
TRAIN_SIZE = 4000   # stratified subsample of the original 16k train split
VAL_SIZE = 1000
TEST_SIZE = 1000

LLM_NAME = "google/flan-t5-base"
EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
