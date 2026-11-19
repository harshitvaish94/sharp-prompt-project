"""Sharpness evaluation module for prompt-based emotion classification.

Measures two distinct forms of black-box model sensitivity:
1. Prompt Template Sensitivity:
   - Evaluates the original prompt template against alternative template phrasings.
   - Computes Macro-F1 for every template.
   - Measures performance drop (absolute and relative) relative to the original template.
   - Aggregates template sensitivity (mean drop, max drop, Macro-F1 variance).

2. Input Paraphrase Sensitivity:
   - Evaluates prediction stability across deterministic input paraphrases.
   - Compares predictions on original vs. paraphrased sentences.
   - Computes overall flip rate and per-class flip rates.

Outputs:
- results/sharpness.json
- results/paraphrase_predictions.csv
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.paraphrase import paraphrase_batch
from src.prompt_classifier import PROMPT_TEMPLATES, PromptClassifier


def evaluate_template_sensitivity(
    classifier: PromptClassifier,
    df: pd.DataFrame,
    batch_size: int = 64,
) -> Dict:
    """Evaluate performance variations across all prompt templates."""
    print("\n--- Evaluating Prompt Template Sensitivity ---")

    # Check if pre-computed validation template metrics exist in prompt_metrics.json
    cached_metrics_path = C.RESULTS_DIR / "prompt_metrics.json"
    if cached_metrics_path.exists():
        try:
            with open(cached_metrics_path, "r") as f:
                c_data = json.load(f)
            cached_val = c_data.get("val_template_metrics")
            if cached_val and all(t in cached_val for t in PROMPT_TEMPLATES):
                print("Loading verified validation template metrics from prompt_metrics.json...")
                template_results = {}
                f1_scores = []
                for name, m in cached_val.items():
                    f1 = float(m["macro_f1"])
                    f1_scores.append(f1)
                    template_results[name] = {
                        "macro_f1": f1,
                        "accuracy": float(m["accuracy"]),
                        "per_class_f1": m["per_class_f1"],
                        "eval_time_seconds": round(m.get("eval_time_seconds", 0.0), 2),
                    }
                    print(f"  Template: {name:20s} | Macro-F1: {f1:.4f} | Accuracy: {m['accuracy']:.4f}")

                orig_f1 = template_results["original"]["macro_f1"]
                for name, res in template_results.items():
                    drop = orig_f1 - res["macro_f1"]
                    rel_drop = (drop / orig_f1) if orig_f1 > 0 else 0.0
                    res["drop_relative_to_original"] = round(float(drop), 4)
                    res["relative_drop"] = round(float(rel_drop), 4)

                alt_drops = [res["drop_relative_to_original"] for k, res in template_results.items() if k != "original"]
                mean_drop = float(np.mean(alt_drops)) if alt_drops else 0.0
                max_drop = float(np.max(alt_drops)) if alt_drops else 0.0
                macro_f1_std = float(np.std(f1_scores))

                print(f"\nTemplate Sensitivity Summary:")
                print(f"  Original Macro-F1: {orig_f1:.4f}")
                print(f"  Mean Drop (alt templates): {mean_drop:.4f}")
                print(f"  Max Drop (worst template): {max_drop:.4f}")
                print(f"  Macro-F1 Std Dev: {macro_f1_std:.4f}")

                return {
                    "original_macro_f1": round(orig_f1, 4),
                    "templates": template_results,
                    "mean_drop": round(mean_drop, 4),
                    "max_drop": round(max_drop, 4),
                    "macro_f1_std": round(macro_f1_std, 4),
                }
        except Exception as e:
            print(f"Notice: re-evaluating templates ({e})")

    texts = df["text"].tolist()
    y_true = df["label"].values.astype(int)

    template_results = {}
    f1_scores = []

    for name in PROMPT_TEMPLATES:
        t0 = time.time()
        preds = classifier.predict(texts, template_name=name, batch_size=batch_size)
        metrics = compute_metrics(y_true, preds, C.LABELS)
        f1 = metrics["macro_f1"]
        f1_scores.append(f1)
        duration = time.time() - t0

        template_results[name] = {
            "macro_f1": float(f1),
            "accuracy": float(metrics["accuracy"]),
            "per_class_f1": metrics["per_class_f1"],
            "eval_time_seconds": round(duration, 2),
        }
        print(f"  Template: {name:20s} | Macro-F1: {f1:.4f} | Accuracy: {metrics['accuracy']:.4f}")

    orig_f1 = template_results["original"]["macro_f1"]

    # Calculate drops relative to original template
    for name, res in template_results.items():
        drop = orig_f1 - res["macro_f1"]
        rel_drop = (drop / orig_f1) if orig_f1 > 0 else 0.0
        res["drop_relative_to_original"] = round(float(drop), 4)
        res["relative_drop"] = round(float(rel_drop), 4)

    alt_drops = [res["drop_relative_to_original"] for k, res in template_results.items() if k != "original"]
    mean_drop = float(np.mean(alt_drops)) if alt_drops else 0.0
    max_drop = float(np.max(alt_drops)) if alt_drops else 0.0
    macro_f1_std = float(np.std(f1_scores))

    print(f"\nTemplate Sensitivity Summary:")
    print(f"  Original Macro-F1: {orig_f1:.4f}")
    print(f"  Mean Drop (alt templates): {mean_drop:.4f}")
    print(f"  Max Drop (worst template): {max_drop:.4f}")
    print(f"  Macro-F1 Std Dev: {macro_f1_std:.4f}")

    return {
        "original_macro_f1": round(orig_f1, 4),
        "templates": template_results,
        "mean_drop": round(mean_drop, 4),
        "max_drop": round(max_drop, 4),
        "macro_f1_std": round(macro_f1_std, 4),
    }


def evaluate_paraphrase_sensitivity(
    classifier: PromptClassifier,
    df: pd.DataFrame,
    template_name: str = "original",
    batch_size: int = 32,
    csv_save_path: Optional[Path] = C.RESULTS_DIR / "paraphrase_predictions.csv",
) -> Tuple[Dict, pd.DataFrame]:
    """Evaluate prediction flip rate between original and paraphrased inputs."""
    print("\n--- Evaluating Input Paraphrase Sensitivity ---")
    texts = df["text"].tolist()
    y_true = df["label"].values.astype(int)
    n_samples = len(texts)

    print(f"Generating deterministic paraphrases for {n_samples} texts...")
    para_texts = paraphrase_batch(texts, seed=C.SEED)

    print(f"Running inference on original texts...")
    orig_preds = classifier.predict(texts, template_name=template_name, batch_size=batch_size)

    print(f"Running inference on paraphrased texts...")
    para_preds = classifier.predict(para_texts, template_name=template_name, batch_size=batch_size)

    flipped = (orig_preds != para_preds)
    n_flipped = int(np.sum(flipped))
    flip_rate = float(n_flipped / n_samples)

    print(f"Flip Rate: {flip_rate:.4f} ({n_flipped} / {n_samples} predictions changed)")

    # Per-class flip rate (conditioned on ground truth class)
    per_class_flip = {}
    for class_id, class_name in enumerate(C.LABELS):
        mask = (y_true == class_id)
        class_total = int(np.sum(mask))
        if class_total > 0:
            class_flipped = int(np.sum(flipped[mask]))
            per_class_flip[class_name] = round(float(class_flipped / class_total), 4)
        else:
            per_class_flip[class_name] = 0.0

    # Build predictions dataframe
    pred_df = pd.DataFrame({
        "text": texts,
        "paraphrase": para_texts,
        "label": y_true,
        "label_name": [C.LABELS[i] for i in y_true],
        "original_pred": orig_preds,
        "paraphrase_pred": para_preds,
        "original_pred_name": [C.LABELS[i] for i in orig_preds],
        "paraphrase_pred_name": [C.LABELS[i] for i in para_preds],
        "flipped": flipped.tolist(),
    })

    if csv_save_path:
        csv_save_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(csv_save_path, index=False)
        print(f"Wrote paraphrase predictions table to {csv_save_path}")

    # Compute Macro-F1 on original vs paraphrase
    orig_metrics = compute_metrics(y_true, orig_preds, C.LABELS)
    para_metrics = compute_metrics(y_true, para_preds, C.LABELS)

    res = {
        "n_samples": n_samples,
        "n_flipped": n_flipped,
        "flip_rate": round(flip_rate, 4),
        "per_class_flip_rate": per_class_flip,
        "original_accuracy": round(orig_metrics["accuracy"], 4),
        "paraphrase_accuracy": round(para_metrics["accuracy"], 4),
        "original_macro_f1": round(orig_metrics["macro_f1"], 4),
        "paraphrase_macro_f1": round(para_metrics["macro_f1"], 4),
        "macro_f1_gap": round(float(orig_metrics["macro_f1"] - para_metrics["macro_f1"]), 4),
    }

    return res, pred_df


def run_sharpness_evaluation(
    classifier: Optional[PromptClassifier] = None,
    split_path: Path = C.TEST_CSV,
    json_save_path: Path = C.RESULTS_DIR / "sharpness.json",
    csv_save_path: Path = C.RESULTS_DIR / "paraphrase_predictions.csv",
) -> Dict:
    """Execute full black-box sharpness evaluation."""
    if classifier is None:
        classifier = PromptClassifier()

    df = pd.read_csv(split_path)
    val_df = pd.read_csv(C.VAL_CSV)

    # 1. Template sensitivity on validation set
    template_sens = evaluate_template_sensitivity(classifier, val_df)

    # 2. Input paraphrase sensitivity on test set
    paraphrase_sens, pred_df = evaluate_paraphrase_sensitivity(
        classifier, df, template_name="original", csv_save_path=csv_save_path
    )

    # 3. Overall sharpness summary
    overall_sharpness = float(0.5 * paraphrase_sens["flip_rate"] + 0.5 * max(0.0, template_sens["mean_drop"]))

    sharpness_out = {
        "template_sensitivity": template_sens,
        "paraphrase_sensitivity": paraphrase_sens,
        "sharpness_summary": {
            "paraphrase_flip_rate": paraphrase_sens["flip_rate"],
            "template_mean_drop": template_sens["mean_drop"],
            "template_max_drop": template_sens["max_drop"],
            "macro_f1_std": template_sens["macro_f1_std"],
            "overall_sharpness_score": round(overall_sharpness, 4),
        },
    }

    json_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_save_path, "w") as f:
        json.dump(sharpness_out, f, indent=2)

    print(f"\nWrote sharpness evaluation results to {json_save_path}")
    return sharpness_out


if __name__ == "__main__":
    run_sharpness_evaluation()
