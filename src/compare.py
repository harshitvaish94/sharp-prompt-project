"""Comparison module across all four classification approaches.

Approaches compared:
1. TF-IDF + Logistic Regression (baseline)
2. FLAN-T5 Prompt-based Classifier (with validation-selected template)
3. MiniLM Embeddings + Cross-Entropy Loss
4. MiniLM Embeddings + Focal Loss (gamma = 2.0)

Selection rule for prompt templates (validation only):
  Score = Macro-F1 - lambda * Sharpness

Sharpness is set to null for methods where it is not applicable.
Generates: results/comparison.json
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def select_best_prompt_template(
    prompt_metrics: Dict[str, Any],
    sharpness_metrics: Optional[Dict[str, Any]] = None,
    lambda_param: float = 0.5,
) -> Dict[str, Any]:
    """Select the best prompt template using validation data only.

    Score = Macro-F1 - lambda * Sharpness
    Sharpness here represents template sensitivity / paraphrase flip rate.
    """
    val_metrics = prompt_metrics.get("val_template_metrics", {})
    if not val_metrics:
        # Fallback if only original was evaluated
        orig_val = prompt_metrics.get("val_macro_f1", prompt_metrics.get("macro_f1", 0.0))
        return {
            "selected_template": "original",
            "val_score": orig_val,
            "lambda": lambda_param,
            "val_template_scores": {"original": {"val_macro_f1": orig_val, "sharpness": 0.0, "score": orig_val}},
        }

    template_sens = {}
    paraphrase_flip = 0.0
    if sharpness_metrics:
        template_sens = sharpness_metrics.get("template_sensitivity", {}).get("templates", {})
        paraphrase_flip = sharpness_metrics.get("paraphrase_sensitivity", {}).get("flip_rate", 0.0)

    orig_val_f1 = val_metrics.get("original", {}).get("macro_f1", 0.0)

    scored_templates = {}
    for t_name, m in val_metrics.items():
        val_f1 = m["macro_f1"]
        # Template-specific sharpness: drop relative to original template plus overall flip rate
        t_drop = max(0.0, orig_val_f1 - val_f1)
        sharpness_val = float(0.5 * t_drop + 0.5 * paraphrase_flip)
        score = float(val_f1 - lambda_param * sharpness_val)

        scored_templates[t_name] = {
            "val_macro_f1": round(val_f1, 4),
            "val_accuracy": round(m["accuracy"], 4),
            "sharpness": round(sharpness_val, 4),
            "selection_score": round(score, 4),
        }

    # Best template by selection score
    best_template = max(scored_templates.keys(), key=lambda k: scored_templates[k]["selection_score"])

    return {
        "selected_template": best_template,
        "lambda": lambda_param,
        "selection_formula": "Score = Macro-F1 - lambda * Sharpness",
        "best_selection_score": scored_templates[best_template]["selection_score"],
        "scored_templates": scored_templates,
    }


def compute_minority_f1(per_class_f1: Dict[str, float]) -> float:
    """Average F1 for the two minority classes (love and surprise)."""
    minority_classes = ["love", "surprise"]
    scores = [per_class_f1.get(c, 0.0) for c in minority_classes if c in per_class_f1]
    return float(sum(scores) / len(scores)) if scores else 0.0


def build_comparison(
    baseline_path: Path = C.BASELINE_JSON,
    prompt_path: Path = C.RESULTS_DIR / "prompt_metrics.json",
    focal_path: Path = C.RESULTS_DIR / "focal_metrics.json",
    sharpness_path: Path = C.RESULTS_DIR / "sharpness.json",
    output_path: Path = C.RESULTS_DIR / "comparison.json",
    lambda_param: float = 0.5,
) -> Dict[str, Any]:
    """Compile comprehensive cross-method comparison."""
    baseline_data = load_json(baseline_path)
    prompt_data = load_json(prompt_path)
    focal_data = load_json(focal_path)
    sharpness_data = load_json(sharpness_path) if sharpness_path.exists() else None

    # Step 1: Prompt selection via validation data only
    selection_info = select_best_prompt_template(
        prompt_data, sharpness_data, lambda_param=lambda_param
    )
    selected_t = selection_info["selected_template"]

    # Retrieve test metrics for selected template
    if selected_t == "original" or "best_template_test_metrics" not in prompt_data or prompt_data["best_template_test_metrics"] is None:
        prompt_test_acc = prompt_data["accuracy"]
        prompt_test_f1 = prompt_data["macro_f1"]
        prompt_test_per_class = prompt_data["per_class_f1"]
        prompt_confusion = prompt_data.get("confusion_matrix")
    else:
        best_test_m = prompt_data["best_template_test_metrics"]
        prompt_test_acc = best_test_m["accuracy"]
        prompt_test_f1 = best_test_m["macro_f1"]
        prompt_test_per_class = best_test_m["per_class_f1"]
        prompt_confusion = best_test_m.get("confusion_matrix")

    prompt_sharpness_score = (
        sharpness_data["sharpness_summary"]["overall_sharpness_score"]
        if sharpness_data else None
    )

    # MiniLM configurations from focal_data
    ce_data = focal_data.get("configurations", {}).get("cross_entropy", {})
    focal_cfg_data = focal_data.get("configurations", {}).get("focal_loss", {})

    methods_dict = {
        "tfidf_logreg_baseline": {
            "name": "TF-IDF + Logistic Regression",
            "type": "classical_baseline",
            "accuracy": round(baseline_data["accuracy"], 4),
            "macro_f1": round(baseline_data["macro_f1"], 4),
            "minority_f1_love_surprise": round(compute_minority_f1(baseline_data["per_class_f1"]), 4),
            "per_class_f1": {k: round(v, 4) for k, v in baseline_data["per_class_f1"].items()},
            "sharpness": None,
            "n_test": baseline_data.get("n_test", 998),
        },
        "flan_t5_prompt": {
            "name": f"FLAN-T5 Prompt Classifier ({selected_t})",
            "type": "prompt_based_llm",
            "selected_template": selected_t,
            "accuracy": round(prompt_test_acc, 4),
            "macro_f1": round(prompt_test_f1, 4),
            "minority_f1_love_surprise": round(compute_minority_f1(prompt_test_per_class), 4),
            "per_class_f1": {k: round(v, 4) for k, v in prompt_test_per_class.items()},
            "sharpness": prompt_sharpness_score,
            "n_test": prompt_data.get("n_test", 998),
        },
        "minilm_cross_entropy": {
            "name": "MiniLM + Cross-Entropy",
            "type": "embedding_mlp",
            "accuracy": round(ce_data.get("accuracy", 0.0), 4),
            "macro_f1": round(ce_data.get("macro_f1", 0.0), 4),
            "minority_f1_love_surprise": round(compute_minority_f1(ce_data.get("per_class_f1", {})), 4),
            "per_class_f1": {k: round(v, 4) for k, v in ce_data.get("per_class_f1", {}).items()},
            "sharpness": None,
            "n_test": focal_data.get("n_test", 998),
        },
        "minilm_focal_loss": {
            "name": "MiniLM + Focal Loss (gamma=2)",
            "type": "embedding_mlp",
            "gamma": 2.0,
            "accuracy": round(focal_cfg_data.get("accuracy", focal_data.get("accuracy", 0.0)), 4),
            "macro_f1": round(focal_cfg_data.get("macro_f1", focal_data.get("macro_f1", 0.0)), 4),
            "minority_f1_love_surprise": round(compute_minority_f1(focal_cfg_data.get("per_class_f1", focal_data.get("per_class_f1", {}))), 4),
            "per_class_f1": {k: round(v, 4) for k, v in focal_cfg_data.get("per_class_f1", focal_data.get("per_class_f1", {})).items()},
            "sharpness": None,
            "n_test": focal_data.get("n_test", 998),
        },
    }

    # Rankings
    rankings = sorted(
        methods_dict.keys(),
        key=lambda k: methods_dict[k]["macro_f1"],
        reverse=True,
    )

    ranking_summary = [
        {
            "rank": rank + 1,
            "key": k,
            "name": methods_dict[k]["name"],
            "macro_f1": methods_dict[k]["macro_f1"],
            "accuracy": methods_dict[k]["accuracy"],
            "minority_f1": methods_dict[k]["minority_f1_love_surprise"],
            "improvement_over_baseline_macro_f1": round(
                methods_dict[k]["macro_f1"] - baseline_data["macro_f1"], 4
            ),
        }
        for rank, k in enumerate(rankings)
    ]

    out = {
        "prompt_template_selection": selection_info,
        "methods": methods_dict,
        "rankings_by_macro_f1": ranking_summary,
        "summary": {
            "best_method": ranking_summary[0]["name"],
            "best_macro_f1": ranking_summary[0]["macro_f1"],
            "baseline_macro_f1": round(baseline_data["macro_f1"], 4),
            "max_improvement": round(ranking_summary[0]["macro_f1"] - baseline_data["macro_f1"], 4),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n================ METHOD COMPARISON ================")
    for item in ranking_summary:
        print(f"Rank {item['rank']}: {item['name']:38s} | Macro-F1: {item['macro_f1']:.4f} | Acc: {item['accuracy']:.4f} | Minority-F1: {item['minority_f1']:.4f}")
    print(f"====================================================\nWrote comparison to {output_path}")

    return out


if __name__ == "__main__":
    build_comparison()
