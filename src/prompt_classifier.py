"""FLAN-T5 Prompt-based Classifier for Emotion Detection.

Uses google/flan-t5-base with likelihood-based candidate label scoring.
Computes the exact log-likelihood of each candidate emotion token
under the model's conditional language modeling distribution.

Zero reliance on unconstrained text generation or brittle string parsing.
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.metrics import compute_metrics, save_metrics  # noqa: E402

# Suppress HF symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 6 Manually written prompt templates, including "original"
PROMPT_TEMPLATES: Dict[str, str] = {
    "original": (
        "Classify the emotion of the following text into one of sadness, joy, love, anger, fear, surprise:\n"
        "Text: {text}\n"
        "Emotion:"
    ),
    "template_concise": (
        "Text: {text}\n"
        "Emotion (sadness, joy, love, anger, fear, surprise):"
    ),
    "template_question": (
        "What emotion is expressed in this text: \"{text}\"? Choose from sadness, joy, love, anger, fear, surprise.\n"
        "Answer:"
    ),
    "template_context": (
        "Identify the sentiment or feeling of the author. Options: sadness, joy, love, anger, fear, surprise.\n"
        "Author's text: \"{text}\"\n"
        "Detected feeling:"
    ),
    "template_instruction": (
        "Given the input text, select the single best matching emotion from sadness, joy, love, anger, fear, surprise.\n"
        "Input: {text}\n"
        "Selection:"
    ),
    "template_roleplay": (
        "Analyze the following message and determine its primary emotion among sadness, joy, love, anger, fear, surprise.\n"
        "Message: {text}\n"
        "Emotion category:"
    ),
}


class PromptClassifier:
    """Likelihood-based Prompt Classifier using FLAN-T5."""

    def __init__(self, model_name: str = C.LLM_NAME, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Pre-extract single token IDs for each candidate emotion class
        self.candidate_tokens = [
            self.tokenizer.encode(label_str, add_special_tokens=False)[0]
            for label_str in C.LABELS
        ]
        self.decoder_start_token_id = self.model.config.decoder_start_token_id or 0

    def format_prompts(self, texts: List[str], template_name: str = "original") -> List[str]:
        template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["original"])
        return [template.format(text=t) for t in texts]

    @torch.inference_mode()
    def predict_log_likelihoods(
        self, texts: List[str], template_name: str = "original", batch_size: int = 32
    ) -> np.ndarray:
        """Compute exact log-likelihood scores for each candidate emotion token."""
        prompts = self.format_prompts(texts, template_name=template_name)
        n = len(prompts)
        all_scores = []

        for i in range(0, n, batch_size):
            batch_prompts = prompts[i : i + batch_size]
            enc = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            # 1. Run encoder forward pass once per batch
            encoder_outputs = self.model.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )

            # 2. Run decoder step 0 for candidate label tokens
            b_size = len(batch_prompts)
            decoder_input_ids = torch.full(
                (b_size, 1),
                self.decoder_start_token_id,
                dtype=torch.long,
                device=self.device,
            )

            decoder_out = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=enc["attention_mask"],
            )

            # 3. Logits over vocabulary for step 0
            step_logits = self.model.lm_head(decoder_out.last_hidden_state[:, 0, :])
            log_probs = F.log_softmax(step_logits, dim=-1)

            # 4. Extract log probabilities for the 6 candidate emotion tokens
            cand_log_probs = log_probs[:, self.candidate_tokens]
            all_scores.append(cand_log_probs.cpu().numpy())

        return np.concatenate(all_scores, axis=0)

    def predict(
        self, texts: List[str], template_name: str = "original", batch_size: int = 32
    ) -> np.ndarray:
        """Return integer predictions from 0 to 5."""
        scores = self.predict_log_likelihoods(texts, template_name=template_name, batch_size=batch_size)
        return np.argmax(scores, axis=-1).astype(int)

    def evaluate(
        self, df: pd.DataFrame, template_name: str = "original", batch_size: int = 32
    ) -> Tuple[dict, np.ndarray]:
        """Compute metrics for a given dataframe and template."""
        texts = df["text"].tolist()
        y_true = df["label"].values.astype(int)
        preds = self.predict(texts, template_name=template_name, batch_size=batch_size)
        metrics = compute_metrics(y_true, preds, C.LABELS)
        return metrics, preds


def benchmark_speed(classifier: PromptClassifier, val_df: pd.DataFrame, n_samples: int = 200) -> dict:
    """Benchmark inference speed on n_samples and estimate full evaluation time."""
    print(f"\n[Benchmarking] Running inference on {n_samples} validation samples...")
    sample_df = val_df.head(n_samples)
    texts = sample_df["text"].tolist()

    # Warmup
    _ = classifier.predict(texts[:5], template_name="original", batch_size=5)

    t0 = time.time()
    preds = classifier.predict(texts, template_name="original", batch_size=32)
    elapsed = time.time() - t0

    samples_per_sec = n_samples / elapsed
    est_1000 = 1000.0 / samples_per_sec

    print(f"Benchmark completed in {elapsed:.2f}s ({samples_per_sec:.2f} samples/sec).")
    print(f"Estimated time for 1,000 samples: {est_1000:.2f}s (~{est_1000/60:.2f} minutes).")

    return {
        "n_samples": n_samples,
        "elapsed_seconds": round(elapsed, 3),
        "samples_per_second": round(samples_per_sec, 2),
        "est_time_per_1000_seconds": round(est_1000, 2),
    }


def run_prompt_classifier_pipeline(
    save_path: Path = C.RESULTS_DIR / "prompt_metrics.json",
) -> dict:
    """Run full prompt classifier pipeline: benchmark, template evaluation, and test evaluation."""
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    val_df = pd.read_csv(C.VAL_CSV)
    test_df = pd.read_csv(C.TEST_CSV)

    classifier = PromptClassifier()

    # Step 1: Benchmark on 200 samples
    benchmark_info = benchmark_speed(classifier, val_df, n_samples=200)

    # Step 2: Evaluate all templates on validation set to measure template sensitivity
    print("\n[Validation] Evaluating all 6 prompt templates on validation set...")
    val_template_metrics = {}
    for t_name in PROMPT_TEMPLATES:
        t_start = time.time()
        m, _ = classifier.evaluate(val_df, template_name=t_name, batch_size=32)
        duration = time.time() - t_start
        val_template_metrics[t_name] = m
        print(f"  Template '{t_name:20s}' -> Val Macro-F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']:.4f} ({duration:.1f}s)")

    # Identify best template on validation
    best_template = max(val_template_metrics.keys(), key=lambda k: val_template_metrics[k]["macro_f1"])
    print(f"\nBest template on validation set: '{best_template}' (Macro-F1: {val_template_metrics[best_template]['macro_f1']:.4f})")

    # Step 3: Final evaluation on test data using original and best template
    print(f"\n[Test] Evaluating on test set ({len(test_df)} samples) with original template...")
    test_metrics_orig, test_preds_orig = classifier.evaluate(test_df, template_name="original", batch_size=32)
    print(f"Original Template Test -> Macro-F1: {test_metrics_orig['macro_f1']:.4f} | Accuracy: {test_metrics_orig['accuracy']:.4f}")

    test_metrics_best, test_preds_best = None, None
    if best_template != "original":
        print(f"[Test] Evaluating on test set with best validation template '{best_template}'...")
        test_metrics_best, test_preds_best = classifier.evaluate(test_df, template_name=best_template, batch_size=32)
        print(f"Best Template Test ('{best_template}') -> Macro-F1: {test_metrics_best['macro_f1']:.4f} | Accuracy: {test_metrics_best['accuracy']:.4f}")

    # Use save_metrics to write results/prompt_metrics.json
    out = save_metrics(
        save_path,
        method="FLAN-T5 Prompt Classifier",
        y_true=test_df["label"].values.astype(int),
        y_pred=test_preds_orig,
        labels=C.LABELS,
        template="original",
        benchmark=benchmark_info,
        val_template_metrics=val_template_metrics,
        best_val_template=best_template,
        best_template_test_metrics=test_metrics_best,
    )

    print(f"\nWrote results to {save_path}")
    return out


if __name__ == "__main__":
    run_prompt_classifier_pipeline()
