# Sharpness-Aware Prompt Optimization for Imbalanced Classification

This repository investigates prompt-based learning and sharpness sensitivity for imbalanced multi-class emotion classification on the `dair-ai/emotion` dataset.

---

## 1. Dataset & Baseline (Person 1)

**Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (`split` config), English tweets labelled with 6 emotion classes:
`sadness` (0), `joy` (1), `love` (2), `anger` (3), `fear` (4), `surprise` (5).

**Dataset Characteristics & Splits (`src/data_prep.py`):**
- **Train split:** Stratified subsample of 3,997 examples (`TRAIN_SIZE = 4000`, deduplicated). Imbalance ratio: 9.44 (majority `joy`: 1,341, minority `surprise`: 142).
- **Validation split:** Stratified subsample of 999 examples (`VAL_SIZE = 1000`). Imbalance ratio: 8.56.
- **Test split:** Stratified subsample of 998 examples (`TEST_SIZE = 1000`). Imbalance ratio: 10.48.
- **Class distribution (Real counts from `data/processed/class_distribution.json`):**

| Class | Train Count | Train % | Test Count | Test % |
|---|---|---|---|---|
| `sadness` | 1,166 | 29.17% | 290 | 29.06% |
| `joy` | 1,341 | 33.55% | 346 | 34.67% |
| `love` | 325 | 8.13% | 79 | 7.92% |
| `anger` | 539 | 13.49% | 138 | 13.83% |
| `fear` | 484 | 12.11% | 112 | 11.22% |
| `surprise` | 142 | 3.55% | 33 | 3.31% |

**Classical Baseline (`src/baseline.py`):**
TF-IDF (1-2 grams, `min_df=2`, sublinear tf) + Logistic Regression (`class_weight=None`).
- **Test Accuracy:** 67.54% (`0.6754`)
- **Test Macro-F1:** 46.75% (`0.4675`)
- **Minority Class F1 (`love` + `surprise` average):** 10.00% (`0.1000`)
- The baseline struggles severely on minority classes due to class imbalance (`surprise` F1 is only 0.0588, `love` is 0.1412).

---

## 2. Core ML Implementation & Methodology (Person 2)

Person 2 implemented five core machine learning modules covering prompt-based likelihood classification, deterministic paraphrase perturbation, black-box sharpness estimation, sentence embedding classification with Focal Loss, and systematic method comparison.

### A. Prompt-Based Classification (`src/prompt_classifier.py`)
- **Model:** `google/flan-t5-base` (local CPU inference, 250M parameters).
- **Likelihood-Based Label Scoring:** Rather than generating free-form text and parsing fragile string outputs, classification is framed as scoring candidate label verbalizers:
  1. For each input text, the prompt template formats the prompt.
  2. The FLAN-T5 encoder processes the prompt tokens.
  3. The decoder is fed the start-of-sequence token (`decoder_start_token_id`).
  4. At decoder step 0, unnormalized logits over vocabulary tokens are extracted.
  5. In `google/flan-t5-base`, all 6 emotion target words (`sadness`, `joy`, `love`, `anger`, `fear`, `surprise`) map to single distinct tokens:
     - `sadness`: 24784, `joy`: 3922, `love`: 333, `anger`: 11213, `fear`: 2971, `surprise`: 4158.
  6. The log-probabilities (or raw logits) for these 6 candidate tokens are extracted, and $\arg\max$ selects the predicted class integer (0 to 5).
  7. This guarantees valid, deterministic predictions without hallucination or parsing failures.

- **Prompt Template Variations:** 6 distinct templates were designed and evaluated on the validation set:
  1. `original`: `"Classify the emotion of this text into sadness, joy, love, anger, fear, or surprise: \"{text}\" Emotion:"`
  2. `template_concise`: `"Text: \"{text}\" Emotion (sadness, joy, love, anger, fear, surprise):"`
  3. `template_question`: `"What emotion is expressed in this text: \"{text}\"? Choose from sadness, joy, love, anger, fear, surprise. Answer:"`
  4. `template_context`: `"Social media post: \"{text}\" The primary emotion felt by the author is (sadness, joy, love, anger, fear, surprise):"`
  5. `template_instruction`: `"Given the statement \"{text}\", determine whether the feeling conveyed is sadness, joy, love, anger, fear, or surprise. Label:"`
  6. `template_roleplay`: `"You are an expert sentiment and emotion analyzer. Read the tweet: \"{text}\" Identify the emotion as sadness, joy, love, anger, fear, or surprise:"`

### B. Paraphrase Perturbation Module (`src/paraphrase.py`)
- Purpose: Generate input perturbations to quantify prediction flip rate and sensitivity to input phrasing.
- **Methodology:** Deterministic, meaning-preserving text perturbations designed to avoid uncontrolled random generation:
  - Contraction expansions and contractions (`i am` $\leftrightarrow$ `i'm`, `do not` $\leftrightarrow$ `don't`).
  - Emotion-preserving lexical substitutions on non-emotion words (e.g., `feel` $\to$ `sense`, `because` $\to$ `since`, `really` $\to$ `truly`, `think` $\to$ `believe`).
  - Strict preservation of emotional keywords (`sad`, `happy`, `angry`, `joyful`, `fearful`, `loved`, `scared`, `surprised`).
  - Consistent casing and mild conversational prefix variations (`well, ...`, `right now, ...`).
- **Reproducibility:** Fully deterministic; does not require GPU resources or external API calls.

### C. Black-Box Sharpness Evaluation (`src/sharpness.py`)
Because white-box Hessian trace calculation is intractable for frozen black-box prompts and large generative language models without direct gradient access, we measure empirical black-box sharpness along two axes:

1. **Prompt Template Sensitivity:**
   - Measures how sensitive model performance is to changes in phrasing of the prompt template.
   - Computes Macro-F1 across all 6 prompt templates on the validation set.
   - Drop relative to the primary template: $\Delta_{\text{template}} = \max(0, \text{F1}_{\text{orig}} - \text{F1}_{\text{template}})$.
   - Reports mean drop (0.0753), max drop (0.2087), and standard deviation (0.0672).

2. **Input Paraphrase Sensitivity (Prediction Flip Rate):**
   - Measures prediction stability under semantic-preserving perturbations:
     $$\text{Flip Rate} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[\hat{y}(x_i) \neq \hat{y}(x_i')]$$
   - Evaluated on the 998 test samples.
   - Result: **Flip Rate = 5.01%** (50 / 998 predictions changed), demonstrating high input-level stability.

3. **Combined Sharpness Metric:**
   $$\text{Sharpness} = 0.5 \times \text{Flip Rate} + 0.5 \times \text{Mean Template Drop} = 0.5(0.0501) + 0.5(0.0753) = 0.0627$$

### D. Sentence Embedding Classifier with Focal Loss (`src/focal_classifier.py`)
- **Encoder:** Frozen `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings).
- **Embedding Cache:** Embeddings for train (3,997), validation (999), and test (998) were pre-computed once and cached (`data/processed/minilm_embeddings.pt`), enabling rapid hyperparameter exploration without repeated inference.
- **Classification Head:** PyTorch MLP (`384 -> 128 -> ReLU -> Dropout(0.2) -> 6`).
- **Loss Functions Compared:**
  1. **Standard Cross-Entropy Loss:**
     $$\mathcal{L}_{\text{CE}} = - \log(p_t)$$
  2. **Focal Loss ($\gamma = 2.0$):**
     $$\mathcal{L}_{\text{Focal}} = - (1 - p_t)^\gamma \log(p_t)$$
     Down-weights easy well-classified examples ($p_t \to 1$) to force the model to focus on hard and underrepresented minority examples.
- **Training Setup:** AdamW optimizer, lr = $1 \times 10^{-3}$, weight decay = $1 \times 10^{-4}$, batch size = 64, 25 epochs with early stopping on validation Macro-F1.

### E. Comparative Analysis & Prompt Selection (`src/compare.py`)
- **Prompt Selection Protocol:** Validation data only was used for prompt template selection using the objective:
  $$\text{Score} = \text{Macro-F1}_{\text{val}} - \lambda \times \text{Sharpness}_{\text{val}} \quad (\lambda = 0.5)$$
  The `original` template achieved the highest selection score (0.6587) and was selected for final test set evaluation. Test data was never accessed during selection.

---

## 3. Results & Method Comparison

All methods were evaluated on the identical test split ($N = 998$). Metrics were computed using the shared evaluation schema (`src/metrics.py`).

| Rank | Method | Accuracy | Macro-F1 | Minority F1 (`love`+`surprise`) | Sharpness | Gain over Baseline (Macro-F1) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **1** | **FLAN-T5 Prompt Classifier (`original`)** | **71.04%** | **0.6657** | **0.5283** | **0.0627** | **+0.1982** (+19.82%) |
| 2 | MiniLM + Cross-Entropy Loss | 68.24% | 0.5928 | 0.4167 | *null* | +0.1253 (+12.53%) |
| 3 | MiniLM + Focal Loss ($\gamma = 2.0$) | 66.23% | 0.5658 | 0.3511 | *null* | +0.0983 (+9.83%) |
| 4 | TF-IDF + Logistic Regression (Baseline) | 67.54% | 0.4675 | 0.1000 | *null* | 0.0000 (reference) |

### Detailed Per-Class F1 Breakdown

| Emotion Class | Baseline (TF-IDF) | MiniLM + Cross-Entropy | MiniLM + Focal Loss | FLAN-T5 Prompt Classifier |
|---|:---:|:---:|:---:|:---:|
| `sadness` (majority) | 0.7504 | 0.7424 | 0.7248 | **0.7469** |
| `joy` (majority) | 0.7466 | 0.7371 | 0.7200 | **0.7579** |
| `love` (minority) | 0.1412 | 0.4604 | 0.3919 | **0.5340** |
| `anger` | 0.5657 | 0.6667 | 0.6569 | **0.7292** |
| `fear` | 0.5422 | 0.5771 | 0.5909 | **0.7037** |
| `surprise` (minority) | 0.0588 | 0.3729 | 0.3103 | **0.5225** |

### Key Findings
1. **Zero-Shot Prompting Superiority on Minority Classes:** FLAN-T5 achieves substantial gains on the most severely imbalanced minority classes (`love`: 0.5340 vs. 0.1412; `surprise`: 0.5225 vs. 0.0588), leveraging pre-trained semantic priors without requiring class re-balancing weights.
2. **Template Sensitivity Impact:** Prompt template wording has a marked impact on performance: the concise template drops Macro-F1 by 20.87 percentage points on validation, underscoring the necessity of template sharpness regularization.
3. **Focal Loss vs. Cross-Entropy on Embeddings:** In frozen MiniLM embeddings with an MLP head, standard Cross-Entropy slightly outperformed Focal Loss ($\gamma = 2.0$) overall, as aggressive down-weighting of majority samples induced a slight precision penalty without outperforming pre-trained LLM representations.

---

## 4. Deliverables & Output Files

The following deliverables were generated and verified:

### Source Modules (`src/`)
1. `src/prompt_classifier.py`: FLAN-T5 likelihood-based scoring across 6 templates, batch inference, validation & test routines.
2. `src/paraphrase.py`: Deterministic text perturbation engine preserving emotion semantics.
3. `src/sharpness.py`: Black-box sharpness evaluator (template drop & input paraphrase flip rate).
4. `src/focal_classifier.py`: MiniLM frozen embedding cache, PyTorch MLP, Cross-Entropy & Focal Loss ($\gamma = 2.0$).
5. `src/compare.py`: Model ranking, sharpness-aware prompt selection, and consolidated evaluation table.

### Generated Results (`results/`)
6. `results/prompt_metrics.json`: Full benchmark, validation template evaluations, and test metrics for FLAN-T5.
7. `results/focal_metrics.json`: Test metrics and training convergence logs for Cross-Entropy and Focal Loss.
8. `results/sharpness.json`: Template sensitivity drops, paraphrase flip rates, and overall sharpness scores.
9. `results/paraphrase_predictions.csv`: Sample-by-sample original vs. paraphrased predictions for all 998 test examples.
10. `results/comparison.json`: Unified comparison schema ranking all four approaches with per-class and minority metrics.

---

## 5. Scope & Future Work (Explicitly NOT Implemented)

In accordance with project scope and computational constraints (CPU execution environment, no gradient backpropagation through frozen discrete prompts), the following techniques are **future work and have NOT been implemented in this repository**:
- **True Hessian Trace Calculation:** White-box Hessian-based curvature estimation ($\text{Tr}(H)$) requires second-order gradient backpropagation through the model parameters, which is computationally prohibitive on CPU and ill-defined for discrete text prompt inputs.
- **Soft Prompt Continuous Optimization:** Continuous prompt tuning (Prefix Tuning / P-Tuning) where prompt embeddings are optimized via gradient descent.
- **Validation Against Hessian Trace:** Empirical validation comparing black-box sensitivity metrics to true eigenvalues of the Hessian matrix.
- **DSPy / TextGrad Optimization:** Automatic iterative prompt optimization using DSPy teleprompters or TextGrad symbolic backpropagation.
