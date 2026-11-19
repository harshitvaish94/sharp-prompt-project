"""Paraphrase module for input sensitivity and sharpness evaluation.

Generates meaning-preserving input variations to evaluate model prediction stability (flip rate).
Uses a deterministic, reproducible perturbation approach that:
1. Expands / contracts common English contractions (e.g., "i am" <-> "i'm", "do not" <-> "don't").
2. Replaces non-emotion context words with controlled synonyms, strictly preserving emotion cues.
3. Modulates non-semantic sentence prefixes / filler phrases.
4. Normalizes casing and whitespace.

All transformations are deterministic and reproducible given a random seed.
"""
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

# Emotion-carrying keywords that must NEVER be mutated to preserve true class label
PROTECTED_EMOTION_WORDS = {
    "sad", "sadness", "depressed", "depressing", "gloomy", "miserable", "heartbroken", "grief", "unhappy",
    "joy", "happy", "joyful", "delighted", "cheerful", "glad", "ecstatic", "content", "blessed",
    "love", "loving", "beloved", "caring", "affection", "adore", "passionate", "fond",
    "anger", "angry", "furious", "mad", "irritated", "annoyed", "enraged", "pissed", "bitter", "hate",
    "fear", "scared", "afraid", "terrified", "frightened", "anxious", "nervous", "panicked",
    "surprise", "surprised", "shocked", "amazed", "astonished", "stunned", "unexpected",
}

# Contraction expansion / compression mappings
CONTRACTIONS_EXPAND = {
    r"\bi'm\b": "i am",
    r"\bi've\b": "i have",
    r"\bi'll\b": "i will",
    r"\bi'd\b": "i would",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
}

CONTRACTIONS_COMPACT = {
    r"\bi am\b": "i'm",
    r"\bi have\b": "i've",
    r"\bi will\b": "i'll",
    r"\bi would\b": "i'd",
    r"\bdo not\b": "don't",
    r"\bdoes not\b": "doesn't",
    r"\bdid not\b": "didn't",
    r"\bcannot\b": "can't",
    r"\bwill not\b": "won't",
    r"\bis not\b": "isn't",
    r"\bare not\b": "aren't",
    r"\bwas not\b": "wasn't",
    r"\bwere not\b": "weren't",
    r"\bhave not\b": "haven't",
    r"\bhas not\b": "hasn't",
    r"\bit is\b": "it's",
    r"\bthat is\b": "that's",
}

# Safe context synonyms that do not change sentiment/emotion
SAFE_SYNONYMS = {
    "feel": "sense",
    "feeling": "experiencing",
    "really": "truly",
    "very": "extremely",
    "quite": "rather",
    "pretty": "fairly",
    "lot of": "plenty of",
    "a lot": "a great deal",
    "going to": "about to",
    "little": "slight",
    "always": "constantly",
    "because": "since",
    "think": "believe",
    "actually": "in reality",
    "start": "begin",
    "difficult": "hard",
    "make": "render",
    "look": "appear",
    "around": "about",
    "someone": "somebody",
    "anyone": "anybody",
}


def perturb_text(text: str, variant_idx: int = 0) -> str:
    """Apply deterministic, meaning-preserving perturbations to text.

    variant_idx selects a reproducible perturbation strategy:
      0: Safe synonym replacement + contraction swap
      1: Prefix insertion / removal + contraction expansion
      2: Combined contextual substitution + phrasing variation
    """
    cleaned = text.strip()
    if not cleaned:
        return text

    words = cleaned.split()

    if variant_idx % 3 == 0:
        # Strategy 0: Context synonym replacement + contraction toggle
        res = cleaned.lower()
        for k, v in CONTRACTIONS_COMPACT.items():
            if re.search(k, res):
                res = re.sub(k, v, res, count=1)
                break
        else:
            for k, v in CONTRACTIONS_EXPAND.items():
                if re.search(k, res):
                    res = re.sub(k, v, res, count=1)
                    break

        for orig_word, syn_word in SAFE_SYNONYMS.items():
            pattern = rf"\b{re.escape(orig_word)}\b"
            if re.search(pattern, res, flags=re.IGNORECASE):
                # Ensure it's not protected
                if orig_word not in PROTECTED_EMOTION_WORDS:
                    res = re.sub(pattern, syn_word, res, count=1, flags=re.IGNORECASE)
                    break
        return res

    elif variant_idx % 3 == 1:
        # Strategy 1: Contraction expansion + slight structural variation
        res = cleaned
        for k, v in CONTRACTIONS_EXPAND.items():
            res = re.sub(k, v, res, flags=re.IGNORECASE)
        # Prefix perturbation if text starts with "i "
        if res.lower().startswith("i ") and not res.lower().startswith("i feel"):
            res = "well, " + res
        elif res.lower().startswith("i feel"):
            res = "right now, " + res
        return res.strip()

    else:
        # Strategy 2: Context synonym + casing / sentence ending normalization
        res = cleaned.lower()
        for orig_word, syn_word in list(SAFE_SYNONYMS.items())[5:]:
            pattern = rf"\b{re.escape(orig_word)}\b"
            if re.search(pattern, res):
                res = re.sub(pattern, syn_word, res, count=1)
                break
        if not res.endswith((".", "!", "?")):
            res = res + "."
        return res


def paraphrase_text(text: str, seed: int = C.SEED) -> str:
    """Return a single high-quality deterministic paraphrase of text."""
    # Hash text to pick variant deterministically but varied across samples
    h = abs(hash(f"{seed}_{text}")) % 3
    paraphrased = perturb_text(text, variant_idx=h)
    # If identical, try another variant
    if paraphrased.strip().lower() == text.strip().lower():
        paraphrased = perturb_text(text, variant_idx=(h + 1) % 3)
    # If still identical, prepend subtle marker
    if paraphrased.strip().lower() == text.strip().lower():
        paraphrased = "honestly, " + text.strip()
    return paraphrased


def paraphrase_batch(texts: List[str], seed: int = C.SEED) -> List[str]:
    """Paraphrase a list of texts deterministically."""
    return [paraphrase_text(t, seed=seed + i) for i, t in enumerate(texts)]


if __name__ == "__main__":
    examples = [
        "i feel that this leads to not many people caring who get s the real job as sin cara",
        "i go to sleep i feel as if i m giving up precious time to do something else with my life",
        "i want to take a shower without feeling like i was beaten with a baseball bat",
        "i dont know what crazy girl i think her name was katja does for a living",
    ]
    print("Testing Paraphrase Module:")
    for ex in examples:
        p = paraphrase_text(ex)
        print(f"Orig: {ex}")
        print(f"Para: {p}\n")
