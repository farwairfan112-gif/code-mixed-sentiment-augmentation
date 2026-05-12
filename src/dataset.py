"""
dataset.py — Data loading, preprocessing, and PyTorch Dataset classes.
"""

import os
import re
import subprocess

import emoji
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ── Label mappings ─────────────────────────────────────────────────────────
LABEL2ID = {"Positive": 0, "Negative": 1, "Neutral": 2}
ID2LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}

LABEL_NORMALIZE = {
    "positive": "Positive", "negative": "Negative", "neutral": "Neutral",
    "Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral",
    "POSITIVE": "Positive", "NEGATIVE": "Negative", "NEUTRAL": "Neutral",
    "pos": "Positive", "neg": "Negative", "neu": "Neutral",
    "0": "Positive", "1": "Negative", "2": "Neutral",
    0: "Positive",   1: "Negative",   2: "Neutral",
    "Mixed_feelings": "Neutral", "mixed_feelings": "Neutral",
    "Non malayalam": None, "not-malayalam": None,
    "Unknown": None, "unknown": None, "non_malayalam": None, "nan": None,
}


def normalize_label(x):
    if x is None:
        return None
    if x in LABEL_NORMALIZE:
        return LABEL_NORMALIZE[x]
    s = str(x).strip()
    return LABEL_NORMALIZE.get(s, LABEL_NORMALIZE.get(s.capitalize(), None))


def clean_text(text: str) -> str:
    """Preprocessing per paper Section 3.1."""
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s\'\".,!?;:\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_mix_score(text: str) -> float:
    """
    Lingua-based word-level CM degree score.
    Returns proportion of non-English words (proxy for L2 tokens).
    Falls back to ASCII heuristic if lingua unavailable.
    """
    try:
        from lingua import Language, LanguageDetectorBuilder
        detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.HINDI, Language.SPANISH
        ).build()
        words = text.split()
        if not words:
            return 0.0
        non_eng = sum(
            1 for w in words
            if detector.detect_language_of(w) not in (Language.ENGLISH, None)
        )
        return non_eng / len(words)
    except Exception:
        # fallback: fraction of non-ASCII chars
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return min(non_ascii / max(len(text), 1), 1.0)


def preprocess_df(df: pd.DataFrame, name: str = "",
                  add_mix_score: bool = False) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed")]
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text"})
    if "text" not in df.columns and len(df.columns) >= 2:
        df.columns = ["text", "label"] + list(df.columns[2:])
    if "label" not in df.columns:
        print(f"  [{name}] No label column — skipping.")
        return None
    df["label"] = df["label"].apply(normalize_label)
    df = df[df["label"].notna()].copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].copy()
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)
    if add_mix_score and "mix_score" not in df.columns:
        df["mix_score"] = df["text"].apply(compute_mix_score).clip(0, 1)
    print(f"  [{name}] {len(df)} rows")
    return df.reset_index(drop=True)


def download_data(repo_url="https://github.com/lindazeng979/LLM-CMSA.git",
                  repo_dir="LLM-CMSA"):
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url], check=True)
        print("Repository cloned.")
    else:
        print(f"Repository already at ./{repo_dir}")


# ── PyTorch Datasets ───────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """Standard dataset: input_ids, attention_mask, labels."""
    def __init__(self, df, tokenizer, max_len=64):
        self.texts     = df["text"].tolist()
        self.labels    = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CMDataset(SentimentDataset):
    """Adds mix_score field for CDWL training."""
    def __init__(self, df, tokenizer, max_len=64):
        super().__init__(df, tokenizer, max_len)
        if "mix_score" not in df.columns:
            df = df.copy()
            df["mix_score"] = df["text"].apply(compute_mix_score).clip(0, 1)
        self.mix_scores = df["mix_score"].tolist()

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["mix_score"] = torch.tensor(self.mix_scores[idx], dtype=torch.float)
        return item


class LAAFDataset(SentimentDataset):
    """Adds token_lang field (per-token P(English)) for LAAF Phase-1 supervision."""
    def __init__(self, df, tokenizer, max_len=64):
        super().__init__(df, tokenizer, max_len)

    def _get_token_lang(self, text, encoding):
        """
        Assign 1.0 (English) or 0.0 (L2) per token using simple heuristic.
        Words that are pure ASCII → English; others → L2.
        """
        try:
            tokens     = self.tokenizer.convert_ids_to_tokens(
                encoding["input_ids"].squeeze().tolist()
            )
            token_lang = []
            for tok in tokens:
                clean = tok.replace("▁", "").replace("##", "")
                token_lang.append(1.0 if clean.isascii() else 0.0)
            return torch.tensor(token_lang[:self.max_len], dtype=torch.float)
        except Exception:
            return torch.ones(self.max_len, dtype=torch.float)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
            "token_lang":     self._get_token_lang(self.texts[idx], enc),
        }
        return item


def build_loader(df, tokenizer, batch_size=32, shuffle=True,
                 max_len=64, num_workers=2):
    ds = SentimentDataset(df, tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)


# ── Sample data ────────────────────────────────────────────────────────────
def create_sample_data(path="data/sample_data.csv"):
    rows = [
        ("I love this! It's amazing",       "Positive"),
        ("Odio esto completamente horrible", "Negative"),
        ("The product is okay nothing special", "Neutral"),
        ("Me encanta que bueno que lo hicieron", "Positive"),
        ("This is the worst experience ever",  "Negative"),
        ("No me parece ni bueno ni malo",      "Neutral"),
        ("Absolutely fantastic result",        "Positive"),
        ("Terribly disappointed with this",    "Negative"),
        ("yaar ye toh bahut achha hai",        "Positive"),
        ("bilkul bekar hai ye cheez",          "Negative"),
    ]
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["label_id"] = df["label"].map(LABEL2ID)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Sample data saved → {path}")
