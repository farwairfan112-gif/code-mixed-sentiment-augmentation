"""
Dataset classes for Sentiment Analysis with Code-Mixed text
Supports:
- Standard dataset (text, labels)
- CDWL dataset (with mix_score for loss weighting)
- LAAF dataset (with token-level language labels)
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from lingua import Language, LanguageDetectorBuilder
import re
import emoji


# Global language detector (English vs Hindi)
LANG_DETECTOR = None

LABEL2ID = {'Positive': 0, 'Negative': 1, 'Neutral': 2}


def init_language_detector():
    """Initialize the lingua language detector for English vs Hindi"""
    global LANG_DETECTOR
    if LANG_DETECTOR is None:
        LANG_DETECTOR = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.HINDI
        ).with_low_accuracy_mode().build()
    return LANG_DETECTOR


# ============================================================
# Preprocessing functions (copy from your notebook)
# ============================================================

LABEL_NORMALIZE = {
    'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral',
    'Positive': 'Positive', 'Negative': 'Negative', 'Neutral': 'Neutral',
    'POSITIVE': 'Positive', 'NEGATIVE': 'Negative', 'NEUTRAL': 'Neutral',
    'pos': 'Positive', 'neg': 'Negative', 'neu': 'Neutral',
    'POS': 'Positive', 'NEG': 'Negative', 'NEU': 'Neutral',
    '0': 'Positive', '1': 'Negative', '2': 'Neutral',
    0: 'Positive', 1: 'Negative', 2: 'Neutral',
    'Mixed_feelings': 'Neutral', 'mixed_feelings': 'Neutral',
    'Non malayalam': None, 'not-malayalam': None,
    'Unknown': None, 'unknown': None, 'nan': None,
}


def normalize_label(x):
    if x is None:
        return None
    if x in LABEL_NORMALIZE:
        return LABEL_NORMALIZE[x]
    s = str(x).strip()
    if s in LABEL_NORMALIZE:
        return LABEL_NORMALIZE[s]
    if s.capitalize() in LABEL_NORMALIZE:
        return LABEL_NORMALIZE[s.capitalize()]
    return None


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s\'\".,!?;:\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def code_mix_score(text, lang_detector, cap_at_half=False):
    """
    CM degree: fraction of words detected as Hindi (non-English).
    FIX 1: Uses lingua for proper Romanized Hindi detection.
    cap_at_half: FIX 5 - for Ma-En data to prevent script inflation.
    """
    words = text.split()
    if not words:
        return 0.0
    hindi_count = 0
    for w in words:
        detected = lang_detector.detect_language_of(w)
        if detected == Language.HINDI:
            hindi_count += 1
    score = hindi_count / len(words)
    if cap_at_half:
        score = min(score, 0.5)
    return score


def preprocess_df(df, name='', cap_mix_at_half=False):
    """Preprocess dataframe: clean text, normalize labels, compute mix_score"""
    detector = init_language_detector()
    
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith('unnamed')]
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    if 'sentence' in df.columns:
        df = df.rename(columns={'sentence': 'text'})
    if 'text' not in df.columns:
        df.columns = ['text', 'label'] + list(df.columns[2:])
    
    if 'label' not in df.columns:
        print(f'  [{name}] No label column — blind test set, skipping')
        return None
    
    df['label'] = df['label'].apply(normalize_label)
    before = len(df)
    df = df[df['label'].notna()].copy()
    if before != len(df):
        print(f'  [{name}] Dropped {before - len(df)} invalid rows')
    
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0].copy()
    df['label_id'] = df['label'].map(LABEL2ID).astype(int)
    df['mix_score'] = df['text'].apply(lambda t: code_mix_score(t, detector, cap_at_half=cap_mix_at_half))
    
    print(f'  [{name}] {len(df)} samples | avg_mix: {df["mix_score"].mean():.3f}')
    return df[['text', 'label', 'label_id', 'mix_score']].reset_index(drop=True)


def load_conll(file_path):
    """Parse CONLL format file for SentiMix dataset"""
    sentences, labels = [], []
    current_sentence, current_label = [], None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence and current_label is not None:
                    sentences.append(' '.join(current_sentence))
                    labels.append(current_label)
                    current_sentence = []
                    current_label = None
            elif line.startswith('meta\t'):
                parts = line.split('\t')
                current_label = parts[-1].strip().capitalize()
            else:
                word = line.split()[0]
                current_sentence.append(word)

    if current_sentence and current_label is not None:
        sentences.append(' '.join(current_sentence))
        labels.append(current_label)

    return pd.DataFrame({'text': sentences, 'label': labels})


# ============================================================
# Dataset Classes
# ============================================================

class SentimentDataset(Dataset):
    """Standard dataset for sentiment classification"""
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df['text'].tolist()
        self.labels = df['label_id'].tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class CMDataset(Dataset):
    """Dataset that returns mix_score for CDWL loss weighting."""
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df['text'].tolist()
        self.labels = df['label_id'].tolist()
        self.mix_scores = df['mix_score'].tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'mix_score': torch.tensor(self.mix_scores[idx], dtype=torch.float)
        }


class LAAFDataset(Dataset):
    """
    Dataset for LAAF that also returns per-token language labels.
    FIX 2: Used in Phase 1 warmup to provide supervision for lang_scorer.
    """
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df['text'].tolist()
        self.labels = df['label_id'].tolist()
        self.tok = tokenizer
        self.max_len = max_len
        
        detector = init_language_detector()
        
        # Pre-compute per-sentence word language labels
        self.word_lang_labels = []
        for t in self.texts:
            wls = []
            for w in t.split():
                det = detector.detect_language_of(w)
                wls.append(1.0 if det == Language.ENGLISH else 0.0)
            self.word_lang_labels.append(wls)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       padding='max_length', truncation=True,
                       return_tensors='pt', return_offsets_mapping=False)
        input_ids = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()

        # Build token-level language labels by word alignment
        word_labels = self.word_lang_labels[idx]
        token_lang = torch.zeros(self.max_len, dtype=torch.float)
        words = self.texts[idx].split()
        pos = 1  # skip [CLS] / <s>
        
        for wi, w in enumerate(words):
            if pos >= self.max_len - 1:
                break
            wtoks = self.tok(w, add_special_tokens=False)['input_ids']
            lang_val = word_labels[wi] if wi < len(word_labels) else 0.5
            for _ in wtoks:
                if pos < self.max_len - 1:
                    token_lang[pos] = lang_val
                    pos += 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'token_lang': token_lang
        }


def build_loader(df, tokenizer, batch_size=32, shuffle=True, 
                 with_mix_score=False, with_token_lang=False, max_len=64):
    """Build dataloader with appropriate dataset class"""
    if with_token_lang:
        cls = LAAFDataset
    elif with_mix_score:
        cls = CMDataset
    else:
        cls = SentimentDataset
    return DataLoader(cls(df, tokenizer, max_len), batch_size=batch_size,
                      shuffle=shuffle, num_workers=2)