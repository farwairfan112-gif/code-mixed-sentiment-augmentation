"""
utils.py  —  Training loops, evaluation, and experiment runners.

Implements:
  - train_one_epoch()         : standard CE training step
  - train_one_epoch_cdwl()    : CDWL-weighted training step
  - evaluate()                : weighted F1 evaluation
  - run_standard_training()   : single-stage fine-tuning (CLAT / reproduction)
  - run_cdwl_training()       : CDWL training
  - run_laaf_training()       : 2-phase LAAF training
  - run_combined_training()   : CDWL + LAAF combined
  - run_gradual_finetuning()  : 5-stage GFT (paper's Spanish-English method)
  - save_checkpoint()         : save model + tokenizer + metadata
"""

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

from src.dataset import SentimentDataset, build_loader
from src.model import (
    MODEL_NAMES, XLMTwithLAAF, mBERTwithLAAF, build_standard_model,
)

# ── Gradual fine-tuning stages (paper Appendix B.2) ──────────────────────────
GRADUAL_STAGES_LR  = [1e-6, 2e-6, 2e-6, 4e-6, 2e-6]
GRADUAL_SYN_SIZES  = [50000, 25000, 15000, 5000, 0]

ID2LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader: DataLoader, device) -> tuple:
    """
    Evaluate model on a DataLoader.
    Returns: (weighted_f1, predictions, true_labels)
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids,
                                   attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return f1, all_preds, all_labels


def print_classification_report(labels, preds):
    """Safe classification_report that always passes all 3 label ids."""
    print(classification_report(
        labels, preds,
        labels=[0, 1, 2],
        target_names=["Positive", "Negative", "Neutral"],
        zero_division=0,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, checkpoint_dir: str,
                    metadata: Optional[dict] = None) -> None:
    """
    Save model weights, tokenizer, and metadata to checkpoint_dir.
    For LAAF models (XLMTwithLAAF / mBERTwithLAAF) we save state_dict
    and encoder config separately because they are not plain
    AutoModelForSequenceClassification instances.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Detect LAAF vs standard
    if hasattr(model, "laaf"):
        # Custom LAAF model — save state_dict + encoder config
        torch.save(model.state_dict(),
                   os.path.join(checkpoint_dir, "model_state_dict.pt"))
        model.encoder.config.save_pretrained(checkpoint_dir)
        print(f"  [LAAF] state_dict saved to {checkpoint_dir}/model_state_dict.pt")
    else:
        # Standard HuggingFace model
        model.save_pretrained(checkpoint_dir)
        print(f"  Model saved to {checkpoint_dir}")

    tokenizer.save_pretrained(checkpoint_dir)
    print(f"  Tokenizer saved to {checkpoint_dir}")

    if metadata:
        meta_path = os.path.join(checkpoint_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved to {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Standard training (CLAT / paper reproduction)
# ─────────────────────────────────────────────────────────────────────────────

def run_standard_training(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    experiment_name: str,
    device,
    num_epochs:   int   = 3,
    lr:           float = 5e-5,
    weight_decay: float = 0.01,
    batch_size:   int   = 32,
    max_seq_len:  int   = 64,
    save_dir:     Optional[str] = None,
    seed:         int   = 42,
):
    """
    Standard single-stage fine-tuning (cross-entropy, no custom head).
    Used for CLAT baselines (C1–C4) and paper reproduction experiments.

    Returns: (test_f1, train_losses, dev_f1s, test_preds, test_labels)
    """
    print(f"\n{'='*65}")
    print(f"STANDARD [{model_name}]: {experiment_name}")
    print(f"  Train: {len(train_df)}  Test: {len(test_df)}  Epochs: {num_epochs}")
    print(f"{'='*65}")

    torch.manual_seed(seed)
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    model        = build_standard_model(model_name, device=device)
    train_loader = build_loader(train_df, tokenizer, batch_size, shuffle=True,
                                max_len=max_seq_len)
    dev_loader   = build_loader(dev_df,   tokenizer, batch_size, shuffle=False,
                                max_len=max_seq_len)
    test_loader  = build_loader(test_df,  tokenizer, batch_size, shuffle=False,
                                max_len=max_seq_len)

    optimizer = AdamW(model.parameters(), lr=lr,
                      weight_decay=weight_decay, eps=1e-8)
    total_steps = len(train_loader) * num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    train_losses, dev_f1s = [], []
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        loss = total_loss / len(train_loader)
        dev_f1, _, _ = evaluate(model, dev_loader, device)
        train_losses.append(loss)
        dev_f1s.append(dev_f1)
        print(f"  Epoch {epoch}/{num_epochs} | Loss: {loss:.4f} | Dev F1: {dev_f1:.4f}")

    test_f1, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\n  TEST F1: {test_f1:.4f}")
    print_classification_report(test_labels, test_preds)

    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, metadata={
            "experiment": experiment_name, "model": model_name,
            "test_f1": test_f1, "epochs": num_epochs, "lr": lr,
            "saved_at": datetime.now().isoformat(),
        })

    del model
    torch.cuda.empty_cache()
    return test_f1, train_losses, dev_f1s, test_preds, test_labels


# ─────────────────────────────────────────────────────────────────────────────
# CDWL training
# ─────────────────────────────────────────────────────────────────────────────

def run_cdwl_training(
    model_name: str,
    train_df:   pd.DataFrame,
    dev_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    experiment_name: str,
    device,
    num_epochs:   int   = 3,
    lr:           float = 5e-5,
    weight_decay: float = 0.01,
    batch_size:   int   = 32,
    max_seq_len:  int   = 64,
    save_dir:     Optional[str] = None,
    seed:         int   = 42,
):
    """
    CDWL: Code-Mix Degree Weighted Loss.

    Loss per sample = CrossEntropy * (1 + mix_score)
    mix_score is computed by lingua word-level language detection (FIX 1),
    replacing the inaccurate ASCII heuristic used in early experiments.
    Ma-En mix_scores are capped at 0.5 during preprocessing (FIX 5).

    Returns: (test_f1, train_losses, dev_f1s, test_preds, test_labels)
    """
    from src.dataset import CMDataset  # imports CMDataset with mix_score field

    print(f"\n{'='*65}")
    print(f"CDWL [{model_name}]: {experiment_name}")
    print(f"  Train: {len(train_df)}  Test: {len(test_df)}  Epochs: {num_epochs}")
    print(f"{'='*65}")

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])

    def build_cm_loader(df, shuffle):
        ds = CMDataset(df, tokenizer, max_len=max_seq_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=2)

    model        = build_standard_model(model_name, device=device)
    train_loader = build_cm_loader(train_df, shuffle=True)
    dev_loader   = build_loader(dev_df,   tokenizer, batch_size, shuffle=False,
                                max_len=max_seq_len)
    test_loader  = build_loader(test_df,  tokenizer, batch_size, shuffle=False,
                                max_len=max_seq_len)

    optimizer = AdamW(model.parameters(), lr=lr,
                      weight_decay=weight_decay, eps=1e-8)
    total_steps = len(train_loader) * num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    train_losses, dev_f1s = [], []
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            labels      = batch["labels"].to(device)
            mix_scores  = batch["mix_score"].to(device)
            per_sample  = ce_loss(logits, labels)
            weighted    = (per_sample * (1.0 + mix_scores)).mean()
            weighted.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += weighted.item()
        loss = total_loss / len(train_loader)
        dev_f1, _, _ = evaluate(model, dev_loader, device)
        train_losses.append(loss)
        dev_f1s.append(dev_f1)
        print(f"  Epoch {epoch}/{num_epochs} | Loss: {loss:.4f} | Dev F1: {dev_f1:.4f}")

    test_f1, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\n  TEST F1: {test_f1:.4f}")
    print_classification_report(test_labels, test_preds)

    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, metadata={
            "experiment": experiment_name, "model": model_name,
            "method": "CDWL", "test_f1": test_f1,
            "saved_at": datetime.now().isoformat(),
        })

    del model
    torch.cuda.empty_cache()
    return test_f1, train_losses, dev_f1s, test_preds, test_labels


# ─────────────────────────────────────────────────────────────────────────────
# LAAF 2-phase training
# ─────────────────────────────────────────────────────────────────────────────

def run_laaf_training(
    train_df: pd.DataFrame,
    dev_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    experiment_name: str,
    device,
    model_class=None,    # XLMTwithLAAF or mBERTwithLAAF
    warmup_epochs: int   = 2,
    joint_epochs:  int   = 3,
    lr:            float = 5e-5,
    weight_decay:  float = 0.01,
    batch_size:    int   = 32,
    max_seq_len:   int   = 64,
    save_dir:      Optional[str] = None,
    seed:          int   = 42,
):
    """
    2-Phase LAAF training.

    Phase 1 (warmup_epochs): encoder FROZEN, only LAAF head trained.
      - Uses LAAFDataset with token_lang labels for auxiliary BCE supervision (FIX 2).
      - LR = lr × 3  (FIX 3: was lr × 10, reduced to prevent head overfitting).
    Phase 2 (joint_epochs): full model unfrozen, joint fine-tuning.
      - Encoder LR = lr, LAAF head LR = lr × 3.

    Returns: (test_f1, all_losses, all_dev_f1s, test_preds, test_labels)
    """
    from src.dataset import LAAFDataset

    if model_class is None:
        model_class = XLMTwithLAAF

    tok_name = (MODEL_NAMES["mBERT"] if model_class == mBERTwithLAAF
                else MODEL_NAMES["XLM-T"])
    print(f"\n{'='*65}")
    print(f"LAAF [{model_class.__name__}]: {experiment_name}")
    print(f"  Train: {len(train_df)}  Warmup: {warmup_epochs}  Joint: {joint_epochs}")
    print(f"{'='*65}")

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model     = model_class().to(device)

    tr_loader_p1 = DataLoader(
        LAAFDataset(train_df, tokenizer, max_len=max_seq_len),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    tr_loader_p2 = build_loader(train_df, tokenizer, batch_size, shuffle=True,
                                max_len=max_seq_len)
    dev_loader   = build_loader(dev_df,   tokenizer, batch_size, shuffle=False,
                                max_len=max_seq_len)
    test_loader  = build_loader(test_df,  tokenizer, batch_size, shuffle=False,
                                max_len=max_seq_len)

    all_losses, all_dev_f1s = [], []

    # ── Phase 1: frozen encoder ──────────────────────────────────────────────
    WARMUP_LR = lr * 3   # FIX 3
    print(f"[Phase 1] Encoder frozen | LR={WARMUP_LR:.2e}")
    model.freeze_encoder()
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=WARMUP_LR, weight_decay=weight_decay, eps=1e-8)
    sch1 = get_linear_schedule_with_warmup(
        opt1,
        num_warmup_steps=int(0.1 * len(tr_loader_p1) * warmup_epochs),
        num_training_steps=len(tr_loader_p1) * warmup_epochs,
    )

    for ep in range(1, warmup_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tr_loader_p1:
            opt1.zero_grad()
            out = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
                batch["token_lang"].to(device),   # FIX 2: aux supervision
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            sch1.step()
            total_loss += out.loss.item()
        loss = total_loss / len(tr_loader_p1)
        dev_f1, _, _ = evaluate(model, dev_loader, device)
        all_losses.append(loss)
        all_dev_f1s.append(dev_f1)
        print(f"  [W{ep}] Loss: {loss:.4f} | Dev F1: {dev_f1:.4f}")

    # ── Phase 2: full model ──────────────────────────────────────────────────
    print("[Phase 2] Full model unfrozen — joint fine-tuning")
    model.unfreeze_encoder()
    opt2 = AdamW([
        {"params": model.encoder.parameters(), "lr": lr},
        {"params": model.laaf.parameters(),    "lr": lr * 3},
    ], weight_decay=weight_decay, eps=1e-8)
    sch2 = get_linear_schedule_with_warmup(
        opt2,
        num_warmup_steps=int(0.1 * len(tr_loader_p2) * joint_epochs),
        num_training_steps=len(tr_loader_p2) * joint_epochs,
    )

    for ep in range(1, joint_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tr_loader_p2:
            opt2.zero_grad()
            out = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            sch2.step()
            total_loss += out.loss.item()
        loss = total_loss / len(tr_loader_p2)
        dev_f1, _, _ = evaluate(model, dev_loader, device)
        all_losses.append(loss)
        all_dev_f1s.append(dev_f1)
        print(f"  [J{ep}] Loss: {loss:.4f} | Dev F1: {dev_f1:.4f}")

    test_f1, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\n  TEST F1: {test_f1:.4f}")
    print_classification_report(test_labels, test_preds)

    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, metadata={
            "experiment": experiment_name,
            "model": model_class.__name__,
            "method": "LAAF",
            "test_f1": test_f1,
            "warmup_epochs": warmup_epochs,
            "joint_epochs": joint_epochs,
            "saved_at": datetime.now().isoformat(),
        })

    del model
    torch.cuda.empty_cache()
    return test_f1, all_losses, all_dev_f1s, test_preds, test_labels


# ─────────────────────────────────────────────────────────────────────────────
# Combined CDWL + LAAF training
# ─────────────────────────────────────────────────────────────────────────────

def run_combined_training(
    train_df: pd.DataFrame,
    dev_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    experiment_name: str,
    device,
    model_class=None,
    warmup_epochs: int   = 2,
    joint_epochs:  int   = 3,
    lr:            float = 5e-5,
    weight_decay:  float = 0.01,
    batch_size:    int   = 32,
    max_seq_len:   int   = 64,
    save_dir:      Optional[str] = None,
    seed:          int   = 42,
):
    """
    Combined CDWL + LAAF with all 5 fixes applied.

    Phase 1: encoder frozen, LAAF head + lang_scorer trained with aux BCE,
             AND CDWL-weighted CE loss on sentiment.
    Phase 2: full model, CDWL-weighted CE only (no aux loss).

    Returns: (test_f1, all_losses, all_dev_f1s, test_preds, test_labels)
    """
    from src.dataset import LAAFDataset

    if model_class is None:
        model_class = XLMTwithLAAF

    tok_name = (MODEL_NAMES["mBERT"] if model_class == mBERTwithLAAF
                else MODEL_NAMES["XLM-T"])
    print(f"\n{'='*65}")
    print(f"COMBINED CDWL+LAAF [{model_class.__name__}]: {experiment_name}")
    print(f"  Train: {len(train_df)}  Warmup: {warmup_epochs}  Joint: {joint_epochs}")
    print(f"{'='*65}")

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model     = model_class().to(device)
    ce_loss   = nn.CrossEntropyLoss(reduction="none")

    # Combined Dataset: LAAFDataset extended with mix_score
    class CombinedDataset(Dataset):
        def __init__(self, df, tok, max_len):
            self.base       = LAAFDataset(df, tok, max_len=max_len)
            self.mix_scores = df["mix_score"].tolist()
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            item = self.base[idx]
            item["mix_score"] = torch.tensor(self.mix_scores[idx], dtype=torch.float)
            return item

    tr_loader_p1 = DataLoader(
        CombinedDataset(train_df, tokenizer, max_seq_len),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    tr_loader_p2 = DataLoader(
        CombinedDataset(train_df, tokenizer, max_seq_len),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    dev_loader  = build_loader(dev_df,  tokenizer, batch_size, shuffle=False,
                               max_len=max_seq_len)
    test_loader = build_loader(test_df, tokenizer, batch_size, shuffle=False,
                               max_len=max_seq_len)

    all_losses, all_dev_f1s = [], []

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    WARMUP_LR = lr * 3
    model.freeze_encoder()
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=WARMUP_LR, weight_decay=weight_decay, eps=1e-8)
    sch1 = get_linear_schedule_with_warmup(
        opt1,
        num_warmup_steps=int(0.1 * len(tr_loader_p1) * warmup_epochs),
        num_training_steps=len(tr_loader_p1) * warmup_epochs,
    )
    print(f"[Phase 1] Encoder frozen | LR={WARMUP_LR:.2e}")

    for ep in range(1, warmup_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tr_loader_p1:
            opt1.zero_grad()
            logits, aux_loss = model.laaf(
                model.encoder(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).last_hidden_state,
                batch["attention_mask"].to(device),
                batch["token_lang"].to(device),
            )
            labels     = batch["labels"].to(device)
            mix_scores = batch["mix_score"].to(device)
            per_sample = ce_loss(logits, labels)
            loss = (per_sample * (1.0 + mix_scores)).mean()
            if aux_loss is not None:
                loss = loss + 0.3 * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            sch1.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(tr_loader_p1)
        dev_f1, _, _ = evaluate(model, dev_loader, device)
        all_losses.append(epoch_loss)
        all_dev_f1s.append(dev_f1)
        print(f"  [W{ep}] Loss: {epoch_loss:.4f} | Dev F1: {dev_f1:.4f}")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    model.unfreeze_encoder()
    opt2 = AdamW([
        {"params": model.encoder.parameters(), "lr": lr},
        {"params": model.laaf.parameters(),    "lr": lr * 3},
    ], weight_decay=weight_decay, eps=1e-8)
    sch2 = get_linear_schedule_with_warmup(
        opt2,
        num_warmup_steps=int(0.1 * len(tr_loader_p2) * joint_epochs),
        num_training_steps=len(tr_loader_p2) * joint_epochs,
    )
    print("[Phase 2] Full model unfrozen — joint CDWL+LAAF")

    for ep in range(1, joint_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tr_loader_p2:
            opt2.zero_grad()
            logits, _ = model.laaf(
                model.encoder(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).last_hidden_state,
                batch["attention_mask"].to(device),
            )
            labels     = batch["labels"].to(device)
            mix_scores = batch["mix_score"].to(device)
            per_sample = ce_loss(logits, labels)
            loss = (per_sample * (1.0 + mix_scores)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            sch2.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(tr_loader_p2)
        dev_f1, _, _ = evaluate(model, dev_loader, device)
        all_losses.append(epoch_loss)
        all_dev_f1s.append(dev_f1)
        print(f"  [J{ep}] Loss: {epoch_loss:.4f} | Dev F1: {dev_f1:.4f}")

    test_f1, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\n  TEST F1: {test_f1:.4f}")
    print_classification_report(test_labels, test_preds)

    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, metadata={
            "experiment": experiment_name,
            "model": model_class.__name__,
            "method": "CDWL+LAAF",
            "test_f1": test_f1,
            "saved_at": datetime.now().isoformat(),
        })

    del model
    torch.cuda.empty_cache()
    return test_f1, all_losses, all_dev_f1s, test_preds, test_labels


# ─────────────────────────────────────────────────────────────────────────────
# Gradual Fine-Tuning (paper's Spanish-English best method)
# ─────────────────────────────────────────────────────────────────────────────

def run_gradual_finetuning(
    model_name:       str,
    natural_train_df: pd.DataFrame,
    synthetic_df:     pd.DataFrame,
    dev_df:           pd.DataFrame,
    test_df:          pd.DataFrame,
    experiment_name:  str,
    device,
    num_epochs:   int   = 3,
    weight_decay: float = 0.01,
    batch_size:   int   = 32,
    max_seq_len:  int   = 40,
    seed:         int   = 42,
    save_dir:     Optional[str] = None,
):
    """
    5-stage Gradual Fine-Tuning (paper Section 3.3).

    Stage | Synthetic size | LR
    ------+----------------+------
      1   |    50,000      | 1e-6
      2   |    25,000      | 2e-6
      3   |    15,000      | 2e-6
      4   |     5,000      | 4e-6
      5   |         0      | 2e-6

    Returns: (test_f1, all_losses, all_dev_f1s, test_preds, test_labels)
    """
    import random
    print(f"\n{'='*65}")
    print(f"GRADUAL FT [{model_name}]: {experiment_name}")
    print(f"  Natural: {len(natural_train_df)}  Epochs/stage: {num_epochs}")
    print(f"{'='*65}")

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    model     = build_standard_model(model_name, device=device)

    dev_loader  = build_loader(dev_df,  tokenizer, batch_size, shuffle=False,
                               max_len=max_seq_len)
    test_loader = build_loader(test_df, tokenizer, batch_size, shuffle=False,
                               max_len=max_seq_len)

    all_losses, all_dev_f1s = [], []

    for stage_idx, (syn_size, stage_lr) in enumerate(
        zip(GRADUAL_SYN_SIZES, GRADUAL_STAGES_LR)
    ):
        print(f"\n  [Stage {stage_idx+1}/5]  Synthetic: {syn_size}  LR: {stage_lr:.0e}")
        if syn_size > 0:
            syn_sample = synthetic_df.sample(
                n=min(syn_size, len(synthetic_df)), random_state=seed
            )
            stage_df = pd.concat([natural_train_df, syn_sample],
                                  ignore_index=True)
        else:
            stage_df = natural_train_df.copy()

        train_loader = build_loader(stage_df, tokenizer, batch_size,
                                    shuffle=True, max_len=max_seq_len)
        optimizer = AdamW(model.parameters(), lr=stage_lr,
                          weight_decay=weight_decay, eps=1e-8)
        total_steps = len(train_loader) * num_epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for ep in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += out.loss.item()
            loss = total_loss / len(train_loader)
            dev_f1, _, _ = evaluate(model, dev_loader, device)
            all_losses.append(loss)
            all_dev_f1s.append(dev_f1)
            print(f"    Epoch {ep} | Loss: {loss:.4f} | Dev F1: {dev_f1:.4f}")

    test_f1, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\n  FINAL TEST F1: {test_f1:.4f}")
    print_classification_report(test_labels, test_preds)

    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, metadata={
            "experiment": experiment_name, "model": model_name,
            "method": "GradualFT", "test_f1": test_f1,
            "saved_at": datetime.now().isoformat(),
        })

    del model
    torch.cuda.empty_cache()
    return test_f1, all_losses, all_dev_f1s, test_preds, test_labels
