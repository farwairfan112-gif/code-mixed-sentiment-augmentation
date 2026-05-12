"""
utils.py — Training loops, evaluation, and checkpoint utilities.
"""

import json, os
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.dataset import (
    CMDataset, LAAFDataset, SentimentDataset, build_loader,
)
from src.model import (
    MODEL_NAMES, XLMTwithLAAF, mBERTwithLAAF, build_standard_model,
)

GRADUAL_STAGE_LRS  = [1e-6, 2e-6, 2e-6, 4e-6, 2e-6]
GRADUAL_SYN_SIZES  = [50000, 25000, 15000, 5000, 0]


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for b in loader:
            out = model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
            )
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labs.extend(b["labels"].cpu().numpy())
    return f1_score(labs, preds, average="weighted"), preds, labs


def print_report(labels, preds):
    print(classification_report(
        labels, preds, labels=[0,1,2],
        target_names=["Positive","Negative","Neutral"], zero_division=0,
    ))


# ── Checkpoint ────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, checkpoint_dir, metadata=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if hasattr(model, "laaf"):
        torch.save(model.state_dict(),
                   os.path.join(checkpoint_dir, "model_state_dict.pt"))
        model.encoder.config.save_pretrained(checkpoint_dir)
    else:
        model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    if metadata:
        with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    print(f"  Checkpoint saved → {checkpoint_dir}")


# ── Standard training ─────────────────────────────────────────────────────

def run_standard_training(
    model_name, train_df, dev_df, test_df, experiment_name, device,
    num_epochs=3, lr=5e-5, weight_decay=0.01, batch_size=32,
    max_seq_len=64, save_dir=None, seed=42,
):
    torch.manual_seed(seed)
    print(f"\n{'='*60}\nSTANDARD [{model_name}]: {experiment_name}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    model     = build_standard_model(model_name, device=device)
    tr_ldr    = build_loader(train_df, tokenizer, batch_size, True,  max_seq_len)
    dv_ldr    = build_loader(dev_df,   tokenizer, batch_size, False, max_seq_len)
    ts_ldr    = build_loader(test_df,  tokenizer, batch_size, False, max_seq_len)
    opt       = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
    total     = len(tr_ldr) * num_epochs
    sch       = get_linear_schedule_with_warmup(opt, int(0.1*total), total)

    losses, dev_f1s = [], []
    for ep in range(1, num_epochs+1):
        model.train(); tl = 0
        for b in tr_ldr:
            opt.zero_grad()
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tl += out.loss.item()
        l = tl/len(tr_ldr); df1,_,_ = evaluate(model, dv_ldr, device)
        losses.append(l); dev_f1s.append(df1)
        print(f"  Ep{ep} | Loss:{l:.4f} | DevF1:{df1:.4f}")

    tf1, tp, tl = evaluate(model, ts_ldr, device)
    print(f"  TEST F1: {tf1:.4f}"); print_report(tl, tp)
    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, {
            "experiment": experiment_name, "model": model_name,
            "test_f1": tf1, "saved_at": datetime.now().isoformat(),
        })
    del model; torch.cuda.empty_cache()
    return tf1, losses, dev_f1s, tp, tl


# ── CDWL training ─────────────────────────────────────────────────────────

def run_cdwl_training(
    model_name, train_df, dev_df, test_df, experiment_name, device,
    num_epochs=3, lr=5e-5, weight_decay=0.01, batch_size=32,
    max_seq_len=64, save_dir=None, seed=42,
):
    torch.manual_seed(seed)
    print(f"\n{'='*60}\nCDWL [{model_name}]: {experiment_name}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    model     = build_standard_model(model_name, device=device)
    tr_ldr    = DataLoader(CMDataset(train_df, tokenizer, max_seq_len),
                           batch_size=batch_size, shuffle=True, num_workers=2)
    dv_ldr    = build_loader(dev_df,  tokenizer, batch_size, False, max_seq_len)
    ts_ldr    = build_loader(test_df, tokenizer, batch_size, False, max_seq_len)
    opt       = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
    total     = len(tr_ldr) * num_epochs
    sch       = get_linear_schedule_with_warmup(opt, int(0.1*total), total)
    ce        = nn.CrossEntropyLoss(reduction="none")

    losses, dev_f1s = [], []
    for ep in range(1, num_epochs+1):
        model.train(); tl = 0
        for b in tr_ldr:
            opt.zero_grad()
            logits     = model(input_ids=b["input_ids"].to(device),
                               attention_mask=b["attention_mask"].to(device)).logits
            per_sample = ce(logits, b["labels"].to(device))
            loss       = (per_sample * (1.0 + b["mix_score"].to(device))).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tl += loss.item()
        l = tl/len(tr_ldr); df1,_,_ = evaluate(model, dv_ldr, device)
        losses.append(l); dev_f1s.append(df1)
        print(f"  Ep{ep} | Loss:{l:.4f} | DevF1:{df1:.4f}")

    tf1, tp, tl = evaluate(model, ts_ldr, device)
    print(f"  TEST F1: {tf1:.4f}"); print_report(tl, tp)
    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, {
            "experiment": experiment_name, "method": "CDWL",
            "test_f1": tf1, "saved_at": datetime.now().isoformat(),
        })
    del model; torch.cuda.empty_cache()
    return tf1, losses, dev_f1s, tp, tl


# ── LAAF 2-phase training ──────────────────────────────────────────────────

def run_laaf_training(
    train_df, dev_df, test_df, experiment_name, device,
    model_class=None, warmup_epochs=2, joint_epochs=3,
    lr=5e-5, weight_decay=0.01, batch_size=32, max_seq_len=64,
    save_dir=None, seed=42,
):
    if model_class is None: model_class = XLMTwithLAAF
    tok_name = MODEL_NAMES["mBERT"] if model_class == mBERTwithLAAF else MODEL_NAMES["XLM-T"]
    torch.manual_seed(seed)
    print(f"\n{'='*60}\nLAAF [{model_class.__name__}]: {experiment_name}\n{'='*60}")
    tokenizer   = AutoTokenizer.from_pretrained(tok_name)
    model       = model_class().to(device)
    tr_ldr_p1   = DataLoader(LAAFDataset(train_df, tokenizer, max_seq_len),
                             batch_size=batch_size, shuffle=True, num_workers=2)
    tr_ldr_p2   = build_loader(train_df, tokenizer, batch_size, True,  max_seq_len)
    dv_ldr      = build_loader(dev_df,   tokenizer, batch_size, False, max_seq_len)
    ts_ldr      = build_loader(test_df,  tokenizer, batch_size, False, max_seq_len)

    losses, dev_f1s = [], []

    # Phase 1: frozen encoder
    WARMUP_LR = lr * 3
    model.freeze_encoder()
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=WARMUP_LR, weight_decay=weight_decay, eps=1e-8)
    sch1 = get_linear_schedule_with_warmup(opt1,
        int(0.1*len(tr_ldr_p1)*warmup_epochs), len(tr_ldr_p1)*warmup_epochs)
    print(f"[P1] Frozen encoder | LR={WARMUP_LR:.2e}")
    for ep in range(1, warmup_epochs+1):
        model.train(); tl = 0
        for b in tr_ldr_p1:
            opt1.zero_grad()
            out = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                        b["labels"].to(device), b["token_lang"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step(); sch1.step(); tl += out.loss.item()
        l = tl/len(tr_ldr_p1); df1,_,_ = evaluate(model, dv_ldr, device)
        losses.append(l); dev_f1s.append(df1)
        print(f"  [W{ep}] Loss:{l:.4f} | DevF1:{df1:.4f}")

    # Phase 2: full model
    model.unfreeze_encoder()
    opt2 = AdamW([
        {"params": model.encoder.parameters(), "lr": lr},
        {"params": model.laaf.parameters(),    "lr": lr*3},
    ], weight_decay=weight_decay, eps=1e-8)
    sch2 = get_linear_schedule_with_warmup(opt2,
        int(0.1*len(tr_ldr_p2)*joint_epochs), len(tr_ldr_p2)*joint_epochs)
    print("[P2] Full model unfrozen")
    for ep in range(1, joint_epochs+1):
        model.train(); tl = 0
        for b in tr_ldr_p2:
            opt2.zero_grad()
            out = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                        b["labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step(); sch2.step(); tl += out.loss.item()
        l = tl/len(tr_ldr_p2); df1,_,_ = evaluate(model, dv_ldr, device)
        losses.append(l); dev_f1s.append(df1)
        print(f"  [J{ep}] Loss:{l:.4f} | DevF1:{df1:.4f}")

    tf1, tp, tl = evaluate(model, ts_ldr, device)
    print(f"  TEST F1: {tf1:.4f}"); print_report(tl, tp)
    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, {
            "experiment": experiment_name, "method": "LAAF",
            "test_f1": tf1, "saved_at": datetime.now().isoformat(),
        })
    del model; torch.cuda.empty_cache()
    return tf1, losses, dev_f1s, tp, tl


# ── Combined CDWL+LAAF ─────────────────────────────────────────────────────

def run_combined_training(
    train_df, dev_df, test_df, experiment_name, device,
    model_class=None, warmup_epochs=2, joint_epochs=3,
    lr=5e-5, weight_decay=0.01, batch_size=32, max_seq_len=64,
    save_dir=None, seed=42,
):
    if model_class is None: model_class = XLMTwithLAAF
    tok_name = MODEL_NAMES["mBERT"] if model_class == mBERTwithLAAF else MODEL_NAMES["XLM-T"]
    torch.manual_seed(seed)
    print(f"\n{'='*60}\nCOMBINED CDWL+LAAF [{model_class.__name__}]: {experiment_name}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model     = model_class().to(device)
    ce        = nn.CrossEntropyLoss(reduction="none")

    class CombinedDS(Dataset):
        def __init__(self, df, tok, ml):
            self.base = LAAFDataset(df, tok, ml)
            self.ms   = df["mix_score"].tolist() if "mix_score" in df.columns else [0.0]*len(df)
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            item = self.base[i]
            item["mix_score"] = torch.tensor(self.ms[i], dtype=torch.float)
            return item

    tr_ldr_p1 = DataLoader(CombinedDS(train_df, tokenizer, max_seq_len),
                           batch_size=batch_size, shuffle=True, num_workers=2)
    tr_ldr_p2 = DataLoader(CombinedDS(train_df, tokenizer, max_seq_len),
                           batch_size=batch_size, shuffle=True, num_workers=2)
    dv_ldr    = build_loader(dev_df,  tokenizer, batch_size, False, max_seq_len)
    ts_ldr    = build_loader(test_df, tokenizer, batch_size, False, max_seq_len)

    losses, dev_f1s = [], []
    WARMUP_LR = lr * 3
    model.freeze_encoder()
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=WARMUP_LR, weight_decay=weight_decay, eps=1e-8)
    sch1 = get_linear_schedule_with_warmup(opt1,
        int(0.1*len(tr_ldr_p1)*warmup_epochs), len(tr_ldr_p1)*warmup_epochs)
    print(f"[P1] Frozen | LR={WARMUP_LR:.2e}")
    for ep in range(1, warmup_epochs+1):
        model.train(); tl = 0
        for b in tr_ldr_p1:
            opt1.zero_grad()
            h   = model.encoder(input_ids=b["input_ids"].to(device),
                                 attention_mask=b["attention_mask"].to(device)).last_hidden_state
            logits, aux = model.laaf(h, b["attention_mask"].to(device), b["token_lang"].to(device))
            loss = (ce(logits, b["labels"].to(device)) * (1+b["mix_score"].to(device))).mean()
            if aux is not None: loss = loss + 0.3*aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step(); sch1.step(); tl += loss.item()
        l = tl/len(tr_ldr_p1); df1,_,_ = evaluate(model, dv_ldr, device)
        losses.append(l); dev_f1s.append(df1)
        print(f"  [W{ep}] Loss:{l:.4f} | DevF1:{df1:.4f}")

    model.unfreeze_encoder()
    opt2 = AdamW([
        {"params": model.encoder.parameters(), "lr": lr},
        {"params": model.laaf.parameters(),    "lr": lr*3},
    ], weight_decay=weight_decay, eps=1e-8)
    sch2 = get_linear_schedule_with_warmup(opt2,
        int(0.1*len(tr_ldr_p2)*joint_epochs), len(tr_ldr_p2)*joint_epochs)
    print("[P2] Full model")
    for ep in range(1, joint_epochs+1):
        model.train(); tl = 0
        for b in tr_ldr_p2:
            opt2.zero_grad()
            h   = model.encoder(input_ids=b["input_ids"].to(device),
                                 attention_mask=b["attention_mask"].to(device)).last_hidden_state
            logits, _ = model.laaf(h, b["attention_mask"].to(device))
            loss = (ce(logits, b["labels"].to(device)) * (1+b["mix_score"].to(device))).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step(); sch2.step(); tl += loss.item()
        l = tl/len(tr_ldr_p2); df1,_,_ = evaluate(model, dv_ldr, device)
        losses.append(l); dev_f1s.append(df1)
        print(f"  [J{ep}] Loss:{l:.4f} | DevF1:{df1:.4f}")

    tf1, tp, tl = evaluate(model, ts_ldr, device)
    print(f"  TEST F1: {tf1:.4f}"); print_report(tl, tp)
    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, {
            "experiment": experiment_name, "method": "CDWL+LAAF",
            "test_f1": tf1, "saved_at": datetime.now().isoformat(),
        })
    del model; torch.cuda.empty_cache()
    return tf1, losses, dev_f1s, tp, tl


# ── Gradual Fine-Tuning ────────────────────────────────────────────────────

def run_gradual_finetuning(
    model_name, natural_train_df, synthetic_df, dev_df, test_df,
    experiment_name, device, num_epochs=3, weight_decay=0.01,
    batch_size=32, max_seq_len=40, seed=42, save_dir=None,
):
    torch.manual_seed(seed)
    print(f"\n{'='*60}\nGRADUAL FT [{model_name}]: {experiment_name}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    model     = build_standard_model(model_name, device=device)
    dv_ldr    = build_loader(dev_df,  tokenizer, batch_size, False, max_seq_len)
    ts_ldr    = build_loader(test_df, tokenizer, batch_size, False, max_seq_len)

    all_losses, all_dev_f1s = [], []
    for s, (syn_size, stage_lr) in enumerate(zip(GRADUAL_SYN_SIZES, GRADUAL_STAGE_LRS), 1):
        print(f"\n  [Stage {s}/5] Syn:{syn_size}  LR:{stage_lr:.0e}")
        if syn_size > 0:
            syn = synthetic_df.sample(n=min(syn_size, len(synthetic_df)), random_state=seed)
            stage_df = pd.concat([natural_train_df, syn], ignore_index=True)
        else:
            stage_df = natural_train_df
        tr_ldr = build_loader(stage_df, tokenizer, batch_size, True, max_seq_len)
        opt    = AdamW(model.parameters(), lr=stage_lr, weight_decay=weight_decay, eps=1e-8)
        total  = len(tr_ldr)*num_epochs
        sch    = get_linear_schedule_with_warmup(opt, int(0.1*total), total)
        for ep in range(1, num_epochs+1):
            model.train(); tl = 0
            for b in tr_ldr:
                opt.zero_grad()
                out = model(input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device),
                            labels=b["labels"].to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); tl += out.loss.item()
            l = tl/len(tr_ldr); df1,_,_ = evaluate(model, dv_ldr, device)
            all_losses.append(l); all_dev_f1s.append(df1)
            print(f"    Ep{ep} | Loss:{l:.4f} | DevF1:{df1:.4f}")

    tf1, tp, tl = evaluate(model, ts_ldr, device)
    print(f"  FINAL TEST F1: {tf1:.4f}"); print_report(tl, tp)
    if save_dir:
        save_checkpoint(model, tokenizer, save_dir, {
            "experiment": experiment_name, "method": "GradualFT",
            "test_f1": tf1, "saved_at": datetime.now().isoformat(),
        })
    del model; torch.cuda.empty_cache()
    return tf1, all_losses, all_dev_f1s, tp, tl
