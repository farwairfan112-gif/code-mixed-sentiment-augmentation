"""
Main training script for all experiments
Run: python train.py --experiment C1 --save_checkpoints
"""

import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

from src.dataset import preprocess_df, load_conll, build_loader
from src.model import XLMTwithLAAF, mBERTwithLAAF
from src.utils import evaluate, safe_report

# Hyperparameters (same as notebook)
NUM_LABELS = 3
MAX_SEQ_LEN = 64
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
EPSILON = 1e-8
NUM_EPOCHS = 3
WARMUP_EPOCHS = 2
JOINT_EPOCHS = 3

MODEL_NAMES = {
    'mBERT': 'bert-base-multilingual-cased',
    'XLM-T': 'cardiffnlp/twitter-xlm-roberta-base'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Training Functions (copied from your notebook)
# ============================================================

def run_standard(model_name, train_df, dev_df, test_df, name, save_checkpoints=False):
    """Standard cross-entropy fine-tuning (baseline & CLAT)"""
    print(f'\n{"="*65}\nSTANDARD [{model_name}]: {name}')
    print(f'Train:{len(train_df)} Test:{len(test_df)} Epochs:{NUM_EPOCHS}\n{"="*65}')
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAMES[model_name], num_labels=NUM_LABELS).to(DEVICE)
    
    tr_ldr = build_loader(train_df, tok, BATCH_SIZE, shuffle=True)
    dev_ldr = build_loader(dev_df, tok, BATCH_SIZE, shuffle=False)
    tst_ldr = build_loader(test_df, tok, BATCH_SIZE, shuffle=False)
    
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    opt = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=EPSILON)
    total = len(tr_ldr) * NUM_EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    
    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        tl = 0
        for b in tr_ldr:
            opt.zero_grad()
            out = model(
                input_ids=b['input_ids'].to(DEVICE),
                attention_mask=b['attention_mask'].to(DEVICE),
                labels=b['labels'].to(DEVICE)
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tl += out.loss.item()
        
        loss = tl / len(tr_ldr)
        df1, _, _ = evaluate(model, dev_ldr, DEVICE)
        print(f'  Epoch {ep}/{NUM_EPOCHS} | Loss:{loss:.4f} | DevF1:{df1:.4f}')
        
        if save_checkpoints:
            from src.utils import save_checkpoint
            save_checkpoint(model, opt, ep, df1, name)
    
    tf1, tpreds, tlabs = evaluate(model, tst_ldr, DEVICE)
    print(f'TEST F1: {tf1:.4f}')
    safe_report(tlabs, tpreds)
    
    return tf1


def run_cdwl(model_name, train_df, dev_df, test_df, name, save_checkpoints=False):
    """CDWL: Code-Mix Degree Weighted Loss"""
    print(f'\n{"="*65}\nCDWL [{model_name}]: {name}')
    print(f'Train:{len(train_df)} Test:{len(test_df)} Epochs:{NUM_EPOCHS}\n{"="*65}')
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAMES[model_name], num_labels=NUM_LABELS).to(DEVICE)
    
    tr_ldr = build_loader(train_df, tok, BATCH_SIZE, shuffle=True, with_mix_score=True)
    dev_ldr = build_loader(dev_df, tok, BATCH_SIZE, shuffle=False)
    tst_ldr = build_loader(test_df, tok, BATCH_SIZE, shuffle=False)
    
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    opt = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=EPSILON)
    total = len(tr_ldr) * NUM_EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    
    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        tl = 0
        for b in tr_ldr:
            opt.zero_grad()
            logits = model(
                input_ids=b['input_ids'].to(DEVICE),
                attention_mask=b['attention_mask'].to(DEVICE)
            ).logits
            labels = b['labels'].to(DEVICE)
            mix_scores = b['mix_score'].to(DEVICE)
            
            per_sample_loss = ce(logits, labels)
            weighted_loss = (per_sample_loss * (1.0 + mix_scores)).mean()
            weighted_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tl += weighted_loss.item()
        
        loss = tl / len(tr_ldr)
        df1, _, _ = evaluate(model, dev_ldr, DEVICE)
        print(f'  Epoch {ep}/{NUM_EPOCHS} | Loss:{loss:.4f} | DevF1:{df1:.4f}')
        
        if save_checkpoints:
            from src.utils import save_checkpoint
            save_checkpoint(model, opt, ep, df1, name)
    
    tf1, tpreds, tlabs = evaluate(model, tst_ldr, DEVICE)
    print(f'TEST F1: {tf1:.4f}')
    safe_report(tlabs, tpreds)
    
    return tf1


def run_laaf(train_df, dev_df, test_df, name, model_class=XLMTwithLAAF, save_checkpoints=False):
    """2-Phase LAAF training with auxiliary language supervision"""
    tok_name = MODEL_NAMES['mBERT'] if model_class == mBERTwithLAAF else MODEL_NAMES['XLM-T']
    print(f'\n{"="*65}\nLAAF [{model_class.__name__}]: {name}')
    print(f'Train:{len(train_df)} Test:{len(test_df)} Warmup:{WARMUP_EPOCHS} Joint:{JOINT_EPOCHS}\n{"="*65}')
    
    tok = AutoTokenizer.from_pretrained(tok_name)
    model = model_class().to(DEVICE)
    
    tr_ldr_phase1 = build_loader(train_df, tok, BATCH_SIZE, shuffle=True, with_token_lang=True)
    tr_ldr_phase2 = build_loader(train_df, tok, BATCH_SIZE, shuffle=True)
    dev_ldr = build_loader(dev_df, tok, BATCH_SIZE, shuffle=False)
    tst_ldr = build_loader(test_df, tok, BATCH_SIZE, shuffle=False)
    
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    # Phase 1: Warmup - encoder frozen
    WARMUP_LR = LEARNING_RATE * 3  # FIX 3
    print(f'[Phase 1] Encoder frozen | LR={WARMUP_LR:.2e}')
    model.freeze_encoder()
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=WARMUP_LR, weight_decay=WEIGHT_DECAY, eps=EPSILON)
    sch1 = get_linear_schedule_with_warmup(opt1,
        int(0.1 * len(tr_ldr_phase1) * WARMUP_EPOCHS),
        len(tr_ldr_phase1) * WARMUP_EPOCHS)
    
    for ep in range(1, WARMUP_EPOCHS + 1):
        model.train()
        tl = 0
        for b in tr_ldr_phase1:
            opt1.zero_grad()
            out = model(
                b['input_ids'].to(DEVICE),
                b['attention_mask'].to(DEVICE),
                b['labels'].to(DEVICE),
                b['token_lang'].to(DEVICE)
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            sch1.step()
            tl += out.loss.item()
        
        loss = tl / len(tr_ldr_phase1)
        df1, _, _ = evaluate(model, dev_ldr, DEVICE)
        print(f'  [W] Ep{ep} Loss:{loss:.4f} DevF1:{df1:.4f}')
        
        if save_checkpoints:
            from src.utils import save_checkpoint
            save_checkpoint(model, opt1, ep, df1, f"{name}_phase1")
    
    # Phase 2: Joint training
    print('[Phase 2] Full model unfrozen — joint training')
    model.unfreeze_encoder()
    opt2 = AdamW([
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': model.laaf.parameters(), 'lr': LEARNING_RATE * 3}
    ], weight_decay=WEIGHT_DECAY, eps=EPSILON)
    sch2 = get_linear_schedule_with_warmup(opt2,
        int(0.1 * len(tr_ldr_phase2) * JOINT_EPOCHS),
        len(tr_ldr_phase2) * JOINT_EPOCHS)
    
    for ep in range(1, JOINT_EPOCHS + 1):
        model.train()
        tl = 0
        for b in tr_ldr_phase2:
            opt2.zero_grad()
            out = model(
                b['input_ids'].to(DEVICE),
                b['attention_mask'].to(DEVICE),
                b['labels'].to(DEVICE)
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            sch2.step()
            tl += out.loss.item()
        
        loss = tl / len(tr_ldr_phase2)
        df1, _, _ = evaluate(model, dev_ldr, DEVICE)
        print(f'  [J] Ep{ep} Loss:{loss:.4f} DevF1:{df1:.4f}')
        
        if save_checkpoints:
            from src.utils import save_checkpoint
            save_checkpoint(model, opt2, ep, df1, f"{name}_phase2")
    
    tf1, tpreds, tlabs = evaluate(model, tst_ldr, DEVICE)
    print(f'TEST F1: {tf1:.4f}')
    safe_report(tlabs, tpreds)
    
    return tf1


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, 
                        help='C1, C2, C3, C4, W1, W2, W3, W4, L1, L2, L3, L4, CL1, CL2, CL3, CL4, CL5, CL6')
    parser.add_argument('--save_checkpoints', action='store_true', help='Save model checkpoints')
    args = parser.parse_args()
    
    print(f"Running experiment: {args.experiment}")
    print(f"Device: {DEVICE}")
    
    # Note: You need to load your data here
    # This is a placeholder - you'll need to adapt paths to your data location
    print("\nNOTE: Please update data paths in this script before running")
    print("Expected data files:")
    print("  - Hinglish_train_14k_split_conll.txt")
    print("  - Hinglish_dev_3k_split_conll.txt")
    print("  - Hinglish_test_unlabelled_conll_updated.txt")
    print("  - Hinglish_test_labels.txt")
    
    # Example of how to run (uncomment after setting paths):
    # hi_train = preprocess_df(load_conll('path/to/Hinglish_train...'), 'hi_train')
    # hi_dev = preprocess_df(load_conll('path/to/Hinglish_dev...'), 'hi_dev')
    # hi_test = preprocess_df(...)
    # 
    # if args.experiment == 'C1':
    #     run_standard('XLM-T', hi_train, hi_dev, hi_test, 'C1', args.save_checkpoints)