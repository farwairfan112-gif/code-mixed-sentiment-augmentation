# ============================================================
# PART 1: STANDARD MODELS (for reproduction + CLAT baselines)
# ============================================================
# Used for:
#   - Sp-En baseline (reproduction)
#   - Ma-En baseline (reproduction)
#   - C1: XLM-T | Hinglish natural
#   - C2: XLM-T | Hinglish + Sp-En synthetic (CLAT)
#   - C3: XLM-T | Hinglish + Ma-En synthetic (CLAT ctrl)
#   - C4: mBERT | Hinglish natural

def run_standard(model_name, train_df, dev_df, test_df, name, epochs=NUM_EPOCHS):
    """Standard cross-entropy fine-tuning (same as paper setup)."""
    print(f'\n{"="*65}\nSTANDARD [{model_name}]: {name}\n'
          f'Train:{len(train_df)} Test:{len(test_df)} Epochs:{epochs}\n{"="*65}')
    tok   = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAMES[model_name], num_labels=NUM_LABELS).to(DEVICE)
    tr_ldr  = build_loader(train_df, tok, shuffle=True)
    dev_ldr = build_loader(dev_df,   tok, shuffle=False)
    tst_ldr = build_loader(test_df,  tok, shuffle=False)
    opt = AdamW(model.parameters(), lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY, eps=EPSILON)
    total = len(tr_ldr) * epochs
    sch   = get_linear_schedule_with_warmup(opt, int(0.1*total), total)
    losses, dev_f1s = [], []
    for ep in range(1, epochs+1):
        model.train(); tl = 0
        for b in tr_ldr:
            opt.zero_grad()
            out  = model(input_ids=b['input_ids'].to(DEVICE),
                         attention_mask=b['attention_mask'].to(DEVICE),
                         labels=b['labels'].to(DEVICE))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tl += out.loss.item()
        loss = tl / len(tr_ldr)
        df1, _, _ = evaluate(model, dev_ldr)
        losses.append(loss); dev_f1s.append(df1)
        print(f'  Epoch {ep}/{epochs} | Loss:{loss:.4f} | DevF1:{df1:.4f}')
    tf1, tpreds, tlabs = evaluate(model, tst_ldr)
    print(f'TEST F1: {tf1:.4f}'); safe_report(tlabs, tpreds)
    del model; torch.cuda.empty_cache()
    return tf1, losses, dev_f1s, tpreds, tlabs

results = {}
print('Standard training function defined!')

# ============================================================
# PART 2: CDWL MODELS (same architecture, loss weighting happens in training)
# ============================================================
# Note: CDWL doesn't change model architecture, only the loss function.
# So same standard models are used with custom loss weighting in train_cdwl()


# ============================================================
# PART 3: LAAF MODELS (Language-Aware Attention Fusion)
# ============================================================
# Used for:
#   - L1: XLM-T + LAAF | Hinglish natural
#   - L2: XLM-T + LAAF | Hinglish + Sp-En synthetic
#   - L3: XLM-T + LAAF | Hinglish + Ma-En synthetic
#   - L4: mBERT + LAAF | Hinglish natural
#   - CL1-CL6: Combined CDWL+LAAF

class LAAFModule(nn.Module):
    """Separate attention heads for English vs L2 tokens, fused with CLS.
    FIX 2: lang_scorer is supervised by auxiliary BCE loss in Phase 1.
    """
    def __init__(self, hidden_size=768, num_labels=3, dropout=0.1):
        super().__init__()
        self.lang_scorer = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, 1))
        self.eng_attn   = nn.Linear(hidden_size, 1)
        self.l2_attn    = nn.Linear(hidden_size, 1)
        self.fusion     = nn.Linear(hidden_size * 3, hidden_size)
        self.norm       = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, attention_mask, token_lang=None):
        """
        hidden_states: (B, T, H)
        attention_mask: (B, T)
        token_lang: (B, T) float — 1.0=English, 0.0=L2 — used for aux loss in Phase 1
        Returns: (logits, aux_loss or None)
        """
        cls_vec    = hidden_states[:, 0, :]                          # (B, H)
        lang_score = torch.sigmoid(self.lang_scorer(hidden_states))  # (B, T, 1)
        l2_score   = 1.0 - lang_score

        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        mask[:, 0, :] = 0.0  # exclude CLS

        eng_w = (self.eng_attn(hidden_states) * lang_score * mask
                 ).masked_fill(mask == 0, -1e9)
        eng_w = torch.softmax(eng_w, dim=1)
        l2_w  = (self.l2_attn(hidden_states) * l2_score * mask
                 ).masked_fill(mask == 0, -1e9)
        l2_w  = torch.softmax(l2_w, dim=1)

        eng_ctx = (eng_w * hidden_states).sum(1)   # (B, H)
        l2_ctx  = (l2_w  * hidden_states).sum(1)   # (B, H)
        fused   = self.fusion(torch.cat([cls_vec, eng_ctx, l2_ctx], -1))
        logits  = self.classifier(self.dropout(self.norm(fused)))    # (B, num_labels)

        # FIX 2: Auxiliary loss — supervise lang_scorer with true token labels
        aux_loss = None
        if token_lang is not None:
            # lang_score: (B, T, 1) → (B, T); token_lang: (B, T)
            ls_flat  = lang_score.squeeze(-1)  # (B, T)
            tl_mask  = (attention_mask.float() * mask.squeeze(-1))  # non-CLS active tokens
            bce_loss = F.binary_cross_entropy(ls_flat, token_lang, reduction='none')
            # Average only over non-padding, non-CLS tokens
            active   = tl_mask.sum().clamp(min=1)
            aux_loss = (bce_loss * tl_mask).sum() / active

        return logits, aux_loss


class XLMTwithLAAF(nn.Module):
    """XLM-T encoder + LAAF head."""
    def __init__(self, num_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAMES['XLM-T'])
        self.laaf    = LAAFModule(self.encoder.config.hidden_size, num_labels, dropout)

    def forward(self, input_ids, attention_mask, labels=None, token_lang=None):
        enc    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits, aux_loss = self.laaf(enc.last_hidden_state, attention_mask, token_lang)
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            # FIX 2: Add auxiliary lang supervision loss (weight=0.3)
            if aux_loss is not None:
                loss = loss + 0.3 * aux_loss
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False
    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True


class mBERTwithLAAF(nn.Module):
    """mBERT encoder + LAAF head."""
    def __init__(self, num_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAMES['mBERT'])
        self.laaf    = LAAFModule(self.encoder.config.hidden_size, num_labels, dropout)

    def forward(self, input_ids, attention_mask, labels=None, token_lang=None):
        enc    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits, aux_loss = self.laaf(enc.last_hidden_state, attention_mask, token_lang)
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if aux_loss is not None:
                loss = loss + 0.3 * aux_loss
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False
    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True


# Test
print('Testing LAAF architectures...')
for cls, name in [(XLMTwithLAAF,'XLM-T+LAAF'),(mBERTwithLAAF,'mBERT+LAAF')]:
    m = cls().to(DEVICE)
    ids  = torch.randint(0, 1000, (2, MAX_SEQ_LEN)).to(DEVICE)
    mask = torch.ones(2, MAX_SEQ_LEN, dtype=torch.long).to(DEVICE)
    labs = torch.randint(0, 3, (2,)).to(DEVICE)
    tl   = torch.rand(2, MAX_SEQ_LEN).to(DEVICE)  # dummy token_lang
    out  = m(ids, mask, labs, tl)
    p    = sum(x.numel() for x in m.laaf.parameters())
    print(f'  {name}: shape={out.logits.shape} LAAF_params={p:,} loss={out.loss.item():.4f}')
    del m; torch.cuda.empty_cache()
print('Both OK!')

# ============================================================
# MODEL FACTORY
# ============================================================

MODEL_CONFIGS = {
    'standard': {
        'XLM-T': 'cardiffnlp/twitter-xlm-roberta-base',
        'mBERT': 'bert-base-multilingual-cased'
    },
    'laaf': {
        'XLM-T': XLMTwithLAAF,
        'mBERT': mBERTwithLAAF
    }
}


def get_model(model_type, model_name, num_labels=3):
    """
    Factory function to get model by type.
    
    Args:
        model_type: 'standard' or 'laaf'
        model_name: 'XLM-T' or 'mBERT'
        num_labels: number of classes (default 3)
    
    Returns:
        PyTorch model
    """
    if model_type == 'standard':
        model_path = MODEL_CONFIGS['standard'][model_name]
        return get_standard_model(model_path, num_labels)
    elif model_type == 'laaf':
        model_class = MODEL_CONFIGS['laaf'][model_name]
        return model_class(num_labels=num_labels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
