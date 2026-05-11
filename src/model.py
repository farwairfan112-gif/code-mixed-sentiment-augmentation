"""
model.py  —  Model architectures for Code-Mixed Sentiment Analysis.

Contains:
  - Standard XLM-T / mBERT (via HuggingFace AutoModel)
  - LAAFModule       : Language-Aware Attention Fusion head
  - XLMTwithLAAF     : XLM-T encoder + LAAF head
  - mBERTwithLAAF    : mBERT encoder + LAAF head

Paper: "Cross-Lingual Synthetic Augmentation and Language-Aware Training
        for Code-Mixed Sentiment Analysis"  (Farwa Irfan et al., 2025)
Base:  "Leveraging LLMs for Code-Mixed Data Augmentation in Sentiment Analysis"
        (Linda Zeng, SICon @ EMNLP 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_NAMES = {
    "mBERT": "bert-base-multilingual-cased",
    "XLM-T": "cardiffnlp/twitter-xlm-roberta-base",
}


# ─────────────────────────────────────────────────────────────────────────────
# Standard models (no custom head)
# ─────────────────────────────────────────────────────────────────────────────

def build_standard_model(model_name: str, num_labels: int = 3, device=None):
    """
    Return a HuggingFace AutoModelForSequenceClassification.
    Used for CLAT baselines and paper reproduction experiments.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAMES[model_name], num_labels=num_labels
    )
    if device is not None:
        model = model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LAAF: Language-Aware Attention Fusion
# ─────────────────────────────────────────────────────────────────────────────

class LAAFModule(nn.Module):
    """
    Language-Aware Attention Fusion classification head.

    Replaces the standard linear head on the CLS token with two separate
    attention pathways — one for English tokens, one for L2 tokens — whose
    context vectors are fused with the CLS representation for classification.

    Architecture (paper Section IV-C):
      1. Language scorer MLP  → λ_i = P(English) per token  (B, T, 1)
      2. English attention     → v_eng aggregates English-weighted tokens
      3. L2 attention          → v_L2 aggregates L2-weighted tokens
      4. Fusion                → f = W_F [v_CLS ; v_eng ; v_L2], LayerNorm
      5. Classifier            → ŷ = softmax(W_c · f)

    FIX 2: Auxiliary BCE loss on lang_scorer when token_lang labels supplied.
            Loss = CE(sentiment) + 0.3 * BCE(lang_scorer, token_lang)
    FIX 3: Phase-1 warmup LR reduced to LR×3 (caller's responsibility).

    Adds ~1.87 M parameters — only +0.67% vs XLM-T's 278 M.
    """

    def __init__(self, hidden_size: int = 768, num_labels: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.lang_scorer = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, 1),
        )
        self.eng_attn   = nn.Linear(hidden_size, 1)
        self.l2_attn    = nn.Linear(hidden_size, 1)
        self.fusion     = nn.Linear(hidden_size * 3, hidden_size)
        self.norm       = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, attention_mask, token_lang=None):
        """
        Args:
            hidden_states : (B, T, H)
            attention_mask: (B, T)  — 1 for real tokens, 0 for padding
            token_lang    : (B, T) float — 1.0=English, 0.0=L2
                            When supplied, also returns auxiliary BCE loss.
        Returns:
            logits   : (B, num_labels)
            aux_loss : scalar tensor or None
        """
        cls_vec    = hidden_states[:, 0, :]                           # (B, H)
        lang_score = torch.sigmoid(self.lang_scorer(hidden_states))   # (B, T, 1)
        l2_score   = 1.0 - lang_score

        mask = attention_mask.unsqueeze(-1).float()   # (B, T, 1)
        mask[:, 0, :] = 0.0                           # exclude CLS token

        # English-weighted attention
        eng_raw = self.eng_attn(hidden_states) * lang_score * mask
        eng_raw = eng_raw.masked_fill(mask == 0, -1e9)
        eng_w   = torch.softmax(eng_raw, dim=1)       # (B, T, 1)

        # L2-weighted attention
        l2_raw = self.l2_attn(hidden_states) * l2_score * mask
        l2_raw = l2_raw.masked_fill(mask == 0, -1e9)
        l2_w   = torch.softmax(l2_raw, dim=1)         # (B, T, 1)

        eng_ctx = (eng_w * hidden_states).sum(dim=1)  # (B, H)
        l2_ctx  = (l2_w  * hidden_states).sum(dim=1)  # (B, H)

        fused  = self.fusion(torch.cat([cls_vec, eng_ctx, l2_ctx], dim=-1))
        fused  = self.norm(fused)
        logits = self.classifier(self.dropout(fused))  # (B, num_labels)

        # FIX 2: Auxiliary language identification loss
        aux_loss = None
        if token_lang is not None:
            ls_flat  = lang_score.squeeze(-1)          # (B, T)
            tl_mask  = attention_mask.float() * mask.squeeze(-1)
            bce_loss = F.binary_cross_entropy(ls_flat, token_lang, reduction="none")
            active   = tl_mask.sum().clamp(min=1)
            aux_loss = (bce_loss * tl_mask).sum() / active

        return logits, aux_loss


class XLMTwithLAAF(nn.Module):
    """XLM-T encoder (cardiffnlp/twitter-xlm-roberta-base) + LAAF head."""

    def __init__(self, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAMES["XLM-T"])
        self.laaf    = LAAFModule(self.encoder.config.hidden_size,
                                  num_labels, dropout)

    def forward(self, input_ids, attention_mask, labels=None, token_lang=None):
        enc    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits, aux_loss = self.laaf(enc.last_hidden_state, attention_mask,
                                     token_lang)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if aux_loss is not None:
                loss = loss + 0.3 * aux_loss   # FIX 2 weight
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


class mBERTwithLAAF(nn.Module):
    """mBERT encoder (bert-base-multilingual-cased) + LAAF head."""

    def __init__(self, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAMES["mBERT"])
        self.laaf    = LAAFModule(self.encoder.config.hidden_size,
                                  num_labels, dropout)

    def forward(self, input_ids, attention_mask, labels=None, token_lang=None):
        enc    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits, aux_loss = self.laaf(enc.last_hidden_state, attention_mask,
                                     token_lang)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if aux_loss is not None:
                loss = loss + 0.3 * aux_loss
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
