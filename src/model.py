"""
model.py — Model architectures: Standard XLM-T/mBERT, XLMTwithLAAF, mBERTwithLAAF.
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


def build_standard_model(model_name: str, num_labels: int = 3, device=None):
    """Return HuggingFace AutoModelForSequenceClassification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAMES[model_name], num_labels=num_labels
    )
    if device is not None:
        model = model.to(device)
    return model


class LAAFModule(nn.Module):
    """
    Language-Aware Attention Fusion head (paper Section IV-C).
    Adds ~1.87M params (+0.67% vs XLM-T 278M).
    """
    def __init__(self, hidden_size=768, num_labels=3, dropout=0.1):
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
        cls_vec    = hidden_states[:, 0, :]
        lang_score = torch.sigmoid(self.lang_scorer(hidden_states))  # (B,T,1)
        l2_score   = 1.0 - lang_score

        mask = attention_mask.unsqueeze(-1).float()
        mask[:, 0, :] = 0.0  # exclude CLS

        eng_w = (self.eng_attn(hidden_states) * lang_score * mask).masked_fill(mask == 0, -1e9)
        eng_w = torch.softmax(eng_w, dim=1)
        l2_w  = (self.l2_attn(hidden_states) * l2_score * mask).masked_fill(mask == 0, -1e9)
        l2_w  = torch.softmax(l2_w, dim=1)

        eng_ctx = (eng_w * hidden_states).sum(dim=1)
        l2_ctx  = (l2_w  * hidden_states).sum(dim=1)

        fused  = self.norm(self.fusion(torch.cat([cls_vec, eng_ctx, l2_ctx], dim=-1)))
        logits = self.classifier(self.dropout(fused))

        aux_loss = None
        if token_lang is not None:
            ls_flat  = lang_score.squeeze(-1)
            tl_mask  = attention_mask.float() * mask.squeeze(-1)
            bce      = F.binary_cross_entropy(ls_flat, token_lang, reduction="none")
            aux_loss = (bce * tl_mask).sum() / tl_mask.sum().clamp(min=1)

        return logits, aux_loss


class XLMTwithLAAF(nn.Module):
    def __init__(self, num_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAMES["XLM-T"])
        self.laaf    = LAAFModule(self.encoder.config.hidden_size, num_labels, dropout)

    def forward(self, input_ids, attention_mask, labels=None, token_lang=None):
        h      = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits, aux = self.laaf(h, attention_mask, token_lang)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if aux is not None:
                loss = loss + 0.3 * aux
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True


class mBERTwithLAAF(nn.Module):
    def __init__(self, num_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAMES["mBERT"])
        self.laaf    = LAAFModule(self.encoder.config.hidden_size, num_labels, dropout)

    def forward(self, input_ids, attention_mask, labels=None, token_lang=None):
        h      = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits, aux = self.laaf(h, attention_mask, token_lang)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if aux is not None:
                loss = loss + 0.3 * aux
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True
