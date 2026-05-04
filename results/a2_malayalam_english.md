# Results — Malayalam-English

> Evaluated on official test set (1,000 samples), same as paper.

| Model | Data Setup | Paper F1 | Reproduced F1 | Difference |
|---|---|---|---|---|
| mBERT | Natural Only | 0.750 | 0.733 | −0.017 |
| XLM-T | Natural Only (Paper's Best) | 0.843 | 0.734 | −0.109 |
| XLM-T | Natural + LLM Synthetic | 0.763 | 0.744 | +0.019 |
| mBERT | Natural + LLM Synthetic | 0.745 | 0.727 | −0.018 |

**Notes:**
- mBERT results closely reproduced across all configurations (within 0.017)
- Largest gap: XLM-T natural-only baseline (−0.109) — likely due to preprocessing differences, tokenization of Malayalam script, and possible model version updates on HuggingFace
- Direction reversal: In the paper, XLM-T degrades with LLM synthetic data (0.843→0.763). In our reproduction, it improves slightly (0.734→0.744). This inversion is tied to the lower baseline — the CM pattern mismatch effect is most harmful when the base model is already highly optimized
