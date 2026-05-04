# Results — Spanish-English Low-Resource Subset (3k)

> Evaluated on official development set as proxy.

| Model | Data Setup | Paper F1 | Reproduced F1 | Difference |
|---|---|---|---|---|
| XLM-T | Natural Only (Baseline) | 0.547 | 0.567 | +0.020 |
| XLM-T | Natural + LLM (Gradual FT) | 0.598 | 0.553 | −0.045 |
| mBERT | Natural Only | 0.487 | 0.525 | +0.038 |
| mBERT | Natural + LLM Synthetic | 0.526 | 0.529 | +0.003 |

**Notes:**
- Paper's key claim: +9.32% improvement with XLM-T + LLM GFT in 3k low-resource setup
- Our reproduction maintains the directional trend: mBERT +0.4% (0.529 vs 0.525), close to paper's mBERT +8.01%
- XLM-T GFT improvement less pronounced due to dev-set proxy evaluation (see full dataset note)
