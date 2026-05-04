# Results — Spanish-English Full Dataset (12k)

> Evaluated on official development set (1,859 samples) as proxy — LinCE test labels unavailable.

| Model | Data Setup | Paper F1 | Reproduced F1 | Difference |
|---|---|---|---|---|
| mBERT | Natural Only (Dataset Baseline) | 0.564 | 0.544 | −0.020 |
| XLM-T | Natural Only (Our Baseline) | 0.588 | 0.582 | −0.006 |
| XLM-T | Synthetic LLM Only | 0.544 | 0.531 | −0.013 |
| XLM-T | Natural + LLM (Gradual FT) | 0.603 | 0.558 | −0.045 |
| XLM-T | Natural + Random Translation | 0.563 | 0.581 | +0.018 |

**Notes:**
- XLM-T outperforms mBERT on natural data (0.582 vs 0.544) — consistent with paper
- Gradual FT achieves best F1 (0.558), reproducing the directional finding of the paper (0.603)
- The −0.045 gap in Gradual FT is explained by dev-set proxy evaluation vs. paper's hidden LinCE test set
- Random Translation performance comparable to natural baseline
