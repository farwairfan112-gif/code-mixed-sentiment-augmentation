# Results — Phase 1: CLAT (Cross-Lingual Augmentation Transfer)

> Evaluated on Hinglish SentiMix official test set (3,000 samples).  
> 15,000 samples selected from source synthetic dataset and combined with Hinglish natural training data (14,000 samples).

| Exp | Model | Data | F1 | vs C1 |
|---|---|---|---|---|
| C1 | XLM-T | Hinglish natural (baseline) | 0.7161 | — |
| C2 | XLM-T | Hinglish + Sp-En synthetic (CLAT) | 0.7071 | −0.0090 |
| C3 | XLM-T | Hinglish + Ma-En synthetic (negative control) | 0.7119 | −0.0041 |
| C4 | mBERT | Hinglish natural | 0.6817 | — |

**Notes:**
- CLAT slightly degrades performance on the official test set (−0.0090)
- This contradicts earlier dev-set results showing +2.56% improvement — highlighting the risk of proxy evaluation
- Negative control (Ma-En synthetic) shows slightly less degradation than the supposedly compatible Sp-En data, suggesting structural CM compatibility is not sufficient for successful transfer when the base dataset is already large (14k)
