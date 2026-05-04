# Results — Phase 2: CDWL (Code-Mix Degree Weighted Loss)

> mix_score computed using word-level language detection via Lingua (fix over ASCII heuristic).  
> Loss: L_CDWL = mean(CE(logits, labels) × (1 + mix_score)), mix_score ∈ [0,1]

| Exp | Model | Data | F1 | vs Standard |
|---|---|---|---|---|
| W1 | XLM-T + CDWL | Hinglish natural | 0.7140 | −0.0021 vs C1 |
| W2 | XLM-T + CDWL | Hinglish + Sp-En synthetic | 0.7022 | −0.0049 vs C2 |
| W3 | XLM-T + CDWL | Hinglish + Ma-En synthetic | 0.7129 | +0.0010 vs C3 |
| W4 | mBERT + CDWL | Hinglish natural | 0.6722 | −0.0094 vs C4 |

**Notes:**
- CDWL has marginal or slightly adverse effects on most Hinglish configurations
- Only W3 benefits (+0.0010), possibly because CDWL correctly downweights non-ASCII Malayalam tokens in the mixed training set
- XLM-T already pre-trained on code-mixed social media data — CM-degree weighting adds little additional signal
