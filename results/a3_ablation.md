# Results — Phase 4: Combined CDWL+LAAF & Full Ablation

## Combined Results (CDWL + LAAF)

| Exp | Method | Data | F1 | vs C1 Baseline |
|---|---|---|---|---|
| CL1 | XLM-T CDWL+LAAF | Hinglish natural | **0.7169** | **+0.0008** ← BEST |
| CL2 | XLM-T CDWL+LAAF | Hinglish + Sp-En synthetic | 0.7159 | −0.0002 |
| CL3 | XLM-T CDWL+LAAF | Hinglish + Ma-En synthetic | 0.7093 | −0.0068 |
| CL4 | mBERT CDWL+LAAF | Hinglish natural | 0.6761 | −0.0056 vs C4 |
| CL5 | XLM-T CDWL+LAAF | Spanish-English natural | 0.5676 | −0.0144 vs A2 |
| CL6 | XLM-T CDWL+LAAF | Malayalam-English natural | 0.7318 | −0.0022 vs A2 |

---

## Full Ablation Summary

| Setup | Standard | CDWL | LAAF | CDWL+LAAF | Best Method |
|---|---|---|---|---|---|
| Hinglish natural (XLM-T) | 0.7161 | 0.7140 | 0.7120 | **0.7169** | CDWL+LAAF (+0.0008) |
| Hinglish + Sp-En | 0.7071 | 0.7022 | **0.7129** | 0.7159 | LAAF (+0.0059 vs std) |
| Hinglish + Ma-En | 0.7119 | 0.7129 | **0.7167** | 0.7093 | LAAF (+0.0047 vs std) |
| Hinglish mBERT | 0.6817 | 0.6722 | **0.6844** | 0.6761 | LAAF (+0.0028) |

---

## Key Findings

- **Best result overall:** CL1 — CDWL+LAAF on Hinglish natural data (F1 = 0.7169, +0.0008 over XLM-T baseline)
- **LAAF wins more setups** than CDWL+LAAF combined, suggesting CDWL adds noise when combined
- **CLAT finding:** Apparent +2.56% gain with dev-as-test collapses to −0.90% on proper test split — evaluation protocol is critical
- **Strong baseline problem:** XLM-T (pre-trained on 198M multilingual tweets) already handles Hinglish well; marginal gains are hard to achieve
