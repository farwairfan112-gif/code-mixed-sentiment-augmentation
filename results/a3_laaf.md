# Results — Phase 3: LAAF (Language-Aware Attention Fusion)

> 2-phase training: Phase 1 (encoder frozen, LAAF head warmup with auxiliary lang supervision, LR=1.5e-4, 2 epochs) → Phase 2 (full model, joint training, 3 epochs).  
> Adds 1.87M parameters (+0.67% over XLM-T base).

| Exp | Model | Data | F1 | vs Standard |
|---|---|---|---|---|
| L1 | XLM-T + LAAF | Hinglish natural | 0.7120 | −0.0041 vs C1 |
| L2 | XLM-T + LAAF | Hinglish + Sp-En synthetic | 0.7129 | +0.0059 vs C2 |
| L3 | XLM-T + LAAF | Hinglish + Ma-En synthetic | 0.7167 | +0.0047 vs C3 |
| L4 | mBERT + LAAF | Hinglish natural | 0.6844 | +0.0028 vs C4 |

**Notes:**
- LAAF improves when synthetic data is present (L2: +0.0059, L3: +0.0047) but slightly hurts without it (L1: −0.0041)
- Language scorer learns useful language boundaries when training data contains CM patterns from multiple sources
- Romanized Hinglish (same Latin script for both English and Hindi) makes unsupervised language boundary detection inherently difficult
