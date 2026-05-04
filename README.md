# LLM-Based Code-Mixed Data Augmentation for Sentiment Analysis

PyTorch + HuggingFace reproduction and extension of [Zeng (2024)](https://aclanthology.org/2024.sicon-1.8/) 
— *Leveraging Large Language Models for Code-Mixed Data Augmentation in Sentiment Analysis* 
(SICon Workshop @ EMNLP 2024).

---

## What We Did

Reproduced 13 experiments from the original paper across Spanish-English and Malayalam-English 
code-mixed sentiment analysis, then proposed three novel extensions evaluated on Hinglish 
(SentiMix) in an 18-experiment ablation study.

**Reproduction** — fine-tuned mBERT and XLM-T with natural and GPT-4 synthetic data, 
including the paper's 5-stage Gradual Fine-Tuning (GFT) protocol. 10 of 13 experiments 
reproduced within 0.04 F1 of reported values.

**Extensions:**
- **CLAT** (Cross-Lingual Augmentation Transfer) — transfer synthetic data from one code-mixed 
  pair to another compatible pair
- **CDWL** (Code-Mix Degree Weighted Loss) — weight training samples by their degree of 
  code-mixing to prioritize linguistically complex examples
- **LAAF** (Language-Aware Attention Fusion) — lightweight architecture add-on with separate 
  attention heads for English and L2 tokens, fused with the CLS representation

---

## Datasets

| Dataset | Language Pair | CM Type | Size |
|---|---|---|---|
| LinCE SentMix | Spanish-English | Alternational | 12k train |
| MalayalamMixSentiment | Malayalam-English | Insertional | 3.4k train |
| SentiMix | Hinglish (Hindi-English) | Alternational | 14k train |
| GPT-4 Synthetic (Zeng 2024) | Sp-En + Ma-En | — | 53k + 24k |

Synthetic datasets from [Zeng (2024) GitHub](https://github.com/lindazelinzeng/llm-cm-augmentation).

---

## Models

- **mBERT** — `bert-base-multilingual-cased` (104 languages)
- **XLM-T** — `cardiffnlp/twitter-xlm-roberta-base` (trained on 198M multilingual tweets)

Both fine-tuned with a 3-class head: Positive / Negative / Neutral.

---

## Key Results

Reproduction: 10/13 experiments within 0.04 F1 of paper values.

Best extension result — **CDWL+LAAF on Hinglish natural data: F1 = 0.7169** (+0.0008 over 
XLM-T baseline).

Key finding: CLAT showed apparent +2.56% gain when evaluated on dev-as-test, 
but −0.90% on the proper held-out test set — highlighting the critical importance 
of evaluation protocol in low-resource NLP.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-codemixed-sentiment.git
pip install -r requirements.txt
```

Notebooks run on Kaggle P100 GPU. Dataset paths are set in the config cell at the top of each notebook.

---

## References

- Zeng, L. (2024). [Leveraging LLMs for Code-Mixed Data Augmentation](https://aclanthology.org/2024.sicon-1.8/). SICon @ EMNLP 2024.
- Patwa et al. (2020). SemEval-2020 Task 9: Sentiment Analysis of Code-Mixed Tweets.
- Barbieri et al. (2022). TweetNLP. EMNLP 2022.
- Devlin et al. (2019). BERT. NAACL 2019.
