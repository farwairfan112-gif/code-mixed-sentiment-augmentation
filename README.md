# Cross-Lingual Synthetic Augmentation and Language-Aware Training for Code-Mixed Sentiment Analysis

**FAST-NUCES Islamabad — NLP Final Project**  
Farwa Irfan (25I-8012) · Annash Ahmed (25I-8005) · Wania Hassan (25I-7629)

Based on: *Leveraging Large Language Models for Code-Mixed Data Augmentation in Sentiment Analysis* — Linda Zeng (SICon @ EMNLP 2024)

---

## Project Overview

Three-phase project covering paper understanding, reproduction of 13 experiments, and three novel extensions (CLAT, CDWL, LAAF) validated across 18 experiments on Hinglish, Spanish-English, and Malayalam-English datasets.

**Best result:** CDWL+LAAF on Hinglish natural data — F1 = **0.7169** (+0.0008 over XLM-T baseline)  
**Key finding:** CLAT shows +2.56% improvement with proxy evaluation but −0.90% with proper test splits — highlighting the critical importance of correct evaluation strategy in CM NLP.

---

## Project Structure

```
project-root/
├── README.md
├── requirements.txt
├── train.py               ← Run experiments / save checkpoint
├── inference.py           ← Load checkpoint and predict
├── config.yaml            ← All hyperparameters and paths
├── data/
│   └── sample_data.csv    ← 10 demo samples (Sp-En + Hinglish)
├── notebooks/
│   └── 01_inference_demo.ipynb  ← Inference demo with visualizations
├── src/
│   ├── dataset.py         ← Preprocessing, PyTorch datasets
│   ├── model.py           ← XLM-T, mBERT, LAAF architectures
│   └── utils.py           ← Training loops, evaluation, checkpointing
├── results/
│   ├── baseline_metrics.json    ← Reproduction results (13 experiments)
│   ├── improved_metrics.json    ← Extension results (18 experiments)
│   └── training_log.csv         ← Per-epoch loss and F1 logs
└── checkpoints/
    └── sp_xlmt_natural/         ← Primary experiment checkpoint (after training)
```

---

## Setup

```bash
# Clone repo and install dependencies
pip install -r requirements.txt

# Clone the paper's data repository
git clone https://github.com/lindazeng979/LLM-CMSA.git
```

For Hinglish experiments, download SentiMix from:  
https://github.com/dipteshkanojia/challengeSentimix  
Place as `data/sentimix_train.csv`, `data/sentimix_dev.csv`, `data/sentimix_test.csv`

---

## Training

### Primary Experiment (Professor-approved single run)

Trains **Spanish-English | XLM-T | Natural Only** and saves checkpoint.  
~9 hours on a T4 GPU.

```bash
python train.py
```

Checkpoint saved to: `checkpoints/sp_xlmt_natural/`

### All 13 Reproduction Experiments

```bash
python train.py --all-reproduction
```

### All 18 Extension Experiments (requires Hinglish data)

```bash
python train.py --all-extension
```

---

## Inference

### Demo mode (sample_data.csv, no GPU required)

```bash
python inference.py --demo
```

### Single text

```bash
python inference.py --text "Me encanta este producto!"
# → Positive (94.2%)
```

### From file (one sentence per line)

```bash
python inference.py --file my_texts.txt
```

### Interactive mode

```bash
python inference.py
# >>> type your text here
```

> **If no checkpoint exists**, inference.py automatically falls back to the base XLM-T model from HuggingFace (no fine-tuning). Run `python train.py` first for the fine-tuned version.

---

## Notebook Demo

```bash
cd notebooks
jupyter notebook 01_inference_demo.ipynb
```

Demonstrates:
- Loading the saved checkpoint
- Running predictions on sample data with confidence scores
- Reproduction results table (paper vs. reproduced F1)
- Ablation heatmap (CLAT / CDWL / LAAF / Combined)
- Interactive single-text prediction

---

## Experiments Summary

### Reproduction (Assignment 2) — 10/13 within 0.04 F1

| Exp | Model | Dataset | Paper F1 | Repro F1 | Diff |
|-----|-------|---------|----------|----------|------|
| A1  | mBERT | Sp-En Natural | 0.564 | 0.544 | −0.020 |
| **A2** | **XLM-T** | **Sp-En Natural** | **0.588** | **0.582** | **−0.006** |
| A3  | XLM-T | Sp-En LLM Syn Only | 0.544 | 0.531 | −0.013 |
| A4  | XLM-T | Sp-En Nat+LLM (GFT) | 0.603 | 0.558 | −0.045 |
| A5  | XLM-T | Sp-En Nat+Random | 0.563 | 0.581 | +0.018 |
| C2  | XLM-T | Ma-En Natural | 0.843 | 0.734 | −0.109 |

### Extension (Assignment 3) — Hinglish Ablation

| Method | Natural | +Sp-En Syn | +Ma-En Syn | mBERT |
|--------|---------|------------|------------|-------|
| Standard | 0.7161 | 0.7071 | 0.7119 | 0.6817 |
| CDWL     | 0.7140 | 0.7022 | 0.7129 | 0.6722 |
| LAAF     | 0.7120 | 0.7129 | 0.7167 | 0.6844 |
| **CDWL+LAAF** | **0.7169** | 0.7159 | 0.7093 | 0.6761 |

---

## Novel Contributions

### 1. CLAT — Cross-Lingual Augmentation Transfer
Zero-cost transfer of LLM-generated Spanish-English synthetic data to Hinglish training. Hypothesis: alternational CM pairs (Sp-En ↔ Hinglish) should transfer better than insertional (Ma-En). **Result:** CLAT fails in full-data regime (−0.90% on proper test split), exposing proxy evaluation as misleading.

### 2. CDWL — Code-Mix Degree Weighted Loss
Per-sample loss weighting by CM degree: `loss = CE × (1 + α_s)` where `α_s` = proportion of non-English tokens (lingua-based). Modest improvements in some settings; limited by romanized script ambiguity for Hinglish.

### 3. LAAF — Language-Aware Attention Fusion
Custom classification head with separate attention paths for English and L2 tokens, supervised by auxiliary BCE loss on a language scorer MLP. Best gains when diverse CM training data is available (+0.0047 to +0.0059).

---

## Citation

```bibtex
@inproceedings{irfan2025clat,
  title     = {Cross-Lingual Synthetic Augmentation and Language-Aware Training
               for Code-Mixed Sentiment Analysis},
  author    = {Irfan, Farwa and Ahmed, Annash and Hassan, Wania},
  booktitle = {FAST-NUCES NLP Final Project Report},
  year      = {2025}
}

@inproceedings{zeng2024llmcmsa,
  title     = {Leveraging Large Language Models for Code-Mixed Data Augmentation
               in Sentiment Analysis},
  author    = {Zeng, Linda},
  booktitle = {Proc. 2nd Workshop on Social Influence in Conversations (SICon @ EMNLP 2024)},
  year      = {2024}
}
```
