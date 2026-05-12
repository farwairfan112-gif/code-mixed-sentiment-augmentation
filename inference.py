"""
inference.py — Load a saved checkpoint and run predictions.

The professor requires inference to work. This script:
  1. Loads the saved checkpoint from checkpoints/sp_xlmt_natural/
  2. Accepts text from CLI, a file, or the sample CSV
  3. Prints predictions with confidence scores

Usage:
    # Single text
    python inference.py --text "Me encanta este producto!"

    # From file (one sentence per line)
    python inference.py --file my_texts.txt

    # Run on sample_data.csv (demo — no GPU needed)
    python inference.py --demo

    # Specify a different checkpoint
    python inference.py --checkpoint checkpoints/sp_xlmt_natural --text "..."
"""

import argparse
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset import clean_text
from src.model import MODEL_NAMES, XLMTwithLAAF, mBERTwithLAAF

ID2LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Checkpoint loader ──────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint_dir: str):
    """
    Auto-detects checkpoint type (standard HF model vs LAAF state_dict)
    and loads accordingly.
    """
    meta_path = os.path.join(checkpoint_dir, "metadata.json")
    method    = "standard"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        method = meta.get("method", "standard")
        print(f"  Experiment : {meta.get('experiment', 'N/A')}")
        print(f"  Method     : {method}")
        print(f"  Test F1    : {meta.get('test_f1', 'N/A')}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    state_dict_path = os.path.join(checkpoint_dir, "model_state_dict.pt")
    if os.path.exists(state_dict_path):
        # LAAF model
        model_name = meta.get("model", "XLMTwithLAAF") if os.path.exists(meta_path) else "XLMTwithLAAF"
        ModelClass = mBERTwithLAAF if "mBERT" in model_name else XLMTwithLAAF
        model      = ModelClass()
        model.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
        print(f"  Loaded LAAF model ({ModelClass.__name__})")
    else:
        # Standard HuggingFace model
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        print(f"  Loaded standard model")

    model.eval()
    model.to(DEVICE)
    return model, tokenizer


# ── Prediction ─────────────────────────────────────────────────────────────

def predict(texts: list, model, tokenizer, max_len: int = 64) -> list:
    """
    Predict sentiment for a list of texts.
    Returns list of dicts: {text, label, confidence, scores}
    """
    cleaned = [clean_text(t) for t in texts]
    enc     = tokenizer(
        cleaned, max_length=max_len, padding=True,
        truncation=True, return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for i, text in enumerate(texts):
        pred_id    = int(probs[i].argmax())
        label      = ID2LABEL[pred_id]
        confidence = float(probs[i][pred_id])
        results.append({
            "text":       text,
            "label":      label,
            "confidence": round(confidence, 4),
            "scores": {
                "Positive": round(float(probs[i][0]), 4),
                "Negative": round(float(probs[i][1]), 4),
                "Neutral":  round(float(probs[i][2]), 4),
            },
        })
    return results


def print_results(results: list):
    print(f"\n{'─'*65}")
    print(f"{'TEXT':<40} {'LABEL':<10} {'CONF':>6}")
    print(f"{'─'*65}")
    for r in results:
        snippet = r["text"][:38] + ".." if len(r["text"]) > 40 else r["text"]
        print(f"{snippet:<40} {r['label']:<10} {r['confidence']:>6.1%}")
        print(f"  Scores → Pos:{r['scores']['Positive']:.3f}  "
              f"Neg:{r['scores']['Negative']:.3f}  "
              f"Neu:{r['scores']['Neutral']:.3f}")
    print(f"{'─'*65}")


# ── Demo mode (no weights needed) ─────────────────────────────────────────

def run_demo_with_sample():
    """
    Demo mode: loads sample_data.csv and runs inference on it.
    Uses the saved primary checkpoint if available, otherwise loads
    the base XLM-T directly from HuggingFace (no fine-tuning).
    """
    sample_path = "data/sample_data.csv"
    if not os.path.exists(sample_path):
        from src.dataset import create_sample_data
        create_sample_data(sample_path)

    df = pd.read_csv(sample_path)
    print(f"\nRunning inference on {len(df)} samples from {sample_path}")
    print("True labels are shown for comparison.\n")

    ckpt_dir = "checkpoints/sp_xlmt_natural"
    if os.path.exists(ckpt_dir):
        print(f"Loading fine-tuned checkpoint: {ckpt_dir}")
        model, tokenizer = load_model_and_tokenizer(ckpt_dir)
    else:
        print("No saved checkpoint found. Loading base XLM-T from HuggingFace.")
        print("(Run  python train.py  first to get the fine-tuned checkpoint.)\n")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES["XLM-T"])
        model     = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAMES["XLM-T"], num_labels=3
        ).eval().to(DEVICE)

    results = predict(df["text"].tolist(), model, tokenizer)

    # Show results with true labels
    print(f"\n{'─'*75}")
    print(f"{'TEXT':<35} {'TRUE':<10} {'PRED':<10} {'CONF':>6}  {'OK':>3}")
    print(f"{'─'*75}")
    correct = 0
    for r, true_label in zip(results, df["label"].tolist()):
        snippet = r["text"][:33] + ".." if len(r["text"]) > 35 else r["text"]
        ok      = "✓" if r["label"] == true_label else "✗"
        if r["label"] == true_label:
            correct += 1
        print(f"{snippet:<35} {true_label:<10} {r['label']:<10} "
              f"{r['confidence']:>6.1%}  {ok:>3}")
    print(f"{'─'*75}")
    print(f"Accuracy on sample: {correct}/{len(results)} = {correct/len(results):.0%}")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Code-Mixed Sentiment Analysis — Inference"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/sp_xlmt_natural",
                        help="Path to checkpoint directory")
    parser.add_argument("--text",  type=str, default=None,
                        help="Single text to classify")
    parser.add_argument("--file",  type=str, default=None,
                        help="File with one sentence per line")
    parser.add_argument("--demo",  action="store_true",
                        help="Run on sample_data.csv (demo mode)")
    parser.add_argument("--max-len", type=int, default=64)
    args = parser.parse_args()

    if args.demo:
        run_demo_with_sample()
        return

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run  python train.py  first to train and save the primary experiment.")
        return

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    if args.text:
        results = predict([args.text], model, tokenizer, args.max_len)
        print_results(results)

    elif args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Running inference on {len(texts)} lines from {args.file}")
        results = predict(texts, model, tokenizer, args.max_len)
        print_results(results)

    else:
        # Interactive mode
        print("\nInteractive mode — type a sentence and press Enter.")
        print("Type 'quit' to exit.\n")
        while True:
            text = input(">>> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                results = predict([text], model, tokenizer, args.max_len)
                r = results[0]
                print(f"  → {r['label']} ({r['confidence']:.1%}) | "
                      f"Pos:{r['scores']['Positive']:.3f} "
                      f"Neg:{r['scores']['Negative']:.3f} "
                      f"Neu:{r['scores']['Neutral']:.3f}\n")


if __name__ == "__main__":
    main()
