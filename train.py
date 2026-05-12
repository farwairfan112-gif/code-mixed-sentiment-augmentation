"""
train.py — Run the primary experiment and save checkpoint.

Professor-approved single experiment:
  Spanish-English  |  XLM-T  |  Natural Only  (Experiment A2 / C1 baseline)
  Paper F1: 0.588   Repro F1: 0.582

Usage:
    python train.py                     # primary experiment only
    python train.py --all-reproduction  # all 13 reproduction experiments
    python train.py --all-extension     # all 18 extension experiments
    python train.py --exp C1            # single named experiment
"""

import argparse
import json
import os

import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

from src.dataset import (
    create_sample_data, download_data, preprocess_df,
)
from src.model import mBERTwithLAAF, XLMTwithLAAF
from src.utils import (
    run_cdwl_training, run_combined_training, run_gradual_finetuning,
    run_laaf_training, run_standard_training,
)

# ── Config ────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_DIR   = CFG["data"]["repo_dir"]
TR         = CFG["training"]
CKPT_DIR   = CFG["output"]["checkpoints"]
RES_DIR    = CFG["output"]["results"]
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR,  exist_ok=True)

print(f"Device: {DEVICE}")


# ── Data loading ──────────────────────────────────────────────────────────

def load_all_data():
    download_data(CFG["data"]["repo_url"], REPO_DIR)
    sp = CFG["data"]["spanish_english"]
    ma = CFG["data"]["malayalam_english"]

    # Spanish-English
    sp_train_raw  = pd.read_csv(sp["train_natural"])
    sp_dev_raw    = pd.read_csv(sp["dev_natural"])
    sp_syn_llm    = preprocess_df(pd.read_csv(sp["synthetic_llm"]),    "sp_syn_llm")
    sp_syn_rnd    = preprocess_df(pd.read_csv(sp["synthetic_random"]), "sp_syn_rnd")
    sp_train_full = preprocess_df(sp_train_raw, "sp_train")
    sp_test       = preprocess_df(sp_dev_raw,   "sp_test(dev proxy)")

    sp_train, sp_dev = train_test_split(
        sp_train_full, test_size=0.10, random_state=TR["seed"],
        stratify=sp_train_full["label_id"],
    )
    sp_train = sp_train.reset_index(drop=True)
    sp_dev   = sp_dev.reset_index(drop=True)

    # Spanish-English 3k low-resource
    sp_train_3k, _ = train_test_split(
        sp_train, train_size=3000, random_state=TR["seed"],
        stratify=sp_train["label_id"],
    )

    # Malayalam-English
    ma_train = preprocess_df(pd.read_csv(ma["train_natural"]), "ma_train")
    ma_dev   = preprocess_df(pd.read_csv(ma["dev_natural"]),   "ma_dev")
    ma_test  = preprocess_df(pd.read_csv(ma["test_natural"]),  "ma_test")
    ma_syn   = preprocess_df(pd.read_csv(ma["synthetic_llm"]), "ma_syn")

    print(f"\nData loaded. Sp-train:{len(sp_train)}  Sp-test:{len(sp_test)}"
          f"  Ma-train:{len(ma_train)}  Ma-test:{len(ma_test)}")
    return (sp_train, sp_dev, sp_test, sp_train_3k,
            sp_syn_llm, sp_syn_rnd,
            ma_train, ma_dev, ma_test, ma_syn)


def load_hinglish():
    """Load SentiMix (Hinglish) dataset — needed for extension experiments."""
    try:
        train = preprocess_df(pd.read_csv("data/sentimix_train.csv"), "hi_train")
        dev   = preprocess_df(pd.read_csv("data/sentimix_dev.csv"),   "hi_dev")
        test  = preprocess_df(pd.read_csv("data/sentimix_test.csv"),  "hi_test")
        return train, dev, test
    except FileNotFoundError:
        print("SentiMix not found. Download from: https://github.com/dipteshkanojia/challengeSentimix")
        return None, None, None


# ── Save results ──────────────────────────────────────────────────────────

def save_result(results: dict, filename: str):
    path = os.path.join(RES_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}")


# ── Primary experiment ────────────────────────────────────────────────────

def run_primary_experiment():
    """
    Exp A2: Spanish-English | XLM-T | Natural Only
    This is the professor-approved single experiment.
    Saves checkpoint to checkpoints/sp_xlmt_natural/
    """
    print("\n" + "="*65)
    print("PRIMARY EXPERIMENT: Sp-En | XLM-T | Natural Only")
    print("Paper F1: 0.588  |  Expected repro: ~0.582")
    print("="*65)

    (sp_train, sp_dev, sp_test, *_) = load_all_data()[:6]  # only need sp splits

    f1, losses, dev_f1s, preds, labs = run_standard_training(
        model_name      = "XLM-T",
        train_df        = sp_train,
        dev_df          = sp_dev,
        test_df         = sp_test,
        experiment_name = "A2_sp_xlmt_natural",
        device          = DEVICE,
        num_epochs      = TR["num_epochs"],
        lr              = TR["learning_rate"],
        weight_decay    = TR["weight_decay"],
        batch_size      = TR["batch_size"],
        max_seq_len     = TR["max_seq_len"],
        save_dir        = os.path.join(CKPT_DIR, "sp_xlmt_natural"),
        seed            = TR["seed"],
    )

    result = {
        "experiment": "A2_sp_xlmt_natural",
        "model": "XLM-T", "dataset": "Spanish-English",
        "setup": "Natural Only", "paper_f1": 0.588,
        "repro_f1": round(f1, 4),
        "diff": round(f1 - 0.588, 4),
        "train_losses": losses, "dev_f1s": dev_f1s,
    }
    save_result(result, "primary_experiment.json")
    print(f"\nPrimary experiment done. F1={f1:.4f}  (paper: 0.588)")
    return result


# ── All reproduction experiments ─────────────────────────────────────────

def run_all_reproduction():
    """Run all 13 paper reproduction experiments (Table II–IV)."""
    (sp_train, sp_dev, sp_test, sp_train_3k,
     sp_syn_llm, sp_syn_rnd,
     ma_train, ma_dev, ma_test, ma_syn) = load_all_data()

    results = {}

    # ── Spanish-English Full 12k (Table II) ──────────────────────────────
    exps_sp = [
        ("A1",  "mBERT", sp_train,    sp_dev, sp_test, None,       None,    0.564),
        ("A2",  "XLM-T", sp_train,    sp_dev, sp_test, None,       None,    0.588),
        ("A3",  "XLM-T", sp_syn_llm,  sp_dev, sp_test, None,       None,    0.544),
        ("A5",  "XLM-T", sp_train,    sp_dev, sp_test, None,       None,    0.563),  # +random
    ]
    for exp_id, model, tr, dv, ts, _, _, paper_f1 in [
        ("A1", "mBERT", sp_train,   sp_dev, sp_test, None, None, 0.564),
        ("A2", "XLM-T", sp_train,   sp_dev, sp_test, None, None, 0.588),
        ("A3", "XLM-T", sp_syn_llm, sp_dev, sp_test, None, None, 0.544),
        ("A5", "XLM-T", pd.concat([sp_train, sp_syn_rnd], ignore_index=True),
                        sp_dev, sp_test, None, None, 0.563),
    ]:
        f1, l, d, p, lb = run_standard_training(
            model, tr, dv, ts, f"{exp_id}_sp_{model}_natural", DEVICE,
            num_epochs=TR["num_epochs"], lr=TR["learning_rate"],
            weight_decay=TR["weight_decay"], batch_size=TR["batch_size"],
            max_seq_len=TR["max_seq_len"], seed=TR["seed"],
        )
        results[exp_id] = {"f1": round(f1,4), "paper_f1": paper_f1,
                           "diff": round(f1-paper_f1,4)}

    # Exp A4: GFT (Nat + LLM Synthetic, gradual fine-tuning)
    f1, l, d, p, lb = run_gradual_finetuning(
        "XLM-T", sp_train, sp_syn_llm, sp_dev, sp_test,
        "A4_sp_xlmt_gft", DEVICE,
        num_epochs=TR["num_epochs"], batch_size=TR["batch_size"],
        max_seq_len=TR["max_seq_len"], seed=TR["seed"],
    )
    results["A4"] = {"f1": round(f1,4), "paper_f1": 0.603, "diff": round(f1-0.603,4)}

    # ── Spanish-English 3k low-resource (Table III) ───────────────────────
    for exp_id, model, tr, paper_f1 in [
        ("B1", "XLM-T", sp_train_3k, 0.547),
        ("B3", "mBERT", sp_train_3k, 0.487),
    ]:
        f1, *_ = run_standard_training(
            model, tr, sp_dev, sp_test, f"{exp_id}_sp3k_{model}", DEVICE,
            num_epochs=TR["num_epochs"], lr=TR["learning_rate"],
            batch_size=TR["batch_size"], max_seq_len=TR["max_seq_len"], seed=TR["seed"],
        )
        results[exp_id] = {"f1": round(f1,4), "paper_f1": paper_f1, "diff": round(f1-paper_f1,4)}

    f1, *_ = run_gradual_finetuning(
        "XLM-T", sp_train_3k, sp_syn_llm, sp_dev, sp_test,
        "B2_sp3k_xlmt_gft", DEVICE,
        num_epochs=TR["num_epochs"], batch_size=TR["batch_size"],
        max_seq_len=TR["max_seq_len"], seed=TR["seed"],
    )
    results["B2"] = {"f1": round(f1,4), "paper_f1": 0.598, "diff": round(f1-0.598,4)}

    for exp_id, model, tr, paper_f1 in [
        ("B4", "mBERT", pd.concat([sp_train_3k, sp_syn_llm], ignore_index=True), 0.526),
    ]:
        f1, *_ = run_standard_training(
            model, tr, sp_dev, sp_test, f"{exp_id}_sp3k_mbert_syn", DEVICE,
            num_epochs=TR["num_epochs"], lr=TR["learning_rate"],
            batch_size=TR["batch_size"], max_seq_len=TR["max_seq_len"], seed=TR["seed"],
        )
        results[exp_id] = {"f1": round(f1,4), "paper_f1": paper_f1, "diff": round(f1-paper_f1,4)}

    # ── Malayalam-English (Table IV) ──────────────────────────────────────
    for exp_id, model, tr, paper_f1 in [
        ("C1", "mBERT", ma_train, 0.750),
        ("C2", "XLM-T", ma_train, 0.843),
        ("C3", "XLM-T", pd.concat([ma_train, ma_syn], ignore_index=True), 0.763),
        ("C4", "mBERT", pd.concat([ma_train, ma_syn], ignore_index=True), 0.745),
    ]:
        f1, *_ = run_standard_training(
            model, tr, ma_dev, ma_test, f"{exp_id}_ma_{model}", DEVICE,
            num_epochs=TR["num_epochs"], lr=TR["learning_rate"],
            batch_size=TR["batch_size"], max_seq_len=TR["max_seq_len"], seed=TR["seed"],
        )
        results[exp_id] = {"f1": round(f1,4), "paper_f1": paper_f1, "diff": round(f1-paper_f1,4)}

    save_result(results, "baseline_metrics.json")
    print("\n=== REPRODUCTION SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: Repro={v['f1']}  Paper={v['paper_f1']}  Δ={v['diff']:+.3f}")
    return results


# ── All extension experiments ─────────────────────────────────────────────

def run_all_extension():
    """Run all 18 extension experiments (CLAT, CDWL, LAAF, Combined)."""
    hi_train, hi_dev, hi_test = load_hinglish()
    if hi_train is None:
        print("Hinglish data not available. Skipping extension experiments.")
        return {}

    (sp_train, sp_dev, sp_test, _,
     sp_syn_llm, _, ma_train, ma_dev, ma_test, ma_syn) = load_all_data()

    # Prepare mix_score columns for CDWL/Combined
    for df in [hi_train, hi_dev, hi_test]:
        if df is not None and "mix_score" not in df.columns:
            from src.dataset import compute_mix_score
            df["mix_score"] = df["text"].apply(compute_mix_score).clip(0, 1)

    hi_plus_sp = pd.concat(
        [hi_train, sp_syn_llm.sample(15000, random_state=TR["seed"])],
        ignore_index=True
    )
    hi_plus_ma = pd.concat(
        [hi_train, ma_syn.sample(min(15000, len(ma_syn)), random_state=TR["seed"])],
        ignore_index=True
    )
    for df in [hi_plus_sp, hi_plus_ma]:
        if "mix_score" not in df.columns:
            from src.dataset import compute_mix_score
            df["mix_score"] = df["text"].apply(compute_mix_score).clip(0, 1)

    results = {}

    # ── CLAT baselines ────────────────────────────────────────────────────
    clat_exps = [
        ("C1", "XLM-T", hi_train),
        ("C2", "XLM-T", hi_plus_sp),
        ("C3", "XLM-T", hi_plus_ma),
        ("C4", "mBERT", hi_train),
    ]
    for eid, model, tr in clat_exps:
        f1, *_ = run_standard_training(
            model, tr, hi_dev, hi_test, f"{eid}_clat_{model}", DEVICE,
            num_epochs=3, lr=5e-5, batch_size=32, max_seq_len=64, seed=TR["seed"],
        )
        results[eid] = {"f1": round(f1, 4)}

    # ── CDWL ──────────────────────────────────────────────────────────────
    cdwl_exps = [
        ("W1", "XLM-T", hi_train),
        ("W2", "XLM-T", hi_plus_sp),
        ("W3", "XLM-T", hi_plus_ma),
        ("W4", "mBERT", hi_train),
    ]
    for eid, model, tr in cdwl_exps:
        f1, *_ = run_cdwl_training(
            model, tr, hi_dev, hi_test, f"{eid}_cdwl_{model}", DEVICE,
            num_epochs=3, lr=5e-5, batch_size=32, max_seq_len=64, seed=TR["seed"],
        )
        results[eid] = {"f1": round(f1, 4)}

    # ── LAAF ──────────────────────────────────────────────────────────────
    laaf_exps = [
        ("L1", XLMTwithLAAF,  hi_train),
        ("L2", XLMTwithLAAF,  hi_plus_sp),
        ("L3", XLMTwithLAAF,  hi_plus_ma),
        ("L4", mBERTwithLAAF, hi_train),
    ]
    for eid, cls, tr in laaf_exps:
        f1, *_ = run_laaf_training(
            tr, hi_dev, hi_test, f"{eid}_laaf", DEVICE,
            model_class=cls, warmup_epochs=2, joint_epochs=3,
            lr=5e-5, batch_size=32, max_seq_len=64, seed=TR["seed"],
        )
        results[eid] = {"f1": round(f1, 4)}

    # ── Combined CDWL+LAAF ────────────────────────────────────────────────
    comb_exps = [
        ("CL1", XLMTwithLAAF,  hi_train,    hi_dev,  hi_test,  None,    None),
        ("CL2", XLMTwithLAAF,  hi_plus_sp,  hi_dev,  hi_test,  None,    None),
        ("CL3", XLMTwithLAAF,  hi_plus_ma,  hi_dev,  hi_test,  None,    None),
        ("CL4", mBERTwithLAAF, hi_train,    hi_dev,  hi_test,  None,    None),
        ("CL5", XLMTwithLAAF,  sp_train,    sp_dev,  sp_test,  None,    None),
        ("CL6", XLMTwithLAAF,  ma_train,    ma_dev,  ma_test,  None,    None),
    ]
    for eid, cls, tr, dv, ts, _, _ in comb_exps:
        f1, *_ = run_combined_training(
            tr, dv, ts, f"{eid}_combined", DEVICE,
            model_class=cls, warmup_epochs=2, joint_epochs=3,
            lr=5e-5, batch_size=32, max_seq_len=64, seed=TR["seed"],
        )
        results[eid] = {"f1": round(f1, 4)}

    save_result(results, "improved_metrics.json")
    print("\n=== EXTENSION SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: F1={v['f1']}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-reproduction", action="store_true")
    parser.add_argument("--all-extension",    action="store_true")
    parser.add_argument("--exp",              type=str, default=None,
                        help="Run single named experiment, e.g. --exp C1")
    args = parser.parse_args()

    if args.all_reproduction:
        run_all_reproduction()
    elif args.all_extension:
        run_all_extension()
    else:
        # Default: professor-approved primary experiment
        run_primary_experiment()


if __name__ == "__main__":
    main()
