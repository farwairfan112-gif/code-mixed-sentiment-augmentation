"""
Utility functions for evaluation, metrics, and logging
"""

import os
import json
import torch
from sklearn.metrics import f1_score, classification_report


def evaluate(model, loader, device='cuda'):
    """Evaluate model on given dataloader"""
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
            labs.extend(batch['labels'].numpy())
    f1 = f1_score(labs, preds, average='weighted', labels=[0, 1, 2], zero_division=0)
    return f1, preds, labs


def safe_report(labs, preds):
    """Print classification report"""
    print(classification_report(labs, preds, labels=[0, 1, 2],
                                target_names=['Positive', 'Negative', 'Neutral'],
                                zero_division=0))


def save_metrics(metrics, filepath):
    """Save metrics to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_checkpoint(model, optimizer, epoch, metric, experiment_name, checkpoint_dir="checkpoints"):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_epoch_{epoch}_f1_{metric:.4f}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'f1_score': metric,
        'experiment': experiment_name
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path