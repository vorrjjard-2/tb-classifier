"""Evaluate a trained checkpoint on the TBX11K test split.

Loads a Lightning checkpoint, runs inference over `data/<root>/test/`, and
prints the headline metrics: per-class sensitivity / specificity / AUROC,
macro averages, the sklearn classification report, and the confusion matrix.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --config experiments/configs/default.yaml --ckpt experiments/results/best.ckpt
    uv run python scripts/evaluate.py --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from tb_classifier.config import load_config
from tb_classifier.data import build_test_loader
from tb_classifier.training import TBLitModule
from tb_classifier.utils.metrics import CLASS_NAMES


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/configs/default.yaml",
        help="Path to the YAML run config (used for data + model shape).",
    )
    parser.add_argument(
        "--ckpt",
        default="experiments/results/best.ckpt",
        help="Path to the .ckpt file to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on: 'auto' | 'cpu' | 'cuda' | 'mps'.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the full metrics as JSON (e.g. a Drive folder).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _pick_device(args.device)
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    print(f"config: {args.config}")
    print(f"ckpt:   {ckpt_path}")
    print(f"device: {device}")

    test_loader = build_test_loader(cfg.data)
    print(
        f"test set: {len(test_loader.dataset)} images, "
        f"class counts: {test_loader.dataset.class_counts()}"
    )

    module = TBLitModule.load_from_checkpoint(
        str(ckpt_path),
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        map_location=device,
    )
    module.eval().to(device)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in test_loader:
            logits = module(batch["image"].to(device, non_blocking=True))
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    n_classes = len(CLASS_NAMES)
    labels_onehot = np.eye(n_classes)[labels]

    # One-vs-rest confusion-matrix metrics per class. For TB screening the key
    # numbers are sensitivity (recall on a class) and specificity; PPV/NPV/F1
    # round out the clinical picture.
    print("\n=== Per-class ===")
    cols = ("sens", "spec", "ppv", "npv", "f1", "auroc", "support")
    print(f"{'class':<14} " + " ".join(f"{c:>8}" for c in cols))
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(CLASS_NAMES):
        cls_pred = preds == i
        cls_true = labels == i
        tp = int((cls_pred & cls_true).sum())
        fp = int((cls_pred & ~cls_true).sum())
        fn = int((~cls_pred & cls_true).sum())
        tn = int((~cls_pred & ~cls_true).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0  # recall / sensitivity
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0  # precision
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
        auc = roc_auc_score(labels_onehot[:, i], probs[:, i])
        per_class[name] = {
            "sensitivity": sens, "specificity": spec, "ppv": ppv,
            "npv": npv, "f1": f1, "auroc": auc, "support": int(cls_true.sum()),
        }
        print(
            f"{name:<14} {sens:8.4f} {spec:8.4f} {ppv:8.4f} {npv:8.4f} "
            f"{f1:8.4f} {auc:8.4f} {int(cls_true.sum()):8d}"
        )

    macro_auroc = roc_auc_score(labels_onehot, probs, multi_class="ovr", average="macro")
    macro = {
        m: float(np.mean([per_class[n][m] for n in CLASS_NAMES]))
        for m in ("sensitivity", "specificity", "ppv", "npv", "f1")
    }
    macro["auroc"] = float(macro_auroc)
    accuracy = float((preds == labels).mean())
    print(
        f"{'macro':<14} {macro['sensitivity']:8.4f} {macro['specificity']:8.4f} "
        f"{macro['ppv']:8.4f} {macro['npv']:8.4f} {macro['f1']:8.4f} "
        f"{macro_auroc:8.4f} {len(labels):8d}"
    )
    print(f"\naccuracy: {accuracy:.4f}")

    # WHO target product profile for a TB triage test: sensitivity >= 0.90,
    # specificity >= 0.70 (https://www.who.int/publications/i/item/9789241514828).
    tb = per_class["tb"]
    who_ok = tb["sensitivity"] >= 0.90 and tb["specificity"] >= 0.70
    print(
        f"WHO triage TPP (tb sens>=0.90, spec>=0.70): "
        f"{'PASS' if who_ok else 'FAIL'} "
        f"(sens={tb['sensitivity']:.4f}, spec={tb['specificity']:.4f})"
    )

    print("\n=== sklearn classification report ===")
    print(classification_report(labels, preds, target_names=list(CLASS_NAMES), digits=4))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    header = " " * 14 + "  ".join(f"{n:>10s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{c:>10d}" for c in cm[i])
        print(f"{name:<14}{row}")

    if args.out:
        import json

        results = {
            "config": str(args.config),
            "ckpt": str(ckpt_path),
            "test_size": len(labels),
            "accuracy": accuracy,
            "macro": macro,
            "per_class": per_class,
            "who_triage_tpp_pass": who_ok,
            "confusion_matrix": cm.tolist(),
            "class_names": list(CLASS_NAMES),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nmetrics written to: {out_path}")


if __name__ == "__main__":
    main()
