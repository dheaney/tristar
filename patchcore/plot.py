"""Plot a confusion matrix for the best PatchCore model saved by patchcore.train.

Expects <out_dir>/best_model/{best_model.ckpt,best_config.json};
writes <out_dir>/plots/patchcore_confusion_matrix.png.
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from anomalib.models import Patchcore
from anomalib.engine import Engine


def _get(batch, key):
    return batch[key] if isinstance(batch, dict) else getattr(batch, key)


def plot_patchcore(datamodule, out_dir="."):
    best_dir = os.path.join(out_dir, "best_model")
    ckpt_path = os.path.join(best_dir, "best_model.ckpt")
    cfg_path = os.path.join(best_dir, "best_config.json")
    out_path = os.path.join(out_dir, "plots", "patchcore_confusion_matrix.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for p in (ckpt_path, cfg_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}. Run that dataset's train_patchcore first.")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model = Patchcore(
        backbone=cfg["backbone"],
        layers=tuple(cfg["layers"]),
        coreset_sampling_ratio=cfg["coreset_sampling_ratio"],
        num_neighbors=cfg["num_neighbors"],
    )
    # num_neighbors isn't stored in the checkpoint, so it comes from the config
    model.model.num_neighbors = cfg["num_neighbors"]

    engine = Engine(enable_progress_bar=False)
    preds = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    y_true = np.concatenate([torch.as_tensor(_get(b, "gt_label")).cpu().numpy() for b in preds]).astype(int)
    y_pred = np.concatenate([torch.as_tensor(_get(b, "pred_label")).cpu().numpy() for b in preds]).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(cm)

    title = (
        f"PatchCore {'+'.join(cfg['layers'])}, "
        f"coreset={cfg['coreset_sampling_ratio']}, k={cfg['num_neighbors']}"
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["good", "defect"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
