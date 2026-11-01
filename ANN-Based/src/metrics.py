"""
Evaluation metrics : accuracy, per-class IoU, mIoU, confusion matrix
"""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    excluded_classes: Optional[List[int]] = None,
) -> dict:
    """
    Compute per-class IoU, mIoU, and overall accuracy
    """
    excluded_classes = set(excluded_classes or [])
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    accuracy = cm.diagonal().sum() / cm.sum() if cm.sum() > 0 else 0.0

    per_class_iou = []
    zero_iou_classes = []
    included_ious = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = float(tp / denom) if denom > 0 else 0.0
        per_class_iou.append(iou)

        if c not in excluded_classes:
            n_true = int(cm[c, :].sum())
            if n_true == 0:
                zero_iou_classes.append(c)
                logger.warning(
                    f"Class {c} ({class_names[c] if class_names else c}) has "
                    f"0 true instances in this split — IoU=0 forced, which "
                    f"will drag down mIoU. This is logged explicitly (not "
                    f"silently suppressed) so you can decide whether to "
                    f"exclude it"
                )
            else:
                included_ious.append(iou)

    miou = float(np.mean(included_ious)) if included_ious else 0.0

    names = class_names or [str(i) for i in range(num_classes)]
    per_class_df = pd.DataFrame({
        "class_idx": list(range(num_classes)),
        "class_name": names,
        "n_true": [int(cm[c, :].sum()) for c in range(num_classes)],
        "n_pred": [int(cm[:, c].sum()) for c in range(num_classes)],
        "IoU": [round(iou, 4) for iou in per_class_iou],
        "excluded": [c in excluded_classes for c in range(num_classes)],
    })

    return {
        "accuracy": round(float(accuracy), 4),
        "mIoU": round(miou, 4),
        "per_class_iou": per_class_df,
        "confusion_matrix": cm,
        "excluded_classes": list(excluded_classes),
        "zero_iou_classes": zero_iou_classes,
    }


def results_to_dict(metrics: dict, model_name: str, dataset: str) -> dict:
    return {
        "model": model_name,
        "dataset": dataset,
        "accuracy": metrics["accuracy"],
        "mIoU": metrics["mIoU"],
        "zero_iou_classes": len(metrics["zero_iou_classes"]),
    }
