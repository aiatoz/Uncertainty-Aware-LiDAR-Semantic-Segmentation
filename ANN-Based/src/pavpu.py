"""
Patch Accuracy vs Patch Uncertainty (PAvPU) metric for 3D point clouds.

Extends the 2D image-based PAvPU to point cloud "patches" defined by K-nearest-neighbour or radius-based neighbourhoods, per the paper
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger

logger = get_logger(__name__)


def compute_patches(
    xyz: np.ndarray,
    k: int,
    query_chunk_size: int = 500_000,
) -> np.ndarray:
    """
    For every point, find its K nearest neighbours (including itself at idx 0).
    Returns an (n_points, k) int32 index array.
    """
    n = len(xyz)
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)
    nn.fit(xyz)

    indices = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, query_chunk_size):
        end = min(start + query_chunk_size, n)
        _, idx_chunk = nn.kneighbors(xyz[start:end])
        indices[start:end] = idx_chunk.astype(np.int32)
    return indices


def compute_pavpu(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    xyz: np.ndarray,
    k: int = 30,
    uncertainty_threshold: Optional[float] = None,
    accuracy_threshold: float = 0.5,
    patch_chunk_size: int = 500_000,
) -> Tuple[float, dict]:
    """
    Computes PAvPU over the point cloud.

    Each point defines a "patch" = its K nearest neighbours. For each patch:
      - patch_accuracy  = fraction of K neighbours correctly classified
      - patch_uncertainty = mean normalised uncertainty across K neighbours

    Patches are then classified into:
      AC (Accurate & Certain), AU (Accurate & Uncertain),
      IC (Inaccurate & Certain), IU (Inaccurate & Uncertain)

    PAvPU = (N_AC + N_IU) / (N_AC + N_AU + N_IC + N_IU)

    Parameters
    ----------
    uncertainty_threshold : float or None
        If None, uses median of patch uncertainties
    accuracy_threshold : float
        Fraction of correctly-classified neighbours for a patch to be
        considered "accurate". Default 0.5.
    patch_chunk_size : int
        Reducing chunk-by-chunk keeps peak memory to
        `patch_chunk_size * k` instead of `n * k`.
    """
    n = len(y_true)
    assert len(y_pred) == n and len(uncertainty) == n and len(xyz) == n, \
        "y_true, y_pred, uncertainty, xyz must all have the same length."

    patch_indices = compute_patches(xyz, k=k)

    correct = (y_true == y_pred).astype(np.float32)
    uncertainty = uncertainty.astype(np.float32)

    patch_acc = np.empty(n, dtype=np.float32)
    patch_unc = np.empty(n, dtype=np.float32)
    for start in range(0, n, patch_chunk_size):
        end = min(start + patch_chunk_size, n)
        idx_chunk = patch_indices[start:end]         # (chunk, k)
        patch_acc[start:end] = correct[idx_chunk].mean(axis=1)
        patch_unc[start:end] = uncertainty[idx_chunk].mean(axis=1)

    if uncertainty_threshold is None:
        uncertainty_threshold = float(np.median(patch_unc))
        logger.warning(
            f"uncertainty_threshold not set — using median of patch "
            f"uncertainties ({uncertainty_threshold:.4f}) for this run. "
            f"This threshold is NOT comparable across experiments/datasets "
            f"Set pavpu.uncertainty_threshold in "
            f"configs/default.yaml for reproducible cross-run comparison."
        )

    is_accurate  = patch_acc >= accuracy_threshold
    is_uncertain = patch_unc >= uncertainty_threshold

    N_AC = int(( is_accurate & ~is_uncertain).sum())
    N_AU = int(( is_accurate &  is_uncertain).sum())
    N_IC = int((~is_accurate & ~is_uncertain).sum())
    N_IU = int((~is_accurate &  is_uncertain).sum())
    total = N_AC + N_AU + N_IC + N_IU

    pavpu = (N_AC + N_IU) / total if total > 0 else 0.0

    details = {
        "PAvPU": round(pavpu, 4),
        "N_AC": N_AC, "N_AU": N_AU, "N_IC": N_IC, "N_IU": N_IU,
        "total_patches": total,
        "uncertainty_threshold": round(uncertainty_threshold, 4),
        "accuracy_threshold": accuracy_threshold,
        "k_neighbors": k,
        "pct_AC": round(100 * N_AC / total, 2) if total else 0,
        "pct_AU": round(100 * N_AU / total, 2) if total else 0,
        "pct_IC": round(100 * N_IC / total, 2) if total else 0,
        "pct_IU": round(100 * N_IU / total, 2) if total else 0,
    }
    return pavpu, details