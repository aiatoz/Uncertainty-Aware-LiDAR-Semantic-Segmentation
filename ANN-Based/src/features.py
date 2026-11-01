"""
Geometric feature extraction via Jakteristics
"""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.paths import get_feature_extraction_params  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_FEATURE_NAMES = [
    "eigenvalue_sum", "omnivariance", "eigenentropy", "anisotropy",
    "planarity", "linearity", "PCA1", "PCA2", "surface_variation",
    "sphericity", "verticality", "nx", "ny", "nz",
]


def extract_features(
    points_df: pd.DataFrame,
    dataset: str,
    radius: Optional[float] = None,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes jakteristics geometric features for every point in `points_df` (must have x, y, z columns) and returns a new DataFrame with 
    [x, y, z, label?, <features...>, __nan_feature__]
    """
    from jakteristics import compute_features

    params = get_feature_extraction_params(dataset)
    radius = radius if radius is not None else params["radius"]
    feature_names = feature_names if feature_names is not None else DEFAULT_FEATURE_NAMES
    num_threads = params.get("num_threads", -1)

    xyz = points_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    logger.info(
        f"[{dataset}] computing {len(feature_names)} features for "
        f"{len(xyz):,} points at radius={radius}m ..."
    )
    feats = compute_features(
        xyz, search_radius=radius, feature_names=feature_names, num_threads=num_threads
    )
    feats_df = pd.DataFrame(feats, columns=feature_names)

    assert len(feats_df) == len(points_df), (
        f"Row count mismatch after feature computation: "
        f"{len(feats_df)} features vs {len(points_df)} points — "
        f"this should never happen (jakteristics preserves point order/"
        f"count); if it does, stop and investigate before joining (F-4)."
    )

    points_reset = points_df.reset_index(drop=True)
    feats_df = feats_df.reset_index(drop=True)

    nan_mask = feats_df.isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        pct = 100 * n_nan / len(feats_df)
        msg = (
            f"[{dataset}] {n_nan:,} / {len(feats_df):,} points ({pct:.2f}%) "
            f"have at least one NaN feature at radius={radius}m — likely "
            f"sparse/boundary regions with too few neighbors. "
        )
        if "label" in points_reset.columns:
            affected_classes = points_reset.loc[nan_mask, "label"].value_counts()
            msg += f"Affected label distribution:\n{affected_classes}"
        logger.warning(msg)

    out = pd.concat([points_reset, feats_df], axis=1)
    out["__nan_feature__"] = nan_mask.values
    return out
