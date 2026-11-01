"""
Normalization / scaling
Usage pattern (always in this order):
    scaler = fit_scaler(train_df, feature_columns, config)
    train_df = apply_scaler(train_df, scaler, feature_columns)
    test_df  = apply_scaler(test_df,  scaler, feature_columns)   # transform only, never re-fit
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class FittedScaler:
    scaler: object
    columns: List[str]
    local_centered_xyz: bool
    xyz_columns: List[str]
    centroids: dict


def _local_center_xyz(df: pd.DataFrame, xyz_columns: List[str]) -> (pd.DataFrame, dict):
    """
    Re-centers x,y,z per source file relative to that file's own centroid
    """
    df = df.copy()
    centroids = {}
    if "__source_file__" not in df.columns:
        centroid = df[xyz_columns].mean()
        df[xyz_columns] = df[xyz_columns] - centroid
        centroids["__all__"] = centroid.to_dict()
        return df, centroids

    for fname, group in df.groupby("__source_file__"):
        centroid = group[xyz_columns].mean()
        df.loc[group.index, xyz_columns] = group[xyz_columns] - centroid
        centroids[fname] = centroid.to_dict()
    return df, centroids


def _apply_known_centroids(df: pd.DataFrame, xyz_columns: List[str], centroids: dict) -> pd.DataFrame:
    df = df.copy()
    if "__source_file__" not in df.columns:
        centroid = df[xyz_columns].mean()
        df[xyz_columns] = df[xyz_columns] - centroid
        return df

    for fname, group in df.groupby("__source_file__"):
        if fname in centroids:
            centroid = pd.Series(centroids[fname])
        else:
            centroid = group[xyz_columns].mean()
        df.loc[group.index, xyz_columns] = group[xyz_columns] - centroid
    return df


def fit_scaler(train_df: pd.DataFrame, feature_columns: List[str], config: dict) -> FittedScaler:
    """
    Fits the scaler on TRAIN data only
    """
    norm_cfg = config["normalization"]
    xyz_columns = [c for c in ["x", "y", "z"] if c in feature_columns]
    non_xyz_columns = [c for c in feature_columns if c not in xyz_columns]

    df = train_df.copy()
    centroids = {}
    if norm_cfg.get("local_center_xyz", True) and xyz_columns:
        df, centroids = _local_center_xyz(df, xyz_columns)

    columns_to_fit = feature_columns if norm_cfg.get("scale_xyz", True) else non_xyz_columns
    if not norm_cfg.get("scale_xyz", True) and xyz_columns:
        logger.warning(
            "normalization.scale_xyz is False - x,y,z will NOT be scaled. "
            "This reproduces the legacy bug (ISSUES.md N-1/N-2) unless you "
            "have a specific, deliberate reason. Default should be True."
        )

    method = norm_cfg.get("method", "minmax")
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    scaler.fit(df[columns_to_fit].values)

    return FittedScaler(
        scaler=scaler,
        columns=columns_to_fit,
        local_centered_xyz=norm_cfg.get("local_center_xyz", True) and bool(xyz_columns),
        xyz_columns=xyz_columns,
        centroids=centroids,
    )


def apply_scaler(df: pd.DataFrame, fitted: FittedScaler, feature_columns: List[str]) -> pd.DataFrame:
    """Transform-only - safe to call on train, test, or any future unseen data."""
    df = df.copy()
    if fitted.local_centered_xyz:
        df = _apply_known_centroids(df, fitted.xyz_columns, fitted.centroids)

    df[fitted.columns] = fitted.scaler.transform(df[fitted.columns].values)
    missing_from_scaling = [c for c in feature_columns if c not in fitted.columns]
    if missing_from_scaling:
        logger.warning(
            f"Columns left unscaled: {missing_from_scaling}. Confirm this is "
            f"intentional (see ISSUES.md N-1/N-2 - this was an accidental "
            f"bug in the legacy pipeline, not a deliberate choice)."
        )
    return df
