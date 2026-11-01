"""
File-based (canonical) train/test splitting 

Canonical splits are declared once, per dataset, in configs/datasets.yaml
under `datasets.<name>.split`, e.g. for paris_lille:
    train_files: [Lille1_1, Lille1_2, Lille2]
    test_files:  [Paris]
"""
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.paths import get_split_config, get_processed_file  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_split(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and concatenates the per-file processed (feature-extracted) CSVs into a single train DataFrame and a single test DataFrame, according to the canonical file-based split declared in configs/datasets.yaml
    """
    split_cfg = get_split_config(dataset)
    if split_cfg["strategy"] != "file_based":
        raise ValueError(
            f"Dataset '{dataset}' is configured with split strategy "
            f"'{split_cfg['strategy']}'. Random splitting is legacy behavior and is NOT recommended for spatial data"
            f"Use random_split() explicitly if you really need it for a controlled experiment/ablation."
        )

    train_df = _load_files(dataset, split_cfg["train_files"])
    test_df = _load_files(dataset, split_cfg["test_files"])

    logger.info(
        f"[{dataset}] file-based split: train={split_cfg['train_files']} "
        f"({len(train_df):,} points), test={split_cfg['test_files']} "
        f"({len(test_df):,} points)"
    )
    return train_df, test_df


def _load_files(dataset: str, file_names: list) -> pd.DataFrame:
    frames = []
    for fname in file_names:
        path = get_processed_file(dataset, fname)
        if not path.exists():
            raise FileNotFoundError(
                f"Expected processed feature file at {path} but it doesn't "
                f"exist. Run the feature-extraction notebook for '{fname}' "
                f"first (see notebooks/01_feature_extraction.ipynb)."
            )
        df = pd.read_csv(path)
        df["__source_file__"] = fname
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def random_split(
    dataset: str,
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legacy-style random row-wise split. NOT recommended
    """
    from sklearn.model_selection import train_test_split

    logger.warning(
        f"[{dataset}] Using random_split() — this leaks spatial "
        f"autocorrelation between train/test. Only use this for a deliberate, labeled ablation, never for headline results."
    )
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
