"""
Raw point cloud I/O
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.paths import get_dataset_config, get_raw_file
from src.utils import get_logger

logger = get_logger(__name__)


def load_raw_point_file(dataset: str, file_name: str) -> pd.DataFrame:
    """
    Parses a raw point cloud file into a DataFrame.

    Format is driven by datasets.yaml:
      raw_extension   : file extension,  e.g. ".csv"  (default)
      raw_has_header  : true if the file has a header row (default: true)
      raw_columns     : expected column names IN ORDER
      drop_columns    : columns to discard after parsing

    Data rows are always read as whitespace separated. If the file has a header row it is sniffed independently (handles CloudCompare's common pattern of a comma-separated header with space-separated data rows).
    """
    cfg = get_dataset_config(dataset)
    raw_columns  = cfg["raw_columns"]
    drop_columns = cfg.get("drop_columns", [])
    extension    = cfg.get("raw_extension", ".csv")
    has_header   = cfg.get("raw_has_header", True)

    path = get_raw_file(dataset, file_name, ext=extension)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"Check raw_dir and raw_extension in configs/datasets.yaml."
        )

    if has_header:
        with open(path, "r") as f:
            first_line = f.readline().strip()

        if "," in first_line:
            file_columns = [c.strip() for c in first_line.split(",")]
        else:
            file_columns = first_line.split()

        #Read data rows (skip the header) with whitespace separator
        raw = pd.read_csv(
            path, sep=r"\s+", header=None, skiprows=1,
            names=file_columns, engine="python",
        )

        # Validate that all expected columns are present
        missing = [c for c in raw_columns if c not in raw.columns]
        if missing:
            raise ValueError(
                f"[{dataset}/{file_name}] Expected columns {missing} not "
                f"found in file header.\n"
                f"Header parsed as: {file_columns}\n"
                f"raw_columns in datasets.yaml: {raw_columns}\n"
                f"Either update raw_columns to match the actual header, "
                f"or rename columns in the CSV. See ISSUES.md F-1."
            )
        #keep only declared columns in declared order
        raw = raw[raw_columns]

    else:
        raw = pd.read_csv(
            path, sep=r"\s+", header=None, engine="python",
        )
        if raw.shape[1] != len(raw_columns):
            raise ValueError(
                f"[{dataset}/{file_name}] Parsed {raw.shape[1]} columns "
                f"but raw_columns declares {len(raw_columns)}: {raw_columns}.\n"
                f"If the file has more columns than x/y/z/label (e.g. cl/cs/cp "
                f"from CloudCompare), add them to raw_columns then list the "
                f"extras in drop_columns. See ISSUES.md F-1."
            )
        raw.columns = raw_columns

    if drop_columns:
        raw = raw.drop(columns=[c for c in drop_columns if c in raw.columns])
        logger.info(f"[{dataset}/{file_name}] dropped columns: {drop_columns}")

    raw = raw.reset_index(drop=True)

    n_dupes = raw.duplicated(subset=["x", "y", "z"]).sum()
    if n_dupes > 0:
        logger.warning(
            f"[{dataset}/{file_name}] {n_dupes:,} duplicate (x,y,z) points "
            f"({100 * n_dupes / len(raw):.2f}% of {len(raw):,}). Can cause "
            f"degenerate covariance neighborhoods (ISSUES.md M-2)."
        )

    if "label" in raw.columns:
        n_nan_labels = int(raw["label"].isna().sum())
        if n_nan_labels > 0:
            raise ValueError(
                f"[{dataset}/{file_name}] {n_nan_labels:,} NaN values in "
                f"the 'label' column after parsing. This usually means the "
                f"data separator doesn't match: the file header says the "
                f"separator is one thing, but the data rows use another "
                f"(e.g. comma header with space-separated values). "
                f"Check that your CloudCompare export and the raw_has_header "
                f"setting in datasets.yaml match the actual file format."
            )

        declared_labels = set(cfg.get("classes", {}).keys())
        actual_labels   = set(int(v) for v in raw["label"].unique())
        unknown = actual_labels - declared_labels
        if declared_labels and unknown:
            raise ValueError(
                f"[{dataset}/{file_name}] Label id(s) {unknown} not declared "
                f"in datasets.yaml `classes` map "
                f"(declared: {sorted(declared_labels)}). Either the class map "
                f"is incomplete, or column parsing is misaligned (F-1/F-8)."
            )

    logger.info(
        f"[{dataset}/{file_name}] loaded {len(raw):,} points, "
        f"columns: {list(raw.columns)}"
    )
    return raw


def save_processed_features(df: pd.DataFrame, dataset: str, file_name: str) -> Path:
    """Saves the feature augmented DataFrame to the canonical path"""
    from data.paths import get_processed_file

    out_path = get_processed_file(dataset, file_name)
    df.to_csv(out_path, index=False)
    logger.info(f"[{dataset}/{file_name}] saved {len(df):,} rows -> {out_path}")
    return out_path
