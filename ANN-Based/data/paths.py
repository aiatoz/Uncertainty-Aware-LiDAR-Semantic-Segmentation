import os
from pathlib import Path
import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_DATASETS_YAML = _CONFIG_DIR / "datasets.yaml"


def _load_registry() -> dict:
    with open(_DATASETS_YAML, "r") as f:
        return yaml.safe_load(f)


_REGISTRY = _load_registry()


def get_base_dir() -> Path:
    root = os.environ.get("UNCERTAINTY_DATA_ROOT", _REGISTRY["base_dir"])
    return Path(root)


def get_dataset_config(name: str) -> dict:
    try:
        return _REGISTRY["datasets"][name]
    except KeyError as e:
        available = list(_REGISTRY["datasets"].keys())
        raise KeyError(
            f"Unknown dataset '{name}'. Available datasets: {available}. "
            f"Add a new entry to configs/datasets.yaml if needed."
        ) from e


def get_raw_dir(dataset: str) -> Path:
    cfg = get_dataset_config(dataset)
    return get_base_dir() / cfg["raw_dir"]


def get_processed_dir(dataset: str) -> Path:
    cfg = get_dataset_config(dataset)
    path = get_base_dir() / cfg["processed_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_processed_file(dataset: str, file_name: str) -> Path:
    return get_processed_dir(dataset) / f"features_{file_name}.csv"


def get_raw_file(dataset: str, file_name: str, ext: str = None) -> Path:
    if ext is None:
        ext = get_dataset_config(dataset).get("raw_extension", ".csv")
    return get_raw_dir(dataset) / f"{file_name}{ext}"


def get_class_map(dataset: str) -> dict:
    """Returns {raw_label_id: class_name} as declared in datasets.yaml."""
    return get_dataset_config(dataset).get("classes", {})


def get_split_config(dataset: str) -> dict:
    return get_dataset_config(dataset)["split"]


def get_checkpoints_dir(dataset: str, experiment: str) -> Path:
    base = Path(__file__).resolve().parent.parent / "checkpoints" / dataset / experiment
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_results_dir(dataset: str, experiment: str) -> Path:
    base = Path(__file__).resolve().parent.parent / "results" / dataset / experiment
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_split_file(dataset: str, split_name: str) -> Path:
    split_dir = get_processed_dir(dataset) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir / f"{split_name}.csv"

def get_feature_extraction_params(dataset: str) -> dict:
    cfg = get_dataset_config(dataset)
    defaults = _REGISTRY.get("feature_extraction", {})
    return {
        "radius": cfg.get("feature_radius", defaults.get("default_radius", 0.25)),
        "num_threads": defaults.get("num_threads", -1),
    }
