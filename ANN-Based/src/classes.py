from dataclasses import dataclass
from typing import Dict

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.paths import get_class_map  # noqa: E402


@dataclass
class ClassRegistry:
    dataset: str
    raw_to_name: Dict[int, str]          # {0: "unclassified", 1: "ground", ...}
    raw_to_index: Dict[int, int]
    index_to_raw: Dict[int, int]
    num_classes: int

    def map_labels(self, raw_labels: np.ndarray) -> np.ndarray:
        """Map raw label ids num_classes-1 indices expected by keras' softmax output / one-hot."""
        raw_labels = np.asarray(raw_labels)
        unmapped = set(np.unique(raw_labels)) - set(self.raw_to_index.keys())
        if unmapped:
            raise ValueError(
                f"Found label id(s) {unmapped} in data that are not declared "
                f"in configs/datasets.yaml under datasets.{self.dataset}.classes. "
                f"Either the config is incomplete or upstream parsing is corrupted "
                f"(see ISSUES.md F-1, F-8)."
            )
        return np.vectorize(self.raw_to_index.get)(raw_labels)

    def class_names_in_order(self):
        return [self.raw_to_name[self.index_to_raw[i]] for i in range(self.num_classes)]


def get_class_registry(dataset: str) -> ClassRegistry:
    raw_to_name = get_class_map(dataset)
    if not raw_to_name:
        raise ValueError(
            f"No classes declared for dataset '{dataset}' in configs/datasets.yaml. "
            f"Fill in the `classes:` section before running this pipeline."
        )
    sorted_raw_ids = sorted(raw_to_name.keys())
    raw_to_index = {raw_id: i for i, raw_id in enumerate(sorted_raw_ids)}
    index_to_raw = {i: raw_id for raw_id, i in raw_to_index.items()}
    return ClassRegistry(
        dataset=dataset,
        raw_to_name={int(k): v for k, v in raw_to_name.items()},
        raw_to_index=raw_to_index,
        index_to_raw=index_to_raw,
        num_classes=len(sorted_raw_ids),
    )
