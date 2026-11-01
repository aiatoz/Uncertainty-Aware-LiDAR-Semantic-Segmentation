"""
Shared utilities: reproducible seeding, config loading, logging setup
"""
import logging
import random
from pathlib import Path

import numpy as np
import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str = "default") -> dict:
    path = _CONFIG_DIR / f"{name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int = 42) -> None:
    """
    Empirically determined between 42,43,44
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def configure_gpu_memory_growth() -> None:
    """
    This was a crucial step in the optmiziation part

    Relevant on WSL2 in particular: WSL2's GPU passthrough (vGPU) adds an
    extra virtualization layer between the CUDA driver and the physical
    card, so the usual "grab everything at process start" behaviour can
    interact badly with anything else touching the GPU (display, other
    processes) and has been observed to manifest as the process hanging
    rather than raising a clean OOM.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        get_logger(__name__).info("No GPU visible to TensorFlow - skipping memory_growth config.")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Must be called before GPUs are initialized - if something
            # already touched the GPU this call raises. Log and continue
            # rather than crash the notebook.
            get_logger(__name__).warning(
                f"Could not set memory_growth on {gpu}: {e}. "
                f"This only works if called before any other GPU op - "
                f"make sure configure_gpu_memory_growth() is the first "
                f"TF-touching call in the notebook."
            )
    get_logger(__name__).info(f"memory_growth enabled on {len(gpus)} GPU(s).")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
