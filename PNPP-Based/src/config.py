'''
Config + overrides (from the corresponding notebooks)
'''

import copy
import math
import os

import yaml

CONFIG_DIR = os.environ.get(
    "PNPP_CONFIG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"),
)
VARIANTS = ("baseline", "uncertainty")


def _deepUpdate(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deepUpdate(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def applyOverride(cfg, expr):
    if "=" not in expr:
        raise ValueError(f"Override must look like a.b.c=value, got: {expr}")
    keyPath, rawValue = expr.split("=", 1)
    node, parts = cfg, keyPath.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"Override path not found in config: {keyPath}")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"Override key not found in config: {keyPath}")
    node[parts[-1]] = yaml.safe_load(rawValue)


def loadConfig(dataset, overrides=None, configDir=None):
    configDir = configDir or CONFIG_DIR
    cfg = {}
    for path in (os.path.join(configDir, "base.yaml"), os.path.join(configDir, "datasets", f"{dataset}.yaml")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing config file: {path}")
        with open(path) as f:
            cfg = _deepUpdate(cfg, yaml.safe_load(f) or {})
    cfg["dataset"]["name"] = dataset
    for expr in overrides or []:
        applyOverride(cfg, expr)
    validate(cfg)
    return cfg


def validate(cfg):
    b, m, t = cfg["blocks"], cfg["model"], cfg["training"]
    if b["cellsPerSide"] % b["inferStrideCells"] != 0:
        raise ValueError("blocks.cellsPerSide must be a multiple of blocks.inferStrideCells")
    if t["classWeights"] not in ("none", "inverse", "inverse_sqrt"):
        raise ValueError("training.classWeights must be none | inverse | inverse_sqrt")
    if t.get("lrSchedule", "cosine") not in ("cosine", "exponential"):
        raise ValueError("training.lrSchedule must be cosine | exponential")
    if cfg["features"]["zMode"] not in ("file", "block_min"):
        raise ValueError("features.zMode must be file | block_min")
    nLevels = len(m["saRatios"])
    for key in ("saRadii", "saNsample", "saMlps", "fpMlps"):
        if len(m[key]) != nLevels:
            raise ValueError(f"model.{key} needs {nLevels} entries, one per set abstraction level")
    if any(float(r) <= 0 for r in m["saRadii"]) or any(not 0 < float(r) <= 1 for r in m["saRatios"]):
        raise ValueError("model.saRadii must be > 0 and model.saRatios in (0, 1]")
    n = float(b["numPoints"])
    for r in m["saRatios"]:
        n = math.ceil(n * float(r))
    if n < 3:                                      # FP interpolates from 3 neighbours on the coarsest level
        raise ValueError(f"the coarsest level keeps {n} points per block; feature propagation needs at least 3")


def checkVariant(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
    return variant == "uncertainty"
