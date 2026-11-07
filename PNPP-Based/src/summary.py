'''
This is script mainly made for the paper summaryt
'''

import glob
import json
import os

import pandas as pd

from src import paths
from src.config import loadConfig

PAVPU_KEYS = ("PAvPU", "AC_pct", "AU_pct", "IC_pct", "IU_pct",
              "P_uncertain_given_wrong", "P_correct_given_certain", "uncertaintyThreshold")


def collectRuns(datasets):
    rows, classRows = [], []
    for ds in datasets:
        cfg = loadConfig(ds)
        for path in sorted(glob.glob(os.path.join(paths.datasetOutDir(cfg), "*", "seed*", "metrics.json"))):
            runDir = os.path.dirname(path)
            with open(path) as f:
                m = json.load(f)
            pav = {}
            if os.path.exists(os.path.join(runDir, "pavpu.json")):
                with open(os.path.join(runDir, "pavpu.json")) as f:
                    pav = json.load(f)
            rows.append({"dataset": ds, "variant": m["variant"], "seed": m["seed"], "epoch": m["epoch"],
                         "OA_all": m["OA_all"], "OA_nonExcluded": m["OA_nonExcluded"], "mIoU": m["mIoU"],
                         **{k: pav.get(k) for k in PAVPU_KEYS}})
            cPath = os.path.join(runDir, "per_class_iou.csv")
            if os.path.exists(cPath):
                c = pd.read_csv(cPath, dtype={"raw_label": str}).dropna(subset=["raw_label"])
                c["dataset"], c["variant"], c["seed"] = ds, m["variant"], m["seed"]
                classRows.append(c)
    runs    = pd.DataFrame(rows)
    classes = pd.concat(classRows, ignore_index=True) if classRows else pd.DataFrame()
    if not runs.empty:
        runs = runs.sort_values(["dataset", "variant", "seed"]).reset_index(drop=True)
    return runs, classes


def aggregate(runs):
    return (runs.groupby(["dataset", "variant"])[["OA_all", "mIoU", "PAvPU"]]
                .agg(["mean", "std", "count"]).round(4))


def perSeedEffect(runs):
    '''Uncertainty minus baseline, paired by seed'''
    piv = runs.pivot_table(index=["dataset", "seed"], columns="variant", values=["mIoU", "PAvPU"])
    if ("mIoU", "uncertainty") not in piv.columns or ("mIoU", "baseline") not in piv.columns:
        return pd.DataFrame()
    return pd.DataFrame({"d_mIoU" : piv[("mIoU", "uncertainty")]  - piv[("mIoU", "baseline")],
                         "d_PAvPU": piv[("PAvPU", "uncertainty")] - piv[("PAvPU", "baseline")]}).dropna()


def pavpuTable(runs):
    return runs.groupby(["dataset", "variant"])[list(PAVPU_KEYS)].mean().round(4)


def aggregateClasses(classes):
    if classes.empty:
        return classes
    return (classes.groupby(["dataset", "raw_label", "variant"])["iou"]
                   .agg(["mean", "std"]).round(4).unstack("variant"))


def thresholdWarnings(runs):
    '''The PAvPU threshold must be identical for every run of a dataset'''
    out = []
    for ds, g in runs.groupby("dataset"):
        vals = g["uncertaintyThreshold"].dropna().unique()
        if len(vals) > 1:
            out.append(f"{ds}: differing PAvPU thresholds across runs : {vals}")
    return out
