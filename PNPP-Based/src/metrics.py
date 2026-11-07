import json
import os

import numpy as np
from scipy.spatial import cKDTree

from src import paths


def computeMetrics(pred, label, numClasses, excludeIdx=()):
    pred, label = np.asarray(pred), np.asarray(label)
    cm = np.bincount(label * numClasses + pred, minlength=numClasses * numClasses).reshape(numClasses, numClasses)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    present  = cm.sum(axis=1) > 0
    denom    = tp + fp + fn
    iou      = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    iou[~present] = np.nan

    excluded  = np.zeros(numClasses, dtype=bool)
    excluded[list(excludeIdx)] = True
    averaged  = present & ~excluded
    keep      = ~np.isin(label, list(excludeIdx))

    return {
        "OA_all"          : float((pred == label).mean()),
        "OA_nonExcluded"  : float((pred[keep] == label[keep]).mean()) if keep.any() else float("nan"),
        "mIoU"            : float(np.nanmean(iou[averaged])) if averaged.any() else float("nan"),
        "classIoU"        : [None if np.isnan(v) else float(v) for v in iou],
        "classesAveraged" : np.flatnonzero(averaged).tolist(),
        "classesAbsent"   : np.flatnonzero(~present).tolist(),
        "classesExcluded" : list(map(int, excludeIdx)),
        "confusion"       : cm.tolist(),
    }


def knnIndices(cfg, sourceFile, xyz, k, chunk=1_000_000):
    '''K nearest neighbours (including the point itself), cached as int32 .npy on disk.'''
    path = paths.knnCache(cfg, sourceFile, k)
    if os.path.exists(path):
        nbr = np.load(path, mmap_mode="r")
        if nbr.shape == (len(xyz), k):
            return nbr
    paths.requireWritableCache(cfg, f"PAvPU neighbour cache for {sourceFile}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tree = cKDTree(xyz)
    tmp  = path + ".tmp.npy"
    nbr  = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int32, shape=(len(xyz), k))
    for s in range(0, len(xyz), chunk):
        _, idx = tree.query(xyz[s:s + chunk], k=k, workers=-1)
        nbr[s:s + chunk] = idx.reshape(-1, k)
    nbr.flush()
    del nbr
    os.replace(tmp, path)
    return np.load(path, mmap_mode="r")


def _rowMean(values, nbr, chunk=1_000_000):
    out = np.empty(len(nbr), dtype=np.float32)
    for s in range(0, len(nbr), chunk):
        out[s:s + chunk] = values[np.asarray(nbr[s:s + chunk])].mean(axis=1)
    return out


def patchUncertainty(cfg, testArrays, k):
    path = os.path.join(paths.cacheDir(cfg), "pavpu", f"patch_unc_k{k}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {f: z[f] for f in testArrays}, float(z["__threshold__"])
    per = {}
    for f, a in testArrays.items():
        nbr    = knnIndices(cfg, f, a["xyz"], k)
        per[f] = _rowMean(a["u"], nbr)
    threshold = float(np.median(np.concatenate(list(per.values()))))
    paths.requireWritableCache(cfg, "PAvPU patch uncertainty cache")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, __threshold__=threshold, **per)
    return per, threshold


def computePavpu(cfg, testArrays, predByFile):
    k      = int(cfg["evaluation"]["pavpuK"])
    accThr = float(cfg["evaluation"]["accuracyThreshold"])
    patchUnc, threshold = patchUncertainty(cfg, testArrays, k)

    nAC = nAU = nIC = nIU = 0
    for f, a in testArrays.items():
        nbr      = knnIndices(cfg, f, a["xyz"], k)
        correct  = (predByFile[f] == a["label"]).astype(np.float32)
        patchAcc = _rowMean(correct, nbr)
        accurate = patchAcc >= accThr
        certain  = patchUnc[f] <= threshold
        nAC += int(np.sum(accurate & certain))
        nAU += int(np.sum(accurate & ~certain))
        nIC += int(np.sum(~accurate & certain))
        nIU += int(np.sum(~accurate & ~certain))

    total = nAC + nAU + nIC + nIU
    return {
        "PAvPU"                : (nAC + nIU) / total,
        "AC_pct"               : 100.0 * nAC / total,
        "AU_pct"               : 100.0 * nAU / total,
        "IC_pct"               : 100.0 * nIC / total,
        "IU_pct"               : 100.0 * nIU / total,
        "P_uncertain_given_wrong"   : nIU / max(nIC + nIU, 1),
        "P_correct_given_certain"   : nAC / max(nAC + nIC, 1),
        "uncertaintyThreshold" : threshold,
        "k"                    : k,
        "accuracyThreshold"    : accThr,
        "nPatches"             : total,
    }


def saveJson(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
