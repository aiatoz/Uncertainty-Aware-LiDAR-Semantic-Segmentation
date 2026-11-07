'''
Builds the per file cache that training and evaluation read
'''

import json
import os

import numpy as np
import pandas as pd

from src import paths

# Maximum allowed deviation from an exact affine map, in scaled units. The split CSVs are written as text, so a small rounding residual is expected
AFFINE_TOL = 1e-4


def _normRaw(series):
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().all() and np.all(np.mod(num.to_numpy(), 1) == 0):
        return num.astype(np.int64).astype(str)
    return series.astype(str)


def _readCsv(path, cols):
    try:
        return pd.read_csv(path, usecols=cols, engine="pyarrow")
    except (ImportError, ValueError):
        return pd.read_csv(path, usecols=cols)


def _loadTrueXyz(cfg, sourceFile):
    path = paths.featuresCsv(cfg, sourceFile)
    if not os.path.exists(path):
        raise FileNotFoundError(f"True-coordinate file not found: {path}")
    df = _readCsv(path, ["x", "y", "z", "__nan_feature__"])
    flag = df["__nan_feature__"]
    if flag.dtype != bool:
        flag = flag.astype(str).str.strip().str.lower().eq("true")
    df = df.loc[~flag.values]
    return df[["x", "y", "z"]].to_numpy(dtype=np.float64)


def checkAffineAlignment(scaled, true, label):
    '''
    Raises if `scaled` is not an affine function of `true` on each axis. A shuffled or offset row order breaks this immediately.
    '''
    for axis, name in enumerate("xyz"):
        t = true[:, axis] - true[:, axis].mean()
        s = scaled[:, axis].astype(np.float64)
        if np.ptp(t) == 0:
            if np.ptp(s) > AFFINE_TOL:
                raise ValueError(f"[{label}] axis {name}: true coords constant but scaled coords vary")
            continue
        A = np.stack([t, np.ones_like(t)], axis=1)
        coef, *_ = np.linalg.lstsq(A, s, rcond=None)
        resid = np.abs(A @ coef - s).max()
        if resid > AFFINE_TOL * max(1.0, np.ptp(s)):
            raise ValueError(
                f"[{label}] axis {name} : split CSV coordinates are not an affine function of features CSV coordinates (max residual {resid:.3g}). Row order is NOT aligned."
            )


def _labelColumns(cfg, header):
    c = cfg["columns"]
    if c["classIdx"] in header:
        return c["classIdx"], True
    print(f"[prepare] WARNING: '{c['classIdx']}' not in split CSV, deriving indices from '{c['rawLabel']}'")
    return c["rawLabel"], False


def _checkFileList(cfg, split, found):
    key      = "expectedTrainFiles" if split == "train" else "expectedTestFiles"
    expected = cfg["dataset"].get(key)
    if expected is None:
        raise ValueError(
            f"[prepare] dataset.{key} is not set. Files found in {split}.csv: {sorted(found)}. "
            f"Copy the canonical list into configs/datasets/{cfg['dataset']['name']}.yaml."
        )
    if sorted(map(str, expected)) != sorted(found):
        raise ValueError(f"[prepare] {split} files {sorted(found)} != expected {sorted(map(str, expected))}")


def _loadSplit(cfg, split):
    c      = cfg["columns"]
    path   = paths.splitCsv(cfg, split)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    labelCol, hasIdx = _labelColumns(cfg, header)
    cols = [c["xScaled"], c["yScaled"], c["zScaled"], c["uncertainty"], c["rawLabel"], c["sourceFile"]]
    if hasIdx:
        cols.append(labelCol)
    print(f"[prepare] reading {path}")
    df = _readCsv(path, cols)
    df[c["sourceFile"]] = df[c["sourceFile"]].astype(str)
    df[c["rawLabel"]]   = _normRaw(df[c["rawLabel"]])
    return df, labelCol, hasIdx


def _buildLabelMap(trainDf, testDf, cfg, labelCol, hasIdx):
    raw = cfg["columns"]["rawLabel"]
    if hasIdx:
        pairs = pd.concat([trainDf[[raw, labelCol]], testDf[[raw, labelCol]]]).drop_duplicates()
        if pairs[raw].duplicated().any() or pairs[labelCol].duplicated().any():
            raise ValueError(f"[prepare] raw label <-> class index mapping is not one-to-one:\n{pairs}")
        labelMap = {str(r): int(i) for r, i in zip(pairs[raw], pairs[labelCol])}
    else:
        uniq     = sorted(set(trainDf[raw]).union(testDf[raw]))
        labelMap = {str(r): i for i, r in enumerate(uniq)}
    numClasses = max(labelMap.values()) + 1
    if numClasses != len(labelMap):
        print(f"[prepare] NOTE: class indices are not contiguous; numClasses={numClasses}, used indices={sorted(labelMap.values())}")
    return labelMap, numClasses


def _writeSplit(cfg, split, df, labelMap):
    c        = cfg["columns"]
    srcCol   = c["sourceFile"]
    rawCol   = c["rawLabel"]
    files    = sorted(df[srcCol].unique())
    _checkFileList(cfg, split, files)

    fileInfo = {}
    srcVals  = df[srcCol].to_numpy()
    for f in files:
        rowIndex = np.flatnonzero(srcVals == f)
        sub      = df.iloc[rowIndex]
        trueXyz  = _loadTrueXyz(cfg, f)
        if len(trueXyz) != len(rowIndex):
            raise ValueError(f"[{split}/{f}] features rows ({len(trueXyz):,}) != split rows ({len(rowIndex):,})")

        scaled = sub[[c["xScaled"], c["yScaled"], c["zScaled"]]].to_numpy()
        checkAffineAlignment(scaled, trueXyz, f"{split}/{f}")

        xyz = (trueXyz - trueXyz.mean(axis=0)).astype(np.float32) # float64 re-centring, then cast
        u   = sub[c["uncertainty"]].to_numpy(dtype=np.float32)
        if not np.isfinite(u).all():
            raise ValueError(f"[{split}/{f}] non-finite uncertainty values")
        rawLabel = sub[rawCol].astype(str).to_numpy()
        label    = np.array([labelMap[r] for r in rawLabel], dtype=np.int64)

        np.savez(paths.cachedFile(cfg, split, f), xyz=xyz, u=u, label=label, rowIndex=rowIndex.astype(np.int64))
        extent = np.ptp(xyz, axis=0)
        fileInfo[f] = {
            "n"        : int(len(rowIndex)),
            "extent_m" : [round(float(e), 2) for e in extent],
            "uMax"     : float(u.max()),
        }
        print(f"[prepare] {split}/{f}: {len(rowIndex):,} pts, extent {extent.round(1)} m, "
              f"u in [{u.min():.4f}, {u.max():.4f}]  (alignment verified)")
    return files, fileInfo


def prepareDataset(cfg):
    paths.requireWritableCache(cfg, "prepareDataset")
    trainDf, labelCol, hasIdx = _loadSplit(cfg, "train")
    testDf,  _, _             = _loadSplit(cfg, "test")

    overlap = set(trainDf[cfg["columns"]["sourceFile"]]) & set(testDf[cfg["columns"]["sourceFile"]])
    if overlap:
        raise ValueError(f"[prepare] source files appear in both train and test: {sorted(overlap)}")

    labelMap, numClasses = _buildLabelMap(trainDf, testDf, cfg, labelCol, hasIdx)
    trainFiles, trainInfo = _writeSplit(cfg, "train", trainDf, labelMap)
    testFiles,  testInfo  = _writeSplit(cfg, "test",  testDf,  labelMap)

    raw = cfg["columns"]["rawLabel"]
    def counts(df):
        vc = df[raw].astype(str).value_counts()
        return {str(k): int(v) for k, v in vc.items()}

    manifest = {
        "dataset"      : cfg["dataset"]["name"],
        "labelMap"     : labelMap, # raw label (str) -> class index
        "numClasses"   : int(numClasses),
        "trainFiles"   : trainFiles,
        "testFiles"    : testFiles,
        "trainInfo"    : trainInfo,
        "testInfo"     : testInfo,
        "trainCounts"  : counts(trainDf),
        "testCounts"   : counts(testDf),
        "nTestRows"    : int(len(testDf)),
        "nTrainRows"   : int(len(trainDf)),
        "labelSource"  : labelCol,
    }
    with open(paths.manifestPath(cfg), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepare] manifest -> {paths.manifestPath(cfg)}")
    print(f"[prepare] classes  : {labelMap}")
    return manifest


def loadManifest(cfg):
    p = paths.manifestPath(cfg)
    if not os.path.exists(p):
        raise FileNotFoundError(f"No cache for {cfg['dataset']['name']}: run notebooks/00_prepare.ipynb first")
    with open(p) as f:
        return json.load(f)


def loadSplitArrays(cfg, split, manifest=None):
    '''Returns {sourceFile: dict(xyz, u, label, rowIndex)} for one split'''
    manifest = manifest or loadManifest(cfg)
    files    = manifest["trainFiles"] if split == "train" else manifest["testFiles"]
    out = {}
    for f in files:
        with np.load(paths.cachedFile(cfg, split, f)) as z:
            out[f] = {k: z[k] for k in ("xyz", "u", "label", "rowIndex")}
    return out


def rawToIdx(manifest, rawLabels, what="labels"):
    '''Maps configured raw labels to class indices. Warns about labels absent from the data.'''
    m   = manifest["labelMap"]
    out = set()
    for r in rawLabels:
        key = _normRaw(pd.Series([r])).iloc[0]
        if key in m:
            out.add(m[key])
        else:
            print(f"[labels] WARNING: {what}: raw label {r!r} does not occur in this dataset "
                  f"(known: {sorted(m)})")
    return sorted(out)
