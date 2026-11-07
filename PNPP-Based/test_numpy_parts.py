'''
Tests everything that does not need a GPU
'''

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.blocks import TrainBlockSampler, iterInferenceChunks
from src.config import loadConfig
from src.features import buildBlockInput
from src.metrics import computeMetrics, computePavpu
from src.prepare import loadSplitArrays, prepareDataset, rawToIdx 

RAW_TO_IDX = {0: 0, 2: 1, 5: 2, 9: 3} #class 9 appears only in train

#temporary test data lives inside the project folder (deleted afterwards)
TMP_ROOT = os.path.join(ROOT, ".tmp_tests")
REAL_CONFIGS = os.path.join(ROOT, "configs")

def localTmp():
    os.makedirs(TMP_ROOT, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TMP_ROOT)


def makeFile(rng, n, offset, withClass9):
    xy = rng.uniform([0, 0], [60, 40], size=(n, 2)) + offset
    z  = 0.05 * xy[:, 0] + rng.normal(0, 0.2, n)
    raw = np.where(xy[:, 1] - offset[1] < 15, 2, 5)
    raw[rng.random(n) < 0.05] = 0
    if withClass9:
        raw[rng.random(n) < 0.02] = 9
    nanFlag = rng.random(n) < 0.1
    return np.column_stack([xy, z]), raw, nanFlag


def buildSyntheticTree(tmp, shuffleTestRows=False):
    rng     = np.random.default_rng(1)
    annRoot = os.path.join(tmp, "ann")
    folder  = os.path.join(annRoot, "synth")
    os.makedirs(os.path.join(folder, "splits"))
    files = {"A": (3000, [5e5, 5.4e6], True), "B": (2500, [5e5 + 300, 5.4e6], True), "T": (2800, [5e5 + 900, 5.4e6], False)}

    kept = {}
    for name, (n, off, c9) in files.items():
        xyz, raw, nan = makeFile(rng, n, np.array(off, dtype=np.float64), c9)
        pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                      "__nan_feature__": np.where(nan, "True", "False")}).to_csv(
            os.path.join(folder, f"features_{name}.csv"), index=False)
        keep = ~nan
        centred = xyz[keep] - xyz[keep].mean(axis=0)
        kept[name] = (centred, raw[keep])

    trainCentred = np.concatenate([kept["A"][0], kept["B"][0]])
    lo, hi = trainCentred.min(axis=0), trainCentred.max(axis=0)#scaler

    def splitDf(names):
        parts = []
        for nm in names:
            c, raw = kept[nm]
            scaled = (c - lo) / (hi - lo)
            parts.append(pd.DataFrame({
                "x": np.round(scaled[:, 0], 7), "y": np.round(scaled[:, 1], 7), "z": np.round(scaled[:, 2], 7),
                "label": raw, "class_idx": [RAW_TO_IDX[r] for r in raw],
                "__u_norm__": rng.uniform(0, 1.3, len(raw)).astype(np.float32),
                "__source_file__": nm,
            }))
        return pd.concat(parts, ignore_index=True)

    splitDf(["A", "B"]).to_csv(os.path.join(folder, "splits", "train.csv"), index=False)
    test = splitDf(["T"])
    if shuffleTestRows:
        test = test.sample(frac=1.0, random_state=0).reset_index(drop=True)
    test.to_csv(os.path.join(folder, "splits", "test.csv"), index=False)

    cfgDir = os.path.join(tmp, "configs")
    os.makedirs(os.path.join(cfgDir, "datasets"))
    shutil.copy(os.path.join(ROOT, "configs", "base.yaml"), os.path.join(cfgDir, "base.yaml"))
    with open(os.path.join(cfgDir, "datasets", "synth.yaml"), "w") as f:
        yaml.safe_dump({
            "dataset": {"annFolder": "synth", "expectedTrainFiles": ["A", "B"], "expectedTestFiles": ["T"]},
            "blocks": {"blockSize": 12.0, "numPoints": 1024, "minTrainPoints": 32},
            "training": {"blocksPerEpoch": 10},
            "model": {"saRatios": [0.125, 0.25, 0.25, 0.5]}, # 1024 points : keep >= 3 on the coarsest level
            "labels": {"metricExcludeRaw": [0]},
        }, f)
    cfg = loadConfig("synth", [f"paths.annDataRoot={annRoot}", f"paths.cacheRoot={tmp}/cache",
                               "paths.sharedCacheRoot=null",
                               f"paths.outputRoot={tmp}/runs"], configDir=cfgDir)
    return cfg, kept


def testPrepareAndAlignment():
    with localTmp() as tmp:
        cfg, kept = buildSyntheticTree(tmp)
        man = prepareDataset(cfg)
        assert man["labelMap"] == {str(k): v for k, v in RAW_TO_IDX.items()}
        assert man["numClasses"] == 4
        test = loadSplitArrays(cfg, "test", man)["T"]
        assert np.allclose(test["xyz"], kept["T"][0], atol=1e-3) # metres, recentred, unscaled
        assert (test["rowIndex"] == np.arange(len(test["xyz"]))).all()
        assert rawToIdx(man, [0]) == [0]

    with localTmp() as tmp:
        cfg, _ = buildSyntheticTree(tmp, shuffleTestRows=True)
        try:
            prepareDataset(cfg)
        except ValueError as e:
            assert "NOT aligned" in str(e)
        else:
            raise AssertionError("shuffled rows were not detected")


def testWrongFileListRejected():
    with localTmp() as tmp:
        cfg, _ = buildSyntheticTree(tmp)
        cfg["dataset"]["expectedTestFiles"] = ["Paris"]
        try:
            prepareDataset(cfg)
        except ValueError as e:
            assert "expected" in str(e)
        else:
            raise AssertionError("wrong file list accepted")


def testInferenceCoverage():
    rng = np.random.default_rng(3)
    xyz = np.column_stack([rng.uniform(-40, 40, 5000), rng.uniform(-10, 30, 5000), rng.normal(0, 1, 5000)])
    cfg = {"blocks": {"blockSize": 8.0, "numPoints": 128, "cellsPerSide": 4, "inferStrideCells": 2}}
    votes = np.zeros(len(xyz), dtype=int)
    for idx, valid, centre, block in iterInferenceChunks(xyz, cfg, np.random.default_rng(0)):
        assert len(idx) == 128
        v = idx[valid]
        assert len(np.unique(v)) == len(v)
        assert np.isin(idx, block).all()
        half = 4.0 + 1e-9
        assert (np.abs(xyz[block, :2] - centre).max(axis=0) <= half).all()
        votes[v] += 1
    assert (votes == 4).all(), np.bincount(votes)


def testTrainSampler():
    rng = np.random.default_rng(4)
    xyzA = np.column_stack([rng.uniform(0, 50, 4000), rng.uniform(0, 50, 4000), np.zeros(4000)])
    xyzB = np.column_stack([rng.uniform(0, 20, 1000), rng.uniform(0, 20, 1000), np.zeros(1000)])
    cfg = {"blocks": {"blockSize": 10.0, "numPoints": 300, "cellsPerSide": 4,
                      "minTrainPoints": 50, "maxTrainRetries": 20}}
    s = TrainBlockSampler([xyzA, xyzB], cfg)
    picks = []
    for seed in range(400):
        f, sel, block, centre = s.sample(np.random.default_rng(seed))
        pts = [xyzA, xyzB][f]
        assert len(sel) == 300 and np.isin(sel, block).all()
        assert (np.abs(pts[block, :2] - centre).max(axis=0) <= 5.0 + 1e-9).all()
        picks.append(f)
    assert 0.7 < np.mean(np.array(picks) == 0) < 0.9
    a = s.sample(np.random.default_rng([42, 3, 7]))
    b = s.sample(np.random.default_rng([42, 3, 7]))
    assert all(np.array_equal(x, y) for x, y in zip(a[1:3], b[1:3]))

def testFeatures():
    rng = np.random.default_rng(5)
    xyz = rng.normal(0, 5, (500, 3)).astype(np.float32)
    u   = rng.uniform(0, 1.5, 500).astype(np.float32)
    cfg = {"blocks": {"blockSize": 10.0}, "features": {"zMode": "file"}}
    sel, block, centre = np.arange(100), np.arange(500), np.array([1.0, -2.0])
    p0, f0 = buildBlockInput(xyz, u, sel, block, centre, cfg, False, 0.7)
    p1, f1 = buildBlockInput(xyz, u, sel, block, centre, cfg, True, 0.7)
    assert f0.shape == (100, 3) and f1.shape == (100, 4)
    assert np.array_equal(p0, p1) and np.array_equal(f0, f1[:, :3])
    assert np.array_equal(f1[:, 3], u[:100])
    assert np.allclose(np.linalg.norm(p0[:, :2], axis=1),
                       np.linalg.norm(xyz[:100, :2] - centre, axis=1), atol=1e-4)


def testMetrics():
    label = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    pred  = np.array([0, 1, 1, 1, 2, 2, 2, 0, 2])
    m = computeMetrics(pred, label, numClasses=4, excludeIdx=[0])
    iou1, iou2 = 2 / 4, 3 / 5
    assert abs(m["classIoU"][1] - iou1) < 1e-12 and abs(m["classIoU"][2] - iou2) < 1e-12
    assert m["classIoU"][3] is None and m["classesAbsent"] == [3]
    assert abs(m["mIoU"] - (iou1 + iou2) / 2) < 1e-12
    assert abs(m["OA_all"] - 6 / 9) < 1e-12
    assert abs(m["OA_nonExcluded"] - 5 / 7) < 1e-12


def testPavpuMatchesBruteForce():
    with localTmp() as tmp:
        cfg, _ = buildSyntheticTree(tmp)
        man  = prepareDataset(cfg)
        test = loadSplitArrays(cfg, "test", man)
        rng  = np.random.default_rng(9)
        pred = {f: np.where(rng.random(len(a["label"])) < 0.7, a["label"], 0) for f, a in test.items()}
        got  = computePavpu(cfg, test, pred)

        a, k = test["T"], 30
        d    = ((a["xyz"][:, None, :].astype(np.float64) - a["xyz"][None, :, :]) ** 2).sum(-1)
        nbr  = np.argsort(d, axis=1, kind="stable")[:, :k]
        pu   = a["u"][nbr].mean(1)
        pa   = (pred["T"] == a["label"])[nbr].mean(1)
        thr  = np.median(pu)
        acc, cer = pa >= 0.5, pu <= thr
        want = (np.sum(acc & cer) + np.sum(~acc & ~cer)) / len(pu)
        assert abs(got["PAvPU"] - want) < 2e-3, (got["PAvPU"], want)
        again = computePavpu(cfg, test, pred)
        assert again["uncertaintyThreshold"] == got["uncertaintyThreshold"]


def testOracleVotingGivesPerfectScores():
    with localTmp() as tmp:
        cfg, _ = buildSyntheticTree(tmp)
        man  = prepareDataset(cfg)
        a    = loadSplitArrays(cfg, "test", man)["T"]
        C    = man["numClasses"]
        prob = np.zeros((len(a["xyz"]), C), dtype=np.float32)
        for idx, valid, centre, block in iterInferenceChunks(a["xyz"], cfg, np.random.default_rng(0)):
            onehot = np.eye(C, dtype=np.float32)[a["label"][idx]]
            prob[idx[valid]] += onehot[valid]
        pred = prob.argmax(1)
        m = computeMetrics(pred, a["label"], C, rawToIdx(man, [0]))
        assert m["OA_all"] == 1.0 and abs(m["mIoU"] - 1.0) < 1e-12


def testDiagnostic():
    from src.diagnostics import checkBlocks
    with localTmp() as tmp:
        cfg, _ = buildSyntheticTree(tmp)
        prepareDataset(cfg)
        cfg["blocks"].update({"blockSize": 30.0, "numPoints": 1024, "minTrainPoints": 64})
        cfg["model"]["saRadii"] = [1.0, 2.0, 4.0, 8.0]
        summary, table, sizes, counts = checkBlocks(cfg, nBlocks=4)
        assert list(table["points_in"]) == [1024, 128, 32, 8]
        assert list(table["centroids"]) == [128, 32, 8, 4]
        sug = table["suggested_r"].to_numpy()
        assert sug[1] > sug[0]
        assert summary["warnings"] == []
    try:
        loadConfig("ahn3", ["blocks.numPoints=64"], configDir=REAL_CONFIGS)
    except ValueError:
        pass
    else:
        raise AssertionError("a coarsest level with fewer than 3 points was accepted")


def testSharedCacheIsReadOnly():
    from src import paths
    with localTmp() as tmp:
        cfg, _ = buildSyntheticTree(tmp)
        prepareDataset(cfg)
        shared = loadConfig("synth", [f"paths.annDataRoot={cfg['paths']['annDataRoot']}",
                                      f"paths.sharedCacheRoot={tmp}/cache",
                                      f"paths.outputRoot={tmp}/runs"],
                            configDir=os.path.join(tmp, "configs"))
        assert paths.manifestPath(shared) == os.path.join(tmp, "cache", "synth", "manifest.json")
        assert len(loadSplitArrays(shared, "test")) == 1
        try:
            prepareDataset(shared)
        except RuntimeError:
            pass
        else:
            raise AssertionError("prepareDataset wrote into the shared cache")


def testStorageStaysInProject():
    from src import paths
    cfg = loadConfig("ahn3", configDir=REAL_CONFIGS)
    assert paths.cacheRoot(cfg) == os.path.join(paths.PROJECT_ROOT, "cache")
    assert paths.outputRoot(cfg) == os.path.join(paths.PROJECT_ROOT, "runs")
    for bad in ("~/runs", "/home/someone/runs", "../outside", "notebooks/runs"):
        c = loadConfig("ahn3", [f"paths.outputRoot={bad}"], configDir=REAL_CONFIGS)
        try:
            paths.outputRoot(c)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print(f"{len(tests)} tests passed")
