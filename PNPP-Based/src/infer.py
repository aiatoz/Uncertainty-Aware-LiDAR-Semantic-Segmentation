'''
Inference and evaluation.

- Every test point lies in several overlapping blocks
- Softmax probabilities are summed over all of them and the argmax is the prediction
- Chunking is seeded, so the result does not depend on CSV row order or chunk boundaries
'''

import json
import os

import numpy as np
import pandas as pd
import torch

from src import paths
from src.blocks import countInferenceChunks, iterInferenceChunks
from src.config import applyOverride, checkVariant, loadConfig
from src.features import buildBlockInput, inChannels
from src.metrics import computeMetrics, computePavpu, saveJson
from src.pointnet2 import buildModel
from src.prepare import loadManifest, loadSplitArrays, rawToIdx


def loadRunConfig(dataset, variant, seed, evalOverrides=()):
    current = loadConfig(dataset)
    runDir  = paths.runDir(current, variant, seed)
    path    = os.path.join(runDir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained run at {runDir} (config.json missing)")
    with open(path) as f:
        cfg = json.load(f)["cfg"]
    cfg["paths"] = current["paths"]
    for expr in evalOverrides:
        if not expr.startswith("evaluation."):
            raise ValueError(f"Only evaluation.* can be overridden at evaluation time: {expr}")
        applyOverride(cfg, expr)
    return cfg


@torch.no_grad()
def predictFile(model, a, cfg, useUnc, numClasses, device, seed):
    xyz, u   = a["xyz"], a["u"]
    n        = len(xyz)
    probSum  = np.zeros((n, numClasses), dtype=np.float32)
    votes    = np.zeros(n, dtype=np.int32)
    rng      = np.random.default_rng([int(seed), 7919])
    bs       = int(cfg["evaluation"]["inferBatchBlocks"])
    total    = countInferenceChunks(xyz, cfg)
    done     = 0

    def run(buffer):
        pos   = torch.from_numpy(np.concatenate([b[0] for b in buffer])).to(device)
        feat  = torch.from_numpy(np.concatenate([b[1] for b in buffer])).to(device)
        batch = torch.cat([torch.full((len(b[0]),), j, dtype=torch.long) for j, b in enumerate(buffer)]).to(device)
        prob  = torch.softmax(model(pos, feat, batch), dim=-1).cpu().numpy()
        off = 0
        for _, _, idx, valid in buffer:
            p = prob[off:off + len(idx)]
            probSum[idx[valid]] += p[valid] #indices are unique within a chunk
            votes[idx[valid]]   += 1
            off += len(idx)

    buffer = []
    for idx, valid, centre, block in iterInferenceChunks(xyz, cfg, rng):
        pos, feat = buildBlockInput(xyz, u, idx, block, centre, cfg, useUnc, 0.0)
        buffer.append((pos, feat, idx, valid))
        if len(buffer) == bs:
            run(buffer)
            done  += len(buffer)
            buffer = []
            if done % (bs * 200) == 0:
                print(f"[infer]   {done:,}/{total:,} chunks")
    if buffer:
        run(buffer)

    if votes.min() == 0:
        raise RuntimeError(f"{int((votes == 0).sum())} points were never predicted")
    return probSum.argmax(axis=1).astype(np.int64), votes


def evaluateRun(cfg, variant, seed, device, split="test"):
    useUnc   = checkVariant(variant)
    outDir   = paths.runDir(cfg, variant, seed)
    manifest = loadManifest(cfg)
    numCls   = manifest["numClasses"]
    ckpt     = torch.load(os.path.join(outDir, "last.pth"), map_location=device, weights_only=True)
    if ckpt["inChannel"] != inChannels(useUnc):
        raise ValueError(f"checkpoint inChannel {ckpt['inChannel']} does not match variant {variant}")

    model = buildModel(cfg, inChannels(useUnc), numCls, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    arrays = loadSplitArrays(cfg, split, manifest)
    nRows  = manifest["nTestRows"] if split == "test" else manifest["nTrainRows"]
    predRows = np.full(nRows, -1, dtype=np.int64)
    predByFile, labelsAll, predsAll = {}, [], []
    for f, a in arrays.items():
        print(f"[infer] {split}/{f}: {len(a['xyz']):,} points")
        pred, votes = predictFile(model, a, cfg, useUnc, numCls, device, seed)
        predByFile[f] = pred
        predRows[a["rowIndex"]] = pred
        labelsAll.append(a["label"])
        predsAll.append(pred)
        print(f"[infer]   votes per point: min {votes.min()}, median {int(np.median(votes))}")
    if (predRows < 0).any():
        raise RuntimeError("some split-CSV rows received no prediction")

    exclude = rawToIdx(manifest, cfg["labels"]["metricExcludeRaw"], "metricExcludeRaw")
    metrics = computeMetrics(np.concatenate(predsAll), np.concatenate(labelsAll), numCls, exclude)
    metrics.update({"dataset": cfg["dataset"]["name"], "variant": variant, "seed": seed,
                    "split": split, "epoch": ckpt["epoch"]})

    tag = "" if split == "test" else f"_{split}"
    np.save(os.path.join(outDir, f"pred{tag}_rows.npy"), predRows.astype(np.int16)) #split CSV row order
    if split == "test":
        pavpu = computePavpu(cfg, arrays, predByFile)
        pavpu.update({"dataset": cfg["dataset"]["name"], "variant": variant, "seed": seed})
        saveJson(pavpu, os.path.join(outDir, "pavpu.json"))
        metrics["PAvPU"] = pavpu["PAvPU"]
    saveJson(metrics, os.path.join(outDir, f"metrics{tag}.json"))

    idxToRaw = {v: k for k, v in manifest["labelMap"].items()}
    pd.DataFrame({
        "class_idx" : list(range(numCls)),
        "raw_label" : [idxToRaw.get(i, "") for i in range(numCls)],
        "iou"       : metrics["classIoU"],
        "averaged"  : [i in metrics["classesAveraged"] for i in range(numCls)],
    }).to_csv(os.path.join(outDir, f"per_class_iou{tag}.csv"), index=False)

    print(f"[eval] {cfg['dataset']['name']}/{variant}/seed{seed} ({split}): "
          f"OA_all={metrics['OA_all']:.4f}  mIoU={metrics['mIoU']:.4f}"
          + (f"  PAvPU={metrics['PAvPU']:.4f}" if split == "test" else ""))
    return metrics
