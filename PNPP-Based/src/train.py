'''
Training loop. The model after the last epoch is evaluated
'''

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader

from src import paths
from src.config import checkVariant
from src.datasets import IGNORE_INDEX, BlockTrainDataset, classWeights, collateBlocks
from src.features import inChannels
from src.prepare import loadManifest, loadSplitArrays, rawToIdx
from src.pointnet2 import buildModel


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trainRun(cfg, variant, seed, device):
    useUnc   = checkVariant(variant)
    setSeed(seed)
    outDir   = paths.runDir(cfg, variant, seed)
    manifest = loadManifest(cfg)
    numCls   = manifest["numClasses"]
    ignore   = rawToIdx(manifest, cfg["labels"]["trainIgnoreRaw"], "trainIgnoreRaw")

    with open(os.path.join(outDir, "config.json"), "w") as f:
        json.dump({"cfg": cfg, "variant": variant, "seed": seed, "trainIgnoreIdx": ignore}, f, indent=2)

    ds     = BlockTrainDataset(loadSplitArrays(cfg, "train", manifest), cfg, useUnc, seed, ignore)
    tcfg   = cfg["training"]
    loader = DataLoader(ds, batch_size=int(tcfg["batchBlocks"]), shuffle=False,
                        num_workers=int(tcfg["numWorkers"]), collate_fn=collateBlocks,
                        pin_memory=True, drop_last=True, persistent_workers=False)

    model     = buildModel(cfg, inChannels(useUnc), numCls, device)
    weights   = classWeights(manifest, numCls, tcfg["classWeights"], ignore)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None,
                                    ignore_index=IGNORE_INDEX)
    optimizer = Adam(model.parameters(), lr=float(tcfg["learningRate"]), weight_decay=float(tcfg["weightDecay"]))
    epochs    = int(tcfg["epochs"])
    if tcfg.get("lrSchedule", "cosine") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(tcfg.get("lrMin", 0.0)))
    else:
        scheduler = ExponentialLR(optimizer, gamma=float(tcfg["lrDecay"]))

    history = {"epoch": [], "loss": [], "oa": [], "lr": [], "seconds": []}
    for epoch in range(epochs):
        ds.setEpoch(epoch)
        model.train()
        t0 = time.time()
        lossSum, nBatches, correct, counted = 0.0, 0, 0, 0
        for pos, feat, label, batch in loader:
            pos, feat = pos.to(device, non_blocking=True), feat.to(device, non_blocking=True)
            label     = label.to(device, non_blocking=True)
            batch     = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(pos, feat, batch)
            loss   = criterion(logits, label)
            loss.backward()
            optimizer.step()

            valid     = label != IGNORE_INDEX
            correct  += (logits.argmax(-1)[valid] == label[valid]).sum().item()
            counted  += int(valid.sum().item())
            lossSum  += loss.item()
            nBatches += 1

        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        dt = time.time() - t0
        history["epoch"].append(epoch + 1)
        history["loss"].append(lossSum / max(nBatches, 1))
        history["oa"].append(correct / max(counted, 1))
        history["lr"].append(lr)
        history["seconds"].append(dt)
        print(f"[train] {cfg['dataset']['name']}/{variant}/seed{seed}  epoch {epoch + 1}/{epochs}  "
              f"loss={history['loss'][-1]:.4f}  trainOA={history['oa'][-1]:.4f}  lr={lr:.2e}  {dt:.0f}s")

        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "variant": variant, "seed": seed,
                    "inChannel": inChannels(useUnc), "numClasses": numCls},
                   os.path.join(outDir, "last.pth.tmp"))
        os.replace(os.path.join(outDir, "last.pth.tmp"), os.path.join(outDir, "last.pth"))
        with open(os.path.join(outDir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    return os.path.join(outDir, "last.pth")
