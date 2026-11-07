'''
Training data: every item is one spatial block, drawn at random

The sampling code is the same RandLA-Net code
'''

import numpy as np
import torch
from torch.utils.data import Dataset

from src.blocks import TrainBlockSampler
from src.features import buildBlockInput

IGNORE_INDEX = -100


class BlockTrainDataset(Dataset):

    def __init__(self, trainArrays, cfg, useUncertainty, seed, ignoreIdx=()):
        self.arrays         = list(trainArrays.values())
        self.cfg            = cfg
        self.useUncertainty = useUncertainty
        self.seed           = int(seed)
        self.epoch          = 0
        self.length         = int(cfg["training"]["blocksPerEpoch"])
        self.rotate         = bool(cfg["training"]["rotateZ"])
        self.sampler        = TrainBlockSampler([a["xyz"] for a in self.arrays], cfg)
        self.labels = []
        for a in self.arrays:
            lbl = a["label"].copy()
            if len(ignoreIdx):
                lbl[np.isin(lbl, list(ignoreIdx))] = IGNORE_INDEX
            self.labels.append(lbl)

    def setEpoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        f, sel, block, centre = self.sampler.sample(rng)
        theta = float(rng.uniform(0.0, 2.0 * np.pi)) if self.rotate else 0.0
        sel   = sel[rng.permutation(len(sel))]
        a = self.arrays[f]
        pos, feat = buildBlockInput(a["xyz"], a["u"], sel, block, centre, self.cfg, self.useUncertainty, theta)
        return torch.from_numpy(pos), torch.from_numpy(feat), torch.from_numpy(self.labels[f][sel])


def collateBlocks(items):
    pos   = torch.cat([it[0] for it in items], dim=0)
    feat  = torch.cat([it[1] for it in items], dim=0)
    label = torch.cat([it[2] for it in items], dim=0)
    batch = torch.cat([torch.full((len(it[0]),), b, dtype=torch.long) for b, it in enumerate(items)])
    return pos, feat, label, batch


def classWeights(manifest, numClasses, mode, ignoreIdx=()):
    '''None, or weights from TRAINING counts only, normalised to mean 1 over trained classes'''
    if mode == "none":
        return None
    counts = np.zeros(numClasses, dtype=np.float64)
    for raw, n in manifest["trainCounts"].items():
        counts[manifest["labelMap"][raw]] += n
    trained = counts > 0
    trained[list(ignoreIdx)] = False
    w = np.zeros(numClasses, dtype=np.float64)
    w[trained] = 1.0 / (counts[trained] if mode == "inverse" else np.sqrt(counts[trained]))
    w[trained] /= w[trained].mean()
    return torch.tensor(w, dtype=torch.float32)
