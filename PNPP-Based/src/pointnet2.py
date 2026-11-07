'''
PointNet++
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import ops


class SharedMLP(nn.Module):
    '''Linear -> BatchNorm -> ReLU over the last dimension, for each width in `widths`'''
    def __init__(self, cin, widths, momentum=0.1, lastAct=True):
        super().__init__()
        layers = []
        for i, w in enumerate(widths):
            layers.append(nn.Linear(cin, w, bias=False))
            layers.append(nn.BatchNorm1d(w, momentum=momentum))
            if i < len(widths) - 1 or lastAct:
                layers.append(nn.ReLU(inplace=True))
            cin = w
        self.net = nn.Sequential(*layers)
        self.cout = cin

    def forward(self, x):
        return self.net(x)


class SetAbstraction(nn.Module):

    def __init__(self, cin, widths, ratio, radius, nsample, normalizeRelPos, momentum):
        super().__init__()
        self.ratio, self.radius, self.nsample = float(ratio), float(radius), int(nsample)
        self.normalizeRelPos = bool(normalizeRelPos)
        self.mlp  = SharedMLP(cin + 3, widths, momentum)
        self.cout = self.mlp.cout

    def forward(self, pos, feat, batch):
        idx            = ops.fps(pos, batch, self.ratio)
        posC, batchC   = pos[idx], batch[idx]
        row, col       = ops.radius(pos, posC, self.radius, batch, batchC, self.nsample)
        rel            = pos[col] - posC[row]
        if self.normalizeRelPos:
            rel = rel / self.radius
        x   = self.mlp(torch.cat([rel, feat[col]], dim=-1))
        out = x.new_zeros(posC.shape[0], x.shape[1])
        out = out.scatter_reduce(0, row.unsqueeze(1).expand_as(x), x, reduce="amax", include_self=False)
        return posC, out, batchC


class FeaturePropagation(nn.Module):
    '''Inverse squared distance interpolation from the 3 nearest coarse points, skip concat, shared MLP'''
    def __init__(self, cin, widths, momentum):
        super().__init__()
        self.mlp  = SharedMLP(cin, widths, momentum)
        self.cout = self.mlp.cout

    def forward(self, posCoarse, posFine, featCoarse, featFine, batchCoarse, batchFine):
        row, col = ops.knn(posCoarse, posFine, 3, batchCoarse, batchFine) # row -> fine, col -> coarse
        w   = 1.0 / ((posFine[row] - posCoarse[col]) ** 2).sum(-1).clamp_min(1e-10)
        n   = posFine.shape[0]
        den = w.new_zeros(n).index_add_(0, row, w)
        num = featCoarse.new_zeros(n, featCoarse.shape[1]).index_add_(0, row, featCoarse[col] * w.unsqueeze(1))
        x   = num / den.unsqueeze(1)
        if featFine is not None:
            x = torch.cat([x, featFine], dim=-1)
        return self.mlp(x)


class PointNet2(nn.Module):

    def __init__(self, inChannel, numClasses, cfg):
        super().__init__()
        m   = cfg["model"]
        mom = float(m.get("bnMomentum", 0.1))
        self.inChannel = inChannel

        self.sa = nn.ModuleList()
        c, saOut = inChannel, []
        for ratio, r, ns, widths in zip(m["saRatios"], m["saRadii"], m["saNsample"], m["saMlps"]):
            layer = SetAbstraction(c, [int(w) for w in widths], ratio, r, ns, m.get("normalizeRelPos", True), mom)
            self.sa.append(layer)
            c = layer.cout
            saOut.append(c)

        # FP layers from coarsest to finest, skip = SA output one level up, or the input features
        skips = [inChannel] + saOut[:-1]
        self.fp = nn.ModuleList()
        for widths, skip in zip(m["fpMlps"], reversed(skips)):
            layer = FeaturePropagation(c + skip, [int(w) for w in widths], mom)
            self.fp.append(layer)
            c = layer.cout

        self.head       = SharedMLP(c, [int(w) for w in m["head"]], mom)
        self.drop       = nn.Dropout(p=float(m["dropout"]))
        self.classifier = nn.Linear(self.head.cout, numClasses)

    def forward(self, pos, feat, batch):
        posL, featL, batchL = [pos], [feat], [batch]
        for sa in self.sa:
            p, f, b = sa(posL[-1], featL[-1], batchL[-1])
            posL.append(p); featL.append(f); batchL.append(b)

        f = featL[-1]
        for j, fp in enumerate(self.fp):
            lo, hi = len(self.sa) - j, len(self.sa) - j - 1 # coarse level -> next finer level
            f = fp(posL[lo], posL[hi], f, featL[hi], batchL[lo], batchL[hi])

        return self.classifier(self.drop(self.head(f)))


def buildModel(cfg, inChannel, numClasses, device):
    model   = PointNet2(inChannel, numClasses, cfg).to(device)
    nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] PointNet++  inChannel={inChannel}  numClasses={numClasses}  params={nParams:,}")
    return model
