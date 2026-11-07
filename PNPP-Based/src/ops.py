'''
PNPP essentials
- fps :farthest point sampling per block
- radius ball query : up to maxNeighbours points within r of each centroid knn
- k nearest neighbours (used for feature interpolation)

Index convention (same as torch_cluster): radius(x, y, ...) and knn(x, y, ...) return (row, col), where row indexes the queries y and col indexes the reference points x
'''

import torch

try:
    import torch_cluster as _tc
except ImportError:
    _tc = None

_USE_REFERENCE = False


def useReference(flag=True):
    global _USE_REFERENCE
    _USE_REFERENCE = bool(flag)


def _need():
    if _tc is None and not _USE_REFERENCE:
        raise ImportError("torch_cluster is required (pip install torch-cluster)")


@torch.no_grad()
def fps(pos, batch, ratio):
    _need()
    if _USE_REFERENCE:
        return _fpsRef(pos, batch, ratio)
    return _tc.fps(pos, batch, ratio=float(ratio), random_start=False)


@torch.no_grad()
def radius(x, y, r, batchX, batchY, maxNeighbours):
    _need()
    if _USE_REFERENCE:
        return _radiusRef(x, y, r, batchX, batchY, maxNeighbours)
    row, col = _tc.radius(x, y, float(r), batchX, batchY, max_num_neighbors=int(maxNeighbours))
    return row, col


@torch.no_grad()
def knn(x, y, k, batchX, batchY):
    _need()
    if _USE_REFERENCE:
        return _knnRef(x, y, k, batchX, batchY)
    return _tc.knn(x, y, int(k), batchX, batchY)



def _blocks(batch):
    B = int(batch.max().item()) + 1
    return [torch.nonzero(batch == b, as_tuple=True)[0] for b in range(B)]


def _fpsRef(pos, batch, ratio):
    out = []
    for idx in _blocks(batch):
        p = pos[idx]
        m = int(torch.ceil(torch.tensor(len(idx) * float(ratio))).item())
        sel = [0]
        d = ((p - p[0]) ** 2).sum(-1)
        for _ in range(1, m):
            j = int(torch.argmax(d).item())
            sel.append(j)
            d = torch.minimum(d, ((p - p[j]) ** 2).sum(-1))
        out.append(idx[torch.tensor(sel, dtype=torch.long)])
    return torch.cat(out)


def _radiusRef(x, y, r, batchX, batchY, maxNeighbours):
    rows, cols = [], []
    for b in range(int(batchY.max().item()) + 1):
        iy = torch.nonzero(batchY == b, as_tuple=True)[0]
        ix = torch.nonzero(batchX == b, as_tuple=True)[0]
        d2 = torch.cdist(y[iy], x[ix]) ** 2
        for q in range(len(iy)):
            hit = torch.nonzero(d2[q] <= r * r, as_tuple=True)[0][:maxNeighbours]
            rows.append(torch.full((len(hit),), int(iy[q]), dtype=torch.long))
            cols.append(ix[hit])
    return torch.cat(rows), torch.cat(cols)


def _knnRef(x, y, k, batchX, batchY):
    rows, cols = [], []
    for b in range(int(batchY.max().item()) + 1):
        iy = torch.nonzero(batchY == b, as_tuple=True)[0]
        ix = torch.nonzero(batchX == b, as_tuple=True)[0]
        kk = min(int(k), len(ix))
        nn = torch.cdist(y[iy], x[ix]).topk(kk, largest=False, dim=1).indices
        rows.append(iy.repeat_interleave(kk))
        cols.append(ix[nn.reshape(-1)])
    return torch.cat(rows), torch.cat(cols)
