'''
Spatial blocks, shared by training and inference so both see the same kind of input

A block is a square column in x/y (full z extent) of side blockSize metres, made of cellsPerSide x cellsPerSide grid cells. Blocks never span two source files, because each file is re-centred on its own centroid and different files would otherwise overlap in space.

Training : a random point is drawn (uniformly over all training points), a block containing it is chosen at a random cell offset, and numPoints points are drawn from the block

Inference: blocks are laid on a lattice with stride inferStrideCells cells, every block is split into random chunks of numPoints points so that every point in the block is predicted, and each chunk matches the density of a training sample. Every point is covered by (cellsPerSide / inferStrideCells)^2 blocks
'''

import numpy as np


class GridIndex:
    '''Sorts the points of ONE file into square x/y cells for fast block gathers.'''

    def __init__(self, xy, cellSize):
        xy            = np.asarray(xy, dtype=np.float64)
        self.cellSize = float(cellSize)
        self.origin   = xy.min(axis=0)
        ij            = np.floor((xy - self.origin) / self.cellSize).astype(np.int64)
        self.nx       = int(ij[:, 0].max()) + 1
        self.ny       = int(ij[:, 1].max()) + 1
        self.pointKey = ij[:, 0] * self.ny + ij[:, 1]
        self.order    = np.argsort(self.pointKey, kind="stable")
        sortedKeys    = self.pointKey[self.order]
        self.cellKeys, self.cellStart, self.cellCount = np.unique(
            sortedKeys, return_index=True, return_counts=True
        )
        self.nPoints = len(xy)

    def keyToIJ(self, key):
        return np.divmod(key, self.ny)

    def gather(self, i0, j0, k):
        '''Indices of all points in the k x k cells starting at cell (i0, j0)'''
        ii, jj = np.meshgrid(np.arange(i0, i0 + k), np.arange(j0, j0 + k), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        valid  = (ii >= 0) & (ii < self.nx) & (jj >= 0) & (jj < self.ny)
        keys   = ii[valid] * self.ny + jj[valid]
        pos    = np.searchsorted(self.cellKeys, keys)
        ok     = pos < len(self.cellKeys)
        ok[ok] = self.cellKeys[pos[ok]] == keys[ok]
        pos    = pos[ok]
        if len(pos) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([self.order[s:s + c] for s, c in zip(self.cellStart[pos], self.cellCount[pos])])

    def blockCentre(self, i0, j0, k):
        return self.origin + (np.array([i0, j0], dtype=np.float64) + k / 2.0) * self.cellSize


def subsample(idx, numPoints, rng):
    '''Exactly numPoints indicesm without replacement if possible, else padded with repeats'''
    n = len(idx)
    if n >= numPoints:
        return idx[rng.choice(n, numPoints, replace=False)]
    return np.concatenate([idx, idx[rng.integers(0, n, numPoints - n)]])


class TrainBlockSampler:

    def __init__(self, fileXyz, cfg):
        '''fileXyz : list of (N_f, 3) arrays, one per training file'''
        b = cfg["blocks"]
        self.k          = int(b["cellsPerSide"])
        self.blockSize  = float(b["blockSize"])
        self.numPoints  = int(b["numPoints"])
        self.minPoints  = int(b["minTrainPoints"])
        self.maxRetries = int(b["maxTrainRetries"])
        cell            = self.blockSize / self.k
        self.grids      = [GridIndex(x[:, :2], cell) for x in fileXyz]
        sizes           = np.array([g.nPoints for g in self.grids], dtype=np.float64)
        self.fileProb   = sizes / sizes.sum()

    def sample(self, rng):
        '''Returns (fileIdx, selected point indices, full block indices, block centre xy)'''
        for _ in range(self.maxRetries):
            f      = int(rng.choice(len(self.grids), p=self.fileProb))
            g      = self.grids[f]
            ci, cj = g.keyToIJ(g.pointKey[rng.integers(g.nPoints)])
            i0     = int(ci) - int(rng.integers(0, self.k))
            j0     = int(cj) - int(rng.integers(0, self.k))
            block  = g.gather(i0, j0, self.k)
            if len(block) >= self.minPoints:
                break
        sel = subsample(block, self.numPoints, rng)
        return f, sel, block, g.blockCentre(i0, j0, self.k)


def inferenceBlockOrigins(grid, k, stride):
    '''All lattice aligned block origins whose block contains at least one point'''
    ci, cj = grid.keyToIJ(grid.cellKeys)
    bi, bj = (ci // stride) * stride, (cj // stride) * stride
    shifts = np.arange(0, k, stride)
    oi     = (bi[:, None] - shifts[None, :]).ravel()
    oj     = (bj[:, None] - shifts[None, :]).ravel()
    # every combination of the i- and j-shifts for each occupied cell
    nS     = len(shifts)
    oi     = np.repeat(oi.reshape(-1, nS), nS, axis=1).ravel()
    oj     = np.tile(oj.reshape(-1, nS), (1, nS)).ravel()
    return np.unique(np.stack([oi, oj], axis=1), axis=0)


def iterInferenceChunks(xyz, cfg, rng):
    '''
    Yields (chunkIdx[numPoints], validMask[numPoints], blockCentre, blockIdx) covering every point of one file. Padding entries (validMask False) are fed to the network but their predictions are discarded
    '''
    b         = cfg["blocks"]
    k         = int(b["cellsPerSide"])
    stride    = int(b["inferStrideCells"])
    numPoints = int(b["numPoints"])
    grid      = GridIndex(xyz[:, :2], float(b["blockSize"]) / k)

    for i0, j0 in inferenceBlockOrigins(grid, k, stride):
        block = grid.gather(int(i0), int(j0), k)
        n     = len(block)
        if n == 0:
            continue
        centre = grid.blockCentre(int(i0), int(j0), k)
        perm   = block[rng.permutation(n)]
        for s in range(0, n, numPoints):
            part  = perm[s:s + numPoints]
            valid = np.ones(numPoints, dtype=bool)
            if len(part) < numPoints:
                pad   = block[rng.integers(0, n, numPoints - len(part))]
                valid[len(part):] = False
                part  = np.concatenate([part, pad])
            yield part, valid, centre, block


def countInferenceChunks(xyz, cfg):
    '''Number of chunks inference will run for one file (for progress bars).'''
    b      = cfg["blocks"]
    k      = int(b["cellsPerSide"])
    grid   = GridIndex(xyz[:, :2], float(b["blockSize"]) / k)
    total  = 0
    for i0, j0 in inferenceBlockOrigins(grid, k, int(b["inferStrideCells"])):
        n = len(grid.gather(int(i0), int(j0), k))
        total += -(-n // int(b["numPoints"]))
    return total
