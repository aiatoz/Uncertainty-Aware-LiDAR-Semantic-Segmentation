'''
Check of block size and ball radii before training

For sampled training blocks it rebuilds the PointNet++ hierarchy exactly as the network does (farthest point sampling with the configured ratios) and, for each set abstraction level, counts how many points of the previous level fall inside the ball around each centroid.

  median_in_ball  points inside the ball (uncapped). Around saNsample or a
                  little above is good. Far above means the ball is much
                  larger than needed and only a random saNsample of them is
                  used; far below means sparse neighbourhoods.
                  
  pct_below_4     centroids with fewer than 4 points in the ball. Keep small
  
  suggested_r     median distance to the saNsample-th neighbour, i.e. the
                  radius at which a typical ball holds exactly saNsample
                  points. Copy into model.saRadii if the current one is off
'''

import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.blocks import TrainBlockSampler, countInferenceChunks
from src.features import buildBlockInput
from src.prepare import loadManifest, loadSplitArrays


def fpsNumpy(p, m):
    '''Farthest point sampling from index 0, as torch_cluster.fps(random_start=False).'''
    sel = np.empty(m, dtype=np.int64)
    sel[0] = 0
    d = ((p - p[0]) ** 2).sum(1)
    for i in range(1, m):
        j = int(d.argmax())
        sel[i] = j
        d = np.minimum(d, ((p - p[j]) ** 2).sum(1))
    return sel


def checkBlocks(cfg, nBlocks=20, seed=0):
    manifest = loadManifest(cfg)
    arrays   = list(loadSplitArrays(cfg, "train", manifest).values())
    sampler  = TrainBlockSampler([a["xyz"] for a in arrays], cfg)
    rng      = np.random.default_rng(seed)
    m        = cfg["model"]
    ratios   = [float(r) for r in m["saRatios"]]
    radii    = [float(r) for r in m["saRadii"]]
    nsample  = [int(k) for k in m["saNsample"]]
    L        = len(ratios)

    blockSizes = []
    counts     = [[] for _ in range(L)]
    kthDist    = [[] for _ in range(L)]
    nPts       = [int(cfg["blocks"]["numPoints"])]
    for _ in range(nBlocks):
        f, sel, block, centre = sampler.sample(rng)
        blockSizes.append(len(block))
        sel    = sel[rng.permutation(len(sel))]
        pos, _ = buildBlockInput(arrays[f]["xyz"], arrays[f]["u"], sel, block, centre, cfg, False)
        p = pos.astype(np.float64)
        for lvl in range(L):
            c    = p[fpsNumpy(p, math.ceil(len(p) * ratios[lvl]))]
            tree = cKDTree(p)
            counts[lvl].extend(tree.query_ball_point(c, radii[lvl], return_length=True).tolist())
            k = min(nsample[lvl], len(p))
            d, _ = tree.query(c, k=k)
            kthDist[lvl].extend(np.atleast_2d(d)[:, -1].tolist())
            if len(nPts) <= lvl + 1:
                nPts.append(len(c))
            p = c

    rows = []
    for lvl in range(L):
        cnt = np.array(counts[lvl])
        sug = float(np.median(kthDist[lvl]))
        rows.append({
            "level"          : lvl + 1,
            "points_in"      : nPts[lvl],
            "centroids"      : nPts[lvl + 1],
            "radius"         : radii[lvl],
            "median_in_ball" : float(np.median(cnt)),
            "p10_in_ball"    : float(np.percentile(cnt, 10)),
            "pct_below_4"    : round(100 * float(np.mean(cnt < 4)), 1),
            "nsample"        : nsample[lvl],
            "suggested_r"    : float(f"{sug:.2g}"),
        })

    trainExtent = max(float(np.ptp(a["xyz"][:, :2], axis=0).max()) for a in arrays)
    warnings = []
    if trainExtent < float(cfg["blocks"]["blockSize"]):
        warnings.append(f"the largest training file spans only {trainExtent:.2f} units in x/y, less than "
                        f"blockSize={cfg['blocks']['blockSize']}: one block covers a whole file. "
                        f"Check that the coordinates are in metres.")

    blockSizes = np.array(blockSizes)
    test = loadSplitArrays(cfg, "test", manifest)
    summary = {
        "blockSize"         : cfg["blocks"]["blockSize"],
        "numPoints"         : cfg["blocks"]["numPoints"],
        "block_pts_p10"     : float(np.percentile(blockSizes, 10)),
        "block_pts_median"  : float(np.median(blockSizes)),
        "block_pts_p90"     : float(np.percentile(blockSizes, 90)),
        "pct_blocks_padded" : round(100 * float(np.mean(blockSizes < cfg["blocks"]["numPoints"])), 1),
        "train_extent_xy"   : round(trainExtent, 2),
        "inference_chunks"  : {f: countInferenceChunks(a["xyz"], cfg) for f, a in test.items()},
        "warnings"          : warnings,
    }
    return summary, pd.DataFrame(rows), blockSizes, counts
