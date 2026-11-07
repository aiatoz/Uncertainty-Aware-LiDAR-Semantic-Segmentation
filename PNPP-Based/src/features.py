'''
Builds the network input for one block or chunk

pos  : (N, 3) metres, x/y relative to the block centre, z per features.zMode. Used by the ball queries and interpolation, so radii are in metres

feat : (N, 3) pos / (blockSize / 2)                       baseline
       (N, 4) pos / (blockSize / 2) + u_tilde             uncertainty
'''

import numpy as np


def rotationZ(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def buildBlockInput(xyz, u, sel, block, centre, cfg, useUncertainty, theta=0.0):
    '''
    xyz, u  : full file arrays (file centred metres, u_tilde)
    sel     : indices of the points fed to the network
    block   : indices of ALL points in the block (for the z reference)
    centre  : block centre (x, y) in file coordinates
    '''
    pos = xyz[sel].astype(np.float32, copy=True)
    pos[:, 0] -= np.float32(centre[0])
    pos[:, 1] -= np.float32(centre[1])
    if cfg["features"]["zMode"] == "block_min":
        pos[:, 2] -= xyz[block, 2].min()
    if theta != 0.0:
        pos = pos @ rotationZ(theta).T

    featXyz = pos / np.float32(cfg["blocks"]["blockSize"] / 2.0)
    if useUncertainty:
        feat = np.concatenate([featXyz, u[sel, None].astype(np.float32)], axis=1)
    else:
        feat = featXyz
    return pos, np.ascontiguousarray(feat, dtype=np.float32)


def inChannels(useUncertainty):
    return 4 if useUncertainty else 3
