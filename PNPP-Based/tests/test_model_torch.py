'''
PointNet++ + torch
'''

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import ops
from src.config import loadConfig
from src.pointnet2 import buildModel


def makeBlock(rng, n, size):
    pos = np.column_stack([rng.uniform(-size / 2, size / 2, (n, 2)), rng.uniform(0, 5, n)]).astype(np.float32)
    return torch.from_numpy(pos)


def run():
    try:
        import torch_cluster
        n = 8192
    except ImportError:
        ops.useReference(True)
        n = 1024
        print("torch_cluster not installed : using the reference ops with small blocks")

    torch.manual_seed(0)
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg  = loadConfig("paris_lille", ["model.saRadii=[2.0, 4.0, 8.0, 16.0]"])
    rng  = np.random.default_rng(0)
    size = cfg["blocks"]["blockSize"]
    C    = 10

    for inCh in (3, 4):
        model = buildModel(cfg, inCh, C, dev)
        p0, p1 = makeBlock(rng, n, size), makeBlock(rng, n, size)
        f0 = torch.cat([p0 / (size / 2), torch.rand(n, inCh - 3)], 1)
        f1 = torch.cat([p1 / (size / 2), torch.rand(n, inCh - 3)], 1)
        b2 = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)]).to(dev)
        b1 = torch.zeros(n, dtype=torch.long, device=dev)

        model.train()
        opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
        out  = model(torch.cat([p0, p1]).to(dev), torch.cat([f0, f1]).to(dev), b2)
        loss = torch.nn.functional.cross_entropy(out, torch.randint(0, C, (2 * n,), device=dev))
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), "non-finite training loss"
        noGrad = [k for k, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not noGrad, f"parameters without a finite gradient: {noGrad}"

        model.eval()
        with torch.no_grad():
            out = model(torch.cat([p0, p1]).to(dev), torch.cat([f0, f1]).to(dev), b2)
            assert out.shape == (2 * n, C)
            out2 = model(torch.cat([p0, makeBlock(rng, n, size)]).to(dev), torch.cat([f0, f1]).to(dev), b2)
            assert torch.allclose(out[:n], out2[:n], atol=1e-4), "block 0 output depends on block 1"
            assert not torch.allclose(out[n:], out2[n:]), "block 1 output did not change"
            single = model(p0.to(dev), f0.to(dev), b1)
            assert torch.allclose(single, out[:n], atol=1e-4), "batched and single-block outputs differ"
            scaled = model((p0 * 3).to(dev), f0.to(dev), b1)
            assert not torch.allclose(scaled, single), "output ignores metric scale"
        print(f"ok  inChannel={inCh}")

    model = buildModel(cfg, 3, 3, dev)
    p     = makeBlock(rng, n, size)
    y     = torch.bucketize(p[:, 2], torch.tensor([5 / 3, 10 / 3])).to(dev)
    f     = (p / (size / 2)).to(dev)
    b     = torch.zeros(n, dtype=torch.long, device=dev)
    p     = p.to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(p, f, b), y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(p, f, b).argmax(-1) == y).float().mean().item()
    assert acc > 0.9, f"could not fit a single block (accuracy {acc:.3f})"
    print(f"ok  fits a labelled block (accuracy {acc:.3f})")
    print("PointNet++ tests passed")


if __name__ == "__main__":
    run()
