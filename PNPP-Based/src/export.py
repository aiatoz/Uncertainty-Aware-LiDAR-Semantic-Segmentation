'''
Export test-tile point clouds with ground truth and both models' predictions
'''

import glob
import json
import os

import numpy as np
import pandas as pd

from src import paths
from src.config import loadConfig
from src.prepare import _loadTrueXyz, loadManifest, loadSplitArrays

VARIANTS = ("baseline", "uncertainty")


def runTable(cfg):
    rows = []
    for path in sorted(glob.glob(os.path.join(paths.datasetOutDir(cfg), "*", "seed*", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        if m.get("split", "test") == "test" and m["variant"] in VARIANTS:
            rows.append({"variant": m["variant"], "seed": int(m["seed"]), "OA_all": m["OA_all"],
                         "mIoU": m["mIoU"], "PAvPU": m.get("PAvPU")})
    return pd.DataFrame(rows)


def chooseSeeds(cfg, rule="best_each"):
    '''
    rule = "best_each" : each variant's own highest mIoU seed
           "best_pair" : the one seed with the highest mean mIoU over both variants
           an int      : that seed for both variants
    Returns ({variant: seed}, run table).
    '''
    table = runTable(cfg)
    if table.empty:
        raise FileNotFoundError(f"No evaluated runs for {cfg['dataset']['name']}: run notebook 03 first")
    for v in VARIANTS:
        if not (table["variant"] == v).any():
            raise FileNotFoundError(f"No evaluated {v} run for {cfg['dataset']['name']}")

    if isinstance(rule, (int, np.integer)) and not isinstance(rule, bool):
        seeds = {v: int(rule) for v in VARIANTS}
    elif rule == "best_each":
        seeds = {v: int(table[table["variant"] == v].sort_values("mIoU").iloc[-1]["seed"]) for v in VARIANTS}
    elif rule == "best_pair":
        piv    = table.pivot_table(index="seed", columns="variant", values="mIoU").dropna()
        if piv.empty:
            raise ValueError("best_pair needs at least one seed evaluated for both variants")
        best   = int(piv.mean(axis=1).idxmax())
        seeds  = {v: best for v in VARIANTS}
    else:
        raise ValueError(f"seed rule must be 'best_each', 'best_pair' or an int, got {rule!r}")

    for v, s in seeds.items():
        if not ((table["variant"] == v) & (table["seed"] == s)).any():
            raise FileNotFoundError(f"{v} seed {s} has not been evaluated for {cfg['dataset']['name']}")
    return seeds, table


def _savedPredictions(cfg, variant, seed, arrays, manifest):
    '''Per-file class indices from pred_rows.npy (split-CSV row order), or None if unusable.'''
    path = os.path.join(paths.runDir(cfg, variant, seed), "pred_rows.npy")
    if not os.path.exists(path):
        return None
    rows = np.load(path).astype(np.int64)
    if len(rows) != manifest["nTestRows"]:
        print(f"[export]   {variant} s{seed}: pred_rows.npy has {len(rows):,} rows, expected "
              f"{manifest['nTestRows']:,}; ignoring it")
        return None
    out = {f: rows[a["rowIndex"]] for f, a in arrays.items()}
    if any((p < 0).any() for p in out.values()):
        return None
    return out


def _inferPredictions(dataset, variant, seed, arrays, manifest, device, evalOverrides):
    '''Runs the run's last.pth checkpoint on every test file (same seeded inference as notebook 03).'''
    import torch

    from src.features import inChannels
    from src.infer import loadRunConfig, predictFile
    from src.pointnet2 import buildModel

    cfg    = loadRunConfig(dataset, variant, seed, evalOverrides)
    useUnc = variant == "uncertainty"
    ckpt   = torch.load(os.path.join(paths.runDir(cfg, variant, seed), "last.pth"),
                        map_location=device, weights_only=True)
    if ckpt["inChannel"] != inChannels(useUnc):
        raise ValueError(f"checkpoint inChannel {ckpt['inChannel']} does not match variant {variant}")
    model = buildModel(cfg, inChannels(useUnc), manifest["numClasses"], device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out = {}
    for f, a in arrays.items():
        print(f"[export]   inference {variant} s{seed} on {f}: {len(a['xyz']):,} points")
        out[f], _ = predictFile(model, a, cfg, useUnc, manifest["numClasses"], device, seed)
    del model
    if device is not None and getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()
    return out


def getPredictions(cfg, variant, seed, arrays, manifest, mode="auto", device=None, evalOverrides=(),
                   inferFn=None):
    '''
    mode = "auto"  : saved predictions if present and consistent with metrics.json, otherwise inference
           "saved" : saved predictions only (error if missing)
           "infer" : always run the checkpoint
    inferFn replaces the inference call (used by the tests)
    '''
    if mode not in ("auto", "saved", "infer"):
        raise ValueError("predictions must be 'auto', 'saved' or 'infer'")
    with open(os.path.join(paths.runDir(cfg, variant, seed), "metrics.json")) as f:
        expectedOA = json.load(f)["OA_all"]

    preds, source = None, None
    if mode in ("auto", "saved"):
        preds = _savedPredictions(cfg, variant, seed, arrays, manifest)
        source = "saved (notebook 03)"
        if preds is None and mode == "saved":
            raise FileNotFoundError(f"{variant} s{seed}: no usable pred_rows.npy")
    if preds is None:
        run    = inferFn or (lambda: _inferPredictions(cfg["dataset"]["name"], variant, seed, arrays,
                                                       manifest, device, evalOverrides))
        preds  = run()
        source = "checkpoint inference"

    correct = sum(int((preds[f] == a["label"]).sum()) for f, a in arrays.items())
    oa      = correct / sum(len(a["label"]) for a in arrays.values())
    diff    = abs(oa - expectedOA)
    print(f"[export]   {variant} s{seed}: {source}, OA {oa:.4f} (metrics.json {expectedOA:.4f})")
    if diff > (1e-6 if source.startswith("saved") else 2e-3):
        raise ValueError(f"{variant} s{seed}: OA {oa:.6f} does not match metrics.json {expectedOA:.6f}; "
                         f"predictions are not aligned with the evaluated run")
    return preds, source, oa



def rawLabelLookup(manifest):
    '''class index -> integer raw label (falls back to the class index if raw labels are not integers).'''
    idxToRaw = {v: k for k, v in manifest["labelMap"].items()}
    lut      = np.arange(manifest["numClasses"], dtype=np.int64)
    integral = True
    for i, raw in idxToRaw.items():
        try:
            val = float(raw)
            if val != int(val):
                integral = False
            lut[i] = int(val)
        except ValueError:
            integral = False
    if not integral:
        lut = np.arange(manifest["numClasses"], dtype=np.int64)
    return lut, ("raw" if integral else "class_index")

#writers
def _fields(xyz, gt, pb, pu, u, absolute):
    cb, cu = (pb == gt), (pu == gt)
    comp   = np.where(cb & cu, 0, np.where(~cb & ~cu, 1, np.where(cu, 2, 3))).astype(np.uint8)
    lab    = np.uint16 if max(gt.max(), pb.max(), pu.max()) < 65535 else np.uint32
    ctype  = np.float64 if absolute else np.float32
    return [
        ("x", xyz[:, 0].astype(ctype)), ("y", xyz[:, 1].astype(ctype)), ("z", xyz[:, 2].astype(ctype)),
        ("gt", gt.astype(lab)), ("pred_baseline", pb.astype(lab)), ("pred_uncertainty", pu.astype(lab)),
        ("correct_baseline", cb.astype(np.uint8)), ("correct_uncertainty", cu.astype(np.uint8)),
        ("comparison", comp), ("u_tilde", u.astype(np.float32)),
    ]


_PLY_TYPES = {np.dtype(np.float32): "float", np.dtype(np.float64): "double", np.dtype(np.uint8): "uchar",
              np.dtype(np.uint16): "ushort", np.dtype(np.uint32): "uint"}


def writePly(path, fields):
    '''Binary littleendian PLY, readable by CloudCompare (extra fields become scalar fields)'''
    n      = len(fields[0][1])
    dtype  = np.dtype([(name, arr.dtype.newbyteorder("<")) for name, arr in fields])
    data   = np.empty(n, dtype=dtype)
    for name, arr in fields:
        data[name] = arr
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property {_PLY_TYPES[arr.dtype]} {name}" for name, arr in fields]
    header += ["end_header"]
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        data.tofile(f)
    os.replace(tmp, path)


def writeLas(path, fields):
    '''LAS 1.4 / point format 6 (.las or .laz). classification = gt. Other fields are extra dimensions.'''
    try:
        import laspy
    except ImportError as e:
        raise ImportError("LAS/LAZ export needs laspy (and lazrs for .laz): pip install 'laspy[lazrs]'") from e
    d      = dict(fields)
    xyz    = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float64)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.floor(xyz.min(axis=0))
    header.scales  = np.array([0.001, 0.001, 0.001])
    extra = [(k, d[k].dtype) for k in ("gt", "pred_baseline", "pred_uncertainty", "correct_baseline",
                                        "correct_uncertainty", "comparison", "u_tilde")]
    header.add_extra_dims([laspy.ExtraBytesParams(name=k, type=t) for k, t in extra])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = np.clip(d["gt"], 0, 255).astype(np.uint8)
    for k, _ in extra:
        setattr(las, k, d[k])
    tmp = path + ".tmp" + os.path.splitext(path)[1]
    las.write(tmp)
    os.replace(tmp, path)



def exportDataset(dataset, seedRule="best_each", predictions="auto", formats=("ply",), absoluteCoords=False,
                  device=None, evalOverrides=(), cfg=None, inferFn=None):
    cfg      = cfg or loadConfig(dataset)
    manifest = loadManifest(cfg)
    arrays   = loadSplitArrays(cfg, "test", manifest)
    seeds, table = chooseSeeds(cfg, seedRule)
    print(f"[export] {dataset}: baseline seed {seeds['baseline']}, uncertainty seed {seeds['uncertainty']} "
          f"(rule {seedRule!r})")

    preds, sources, oas = {}, {}, {}
    for v in VARIANTS:
        fn = (lambda v=v: inferFn(v, seeds[v], arrays)) if inferFn else None
        preds[v], sources[v], oas[v] = getPredictions(cfg, v, seeds[v], arrays, manifest, predictions,
                                                      device, evalOverrides, fn)

    lut, labelKind = rawLabelLookup(manifest)
    outDir = os.path.join(paths.datasetOutDir(cfg), "exports")
    os.makedirs(outDir, exist_ok=True)
    tag    = f"base-s{seeds['baseline']}_unc-s{seeds['uncertainty']}"

    files, stats = [], []
    for f, a in arrays.items():
        xyz, offset = a["xyz"].astype(np.float64), [0.0, 0.0, 0.0]
        if absoluteCoords:
            true   = _loadTrueXyz(cfg, f)
            centre = true.mean(axis=0)
            if len(true) != len(xyz) or np.abs((true - centre) - xyz).max() > 0.01:
                raise ValueError(f"{f}: features CSV does not match the cache; cannot restore absolute coordinates")
            xyz, offset = true, centre.tolist()
        gt, pb, pu = lut[a["label"]], lut[preds["baseline"][f]], lut[preds["uncertainty"][f]]
        fields = _fields(xyz, gt, pb, pu, a["u"], absoluteCoords)
        comp   = dict(fields)["comparison"]
        n      = len(gt)
        stats.append({"file": f, "points": n,
                      "OA_baseline": round(float((pb == gt).mean()), 4),
                      "OA_uncertainty": round(float((pu == gt).mean()), 4),
                      "pct_both_correct": round(100 * float((comp == 0).mean()), 2),
                      "pct_both_wrong": round(100 * float((comp == 1).mean()), 2),
                      "pct_fixed_by_unc": round(100 * float((comp == 2).mean()), 2),
                      "pct_broken_by_unc": round(100 * float((comp == 3).mean()), 2),
                      "offsetAdded_xyz": [round(o, 3) for o in offset] if absoluteCoords else None})
        for fmt in formats:
            fmt = fmt.lower().lstrip(".")
            path = os.path.join(outDir, f"{dataset}_{f}_{tag}.{fmt}")
            if fmt == "ply":
                writePly(path, fields)
            elif fmt in ("las", "laz"):
                writeLas(path, fields)
            else:
                raise ValueError(f"unknown format {fmt!r}; use ply, las or laz")
            files.append(path)
            print(f"[export]   wrote {path}  ({n:,} points, {os.path.getsize(path) / 2**20:.1f} MB)")

    info = {
        "dataset": dataset, "seedRule": seedRule if not isinstance(seedRule, np.integer) else int(seedRule),
        "seeds": seeds, "predictionSource": sources,
        "metrics": {v: table[(table.variant == v) & (table.seed == seeds[v])].iloc[0].to_dict() for v in VARIANTS},
        "labels": labelKind, "labelMap_raw_to_index": manifest["labelMap"],
        "coordinates": "absolute (features CSV)" if absoluteCoords else "re-centred per source file, metres",
        "comparisonCodes": {"0": "both correct", "1": "both wrong", "2": "fixed by uncertainty",
                            "3": "broken by uncertainty"},
        "files": files, "perFile": stats,
    }
    with open(os.path.join(outDir, f"{dataset}_{tag}_export.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return info, pd.DataFrame(stats), table
