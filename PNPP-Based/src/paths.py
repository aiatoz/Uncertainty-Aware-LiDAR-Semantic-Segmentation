import os

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOTS = {"cacheRoot": "cache", "outputRoot": "runs"}


def _expand(p):
    p = os.path.expanduser(p)
    if not os.path.isabs(p):
        p = os.path.join(PROJECT_ROOT, p)
    return os.path.abspath(p)


def _inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def _storageRoot(cfg, key):
    p = _expand(cfg["paths"].get(key, DEFAULT_ROOTS[key]))
    if not _inside(p, PROJECT_ROOT) or _inside(p, os.path.join(PROJECT_ROOT, "notebooks")):
        raise ValueError(f"paths.{key} resolves to {p}. It must be inside {PROJECT_ROOT} "
                         f"and not inside notebooks/. Use a relative path such as '{DEFAULT_ROOTS[key]}'.")
    return p


def cacheRoot(cfg):
    return _storageRoot(cfg, "cacheRoot")


def outputRoot(cfg):
    return _storageRoot(cfg, "outputRoot")


def annFolder(cfg):
    return os.path.join(_expand(cfg["paths"]["annDataRoot"]), cfg["dataset"]["annFolder"])


def splitCsv(cfg, split):
    return os.path.join(annFolder(cfg), "splits", f"{split}.csv")


def featuresCsv(cfg, sourceFile):
    return os.path.join(annFolder(cfg), f"features_{sourceFile}.csv")


def sharedCacheRoot(cfg):
    p = cfg["paths"].get("sharedCacheRoot")
    return os.path.abspath(os.path.expanduser(p)) if p else None


def usingSharedCache(cfg):
    return sharedCacheRoot(cfg) is not None


def requireWritableCache(cfg, what):
    if usingSharedCache(cfg):
        raise RuntimeError(f"{what} would write into the shared RandLA-Net cache "
                           f"({sharedCacheRoot(cfg)}), which is read only here. Set paths.sharedCacheRoot to null and build a local cache with notebook0")


def cacheDir(cfg):
    shared = sharedCacheRoot(cfg)
    if shared:
        d = os.path.join(shared, cfg["dataset"]["name"])
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Shared cache for {cfg['dataset']['name']} not found: {d}")
        return d
    d = os.path.join(cacheRoot(cfg), cfg["dataset"]["name"])
    os.makedirs(d, exist_ok=True)
    return d


def manifestPath(cfg):
    return os.path.join(cacheDir(cfg), "manifest.json")


def cachedFile(cfg, split, sourceFile):
    d = os.path.join(cacheDir(cfg), split)
    if not usingSharedCache(cfg):
        os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{sourceFile}.npz")


def knnCache(cfg, sourceFile, k):
    return os.path.join(cacheDir(cfg), "pavpu", f"knn_k{k}_{sourceFile}.npy")


def datasetOutDir(cfg):
    d = os.path.join(outputRoot(cfg), cfg["dataset"]["name"])
    os.makedirs(d, exist_ok=True)
    return d


def runDir(cfg, variant, seed):
    d = os.path.join(datasetOutDir(cfg), variant, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d
