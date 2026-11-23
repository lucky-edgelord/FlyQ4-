"""FlyWire v783 loading with explicit provenance (never silently synthetic)."""
from pathlib import Path
import urllib.request
import numpy as np

BASE = "https://github.com/eonsystemspbc/fly-brain/raw/main/data/"
COMP = "2025_Completeness_783.csv"
CONN = "2025_Connectivity_783.parquet"

def _fetch(name, data_dir, min_bytes=1_000_000):
    dest = Path(data_dir) / name
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BASE + name, dest)
    if dest.stat().st_size < min_bytes:  # Git-LFS pointer stub
        raise RuntimeError(f"{name} is only {dest.stat().st_size} bytes (LFS pointer?). "
                           "Pass --data-dir pointing at a full copy.")
    return dest

def load_flywire(data_dir="data"):
    """Return dict(root_ids, u, v, w) with int32 signed synapse counts."""
    import pandas as pd
    comp = pd.read_csv(_fetch(COMP, data_dir, 100_000))
    conn = pd.read_parquet(_fetch(CONN, data_dir))
    root_ids = comp[comp.columns[0]].astype(np.int64).values
    n = len(root_ids)
    need = {"Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"}
    if not need.issubset(conn.columns):
        raise RuntimeError(f"Unexpected schema {list(conn.columns)}; refusing to guess.")
    u = conn["Presynaptic_Index"].values.astype(np.uint32)
    v = conn["Postsynaptic_Index"].values.astype(np.uint32)
    w = conn["Excitatory x Connectivity"].values.astype(np.int32)
    ok = (u < n) & (v < n) & (w != 0)
    return dict(root_ids=root_ids, u=u[ok], v=v[ok], w=w[ok], source="FlyWire v783")

GR5A_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853356, 720575940629176663,
    720575940611875570, 720575940623172843, 720575940627692048, 720575940631954612,
    720575940613786734,
]

def stimulus_indices(root_ids, mode="first150"):
    """first150 reproduces the A/B notebook; gr5a uses real sugar-GRN root IDs."""
    if mode == "gr5a":
        lut = {int(r): i for i, r in enumerate(root_ids)}
        idx = [lut[r] for r in GR5A_IDS if r in lut]
        if not idx:
            raise RuntimeError("No GR5A root IDs matched this dataset.")
        return np.asarray(idx, dtype=np.uint32)
    return np.arange(min(150, len(root_ids)), dtype=np.uint32)
