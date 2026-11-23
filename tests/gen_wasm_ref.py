"""Reference spike-count traces for the WASM test, produced by the C engine (engine.c) on docs/assets/subnet.bin."""
import json, struct, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flyq4 import quant as Q
from flyq4.runtime import load_lib, simulate

raw = (ROOT / "docs/assets/subnet.bin").read_bytes()
n, nnz, ns = struct.unpack("<III", raw[4:16]); o = 16
ptr = np.frombuffer(raw, "<u4", n + 1, o).astype(np.uint32); o += 4 * (n + 1)
idx = np.frombuffer(raw, "<u2", nnz, o).astype(np.uint32); o += 2 * nnz
cnt = np.frombuffer(raw, "<i2", nnz, o).astype(np.int32)
q4 = np.clip(np.round(cnt / 2.0), -8, 7).astype(np.int8)
q4 = np.where((q4 == 0) & (cnt > 0), 1, np.where((q4 == 0) & (cnt < 0), -1, q4)).astype(np.int8)
stim = np.arange(ns, dtype=np.uint32); TICKS = 1500
lib = load_lib(str(ROOT / "build")); ref = {}
r = simulate(lib, "int4", ptr, idx, Q.pack_int4(q4), stim, TICKS)
ref["int4"] = dict(total=r["total_spikes"], counts=r["counts"].tolist(), edge_ops=r["edge_ops"])
for th, g in [(1, "unit"), (3, "drive"), (2, "twn")]:
    p2, i2, t, lut, _ = Q.ternarize(ptr, idx, cnt, q4, th, g)
    r = simulate(lib, "ternary", p2, i2, Q.pack_ternary(t), stim, TICKS, lut=lut)
    ref[f"ternary_{th}_{g}"] = dict(total=r["total_spikes"], counts=r["counts"].tolist(), edge_ops=r["edge_ops"], lut=lut.tolist())
(ROOT / "tests/wasm_ref.json").write_text(json.dumps(dict(ticks=TICKS, ref=ref)))
print({k: v["total"] for k, v in ref.items()})
