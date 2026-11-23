#!/usr/bin/env python3
"""Crop a real FlyWire v783 subnetwork for the browser demo -> docs/assets/subnet.bin (+ .json meta).

Layout (little endian): magic 'FQS1', N u32, nnz u32, n_stim u32,
  indptr u32[N+1] | indices u16[nnz] | counts i16[nnz]   (signed raw synapse counts, clipped to int16)
The page quantizes in JS (INT4 and deadzone-ternary) so the sliders are live.
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyq4 import quant as Q, esp
from flyq4.data import load_flywire, stimulus_indices

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data"); ap.add_argument("--n", type=int, default=1536)
ap.add_argument("--theta", type=float, default=2.0)
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "assets"))
a = ap.parse_args()

d = load_flywire(a.data_dir); N = len(d["root_ids"])
ptr, idx, data = Q.build_csr(d["u"], d["v"], d["w"], N)
seeds = stimulus_indices(d["root_ids"], "first150")
keep = esp.crop_bfs(ptr, idx, data, seeds, a.n, a.theta)
sp, si, sd = esp.induced(ptr, idx, data, keep)
n = len(keep)
assert np.array_equal(keep[:len(seeds)], seeds.astype(np.int64)), "seeds must come first"
counts = np.clip(sd, -32768, 32767).astype("<i2")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "subnet.bin", "wb") as f:
    f.write(b"FQS1" + struct.pack("<III", n, len(si), len(seeds)))
    f.write(sp.astype("<u4").tobytes()); f.write(si.astype("<u2").tobytes()); f.write(counts.tobytes())
meta = dict(source="FlyWire v783 (eonsystemspbc/fly-brain)", neurons=n, edges=int(len(si)),
            stimulated=int(len(seeds)), crop="BFS from first-150 stimulus, |w|>=%g" % a.theta,
            exc_frac=float((sd > 0).mean()), root_ids_first=[int(x) for x in d["root_ids"][keep[:5]]])
(out / "subnet.json").write_text(json.dumps(meta, indent=2))
print(meta, f"{(out/'subnet.bin').stat().st_size/1024:.0f} KiB")
