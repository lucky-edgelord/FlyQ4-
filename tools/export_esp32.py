#!/usr/bin/env python3
"""Crop a stimulus-centred subnetwork from FlyWire v783 and export it to esp32/main/.

  python tools/export_esp32.py --data-dir data --n 4096 --theta 3 --gain drive
Prints the RAM/flash budget and verifies the ESP32 core (host-compiled) is bit-exact with
the reference engine on this exact subnetwork before writing the blob.
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyq4 import quant as Q, esp
from flyq4.data import load_flywire, stimulus_indices
from flyq4.runtime import load_lib, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data"); ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--theta", type=float, default=3); ap.add_argument("--gain", default="drive")
ap.add_argument("--ticks", type=int, default=2000)
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "esp32" / "main"))
ap.add_argument("--stimulus", default="first150")
a = ap.parse_args()
assert a.n <= 65535, "uint16 indices"

d = load_flywire(a.data_dir); N = len(d["root_ids"])
ptr, idx, data = Q.build_csr(d["u"], d["v"], d["w"], N)
q4, _ = Q.quantize_int4(data)
seeds = stimulus_indices(d["root_ids"], a.stimulus)
keep = esp.crop_bfs(ptr, idx, data, seeds, a.n, a.theta)
sp, si, sd = esp.induced(ptr, idx, data, keep)
# INT4 scale is the full-brain one (p95/6 = 2.0); re-derive INT4 steps on the induced weights
sq4 = np.clip(np.where(sd != 0, np.round(sd / 2.0).astype(np.int32), 0), -8, 7)
sq4 = np.where((sq4 == 0) & (sd > 0), 1, np.where((sq4 == 0) & (sd < 0), -1, sq4)).astype(np.int8)
p2, i2, t, lut, gains = Q.ternarize(sp, si, sd, sq4, a.theta, a.gain)
stim = np.arange(min(len(seeds), len(keep)), dtype=np.uint32)   # seeds are first in `keep`
packed = Q.pack_ternary(t); i16 = i2.astype(np.uint16)

lib = load_lib("build"); ref = simulate(lib, "ternary", p2, i2, packed, stim, a.ticks, lut=lut)
port = esp.run_port(esp.build_port_lib("build"), p2, i16, packed, lut, stim, a.ticks)
assert port[0] == ref["total_spikes"] and np.array_equal(port[1], ref["counts"]), "PORT NOT BIT-EXACT"
print(f"bit-exact vs reference: {ref['total_spikes']:,} spikes over {a.ticks} ticks")

r = esp.write_blob(a.out, p2, i16, t, lut, stim.astype(np.uint16), round(0.55 * 65536), ref["total_spikes"], a.ticks)
print(f"N={r['n']} edges={r['nnz']:,} (gains E={gains[0]:.2f} I={gains[1]:.2f})")
print(f"flash/rodata: indices {r['index_bytes']/1024:.0f} KiB + weights {r['weight_bytes']/1024:.0f} KiB + indptr {r['indptr_bytes']/1024:.0f} KiB")
print(f"RAM state:    {r['state_bytes']/1024:.0f} KiB (V,G,refrac,queue,I)")
