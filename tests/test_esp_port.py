"""ESP32 core (host-compiled) must equal the reference engine tick for tick."""
import sys; from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).parent))
from flyq4 import quant as Q, esp
from flyq4.runtime import load_lib, simulate
from test_engine import fixture

def test_port_bit_exact():
    n, ptr, idx, data = fixture(n=300, m=9000, seed=3)
    q4, _ = Q.quantize_int4(data); stim = np.arange(30, dtype=np.uint32)
    lib, port = load_lib("build"), esp.build_port_lib("build")
    for th, g in [(1, "unit"), (2, "twn"), (3, "drive")]:
        p2, i2, t, lut, _ = Q.ternarize(ptr, idx, data, q4, th, g); packed = Q.pack_ternary(t)
        ref = simulate(lib, "ternary", p2, i2, packed, stim, 600, lut=lut)
        tot, counts, ops = esp.run_port(port, p2, i2.astype(np.uint16), packed, lut, stim, 600)
        assert ref["total_spikes"] > 0 and tot == ref["total_spikes"] and np.array_equal(counts, ref["counts"]), (th, g)
        assert ops == (ref["edge_ops"] & 0xFFFFFFFF)

if __name__ == "__main__":
    test_port_bit_exact(); print("ESP PORT BIT-EXACT")
