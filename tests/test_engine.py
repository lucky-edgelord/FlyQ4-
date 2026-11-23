"""C engines vs an independent pure-Python reference, plus codec round-trips."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyq4 import quant as Q
from flyq4.runtime import load_lib, simulate

def ref_run(indptr, indices, wvals, n, stim, ticks, stim_mv=0.55):
    """wvals: signed per-edge PSP in Q16.16 (already multiplied by scale)."""
    V = [Q.V_REST_Q16] * n; G = [0] * n; R = [0] * n
    I = [0] * n
    for s in stim: I[int(s)] = round(stim_mv * 65536)
    queue = [int(s) for s in stim]; counts = []; tot = [0] * n
    GL, VF = 65536 * 64, -65536 * 90
    for _ in range(ticks):
        for u in queue:
            for e in range(indptr[u], indptr[u + 1]):
                d = int(indices[e]); G[d] = max(-GL, min(GL, G[d] + int(wvals[e])))
        nxt = []
        for i in range(n):
            g = G[i]; G[i] = g - (g >> Q.G_SHIFT)
            if R[i] > 0: R[i] -= 1; continue
            v = V[i]; dv = ((Q.V_REST_Q16 - v) * Q.ALPHA_M_Q16) >> 16
            vn = v + dv + g + I[i]
            if vn >= Q.V_TH_Q16: V[i] = Q.V_REST_Q16; R[i] = Q.REFRAC_T; nxt.append(i); tot[i] += 1
            else: V[i] = max(VF, min(vn, Q.V_TH_Q16))
        queue = nxt; counts.append(len(nxt))
    return np.array(counts), np.array(tot)

def fixture(n=200, m=4000, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.integers(0, n, m); v = rng.integers(0, n, m)
    w = (rng.integers(1, 40, m) * rng.choice([-1, 1], m, p=[.4, .6])).astype(np.int32)
    ptr, idx, data = Q.build_csr(u.astype(np.uint32), v.astype(np.uint32), w, n)
    return n, ptr, idx, data

def test_codecs():
    x = np.arange(-8, 8, dtype=np.int8).repeat(3)
    assert np.array_equal(Q.unpack_int4(Q.pack_int4(x), len(x)), x)
    t = np.array([0, 1, -1, 1, -1, -1, 0, 1, 1], dtype=np.int8)
    assert np.array_equal(Q.unpack_ternary(Q.pack_ternary(t), len(t)), t)

def test_engines_match_reference():
    lib = load_lib("build")
    n, ptr, idx, data = fixture()
    stim = np.arange(20, dtype=np.uint32); ticks = 400
    q4, _ = Q.quantize_int4(data)
    r = simulate(lib, "int4", ptr, idx, Q.pack_int4(q4), stim, ticks)
    c, tot = ref_run(ptr, idx, q4.astype(np.int64) * Q.W_SCALE_Q16, n, stim, ticks)
    assert r["total_spikes"] > 0
    assert np.array_equal(r["counts"], c) and np.array_equal(r["per_neuron"], tot), "int4 mismatch"
    for theta, gain in [(1, "unit"), (3, "twn"), (5, "drive")]:
        p2, i2, t, lut, _ = Q.ternarize(ptr, idx, data, q4, theta, gain)
        r = simulate(lib, "ternary", p2, i2, Q.pack_ternary(t), stim, ticks, lut=lut)
        wv = np.where(t > 0, lut[1], np.where(t < 0, lut[2], 0)).astype(np.int64)
        c, tot = ref_run(p2, i2, wv, n, stim, ticks)
        assert np.array_equal(r["counts"], c) and np.array_equal(r["per_neuron"], tot), (theta, gain)

def test_theta1_equals_sign_model():
    n, ptr, idx, data = fixture(seed=7)
    q4, _ = Q.quantize_int4(data)
    p2, i2, t, lut, _ = Q.ternarize(ptr, idx, data, q4, 1, "unit")
    assert np.array_equal(p2, ptr) and np.array_equal(i2, idx) and np.array_equal(t, np.sign(data))
    assert lut[1] == Q.W_SCALE_Q16 == 18022 and lut[2] == -18022

if __name__ == "__main__":
    test_codecs(); test_theta1_equals_sign_model(); test_engines_match_reference(); print("ALL PASS")
