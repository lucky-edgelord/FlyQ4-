"""Weight codecs: INT4 (FlyQ4) and deadzone-ternary with per-polarity gain (FlyQ2)."""
import numpy as np
from scipy.sparse import csr_matrix

W_SCALE_Q16 = round(0.275 * 65536)
V_REST_Q16 = round(-52.0 * 65536)
V_TH_Q16 = round(-45.0 * 65536)
ALPHA_M_Q16 = round((1.0 - np.exp(-0.1 / 20.0)) * 65536)
G_SHIFT, REFRAC_T = 6, 22

def build_csr(u, v, w, n):
    m = csr_matrix((w, (u, v)), shape=(n, n), dtype=np.int32)
    m.sum_duplicates(); m.eliminate_zeros(); m.sort_indices()
    return m.indptr.astype(np.uint32), m.indices.astype(np.uint32), m.data.astype(np.int32)

def quantize_int4(data):
    """Original FlyQ4 rule: 1 INT4 step ~ p95/6 synapses, never round a real edge to 0."""
    div = max(1.0, float(np.percentile(np.abs(data), 95)) / 6.0)
    q = np.round(data / div).astype(np.int32)
    q = np.where((q == 0) & (data > 0), 1, q)
    q = np.where((q == 0) & (data < 0), -1, q)
    return np.clip(q, -8, 7).astype(np.int8), div

def pack_int4(vals):
    vals = np.asarray(vals, dtype=np.int8)
    p = np.zeros(((len(vals) + 1) // 2) * 2, dtype=np.uint8)
    p[:len(vals)] = vals.view(np.uint8) & 0x0F
    return (p[0::2] | (p[1::2] << 4)).astype(np.uint8)

def pack_ternary(vals):
    vals = np.asarray(vals, dtype=np.int8)
    enc = np.zeros(len(vals), dtype=np.uint8)
    enc[vals > 0] = 1
    enc[vals < 0] = 2
    p = np.zeros(((len(enc) + 3) // 4) * 4, dtype=np.uint8)
    p[:len(enc)] = enc
    return (p[0::4] | (p[1::4] << 2) | (p[2::4] << 4) | (p[3::4] << 6)).astype(np.uint8)

def unpack_int4(packed, n):
    e = np.arange(n)
    nib = (packed[e // 2] >> (4 * (e & 1))) & 15
    return np.where(nib >= 8, nib.astype(np.int16) - 16, nib).astype(np.int8)

def unpack_ternary(packed, n):
    e = np.arange(n)
    return np.array([0, 1, -1, 0], dtype=np.int8)[(packed[e // 4] >> (2 * (e & 3))) & 3]

GAIN_MODES = ("unit", "twn", "drive")

def ternarize(indptr, indices, data, q4, theta, gain="unit"):
    """Deadzone ternary: drop edges with |w| < theta, keep sign of the rest.

    Edges are physically removed (CSR rebuilt), so weak-synapse pruning also cuts
    the work done per spike. theta=1 keeps every edge (== the original np.sign model).

    gain selects the Q16.16 PSP for each polarity (a_exc, a_inh, in INT4 steps):
      unit  : 1 / 1                  (plain ternary, what the A/B notebook did)
      twn   : mean |q4| of kept edges, per polarity (Ternary-Weight-Networks style)
      drive : total |q4| of ALL edges per polarity / kept edges, so aggregate
              excitatory and inhibitory drive match INT4 even after pruning
    """
    n = len(indptr) - 1
    keep = np.abs(data) >= theta
    rows = np.repeat(np.arange(n, dtype=np.uint32), np.diff(indptr))
    cnt = np.bincount(rows[keep], minlength=n)
    new_ptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.uint32)
    new_idx = indices[keep]
    t = np.sign(data[keep]).astype(np.int8)

    mag = np.abs(q4).astype(np.float64)
    pos_all, neg_all = data > 0, data < 0
    pos_k, neg_k = keep & pos_all, keep & neg_all
    if gain == "unit":
        a_e = a_i = 1.0
    elif gain == "twn":
        a_e = mag[pos_k].mean() if pos_k.any() else 1.0
        a_i = mag[neg_k].mean() if neg_k.any() else 1.0
    elif gain == "drive":
        a_e = mag[pos_all].sum() / max(1, pos_k.sum())
        a_i = mag[neg_all].sum() / max(1, neg_k.sum())
    else:
        raise ValueError(gain)
    lut = np.array([0, round(a_e * W_SCALE_Q16), -round(a_i * W_SCALE_Q16), 0], dtype=np.int32)
    return new_ptr, new_idx, t, lut, (a_e, a_i)
