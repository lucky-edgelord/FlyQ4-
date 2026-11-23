"""ctypes wrapper around engine.c. One FlyNet struct, two kernels."""
import ctypes, subprocess, time
from pathlib import Path
import numpy as np
from . import quant as Q

HERE = Path(__file__).parent

class FlyNet(ctypes.Structure):
    _fields_ = [
        ("N", ctypes.c_uint32), ("w_scale", ctypes.c_int32), ("v_rest", ctypes.c_int32),
        ("v_thresh", ctypes.c_int32), ("alpha_m", ctypes.c_int32),
        ("g_shift", ctypes.c_uint32), ("refrac_ticks", ctypes.c_uint32),
        ("psp_lut", ctypes.c_int32 * 4),
        ("indptr", ctypes.POINTER(ctypes.c_uint32)), ("indices", ctypes.POINTER(ctypes.c_uint32)),
        ("weights", ctypes.POINTER(ctypes.c_uint8)),
        ("V", ctypes.POINTER(ctypes.c_int32)), ("G", ctypes.POINTER(ctypes.c_int32)),
        ("refrac", ctypes.POINTER(ctypes.c_uint8)), ("queue", ctypes.POINTER(ctypes.c_uint32)),
        ("spike_total", ctypes.POINTER(ctypes.c_uint32)),
        ("edge_ops", ctypes.c_uint64), ("spike_count", ctypes.c_uint32),
    ]

def compile_engine(out_dir="build"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    so = out / "libflyq.so"
    subprocess.run(["gcc", "-O3", "-march=native", "-Wall", "-Wextra", "-shared", "-fPIC",
                    str(HERE / "engine.c"), "-o", str(so)], check=True)
    return so

_P = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))

def simulate(lib, kind, indptr, indices, packed, stim_idx, ticks=10_000, stim_mv=0.55,
             lut=None, seed_spikes=True):
    """Fresh-state run. Returns dict with counts, per-neuron totals, timing, edge_ops."""
    n = len(indptr) - 1
    V = np.full(n, Q.V_REST_Q16, dtype=np.int32)
    G = np.zeros(n, dtype=np.int32)
    R = np.zeros(n, dtype=np.uint8)
    queue = np.zeros(n, dtype=np.uint32)
    queue[:len(stim_idx)] = stim_idx
    totals = np.zeros(n, dtype=np.uint32)
    I = np.zeros(n, dtype=np.int32)
    I[stim_idx] = round(stim_mv * 65536)
    counts = np.zeros(ticks, dtype=np.uint32)
    lut_arr = (ctypes.c_int32 * 4)(*(lut if lut is not None else [0, 0, 0, 0]))
    st = FlyNet(n, Q.W_SCALE_Q16, Q.V_REST_Q16, Q.V_TH_Q16, Q.ALPHA_M_Q16, Q.G_SHIFT, Q.REFRAC_T,
                lut_arr, _P(indptr, ctypes.c_uint32), _P(indices, ctypes.c_uint32),
                _P(packed, ctypes.c_uint8), _P(V, ctypes.c_int32), _P(G, ctypes.c_int32),
                _P(R, ctypes.c_uint8), _P(queue, ctypes.c_uint32), _P(totals, ctypes.c_uint32),
                0, len(stim_idx) if seed_spikes else 0)
    fn = lib.run_int4 if kind == "int4" else lib.run_ternary
    fn.argtypes = [ctypes.POINTER(FlyNet), ctypes.c_uint32,
                   ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32)]
    fn.restype = ctypes.c_uint64
    t0 = time.perf_counter()
    total = fn(ctypes.byref(st), ticks, _P(I, ctypes.c_int32), _P(counts, ctypes.c_uint32))
    wall = time.perf_counter() - t0
    return dict(wall=wall, total_spikes=int(total), counts=counts, per_neuron=totals,
                edge_ops=int(st.edge_ops), V=V, G=G, ticks=ticks)

def load_lib(out_dir="build"):
    return ctypes.CDLL(str(compile_engine(out_dir)))
