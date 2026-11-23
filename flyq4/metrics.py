"""Fidelity metrics between a reference (INT4) run and a candidate run."""
import numpy as np
from scipy.stats import spearmanr
from . import quant as Q

def _corr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")

def _binned(c, k):
    m = len(c) // k * k
    return c[:m].reshape(-1, k).sum(1)

def _jac(a, b):
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else float("nan")

def compare(ref, cand, top_frac=0.10):
    r, c = ref["per_neuron"], cand["per_neuron"]
    k = max(1, int(len(r) * top_frac))
    top_r = np.zeros(len(r), bool); top_r[np.argsort(-r.astype(np.int64), kind="stable")[:k]] = True
    top_c = np.zeros(len(c), bool); top_c[np.argsort(-c.astype(np.int64), kind="stable")[:k]] = True
    rho = spearmanr(r, c).statistic if r.std() > 0 and c.std() > 0 else float("nan")
    return dict(
        spike_ratio=cand["total_spikes"] / max(1, ref["total_spikes"]),
        rate_pearson=_corr(r, c), rate_spearman=float(rho),
        ever_fired_jaccard=_jac(r > 0, c > 0),
        top10_jaccard=_jac(top_r, top_c),
        final_snapshot_jaccard=_jac(ref["V"] != Q.V_REST_Q16, cand["V"] != Q.V_REST_Q16),
        tick_corr_0p1ms=_corr(ref["counts"], cand["counts"]),
        tick_corr_1ms=_corr(_binned(ref["counts"], 10), _binned(cand["counts"], 10)),
        tick_corr_10ms=_corr(_binned(ref["counts"], 100), _binned(cand["counts"], 100)),
        tick_corr_50ms=_corr(_binned(ref["counts"], 500), _binned(cand["counts"], 500)),
    )
