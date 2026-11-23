#!/usr/bin/env python3
"""Deadzone-threshold x gain sweep: find the ternary config that matches INT4.

Usage:  python scripts/threshold_sweep.py --data-dir data [--quick] [--out results]
Selection rule (override with flags): spike ratio within +/-tol of INT4,
top-10% Jaccard >= --min-jaccard, then fastest wall time wins.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyq4 import quant as Q
from flyq4.data import load_flywire, stimulus_indices
from flyq4.runtime import load_lib, simulate
from flyq4.metrics import compare

def best_of(lib, reps, *a, **k):
    runs = [simulate(lib, *a, **k) for _ in range(reps)]
    r = min(runs, key=lambda x: x["wall"]); r["wall_all"] = [x["wall"] for x in runs]
    return r

def pick_winner(rows, tol, min_j, metric):
    """Fastest config whose total spikes are within +/-tol of INT4 and whose Jaccard metric >= min_j."""
    ok = [r for r in rows if abs(r["spike_ratio"] - 1) <= tol + 1e-9 and r[metric] >= min_j]
    return min(ok, key=lambda r: r["wall"]) if ok else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ticks", type=int, default=10_000)
    ap.add_argument("--reps", type=int, default=3, help="timing repeats (best-of)")
    ap.add_argument("--thetas", type=float, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    ap.add_argument("--gains", nargs="+", default=list(Q.GAIN_MODES))
    ap.add_argument("--stimulus", choices=["first150", "gr5a"], default="first150")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--min-jaccard", type=float, default=0.75)
    ap.add_argument("--jaccard-metric", default="final_snapshot_jaccard",
                    choices=["final_snapshot_jaccard", "ever_fired_jaccard", "top10_jaccard"],
                    help="final_snapshot is the A/B notebook's metric (weak: it saturates ~0.79); "
                         "top10_jaccard and rate_spearman are much stricter")
    ap.add_argument("--reselect", help="re-pick the winner from an existing sweep_results.json and exit")
    ap.add_argument("--quick", action="store_true", help="2000 ticks, 1 rep, 4 thetas")
    a = ap.parse_args()
    if a.quick:
        a.ticks, a.reps, a.thetas = 2000, 1, [1, 3, 6, 12]

    if a.reselect:
        d = json.load(open(a.reselect))
        w = pick_winner(d["rows"], a.tol, a.min_jaccard, a.jaccard_metric)
        d["winner"] = w; d["criteria"] = dict(tol=a.tol, min_jaccard=a.min_jaccard, metric=a.jaccard_metric)
        json.dump(d, open(a.reselect, "w"), indent=2, default=float)
        print("WINNER:", w["cfg"] if w else "none"); return 0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); d = load_flywire(a.data_dir); n = len(d["root_ids"])
    ptr, idx, data = Q.build_csr(d["u"], d["v"], d["w"], n)
    q4, div = Q.quantize_int4(data)
    stim = stimulus_indices(d["root_ids"], a.stimulus)
    print(f"{d['source']}: {n:,} neurons, {len(data):,} edges, quant_div={div}, "
          f"stimulus={a.stimulus} ({len(stim)} neurons), load {time.time()-t0:.1f}s", flush=True)
    lib = load_lib(str(out / "build"))

    ref = best_of(lib, a.reps, "int4", ptr, idx, Q.pack_int4(q4), stim, a.ticks)
    base = dict(cfg="INT4 baseline", theta=None, gain=None, edges=len(data), wall=ref["wall"],
                total_spikes=ref["total_spikes"], edge_ops=ref["edge_ops"], spike_ratio=1.0)
    print(f"INT4: {ref['total_spikes']:,} spikes, {ref['edge_ops']:,} edge-ops, "
          f"{ref['wall']:.3f}s (reps {['%.3f' % w for w in ref['wall_all']]})", flush=True)

    rows = []
    for gain in a.gains:
        for th in a.thetas:
            p2, i2, t, lut, (ae, ai) = Q.ternarize(ptr, idx, data, q4, th, gain)
            r = best_of(lib, a.reps, "ternary", p2, i2, Q.pack_ternary(t), stim, a.ticks, lut=lut)
            m = compare(ref, r)
            row = dict(cfg=f"theta={th:g},{gain}", theta=th, gain=gain, a_exc=ae, a_inh=ai,
                       edges=len(t), edge_frac=len(t) / len(data), wall=r["wall"],
                       speedup_vs_int4=ref["wall"] / r["wall"], total_spikes=r["total_spikes"],
                       edge_ops=r["edge_ops"], edge_ops_ratio=r["edge_ops"] / max(1, ref["edge_ops"]), **m)
            rows.append(row)
            print(f"{row['cfg']:<18} edges={row['edge_frac']:6.1%} spikes x{row['spike_ratio']:5.2f} "
                  f"ops x{row['edge_ops_ratio']:5.2f} wall={row['wall']:6.3f}s ({row['speedup_vs_int4']:4.2f}x) "
                  f"rho={row['rate_spearman']:5.2f} top10J={row['top10_jaccard']:4.2f} "
                  f"r10ms={row['tick_corr_10ms']:5.2f}", flush=True)

    best = pick_winner(rows, a.tol, a.min_jaccard, a.jaccard_metric)
    summary = dict(baseline=base, rows=rows, criteria=dict(tol=a.tol, min_jaccard=a.min_jaccard, metric=a.jaccard_metric),
                   winner=best, ticks=a.ticks, stimulus=a.stimulus, source=d["source"], neurons=n)
    (out / "sweep_results.json").write_text(json.dumps(summary, indent=2, default=float))
    import csv
    with open(out / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nWINNER:", best["cfg"] if best else "none met criteria (see sweep_results.csv)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
