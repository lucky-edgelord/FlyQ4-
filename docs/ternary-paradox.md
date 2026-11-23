# The ternary paradox, explained and fixed

**Symptom** (full FlyWire v783 brain, 10,000 ticks = 1 s): the ternary engine fired 17.8 M spikes against INT4's 14.4 M, ran *slower*, and its per-tick spike counts correlated at −0.51 with INT4.

**Reproduced** with `scripts/threshold_sweep.py`: INT4 = 14,487,987 spikes; ternary θ=1 = ×1.23, 0.84× speed.

## What was actually wrong

The A/B notebook built the ternary model as `np.sign(w)`: there was **no threshold**, so `quant_div = 2.0` never touched it (it only scales INT4). Every one of the 15,091,983 edges became ±1. Compared with INT4 (counts ÷ 2, up to ±7) that:

1. leaves weak edges alone (|w|=1 → 1 in both) but flattens strong ones (|w|=40 → 7 in INT4, 1 in ternary);
2. so strong **inhibitory** edges lose most of their strength → the network over-fires;
3. a tick costs ∝ spikes × out-degree (`edge_ops`), so ~23% more spikes (×1.31 edge-ops) cancelled the cheaper 2-bit decode.

## The fix

`flyq4.quant.ternarize(theta, gain)`:

* **Deadzone θ**: drop edges with |count| < θ and rebuild the CSR, so the graph (and per-spike work) actually shrinks.
* **Per-polarity gain** in the 4-entry LUT (`[0, +aE, −aI, 0]`): free at run time.
  * `unit` 1/1 · `twn` mean |q4| of kept edges · `drive` total |q4| per polarity ÷ kept edges (restores aggregate E and I drive).

## Results (real data, 2 timing repeats, best-of)

| config | edges kept | spikes ×INT4 | edge-ops × | speed-up | rate ρ | snapshot J |
|---|---|---|---|---|---|---|
| θ=1 unit (the bug) | 100% | 1.23 | 1.31 | 0.84× | 0.71 | 0.79 |
| θ=2 unit | 50% | 0.98 | 0.55 | 1.29× | 0.69 | 0.80 |
| **θ=3 drive** | 33% | **0.98** | 0.37 | **1.45×** | 0.63 | 0.78 |
| θ=6 drive | 14% | 0.78 | 0.15 | 1.90× | 0.43 | 0.69 |
| θ=12 drive | 5% | 0.46 | 0.05 | 2.54× | 0.27 | 0.39 |

Full table: `results/sweep_results.csv` / `docs/assets/sweep.json`. Winner rule: spikes within ±5% of INT4, snapshot Jaccard ≥ 0.75, then fastest → **θ=3, drive**.

## Caveats (read these)

* **Fidelity is moderate, not high.** Per-neuron firing-rate Spearman ρ is ≈0.63–0.71 at best and falls as θ rises. Ternary preserves total activity and wiring, not individual neuron rates. θ=2 unit is the most faithful config that also matches spike count.
* The notebook's "final-snapshot Jaccard" (neurons ≠ rest at the last tick) sits near 0.79 for θ=1…3 regardless of gain: **a weak metric**. Use `rate_spearman` and `top10_jaccard`.
* Per-tick correlation is phase-sensitive: at θ=1 it was −0.51 with plain gain but +0.77 once gains were calibrated, so the "rhythm problem" was a gain problem. Binned correlations at 10/50 ms vary widely between configs; do not over-read them.
* Timings are from a shared 2-core sandbox (best of 2). The deterministic `edge_ops` ratio is the number to trust.
* Stimulus is "first 150 neurons" (as in the A/B notebook), not the real Gr5a sugar neurons.
