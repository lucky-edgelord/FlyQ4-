/*
 * FlyQ engine: one neuron model, two synaptic weight codecs.
 *
 *   run_int4    : signed INT4 weights [-8,+7], 2 per byte, w * w_scale
 *   run_ternary : 2-bit codes {00:0, 01:+, 10:-, 11:0}, 4 per byte,
 *                 PSP looked up from psp_lut[4] (Q16.16, set at runtime)
 *
 * Both share neuron_phase() so any A/B difference comes from the synapses only.
 * Extra instrumentation (cheap): per-neuron spike totals and a count of
 * synaptic updates ("edge_ops"), which is the hardware-independent cost metric.
 */
#include <stdint.h>

typedef struct {
    uint32_t N;
    int32_t  w_scale, v_rest, v_thresh, alpha_m;
    uint32_t g_shift;
    uint32_t refrac_ticks;
    int32_t  psp_lut[4];
    const uint32_t *indptr, *indices;
    const uint8_t  *weights;
    int32_t  *V, *G;
    uint8_t  *refrac;
    uint32_t *queue;
    uint32_t *spike_total;   /* per-neuron spike counter, may be NULL */
    uint64_t  edge_ops;      /* synaptic updates performed */
    uint32_t  spike_count;
} FlyNet;

#define G_LIMIT ((int32_t)(65536 * 64))
#define V_FLOOR ((int32_t)(-65536 * 90))

static inline int32_t clamp_i32(int64_t x, int32_t lo, int32_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return (int32_t)x;
}

static inline void neuron_phase(FlyNet *n, const int32_t *I) {
    uint32_t ns = 0;
    for (uint32_t i = 0; i < n->N; i++) {
        int32_t g = n->G[i];
        n->G[i] = g - (g >> n->g_shift);
        if (n->refrac[i] > 0) { n->refrac[i]--; continue; }
        int32_t v  = n->V[i];
        int32_t dv = (int32_t)(((int64_t)(n->v_rest - v) * n->alpha_m) >> 16);
        int64_t vn = (int64_t)v + dv + g + (I ? I[i] : 0);
        if (vn >= n->v_thresh) {
            n->V[i] = n->v_rest;
            n->refrac[i] = (uint8_t)n->refrac_ticks;
            n->queue[ns++] = i;
            if (n->spike_total) n->spike_total[i]++;
        } else {
            n->V[i] = clamp_i32(vn, V_FLOOR, n->v_thresh);
        }
    }
    n->spike_count = ns;
}

static void step_int4(FlyNet *n, const int32_t *I) {
    for (uint32_t s = 0; s < n->spike_count; s++) {
        uint32_t u = n->queue[s], a = n->indptr[u], b = n->indptr[u + 1];
        n->edge_ops += (b - a);
        for (uint32_t e = a; e < b; e++) {
            uint8_t by = n->weights[e >> 1];
            int8_t w = (e & 1) ? ((int8_t)(by & 0xF0) >> 4)
                               : ((int8_t)((by & 0x0F) << 4) >> 4);
            uint32_t d = n->indices[e];
            n->G[d] = clamp_i32((int64_t)n->G[d] + (int64_t)w * n->w_scale,
                                -G_LIMIT, G_LIMIT);
        }
    }
    neuron_phase(n, I);
}

static void step_ternary(FlyNet *n, const int32_t *I) {
    const int32_t l0 = n->psp_lut[0], l1 = n->psp_lut[1],
                  l2 = n->psp_lut[2], l3 = n->psp_lut[3];
    const int32_t lut[4] = { l0, l1, l2, l3 };
    for (uint32_t s = 0; s < n->spike_count; s++) {
        uint32_t u = n->queue[s], a = n->indptr[u], b = n->indptr[u + 1];
        n->edge_ops += (b - a);
        for (uint32_t e = a; e < b; e++) {
            uint32_t bits = (n->weights[e >> 2] >> ((e & 3u) << 1)) & 3u;
            uint32_t d = n->indices[e];
            n->G[d] = clamp_i32((int64_t)n->G[d] + lut[bits], -G_LIMIT, G_LIMIT);
        }
    }
    neuron_phase(n, I);
}

uint64_t run_int4(FlyNet *n, uint32_t steps, const int32_t *I, uint32_t *counts) {
    uint64_t total = 0;
    for (uint32_t t = 0; t < steps; t++) {
        step_int4(n, I);
        if (counts) counts[t] = n->spike_count;
        total += n->spike_count;
    }
    return total;
}

uint64_t run_ternary(FlyNet *n, uint32_t steps, const int32_t *I, uint32_t *counts) {
    uint64_t total = 0;
    for (uint32_t t = 0; t < steps; t++) {
        step_ternary(n, I);
        if (counts) counts[t] = n->spike_count;
        total += n->spike_count;
    }
    return total;
}
