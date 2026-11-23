#include "flyq.h"

#ifdef FLYQ_IRAM
#include "esp_attr.h"
#define FLYQ_HOT IRAM_ATTR
#else
#define FLYQ_HOT
#endif

#define G_LIMIT ((int32_t)(65536 * 64))

/* Default keeps host-identical saturation after every edge. Defining
 * FLYQ_DEFERRED_CLAMP clamps once per tick in the neuron phase instead (fewer
 * branches in the hottest loop, NOT bit-exact when a neuron saturates mid-tick). */
#ifdef FLYQ_DEFERRED_CLAMP
#define ACC(d, delta) (net->G[(d)] += (delta))
#else
#define ACC(d, delta) do { int32_t x_ = net->G[(d)] + (delta); \
    x_ = x_ > G_LIMIT ? G_LIMIT : x_; x_ = x_ < -G_LIMIT ? -G_LIMIT : x_; \
    net->G[(d)] = x_; } while (0)
#endif

static FLYQ_HOT void propagate(flyq_t *net) {
    const int32_t l0 = net->psp[0], l1 = net->psp[1], l2 = net->psp[2], l3 = net->psp[3];
    const int32_t lut[4] = { l0, l1, l2, l3 };
    const uint16_t *idx = net->indices;
    const uint8_t *w = net->weights;
    uint32_t ops = 0;
    for (uint32_t s = 0; s < net->spike_count; s++) {
        uint32_t u = net->queue[s], e = net->indptr[u], b = net->indptr[u + 1];
        ops += b - e;
        for (; e < b && (e & 3u); e++) {                         /* unaligned head */
            ACC(idx[e], lut[(w[e >> 2] >> ((e & 3u) << 1)) & 3u]);
        }
        for (; e + 4 <= b; e += 4) {                              /* 4 edges / byte */
            uint32_t by = w[e >> 2];
            ACC(idx[e],     lut[by & 3u]);
            ACC(idx[e + 1], lut[(by >> 2) & 3u]);
            ACC(idx[e + 2], lut[(by >> 4) & 3u]);
            ACC(idx[e + 3], lut[by >> 6]);
        }
        for (; e < b; e++) {                                      /* tail */
            ACC(idx[e], lut[(w[e >> 2] >> ((e & 3u) << 1)) & 3u]);
        }
    }
    net->edge_ops += ops;
}

static FLYQ_HOT void neurons(flyq_t *net, const int32_t *I) {
    const uint32_t n = net->n, sh = net->g_shift;
    const int32_t vr = net->v_rest, vt = net->v_thresh, am = net->alpha_m;
    const int32_t vfloor = -65536 * 90;
    uint32_t ns = 0;
    for (uint32_t i = 0; i < n; i++) {
        int32_t g = net->G[i];
#ifdef FLYQ_DEFERRED_CLAMP
        g = g > G_LIMIT ? G_LIMIT : (g < -G_LIMIT ? -G_LIMIT : g);
#endif
        net->G[i] = g - (g >> sh);
        if (net->refrac[i]) { net->refrac[i]--; continue; }
        int32_t v  = net->V[i];
        int32_t dv = ((vr - v) * am) >> 16;          /* int32-exact, see header */
        int32_t vn = v + dv + g + (I ? I[i] : 0);
        if (vn >= vt) {
            net->V[i] = vr;
            net->refrac[i] = net->refrac_ticks;
            net->queue[ns++] = (uint16_t)i;
        } else {
            net->V[i] = vn < vfloor ? vfloor : vn;
        }
    }
    net->spike_count = ns;
}

FLYQ_HOT uint32_t flyq_run(flyq_t *net, uint32_t steps, const int32_t *I, uint32_t *counts_out) {
    uint32_t total = 0;
    for (uint32_t t = 0; t < steps; t++) {
        propagate(net);
        neurons(net, I);
        if (counts_out) counts_out[t] = net->spike_count;
        total += net->spike_count;
    }
    return total;
}

void flyq_reset(flyq_t *net, const uint16_t *seed, uint32_t seed_n, int32_t v_rest) {
    for (uint32_t i = 0; i < net->n; i++) { net->V[i] = v_rest; net->G[i] = 0; net->refrac[i] = 0; }
    for (uint32_t i = 0; i < seed_n; i++) net->queue[i] = seed[i];
    net->spike_count = seed_n;
    net->edge_ops = 0;
}
