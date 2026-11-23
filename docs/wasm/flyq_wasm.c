/* FlyQ in WebAssembly: freestanding C (no libc), same neuron model as engine.c.
 * Build: clang --target=wasm32 -O3 -nostdlib -Wl,--no-entry -Wl,--export-dynamic -o flyq.wasm flyq_wasm.c
 * 32-bit integer math only (see esp32/components/flyq/include/flyq.h for the overflow bound). */
typedef unsigned char u8; typedef unsigned short u16; typedef unsigned int u32; typedef int i32;
#define EXPORT __attribute__((visibility("default")))
#define G_LIMIT (65536 * 64)
#define V_FLOOR (-65536 * 90)

typedef struct {
    u32 n, mode, g_shift, refrac_ticks, spike_count, edge_ops;
    i32 v_rest, v_thresh, alpha_m, w_scale;
    i32 lut[4];
    const u32 *indptr; const u16 *indices; const u8 *weights;
    i32 *V, *G, *I; u8 *refrac; u16 *queue; u32 *spike_total;
} Net;

extern u8 __heap_base;
static u32 heap_ptr;

EXPORT u32 alloc(u32 bytes) {
    if (!heap_ptr) heap_ptr = (u32)&__heap_base;
    heap_ptr = (heap_ptr + 15u) & ~15u;
    u32 p = heap_ptr; heap_ptr += bytes;
    u32 need = (heap_ptr + 65535u) >> 16, have = __builtin_wasm_memory_size(0);
    if (need > have && __builtin_wasm_memory_grow(0, need - have) == (u32)-1) return 0;
    return p;
}

EXPORT Net *net_create(u32 n, i32 v_rest, i32 v_thresh, i32 alpha_m, i32 w_scale, u32 g_shift, u32 refrac_ticks) {
    Net *s = (Net *)alloc(sizeof(Net));
    s->n = n; s->v_rest = v_rest; s->v_thresh = v_thresh; s->alpha_m = alpha_m;
    s->w_scale = w_scale; s->g_shift = g_shift; s->refrac_ticks = refrac_ticks;
    s->V = (i32 *)alloc(4 * n); s->G = (i32 *)alloc(4 * n); s->I = (i32 *)alloc(4 * n);
    s->refrac = (u8 *)alloc(n); s->queue = (u16 *)alloc(2 * n); s->spike_total = (u32 *)alloc(4 * n);
    return s;
}
EXPORT void net_bind(Net *s, u32 mode, const u32 *indptr, const u16 *indices, const u8 *weights) {
    s->mode = mode; s->indptr = indptr; s->indices = indices; s->weights = weights;
}
EXPORT void net_set_lut(Net *s, i32 l0, i32 l1, i32 l2, i32 l3) { s->lut[0] = l0; s->lut[1] = l1; s->lut[2] = l2; s->lut[3] = l3; }
EXPORT void net_set_stim(Net *s, u32 i, i32 current) { s->I[i] = current; }
EXPORT void net_reset(Net *s, u32 seed_n) {
    for (u32 i = 0; i < s->n; i++) { s->V[i] = s->v_rest; s->G[i] = 0; s->refrac[i] = 0; s->spike_total[i] = 0; }
    for (u32 i = 0; i < seed_n; i++) s->queue[i] = (u16)i;
    s->spike_count = seed_n; s->edge_ops = 0;
}
EXPORT i32 *net_V(Net *s) { return s->V; }
EXPORT i32 *net_G(Net *s) { return s->G; }
EXPORT u32 *net_totals(Net *s) { return s->spike_total; }
EXPORT u32 net_take_edge_ops(Net *s) { u32 e = s->edge_ops; s->edge_ops = 0; return e; }

static inline void acc(Net *s, u32 d, i32 delta) {
    i32 x = s->G[d] + delta;
    s->G[d] = x > G_LIMIT ? G_LIMIT : (x < -G_LIMIT ? -G_LIMIT : x);
}

/* Run `steps` ticks. Per-tick spike counts -> counts[t]. Spike events -> rec as (t<<16 | neuron),
 * up to rec_cap entries. Returns number of events written. */
EXPORT u32 net_run(Net *s, u32 steps, u32 *counts, u32 *rec, u32 rec_cap) {
    u32 nrec = 0;
    const i32 lut[4] = { s->lut[0], s->lut[1], s->lut[2], s->lut[3] };
    for (u32 t = 0; t < steps; t++) {
        for (u32 k = 0; k < s->spike_count; k++) {
            u32 u = s->queue[k], a = s->indptr[u], b = s->indptr[u + 1];
            s->edge_ops += b - a;
            if (s->mode == 1) {
                for (u32 e = a; e < b; e++)
                    acc(s, s->indices[e], lut[(s->weights[e >> 2] >> ((e & 3u) << 1)) & 3u]);
            } else {
                for (u32 e = a; e < b; e++) {
                    u32 by = s->weights[e >> 1], nib = (e & 1u) ? (by >> 4) : (by & 15u);
                    i32 w = (i32)nib - ((nib & 8u) ? 16 : 0);
                    acc(s, s->indices[e], w * s->w_scale);
                }
            }
        }
        u32 ns = 0;
        for (u32 i = 0; i < s->n; i++) {
            i32 g = s->G[i]; s->G[i] = g - (g >> s->g_shift);
            if (s->refrac[i]) { s->refrac[i]--; continue; }
            i32 v = s->V[i];
            i32 dv = ((s->v_rest - v) * s->alpha_m) >> 16;
            i32 vn = v + dv + g + s->I[i];
            if (vn >= s->v_thresh) {
                s->V[i] = s->v_rest; s->refrac[i] = (u8)s->refrac_ticks; s->queue[ns++] = (u16)i;
                s->spike_total[i]++;
                if (rec && nrec < rec_cap) rec[nrec++] = (t << 16) | i;
            } else s->V[i] = vn < V_FLOOR ? V_FLOOR : vn;
        }
        s->spike_count = ns;
        if (counts) counts[t] = ns;
    }
    return nrec;
}
