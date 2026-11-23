/*
 * FlyQ2-ESP: ternary connectome engine for ESP32 / ESP32-S3 (also builds on a host).
 *
 * Differences from the host engine (engine.c), none of which change results:
 *   - 32-bit integer math only. The leak product |v_rest - v| * alpha_m is bounded by
 *     38 mV * 65536 * 327 = 8.1e8 < 2^31, so the host's int64 path is exactly reproduced
 *     without 64-bit multiplies (slow on Xtensa).
 *   - uint16 destination indices (N <= 65535), uint16 spike queue.
 *   - Weights are 2-bit codes, 4 per byte, decoded one byte (4 edges) at a time.
 *   - Per-polarity PSP lookup table lives in a 4-word local array (registers / DRAM).
 *   - Hot functions are placed in IRAM when FLYQ_IRAM is defined (ESP-IDF build does).
 *
 * Bit-exactness vs engine.c::run_ternary is enforced by tests/test_esp_port.py.
 */
#ifndef FLYQ_H
#define FLYQ_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t        n;             /* neurons, <= 65535 */
    const uint32_t *indptr;        /* n+1 */
    const uint16_t *indices;       /* nnz */
    const uint8_t  *weights;       /* ceil(nnz/4) bytes, codes 00=0 01=+ 10=- 11=0 */
    int32_t         psp[4];        /* Q16.16 post-synaptic contribution per code */
    int32_t         v_rest, v_thresh, alpha_m;
    uint32_t        g_shift;
    uint8_t         refrac_ticks;
    int32_t        *V, *G;         /* n each */
    uint8_t        *refrac;        /* n */
    uint16_t       *queue;         /* n */
    uint32_t        spike_count;
    uint32_t        edge_ops;      /* wraps at 2^32; diagnostic only */
} flyq_t;

/* Run `steps` ticks. I (optional, n entries, Q16.16) is added to membrane input.
 * counts_out (optional, steps entries) receives spikes per tick. Returns total spikes. */
uint32_t flyq_run(flyq_t *net, uint32_t steps, const int32_t *I, uint32_t *counts_out);

/* Reset dynamic state to rest and seed the spike queue with `seed_n` neurons. */
void flyq_reset(flyq_t *net, const uint16_t *seed, uint32_t seed_n, int32_t v_rest);

/* Bytes of RAM the dynamic state needs for n neurons (V,G,refrac,queue). */
static inline size_t flyq_state_bytes(uint32_t n) { return (size_t)n * (4 + 4 + 1 + 2); }

#ifdef __cplusplus
}
#endif
#endif
