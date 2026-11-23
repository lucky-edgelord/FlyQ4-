/* FlyQ2-ESP demo: run the exported subnetwork and report us/tick vs the 100 us real-time budget. */
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "flyq.h"
#include "flyq_blob.h"

static const char *TAG = "flyq";

static void *alloc_fast(size_t bytes, bool *internal) {
    void *p = heap_caps_calloc(1, bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    *internal = p != NULL;
    return p ? p : heap_caps_calloc(1, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

/* Copy a flash-resident (rodata) table into RAM if it fits: flash reads go through a
 * small cache and are the slowest access in the hot loop. */
static const void *to_ram(const void *src, size_t bytes, bool *internal) {
    void *p = alloc_fast(bytes, internal);
    if (!p) return src;
    memcpy(p, src, bytes);
    return p;
}

void app_main(void) {
    bool in_state, in_g1, in_g2, in_g3;
    flyq_t net = {
        .n = FLYQ_N, .psp = { FLYQ_PSP0, FLYQ_PSP1, FLYQ_PSP2, FLYQ_PSP3 },
        .v_rest = FLYQ_V_REST, .v_thresh = FLYQ_V_THRESH, .alpha_m = FLYQ_ALPHA_M,
        .g_shift = FLYQ_G_SHIFT, .refrac_ticks = FLYQ_REFRAC,
    };
    net.indptr  = to_ram(flyq_indptr,  sizeof flyq_indptr,  &in_g1);
    net.indices = to_ram(flyq_indices, sizeof flyq_indices, &in_g2);
    net.weights = to_ram(flyq_weights, sizeof flyq_weights, &in_g3);
    net.V = alloc_fast(FLYQ_N * 4, &in_state);
    net.G = alloc_fast(FLYQ_N * 4, &in_state);
    net.refrac = alloc_fast(FLYQ_N, &in_state);
    net.queue = alloc_fast(FLYQ_N * 2, &in_state);
    int32_t *I = alloc_fast(FLYQ_N * 4, &in_state);
    if (!net.V || !net.G || !net.refrac || !net.queue || !I) { ESP_LOGE(TAG, "out of memory"); return; }
    for (uint32_t i = 0; i < FLYQ_STIM_N; i++) I[flyq_stim[i]] = FLYQ_STIM_Q16;

    ESP_LOGI(TAG, "N=%d edges=%d graph: indptr %s, indices %s, weights %s; free internal %u B, psram %u B",
             FLYQ_N, FLYQ_NNZ, in_g1 ? "RAM" : "flash/PSRAM", in_g2 ? "RAM" : "flash/PSRAM",
             in_g3 ? "RAM" : "flash/PSRAM", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    const uint32_t TICKS = 2000;
    flyq_reset(&net, flyq_stim, FLYQ_STIM_N, FLYQ_V_REST);
    int64_t t0 = esp_timer_get_time();
    uint32_t spikes = flyq_run(&net, TICKS, I, NULL);
    int64_t us = esp_timer_get_time() - t0;
    ESP_LOGI(TAG, "%u ticks: %u spikes, %.1f us/tick (budget 100 us => %.2fx real-time), %u edge-ops",
             (unsigned)TICKS, (unsigned)spikes, (double)us / TICKS, 100.0 * TICKS / (double)us,
             (unsigned)net.edge_ops);
    ESP_LOGI(TAG, "host reference for this blob: %u spikes in %d ticks (must match exactly)",
             (unsigned)FLYQ_REF_SPIKES, FLYQ_REF_TICKS);
}
