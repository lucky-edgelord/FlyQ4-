# ESP32 port (FlyQ2-ESP)

`esp32/components/flyq`: portable C99 ternary engine, ESP-IDF component. `esp32/main`: demo app that runs an exported subnetwork and prints µs/tick.

## Optimizations (all bit-exact with `engine.c::run_ternary`)

| Change | Why |
|---|---|
| 32-bit maths only | `\|v_rest−V\|·alpha_m ≤ 8.1e8 < 2³¹`; Xtensa has no fast 64-bit multiply |
| `uint16` indices + `uint16` spike queue | halves index memory; subnetworks ≤ 65,535 neurons |
| 2-bit weights, one byte = 4 edges | one flash/RAM read feeds four updates; unrolled loop |
| 4-entry PSP LUT in locals | no multiply, no sign extension; θ/gain cost nothing at run time |
| `IRAM_ATTR` hot loops | no instruction-cache misses in the two kernels |
| Graph copied to RAM at boot if it fits | flash reads go through a small cache |
| Optional `FLYQ_DEFERRED_CLAMP` | clamp G once per tick instead of per edge (not bit-exact if a neuron saturates mid-tick) |

## Use

```sh
python tools/export_esp32.py --data-dir data --n 4096 --theta 3 --gain drive   # verifies bit-exactness, writes esp32/main/flyq_blob.{h,c}
cd esp32 && idf.py set-target esp32s3 && idf.py build flash monitor
```

Exported default (N=4096, θ=3, drive): 75,746 edges · 148 KiB indices + 18 KiB weights + 16 KiB indptr · 60 KiB RAM state.

## Status

* Verified on host: the port's C core (compiled with `gcc -std=c99 -pedantic`) matches the reference engine tick for tick; see `tests/test_esp_port.py`.
* **Not verified:** the ESP-IDF build and on-device speed. No toolchain or hardware here, so µs/tick is unmeasured. The real-time budget is 100 µs per 0.1 ms tick.
* The whole brain does not fit: at θ=3 it keeps ≈4.9 M edges (≳15 MB of indices with 3-byte ids) plus ≈1.8 MB of state. Estimate, not a build result.
