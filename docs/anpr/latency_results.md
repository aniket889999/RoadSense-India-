# ANPR validation-only latency measurements

## Scope and method

This measures the existing local image service with the launcher's selected
RapidOCR configuration: the pinned plate detector, `resized_bgr` crops, CPU
PP-OCRv4 and review-only schema 1.2 assessment. It changes no detector/OCR
parameters, normalization, candidate-cleanup policy or acceptance threshold.
See the [model card](model_card.md) for the separate reading-quality evidence.

Each device run constructed a fresh local runtime, executed **3 warm-up image
calls**, then timed **all 71 prepared validation PNGs once**, sequentially in the
same sorted order. Both runs made **94 measured OCR crop calls**. MPS ran first,
then explicit CPU; neither ran concurrently with the test suite or the other
benchmark. Execution used host PTY processes outside the sandbox, with MPS
availability confirmed. No silent device fallback was used.

The [script](../../scripts/benchmark_anpr_latency.py) calls the existing
`AnprImageService.recognize_bytes()` with timing wrappers around its decoder and
local detector/recognizer. `torch.mps.synchronize()` brackets MPS detection so
queued GPU completion is included in detector time. Pipeline bounds, crop
copying, quality assessment, cleanup and response construction remain intact.

**Warm end-to-end service time** means decode through response construction,
including wrapper/synchronization overhead. It excludes input-file I/O, package
imports, runtime initialization, HTTP transfer/queueing and browser rendering.
OCR time sums all crop-recognition adapter calls per image, including adapter
preprocessing. `Other` is the residual for crop handling, quality/assessment and
instrumentation overhead, not an independently instrumented processing stage.
Per-stage medians/percentiles need not add up to the total's median/percentile.

The script only enumerates `images/val`, checks its count against prepared
metadata, and confines selected files to that directory. It refuses test/train
inputs, traversal, symlink aliases and existing/outside output targets before
model construction. It does not read the all-split transcription manifest,
labels, test images or predictions. This is not a new full-dataset integrity
audit: the report identifies the declared dataset fingerprint and hashes the
actual ordered validation input bytes. No accuracy metric is recomputed.

## Recorded hardware and environment

| Item | Recorded value |
| --- | --- |
| Host | Apple M4, arm64, 17,179,869,184 bytes memory (16 GiB) |
| Platform | `macOS-27.0.1-arm64-arm-64bit` |
| Python | 3.12.7, project `.venv` |
| PyTorch / torchvision | 2.13.0 / 0.28.0 |
| Ultralytics | 8.4.124 |
| OpenCV package / NumPy / Pillow | 5.0.0.93 / 2.5.2 / 12.3.0 |
| RapidOCR / ONNX Runtime | 1.4.4 / 1.32.0 |
| OCR execution | CPU in both runs; review-only |
| MPS report timestamp (UTC) | 2026-11-22T04:21:17.328296+00:00 |
| CPU report timestamp (UTC) | 2026-11-22T04:22:29.780259+00:00 |

Both runs recorded the same package versions, dataset/input identity, detector
checkpoint and three OCR model hashes. Runtime construction, separately excluded
from warm service timing, was 1023.368 ms for MPS and 765.102 ms for CPU; these
are not complete cold-process startup measurements.

## Measurements

All values are **milliseconds per image**, rounded to three decimal places.
Median and p95 use linear interpolation at `(n - 1) * q`, with `n = 71`.

### Detector on MPS, OCR on CPU

| Timed boundary | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Decode | 61.402 | 81.969 | 151.299 |
| Detect, synchronized | 50.673 | 225.519 | 293.289 |
| OCR, sum of crops | 476.221 | 1632.308 | 2276.969 |
| Other, residual | 3.299 | 8.859 | 11.777 |
| **Warm end-to-end service** | **613.228** | **1779.143** | **2393.285** |

### Detector on CPU, OCR on CPU

| Timed boundary | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Decode | 59.692 | 78.137 | 85.175 |
| Detect | 60.814 | 82.114 | 951.115 |
| OCR, sum of crops | 487.652 | 1263.892 | 1989.207 |
| Other, residual | 2.836 | 7.185 | 13.205 |
| **Warm end-to-end service** | **607.799** | **1443.117** | **2125.490** |

OCR dominates the measured median service work in both runs. The detector's
median alone does not establish which full application configuration will be
faster under another workload. Each device was measured in a single ordered
pass on a working desktop, without controlling other host activity, thermals or
native-library thread scheduling. Image/crop content and cold shape paths affect
tails even after warm-up. These are observed warm-service distributions, not a
concurrency/load benchmark, latency guarantee or camera-throughput claim. No
deployment performance or reading-accuracy claim follows from them.

## Evidence identity and reproduction

The underlying code HEAD during measurement was
`eb2ea5df9bc6e784fb0126155cce61fceda127dd`, plus the new benchmark script later
committed with this report. Its recorded SHA-256 is
`8ca0f866c99e27412c51cb9b2ee6be4eb2b035b11d926c1506f0ff0bfc3c19ac`.
The [data card](data_card.md) records dataset fingerprint
`2c33c828642ba484c4698e561df47193975f491291718a44c2d8d853e0c94a4a`.
The ordered validation inventory digest (SHA-256 over concatenated per-image
SHA-256 digests, not filenames/text) is
`dd972bc9682e75b18a996760c5acb9c71d09e7d4a25b0dd46b68a6ff67742e80`.

| Local timing artifact, excluded from Git | SHA-256 |
| --- | --- |
| `outputs/anpr/g4_latency_mps_v1.json` | `b4026605f2c8ae910a00b2ab8d59d968c688f27722a8856b76dd2380e8dc8582` |
| `outputs/anpr/g4_latency_cpu_v1.json` | `3a6fe757a57301522990097adb9f6f1c96d4129789e7775eae287767d6a967fe` |

These JSON files contain timing samples and runtime/input/model identity only,
with no images, boxes or OCR/registration text. Fresh reproduction requires the
existing permitted local assets and an available MPS device outside a sandbox.
Choose new output names; existing files cannot be overwritten. From the
repository root, run the commands sequentially:

```bash
caffeinate -dims .venv/bin/python scripts/benchmark_anpr_latency.py \
  --device mps --output outputs/anpr/g4_latency_mps_reproduction_v2.json
caffeinate -dims .venv/bin/python scripts/benchmark_anpr_latency.py \
  --device cpu --output outputs/anpr/g4_latency_cpu_reproduction_v2.json
```

`--image` may select a smaller validation-only subset; such a run must not be
presented as the complete sample reported above. If MPS or assets are unavailable,
stop and report that condition rather than install, download or fall back.
The focused [asset-free regression tests](../../tests/test_ci_anpr_latency.py)
cover test-split refusal before image/model access, aliases/traversal, output
preservation, warm-up exclusion, reading redaction and completion-aware timing.
No images or readings were exported, no training was run, and the ANPR held-out
test split remains untouched.
