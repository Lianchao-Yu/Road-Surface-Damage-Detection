# Phase 2 Async Video Pipeline Performance Analysis

**Date:** 2026-11-26

**Scope:** Three-stage asynchronous overlap for the single-stream YOLO pipeline: decode, GPU worker (preprocess + inference + postprocess), and encode. Video decode and encode remain software implementations.

## **1. Test Setup**

| Item | Configuration |
|---|---|
| CPU | Intel Core i7-12700H, 10 cores / 20 threads |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, driver 621.14 |
| Platform | WSL2, Ubuntu, CUDA 13.5, TensorRT 11.4 |
| Video decode | FFmpeg `libavcodec`, software H.264 |
| Video encode | OpenCV `VideoWriter`, software `mpeg4` |
| Model | `models/best.engine` (YOLO, 4 classes) |
| Preprocess | `launchFusedPreprocess`, output directly into TensorRT input buffer |
| Async implementation | `rd_pipeline`, `road_damage_async` |
| Sync baseline | `road_damage_video` |
| Input | `dataset/testVideo/test1.mp4`, 720x1280, 29 fps, 157 frames |
| Measurement | 10 warmup frames, 3 runs, median reported |

The async pipeline uses:

```text
Decoder
  -> free_queue -> infer_queue
GPU worker: H2D -> preprocess -> TensorRT -> postprocess -> draw
  -> encode_queue
Encoder
  -> free_queue
```

`FrameSlot` owns the pinned host BGR frame, device staging buffer, metadata and
stage timings. TensorRT input/output buffers remain backend-owned and are reused.

## **2. End-to-End Results With Encoding**

| Pipeline | Throughput | E2E p50 | E2E p95 | Speedup |
|---|---:|---:|---:|---:|
| Sync | 87.07 fps | 11.31 ms | 13.31 ms | 1.00x |
| Async, 2 slots | **131.76 fps** | 12.67 ms | 17.06 ms | **1.51x** |
| Async, 3 slots | **135.10 fps** | 18.64 ms | 22.92 ms | **1.55x** |
| Async, 4 slots | **136.17 fps** | 24.94 ms | 28.92 ms | **1.56x** |

Representative stage medians:

| Pipeline | Decode | Infer | Encode | Effective ms/frame |
|---|---:|---:|---:|---:|
| Sync | ~0.51 ms | ~3.89 ms | ~6.05 ms | 11.49 ms |
| Async, 2 slots | ~0.78 ms | ~3.50 ms | ~6.00 ms | **7.59 ms** |

The sync pipeline sums the stages; the async pipeline overlaps encoding with
GPU work, so its effective frame cost is closer to:

```text
max(GPU pipeline, encode) + queue/launch overhead
```

## **3. Slot Count Tradeoff**

| Slots | Throughput | E2E p50 | E2E p95 | Use Case |
|---:|---:|---:|---:|---|
| 2 | 131.76 fps | 12.67 ms | 17.06 ms | Real-time / low latency |
| 3 | 135.10 fps | 18.64 ms | 22.92 ms | Balanced |
| 4 | 136.17 fps | 24.94 ms | 28.92 ms | Offline maximum throughput |

More slots add queue depth and therefore increase end-to-end latency.
Two slots already deliver most of the throughput gain while keeping latency
close to the synchronous baseline.

## **4. Results Without Encoding**

| Pipeline | Throughput | E2E p50 | Speedup |
|---|---:|---:|---:|
| Sync | 195.50 fps | 4.73 ms | 1.00x |
| Async, 2 slots | 207.95 fps | 7.27 ms | 1.06x |
| Async, 4 slots | 211.37 fps | 14.44 ms | 1.08x |

Without encoding, the pipeline is dominated by decode + GPU inference, and the
queue overlap yields only a small throughput improvement.

## **5. Detection and Output Validation**

| Check | Result |
|---|---|
| Sync output | 157 frames, 720x1280, `mpeg4` |
| Async output | 157 frames, 720x1280, `mpeg4` |
| Detection count | 97 detections over 157 frames for both paths |
| EOF handling | Async drained all 157 frames and exited cleanly |
| `--max-frames` | Verified with 50 frames / 50 output frames |

The async path uses the same preprocess, TensorRT engine and postprocess code as
the synchronous path, and produces the same detection count.

## **6. Observations and Analysis**

### Where the gain comes from.
The real gain comes from overlapping software encoding with GPU inference.
In the synchronous loop the frame cost is approximately:

```text
decode + preprocess + infer + postprocess + draw + encode
```

In the async loop the encoder consumes completed frames from `encode_queue`
while the GPU worker processes the next frame. With encoding enabled, throughput
improves from `87.07 fps` to `131.76-136.17 fps` (`1.51x-1.56x`). The encoder is
the largest single stage at about `6 ms`, so this overlap is the primary reason
for the improvement.

### Bottleneck.
After encoding is overlapped, the GPU worker remains the next limit:
preprocess and inference are still serialized inside one worker, and inference
is serialized by a single TensorRT execution context. The no-encoding experiment
shows only `1.06x-1.08x` gain, confirming that the original bottleneck was
encoding rather than decoding. Software decode remains about `0.78 ms/frame` and
is not a limiting stage.

### Latency tradeoff.
Async throughput improves at the cost of queueing latency. With 2 slots the p50
latency is `12.67 ms`, close to the sync total of `11.31 ms`; with 4 slots the p50
rises to `24.94 ms` because more frames are buffered in the pipeline. For live
video, 2 slots is the better default. For offline processing, 3-4 slots provide
slightly higher throughput.

### Future optimization directions.
The next useful step is to reduce H2D overhead with pinned-memory transfers and
reusable staging buffers. True four-stage overlap (decode, preprocess, infer,
postprocess) would require per-slot `d_in`, a separate preprocess stream/event,
per-slot output storage, and an asynchronous TensorRT API; its expected gain is
smaller because preprocess is only about `0.4-0.9 ms` and postprocess about
`0.03 ms`. Such a split should only be implemented after profiling shows a
measurable GPU idle gap. NVDEC/NVENC remain optional later experiments; the
current software decode path is not the bottleneck.

## **7. Caveats**

- Measurements are from WSL2, single-stream video, and software decode/encode.
- Async throughput is computed as encoded frames divided by wall time after
  initialization; sync throughput is computed from measured per-frame totals.
- H2D still uses pageable host memory.
- One TensorRT execution context is used; inference calls are serialized.
- The pipeline is three-stage overlap, not full four-stage overlap.
- No RTSP, NVDEC, NVENC, or multi-stream batching is included in this phase.

## **8. Next Steps**

1. Use 2 slots as the default for latency-sensitive runs and 3-4 slots for
   offline throughput.
2. Add pinned host memory and reusable device staging buffers.
3. Tune queue capacities and measure queue depth / GPU idle time.
4. Only then evaluate per-slot TensorRT input/output buffers and separate
   preprocess/postprocess workers.
