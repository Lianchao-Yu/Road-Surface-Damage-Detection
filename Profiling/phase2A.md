# Phase 2A Async Video Pipeline Performance Analysis

**Date:** 2026-11-27

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
| Measurement | 10 warmup frames, 3 runs, wall-clock after initialization |

The async pipeline uses:

```text
Decoder
  -> free_queue -> infer_queue
GPU worker: H2D -> preprocess -> TensorRT -> postprocess -> draw
  -> encode_queue
Encoder
  -> free_queue
```

`FrameSlot` owns the host BGR frame, device staging buffer, metadata and stage
timings. TensorRT input/output buffers remain backend-owned and are reused.

## **2. Pinned Memory / H2D Path**

The async path does use pinned host memory:

| Buffer | Allocation | Current use |
|---|---|---|
| `FrameSlot::h_bgr` | `cudaMallocHost` | Decoder copies BGR frame into it |
| `FrameSlot::d_bgr` | `cudaMalloc` | H2D destination / preprocess input |
| TensorRT input | backend `cudaMalloc` | Preprocess output |

Therefore the H2D source is pinned, not pageable:

```text
FFmpeg pageable frame
  -> CPU memcpy into pinned FrameSlot::h_bgr
  -> cudaMemcpyAsync to FrameSlot::d_bgr
```

There is still one CPU copy from the FFmpeg decode buffer into the pinned slot.
This is not a pageable-H2D problem; it is a remaining copy path. Future work could
decode directly into a pinned slot or use NVDEC to avoid it.

## **3. End-to-End Results With Encoding**

Wall-clock throughput is computed from after pipeline initialization and
includes all frames, including the 10 warmup frames. Throughput varied by roughly
5-10% between runs on WSL2; ranges are shown where needed.

| Pipeline | Throughput | E2E p50 | E2E p95 | Speedup |
|---|---:|---:|---:|---:|
| Sync | 80-85 fps | ~10.8 ms | ~13.1 ms | 1.00x |
| Async, 2 slots | 121-125 fps | ~13.3 ms | ~18.0 ms | **~1.45-1.57x** |
| Async, 3 slots | 126-137 fps | ~18.9 ms | ~22.8 ms | **~1.55-1.61x** |
| Async, 4 slots | 133-138 fps | ~24.8 ms | ~28.0 ms | **~1.56-1.66x** |

Representative stage medians:

| Pipeline | Decode | Infer | Encode | Effective wall ms/frame |
|---|---:|---:|---:|---:|
| Sync | ~0.51 ms | ~3.89 ms | ~6.05 ms | ~12 ms |
| Async, 2 slots | ~0.78 ms | ~3.50 ms | ~6.00 ms | ~8.0 ms |

The sync pipeline sums the stages; the async pipeline overlaps encoding with
GPU work:

```text
max(GPU pipeline, encode) + queue/launch overhead
```

## **4. Slot Count Tradeoff**

| Slots | Throughput | E2E p50 | E2E p95 | Use Case |
|---:|---:|---:|---:|---|
| 2 | 121-125 fps | ~13.3 ms | ~18.0 ms | Real-time / low latency |
| 3 | 126-137 fps | ~18.9 ms | ~22.8 ms | Balanced |
| 4 | 133-138 fps | ~24.8 ms | ~28.0 ms | Offline maximum throughput |

More slots add queue depth and increase end-to-end latency. Two slots already
deliver most of the gain while keeping p50 latency close to the synchronous
baseline.

## **5. Results Without Encoding**

Without encoding, the gain is small and noisier:

| Pipeline | Throughput (approx.) | E2E p50 | Speedup |
|---|---:|---:|---:|
| Sync | 165-186 fps | 4.3-4.8 ms | 1.00x |
| Async, 2 slots | ~197 fps | 7.3-7.7 ms | ~1.06-1.19x |
| Async, 4 slots | 175-208 fps | 14.4-16.8 ms | ~1.06-1.12x |

This confirms that the major Phase 2A gain comes from encoder overlap, not from
decoder overlap.

## **6. Detection and Output Validation**

| Check | Result |
|---|---|
| Sync output | 157 frames, 720x1280, `mpeg4` |
| Async output | 157 frames, 720x1280, `mpeg4` |
| Detection count | 97 detections over 157 frames for both paths |
| EOF handling | Async drained all 157 frames and exited cleanly |
| `--max-frames` | Verified with 50 frames / 50 output frames |

Current validation is aggregate only. A per-frame comparison of class,
confidence, bounding box and frame ID is still pending:

```text
distance(frame_id, class_id, score, x1, y1, x2, y2)
```

The matching detection count is not sufficient to claim full per-frame
equivalence.

## **7. Observations and Analysis**

### Where the gain comes from.
The real gain comes from overlapping software encoding with GPU inference. In
the synchronous loop the frame cost is approximately:

```text
decode + preprocess + infer + postprocess + draw + encode
```

In the async loop the encoder consumes completed frames from `encode_queue`
while the GPU worker processes the next frame. Wall-clock throughput improves
from about 80-85 fps to 121-138 fps, or roughly 1.45x-1.66x depending on slot
count.

### Bottleneck.
After encoding is overlapped, the GPU worker remains the next limit:
preprocess and inference are still serialized inside one worker, and inference
is serialized by a single TensorRT execution context. The no-encoding
experiment shows only a small gain, confirming that encoding was the main
bottleneck. Software decode remains about 0.8 ms/frame and is not limiting.

### Latency tradeoff.
Async throughput improves at the cost of queueing latency. With 2 slots p50 is
about 13.3 ms; with 4 slots it rises to about 24.8 ms because more frames are
buffered. For live video, 2 slots is the better default. For offline processing,
3-4 slots provide slightly higher throughput.

### Future optimization directions.
The next useful steps are to remove the remaining CPU copy into the pinned slot,
tune queue capacities, and measure GPU idle time with Nsight. True four-stage
overlap (decode, preprocess, infer, postprocess) would require per-slot `d_in`,
a separate preprocess stream/event, per-slot output storage, and an asynchronous
TensorRT API; its expected gain is smaller because preprocess is about 0.4-0.9 ms
and postprocess about 0.03 ms. NVDEC/NVENC remain optional later experiments.

## **8. FrameSlot Lifecycle and Thread-Safety Audit**

Current ownership and synchronization:

- Slot objects are owned by `slot_storage`.
- Queues carry non-owning `FrameSlot*` pointers.
- Flow: `free_queue -> infer_queue -> encode_queue -> free_queue`.
- Each queue has a mutex/condition variable and a single producer/consumer.
- One GPU worker means one `TensorRTBackend` context, so inference calls are
  serialized.
- `stop_requested`/`failed` are atomics; statistics and error strings are mutex
  protected.
- Worker threads are joined before slots, source, writer and backend resources
  are released.

Remaining risks:

- The public `init()/run()/stop()` API is not safe for concurrent control calls.
  One control thread must own the lifecycle.
- A blocking `source->read()` cannot be cancelled by `stop()`; MP4 is fine, RTSP
  will need an interrupt callback/timeout.
- Worker exceptions are not propagated; errors are reported through
  `failed/last_err`.
- No explicit slot-state assertions or debug checks exist yet.
- No per-slot CUDA event is needed now because `TensorInfer` synchronizes, but
  one would be required for future asynchronous inference.

## **9. P0 Checklist**

| Task | Status | Notes |
|---|---|---|
| Pinned memory / async H2D | Done | `FrameSlot::h_bgr` uses `cudaMallocHost`; H2D source is pinned. A separate FFmpeg-to-pinned CPU copy remains |
| Uniform benchmark measurement | Done | Sync app now reports wall-clock after init; async uses wall-clock after init |
| Nsight Systems timeline | Done | `Nsight/async_pipeline_2slots.nsys-rep` (2026.5.1-compatible); Nsight Compute blocked by counter permission |
| FrameSlot lifecycle / thread safety | Partial | Ownership and join order audited; stress tests and state assertions pending |
| Per-frame correctness comparison | Pending | Aggregate count matches; compare class/score/box/frame_id per frame |
| Queue latency tuning | Pending | 2 slots recommended for latency; tune queue capacities |

## **10. Caveats**

- Measurements are from WSL2, single-stream video, and software decode/encode.
- Throughput varies by roughly 5-10% between runs; ranges are reported.
- H2D source is pinned, but the decode-to-pinned copy remains.
- One TensorRT execution context is used; inference calls are serialized.
- The pipeline is three-stage overlap, not full four-stage overlap.
- No RTSP, NVDEC, NVENC, or multi-stream batching is included in this phase.

## **11. Next Steps**

1. Add per-frame detection comparison and record the result in this document.
2. Replace the FFmpeg-to-pinned copy with a decode-into-pinned path if profiling
   shows it matters.
3. Add slot-state assertions and queue stress tests.
4. Use 2 slots for latency-sensitive runs and 3-4 slots for offline throughput.
5. Only then evaluate per-slot TensorRT buffers and separate
   preprocess/postprocess workers.
