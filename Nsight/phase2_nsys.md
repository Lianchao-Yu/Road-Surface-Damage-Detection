# Phase 2 Nsight Systems Analysis

**Date:** 2026-11-26

## 1. Generated Files

| File | Version | Purpose |
|---|---|---|
| `Nsight/async_pipeline_2slots.nsys-rep` | 2026.5.1 | Compatible report; open with the current 2026.5.1 GUI |
| `Nsight/async_pipeline_2slots_nsys_stats.txt` | 2026.5.1 | Text export for the compatible report |
| `Nsight/async_pipeline_2slots_2026.5.2.nsys-rep` | 2026.5.2 | Newer report; requires a 2026.5.2+ GUI |
| `Nsight/async_pipeline_2slots_2026.5.2_nsys_stats.txt` | 2026.5.2 | Text export for the newer report |

The default report `async_pipeline_2slots.nsys-rep` was regenerated with the
matching Nsight Systems 2026.5.1 CLI so it can be opened by the installed
2026.5.1 GUI.

## 2. Capture Command

```bash
/tmp/opencode/nsight-2026.5.1/opt/nvidia/nsight-systems/2026.5.1/bin/nsys profile \
  --trace=cuda,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=/tmp/opencode/async_nsys_2026_5_1 \
  ./build/cuda-trt/road_damage_async \
    --video dataset/testVideo/test1.mp4 \
    --max-frames 50 --warmup 10 --slots 2 \
    --out /tmp/opencode/async_nsys_2026_5_1.mp4
```

The captured run contains 50 frames at 720x1280, 2 pipeline slots, software
H.264 decode and software `mpeg4` encode.

## 3. How to Inspect

Nsight Systems GUI is not available in WSL. Download `async_pipeline_2slots.nsys-rep`
and open it on Windows:

```text
File -> Open -> async_pipeline_2slots.nsys-rep
```

Recommended views:

- CUDA HW: kernel execution and H2D/D2H operations
- Threads: decoder, inference worker and encoder activity
- CUDA API: `cudaMemcpyAsync`, `cudaStreamSynchronize`, kernel launches
- Zoom into steady-state after startup to inspect encoder overlap

NVTX ranges are not inserted yet, so stage names must be inferred from thread and
CUDA activity.

## 4. Initial Findings

| Metric | Observation |
|---|---|
| Preprocess kernel | 50 instances, 0.986 ms total, 19.7 us average, 0.9% of GPU kernel time |
| GPU kernel work | Dominated by TensorRT convolution/GEMM kernels |
| H2D | 52 operations, 22.6 ms total, 407 us median |
| D2H | 51 operations, 1.55 ms total |
| CUDA kernel launches | 4004 launches for 50 frames, roughly 80 per frame |
| Stream synchronizations | 56 calls; inference synchronizes once per frame |
| OS runtime | 89.3% in `pthread_cond_wait`, consistent with queue backpressure |

Interpretation:

- The fused preprocess kernel is negligible in the GPU timeline; TensorRT
  inference dominates.
- H2D is visible but much smaller than the inference work in this run.
- One stream synchronization per frame confirms that the GPU worker still
  serializes preprocess -> inference -> postprocess. The asynchronous gain comes
  from the independent encoder thread, matching the Phase 2 benchmark.
- Queue blocking (`pthread_cond_wait`) is expected because software encoding is
  the slowest stage when encoding is enabled.

## 5. Nsight Compute

Nsight Compute is installed, but GPU performance counters are not available:

```text
ERR_NVGPUCTRPERM: permission to access NVIDIA GPU Performance Counters
```

No `.ncu-rep` was generated. Nsight Systems is sufficient for timeline and
overlap analysis. Nsight Compute requires enabling GPU performance counters on
the host driver and is intended for per-kernel metrics rather than pipeline
overlap.

## 6. Caveats

- The report includes process startup, engine loading and TensorRT context
  initialization; zoom to the steady-state region.
- No NVTX ranges are present yet.
- The capture is 50 frames; use a longer capture for queue behavior and thermal
  effects.
- The `.nsys-rep` files are binary and local; they do not need to be committed.
