# Phase 1 Video Pipeline Performance Analysis: CPU vs GPU Preprocessing

**Date:** 2026-11-24  
**Scope:** Current single-stream video inference pipeline with software video decode and software video encode. No code changes were made for this measurement.

## **1. Test Setup**

| Item | Configuration |
|---|---|
| CPU | Intel Core i7-12700H, 10 cores / 20 threads |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, driver 621.14 |
| Platform | WSL2, Ubuntu, CUDA 13.5, TensorRT 11.4 |
| Video decode | FFmpeg `libavcodec`, software H.264 |
| Video encode | OpenCV `VideoWriter`, software `mpeg4` |
| Model | `models/best.engine` (YOLO, 4 classes) |
| CPU preprocess | Hand-written `preprocess_bgr8_to_nchw_f32_letterbox` |
| GPU preprocess | `launchFusedPreprocess` |
| Input | `dataset/testVideo/test1.mp4`, 720x1280, 29 fps, 157 frames |
| Measurement | 10 warmup frames, 147 measured frames, median latency |

`pipeline` excludes video encoding; `total` includes software encoding.

## **2. Stage Latency**

All values are milliseconds.

| Resolution | Mode | Decode | Preprocess | Infer | Post | Draw | Pipeline | Encode | Total | FPS |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 720x1280 | CPU | 0.51 | 5.01 | 4.28 | 0.03 | 0.47 | 10.39 | 5.90 | 16.42 | 60.1 |
| 720x1280 | GPU | 0.48 | `0.39` | 3.45 | 0.03 | 0.45 | 4.94 | 5.84 | **10.95** | `89.0` |
| 1280x720 | CPU | 0.45 | 4.95 | 4.26 | 0.03 | 0.42 | 10.24 | 5.11 | 15.53 | 63.4 |
| 1280x720 | GPU | 0.44 | `0.37` | 3.41 | 0.03 | 0.38 | 4.81 | 5.15 | **9.98** | `97.8` |
| 1920x1080 | CPU | 0.97 | 5.05 | 5.03 | 0.03 | 0.82 | 11.90 | 11.18 | 23.36 | 42.8 |
| 1920x1080 | GPU | 0.96 | `0.88` | 4.17 | 0.03 | 0.81 | 7.08 | 11.29 | **18.54** | `53.9` |

## **3. Speedup Summary**

| Resolution | Preprocess | Pipeline (no encode) | Total (with soft encode) |
|---|---:|---:|---:|
| 720x1280 | 12.9x | 2.10x | 1.50x |
| 1280x720 | 13.2x | 2.13x | 1.56x |
| 1920x1080 | 5.7x | 1.68x | 1.26x |

## **4. Detection Consistency**

| Resolution | Frames with matching classes | Max score delta | Box deviation |
|---|---:|---:|---|
| 720x1280 | 147 / 147 | 2.8e-4 | One large D20 box shifted 13.64 px (IoU 0.95) |
| 1280x720 | 147 / 147 | 9.0e-4 | <= 0.10 px |
| 1920x1080 | 146 / 147 | 5.5e-4 | <= 0.15 px; one frame had +/-1 detection |

The 720x1280 outlier has the same class and nearly identical score (0.266045 vs 0.266021).
It is consistent with a near-tie candidate selection after a very small CPU/GPU input difference.

## **5. Observations and Analysis**

### Where the gain comes from.
GPU preprocessing is 5.7x-13.2x faster than the hand-written CPU path, but the end-to-end gain is smaller: 1.68x-2.13x without encoding and 1.26x-1.56x with software encoding. The gain has two sources: the fused kernel removes about 4.2-4.6 ms of CPU preprocessing per frame, and it writes directly into the TensorRT input buffer, so `TensorInfer` no longer uploads the 4.9 MB NCHW tensor. The second effect is visible in the infer stage (4.17-5.03 ms CPU vs 3.41-4.17 ms GPU). At 1080p the preprocessing speedup falls from about 13x to 5.7x because the larger source frame still has to cross PCIe through pageable H2D memory.

### **Bottleneck.** 
For the phase-1 chain, software encoding is the dominant cost at 1080p: `11.18-11.29 ms` per frame, or 48-61% of total latency, compared with `7.08 ms` for the entire GPU inference pipeline. At 720p/720x1280, encoding costs `5.1-5.9 ms` and still accounts for roughly one third to one half of total time. Software decode is only `0.45-0.97 ms` and is not a bottleneck. Inside the GPU pipeline, TensorRT inference `(3.45-4.17 ms)` is now the largest stage; preprocessing is down to 0.37-0.88 ms. The remaining structural limit is that the loop is fully sequential: CPU decode, drawing and encoding cannot overlap GPU work.

### **Future optimization directions.**
- The highest-leverage next step for the full output chain is not another preprocessing kernel; it is overlapping the CPU software encoder with the GPU pipeline via a bounded queue and a dedicated encoder thread, with double/triple buffering. 
- Pinned host memory and asynchronous H2D would reduce transfer cost, especially for 1080p source upload, and reusable pipeline slots would remove allocation jitter. 
- If inference becomes the limit after encoding is overlapped, the next options are FP16/INT8 engine calibration, GPU-side post-processing, and batching across streams. NVDEC is only justified if decode becomes significant under multi-stream or 4K load; 
- The current software decode baseline is already fast enough. NVENC can be evaluated later if the output policy permits hardware encoding, while phase 1 keeps both decode and encode in software for a fair baseline.

## 6. Caveats

- Measurements are from WSL2, a single stream, and a sequential loop.
- The 1280x720 and 1920x1080 inputs were temporary transcodes of the 720x1280 source.
  Stage timings are representative for resolution scaling, but detection results are
  not directly comparable across resolutions.
- H2D uses pageable host memory; pinned memory and asynchronous buffering are not yet implemented.
- The OpenCV `mpeg4` software encoder is a baseline, not an optimized production encoder.
