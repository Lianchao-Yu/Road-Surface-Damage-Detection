Phase 2 — PC GPU Systems -Asynchronous Video Pipeline / GPU Systems Optimization


Async pipeline + Buffer reuse + encoder thread + Nsight。

优先顺序是：
1. Bounded Queue + 独立 Encoder Thread
   - GPU 完成 Frame N 推理后，不再等待 CPU 完成 Frame N 编码。
   - Encoder Thread 独立消费已经处理好的 Frame。
   - 这是现在最有可能显著改善完整链吞吐的改动。
2. Double / Triple Buffering + Pipeline Slots
   - 每个 slot 保存 image buffer、device input、TRT output、metadata。
   - Frame N GPU 推理时，CPU 可以准备 Frame N+1。
   - 避免每帧重复申请/释放资源。
3. Pinned Host Memory + Async H2D
   - 你现在 H2D 还是 pageable memory。
   - 尤其 1080p 下，报告已经看到源图上传影响增大。
4. CUDA Events / Nsight Systems
   - 真正证明 CPU encode、H2D、K1、TensorRT 是否发生重叠。
   - 重点看 GPU utilization、空洞、queue depth、P95 latency。
5. 之后再考虑 Multi-Batch
   - 对单路实时视频，Batching 可能提高吞吐但增加单帧等待。
   - 更适合多流或离线视频。
   - 目前不是第一优先级


Phase 3 prospect:迁移部署至-Nvidia Orin NX